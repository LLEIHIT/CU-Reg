import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import time
import sys
import os
import tools
from copy import copy
import numpy as np
from torch.nn import init, Sequential
import cv2

from networks.resnext import resnext50_32x4d
from networks.resnext_fvr_ori import ResNeXtBottleneck, downsample_basic_block
from networks.cnn.pspnet import Modified_PSPNet as PSPNet
from networks.layers import MLP, MLP_layer
import networks.torch_utils as pt_utils

psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
}


# Copy from PVN3D: https://github.com/ethnhe/PVN3D
class DenseFusion(nn.Module):
    def __init__(self, num_points=1024):
        super(DenseFusion, self).__init__()
        self.conv2_rgb = torch.nn.Conv1d(64, 256, 1)
        self.conv2_cld = torch.nn.Conv1d(64, 256, 1)

        self.conv3 = torch.nn.Conv1d(128, 512, 1)
        self.conv4 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)

    def forward(self, rgb_emb, cld_emb):
        bs, _, n_pts = cld_emb.size()
        feat_1 = torch.cat((rgb_emb, cld_emb), dim=1)
        rgb = F.relu(self.conv2_rgb(rgb_emb))
        cld = F.relu(self.conv2_cld(cld_emb))

        feat_2 = torch.cat((rgb, cld), dim=1)

        rgbd = F.relu(self.conv3(feat_1))
        rgbd = F.relu(self.conv4(rgbd))

        ap_x = self.ap1(rgbd)

        feat_3 = ap_x.view(-1, 1024, 1).repeat(1, 1, n_pts)

        fused = torch.cat([feat_1, feat_2, feat_3], 1)  # 256+ 512 + 1024 = 1792
        
        return fused

class NLBlockND_cross(nn.Module):
    # Our implementation of the attention block referenced https://github.com/tea1528/Non-Local-NN-Pytorch

    def __init__(self, in_channels, inter_channels=None, mode='embedded',
                 dimension=3, bn_layer=True):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(NLBlockND_cross, self).__init__()

        assert dimension in [1, 2, 3]

        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        ######## gated ########
        self.to_hidden = nn.Sequential(
            nn.Linear(self.inter_channels, self.inter_channels),
            nn.SiLU()
        ) 

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                bn(self.in_channels)
            )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                nn.ReLU()
            )

    def forward(self, x_thisBranch, x_otherBranch):
        #x_thisBranch for g and theta
        #x_otherBranch for phi
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """
        # print(x_thisBranch.shape)

        batch_size = x_thisBranch.size(0)

        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        # g_x = self.g(x_thisBranch).view(batch_size, self.inter_channels, -1)
        g_x = self.g(x_otherBranch).view(batch_size, self.inter_channels, -1).contiguous()
        g_x = g_x.permute(0, 2, 1).contiguous()
        # print("g_x: ", g_x.shape)
        #### gated #####
        g_x, gate = self.to_hidden(x_otherBranch).chunk(2, dim = 1)
        g_x = g_x.view(batch_size, self.inter_channels, -1).contiguous().permute(0, 2, 1).contiguous()
        # print("g_x: and gate: ", g_x.shape, gate.shape)

        if self.mode == "gaussian":
            theta_x = x_thisBranch.view(batch_size, self.in_channels, -1)
            phi_x = x_otherBranch.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x_thisBranch).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x_otherBranch).view(batch_size, self.inter_channels, -1)
            # theta_x = theta_x.permute(0, 2, 1)
            phi_x = phi_x.permute(0, 2, 1)
            f = torch.matmul(phi_x, theta_x)

        # elif self.mode == "concatenate":
        else: #default as concatenate
            theta_x = self.theta(x_thisBranch).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x_otherBranch).view(batch_size, self.inter_channels, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))

        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1)  # number of position in x
            f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)

        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x_thisBranch.size()[2:])

        y = y * gate ### gated

        W_y = self.W_z(y)
        # residual connection
        z = W_y + x_thisBranch

        return z


class SelfAttention(nn.Module):
    """
     Multi-head masked self-attention layer
    """

    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SelfAttention, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        # key, query, value projections for all heads
        self.que_proj = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj = nn.Linear(d_model, h * self.d_v)  # value projection
        self.out_proj = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''

        b_s, nq = x.shape[:2]
        nk = x.shape[1]
        q = self.que_proj(x).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.key_proj(x).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v = self.val_proj(x).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        # Self-Attention
        #  :math:`(\text(Attention(Q,K,V) = Softmax((Q*K^T)/\sqrt(d_k))`
        # print("shape of q and k and d_k: ", q.shape, k.shape, self.d_k)
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        # weight and mask
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        # get attention matrix
        att = torch.softmax(att, -1)
        att = self.attn_drop(att)

        # output
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.resid_drop(self.out_proj(out))  # (b_s, nq, d_model)

        return out


class myTransformerBlock(nn.Module):
    """ Transformer block """

    def __init__(self, d_model, d_k, d_v, h=4, block_exp=2, attn_pdrop=0.1, resid_pdrop=0.1):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)
        """
        super().__init__()
        self.inter_channels = d_model
        # function g in the paper which goes through conv. with kernel size 1
        self.g = nn.Conv3d(in_channels=d_model, out_channels=d_model, kernel_size=1)

        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.sa = SelfAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            # nn.SiLU(),  # changed from GELU
            nn.GELU(),  # changed from GELU
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        ## x [b, c, T, H, W]
        batch_size, c, t, h, w = x.size()
        if x.dim() > 3:
            x = self.g(x).view(batch_size, self.inter_channels, -1)
            x = x.permute(0, 2, 1)
        bs, nx, c = x.size()

        # print("shape: ", x.shape)

        x = x + self.sa(self.ln_input(x))
        x = x + self.mlp(self.ln_output(x))

        x = x.permute(0, 2, 1).contiguous()
        x = x.view(batch_size, self.inter_channels, t, h, w)

        return x

class RegistNetwork(nn.Module):
    """ First working model! """
    def __init__(self, layers):
        self.inplanes = 128
        super(RegistNetwork, self).__init__()
        """ Balance """
        layers = layers
        self.TransLayers = 4
        self.bn_momentum = 0.75

        # frame feature
        self.frameemb = psp_models['resnet34'.lower()]()
        # layers = [3, 4, 6, 3]  # resnext50
        # layers = [3, 4, 23, 3]  # resnext101
        # layers = [3, 8, 36, 3]  # resnext150
        self.conv1_vol = nn.Conv3d(1, 32, kernel_size=9, stride=(2, 2, 2), padding=(4, 4, 4), bias=False)
        self.conv2_vol = nn.Conv3d(32, 32, kernel_size=7, stride=(2, 1, 1), padding=(3, 3, 3), bias=False)
        self.conv3_vol = nn.Conv3d(32, 64, kernel_size=3, stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
        # self.conv4_vol = nn.Conv3d(64, 64, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.bn1_vol = nn.BatchNorm3d(32)
        self.bn2_vol = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        # self.conv_pool = nn.Conv3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        # self.conv1_frame = nn.Conv3d(1, 32, kernel_size=9, stride=(2, 2, 2), padding=(4, 4, 4), bias=False)
        # self.conv2_frame = nn.Conv3d(32, 64, kernel_size=7, stride=(2, 1, 1), padding=(3, 3, 3), bias=False)
        # self.conv3_frame = nn.Conv3d(128, 256, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1), bias=False)

        # self.conv2d_frame = nn.Conv2d(1, 3, kernel_size=5, stride=1, padding=2)
        self.conv2d_frame_pre1 = nn.Conv1d(4, 64, 1)
        self.conv2d_frame_pre2 = nn.Conv1d(64, 3, 1)
        self.convdown_frame_1 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.convdown_frame_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        ### prompt feat ###
        self.convdown_prompt_1 = nn.Conv2d(2, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.convdown_prompt_2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        self.framePath_CrossAtt = NLBlockND_cross(64)
        self.volPath_CrossAtt = NLBlockND_cross(64)
        
        ############# feature fusion layer #######
        self.densefusion = DenseFusion(num_points=4096)
        self.reduce_layer = (
            pt_utils.Seq(1664)
            .conv1d(1024, activation=nn.ReLU())
            .conv1d(512,activation=nn.ReLU())
            .conv1d(128,activation=nn.ReLU())
        )

        # transformer
        # self.trans_blocks = nn.Sequential(*[myTransformerBlock(d_model=1792, d_k=1792, d_v=1792, h=4)
        #                                     for layer in range(self.TransLayers)])

        self.layer1 = self._make_layer(
            ResNeXtBottleneck, 128, layers[0], shortcut_type='B', cardinality=32, stride=2)
        self.layer2 = self._make_layer(
            ResNeXtBottleneck, 256, layers[1], shortcut_type='B', cardinality=32, stride=2)
        self.layer3 = self._make_layer(
            ResNeXtBottleneck, 512, layers[2], shortcut_type='B', cardinality=32, stride=2)
        self.layer4 = self._make_layer(
            ResNeXtBottleneck, 1024, layers[3], shortcut_type='B', cardinality=32, stride=2)
        
        # print("layer1: ", self.layer1)
        
        self.avgpool = nn.AvgPool3d((1, 4, 4), stride=1)
        # self.maxpool = nn.MaxPool3d((1, 4, 7), stride=1)

        # self.fc1 = nn.Linear(1024, 512)
        # self.fc2 = nn.Linear(512, 128)      # original
        # self.fc3 = nn.Linear(128, 6)      # original

        ####################### prediction headers #############################
        self.conv1_t = torch.nn.Conv1d(1024, 512, 1)
        self.conv2_t = torch.nn.Conv1d(512, 256, 1)
        self.conv3_t = torch.nn.Conv1d(256, 128, 1)
        self.conv4_t = torch.nn.Conv1d(128, 3, 1)
        # self.mlp5_t = (MLP_layer(in_channels = 4096, units = [1024,64], bn_momentum = self.bn_momentum))
        # self.mlp6_t = MLP(in_channels = 64, out_channels=1, apply_norm = False)

        self.conv1_r = torch.nn.Conv1d(1024, 512, 1)
        self.conv2_r = torch.nn.Conv1d(512, 256, 1)
        self.conv3_r = torch.nn.Conv1d(256, 128, 1)
        self.conv4_r = torch.nn.Conv1d(128, 3, 1)
        
        ###### inter frame translation layer #####
        self.inter_translation_layer = (
            pt_utils.Seq(1024)
            .conv1d(512, activation=nn.ReLU())
            .conv1d(128,activation=nn.ReLU())
            .conv1d(3, activation=None)
        )
        # print(self.inter_translation_layer)

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def volBranch(self, vol):
        print('\n********* Vol *********')
        print('vol {}'.format(vol.shape))
        vol = self.conv1_vol(vol)
        vol = self.bn1_vol(vol)
        vol = self.relu(vol)
        # print('conv1 {}'.format(vol.shape))


        vol = self.conv2_vol(vol)
        vol = self.relu(vol)
        # print('conv2 {}'.format(vol.shape))

        vol = self.relu(self.conv3_vol(vol)) # [[4, 64, 4, 32, 32]


        return vol
    
    def frameBranch(self, frame):
        print('\n********* Frame *********')
        print('frame {}'.format(frame.shape))
        if frame.shape[1] != 4:
            frame = frame.repeat(1,4,1,1,1)
        frame = frame.squeeze(2)
        # print('squeeze {}'.format(frame.shape))
        Bs, c, h, w = frame.shape

        frame = self.conv2d_frame_pre1(frame.view(Bs, c, -1).contiguous()) ## [bs, 64, H, w]
        # print('conv2d_frame {}'.format(frame.shape))

        frame = self.conv2d_frame_pre2(frame) ## [bs, 3, H, w]

        frame = frame.view(Bs, -1, h, w).contiguous()

        frame, seg_feature = self.frameemb(frame)
        print('seg_feature {}'.format( seg_feature[0][1,:,:].unsqueeze(0))) # [bs, 128, 128, 1]
        cv2.imwrite('/home/jun/Desktop/project/slice2volume/FVR-Net/experiments_Prompt_plusInterfplusDFplusGatedNoTF/logs/CAMUS/test.png', seg_feature[0][1,:,:].unsqueeze(0).permute(1,2,0).cpu().numpy()*255.0)
        # sys.exit()
        # print('frameemb {}'.format(frame)) # [bs, 32, 128, 128]
        # print('seg_feature {}'.format(seg_feature.shape)) # [bs, 128, 128, 1]
        # print('seg_feature {}'.format(seg_feature)) # [bs, 128, 128, 1]

        frame = self.relu(self.convdown_frame_1(frame)) # 

        frame = self.relu(self.convdown_frame_2(frame)) # [bs, 64, 64, 64]

        frame = frame.unsqueeze(2)
        
        # print('unsqueeze {}'.format(frame.shape)) ## [bs, 64, 1, 32, 32]

        # frame = self.conv1_frame(frame)
        # frame = self.bn1_vol(frame)
        # frame = self.relu(frame)
        # # print('conv1 {}'.format(frame.shape))

        # frame = self.conv2_frame(frame)
        # frame = self.relu(frame)
        # # print('conv2 {}\n'.format(frame.shape))

        return (frame, seg_feature)

    def forward(self, vol, frame, device=None):
        input_vol = vol.clone()

        ##### vol and frame feature extraction ###
        vol_emb = self.volBranch(vol)  # [bs, 64, 4, 32, 32]
        T_size = vol_emb.size(2)
        frame_emb, seg_feat = self.frameBranch(frame)
        # print('seg_feature {}'.format(seg_feat.shape)) # [bs, 128, 128, 1]
        frame_emb = frame_emb.repeat(1,1,T_size,1,1) ## [bs, 64, 1, 32, 32] - > [bs, 64, 4, 32, 32]

        ### prompt feat ###
        prompt_fea = self.convdown_prompt_1(seg_feat)
        prompt_fea = self.convdown_prompt_2(prompt_fea) ## [bs, 64, 128, 128]
        prompt_fea = prompt_fea.unsqueeze(2)
        prompt_fea = prompt_fea.repeat(1,1,T_size,1,1) ## [bs, 64, 1, 128, 128] - > [bs, 64, 4, 32, 32]

        # print("shape of vol_emb and frame_emb and prompt_fea: ", vol_emb.shape, frame_emb.shape, prompt_fea.shape)

        ### cross attention ###
        frame = self.framePath_CrossAtt(frame_emb, vol_emb)
        vol = self.volPath_CrossAtt(vol_emb, frame_emb)
        

        ###################################  prompt feature ###################################
        Bs, c, t, h, w = frame.shape
        
        ## vol = torch.concat((vol, prompt_fea), dim=1)#[bs, 128, 4, 32, 32]
        ## frame = torch.concat((frame, prompt_fea), dim=1) #[bs, 128, 4, 32, 32]
        # vol = torch.concat((vol, prompt_fea), dim=1).view(Bs, 128, -1).contiguous() #[bs, 128, 4, 32, 32]
        # frame = torch.concat((frame, prompt_fea), dim=1).view(Bs, 128, -1).contiguous() #[bs, 128, 4, 32, 32] ###### 这里后面可以试试 直接 + 法

        vol = (vol + prompt_fea).view(Bs, 64, -1).contiguous() #[bs, 64, 4, 32, 32]
        frame = (frame + prompt_fea).view(Bs, 64, -1).contiguous() #[bs, 64, 4, 32, 32] ###### 这里后面可以试试 直接 + 法
        ###################################  prompt feature ###################################
        
        ###################################  densefusion  ###################################
        
        densefused_feat = self.densefusion(frame, vol).view(Bs,1664,-1).contiguous() #[bs, 1792, 4, 32, 32]
        
        ###################################  densefusion  ###################################

        # densefused_feat = torch.cat((vol, frame), 1).view(Bs,256,-1).contiguous()  # [bs, 256, 4, 32, 32] 
        # densefused_feat = torch.cat((vol, frame), 1)  # [bs, 256, 4, 32, 32] 
        # print('cat {}'.format(x.shape))

        
        # sys.exit()

        #### Transformer encoder ####
        # fused_fea = self.trans_blocks(densefused_feat)# [bs, 256, 4, 32, 32] 
        # fused_fea = self.trans_blocks(densefused_feat).view(Bs,1792,-1).contiguous()  # [bs, 256, 4, 32, 32] 
        fused_fea = self.reduce_layer(densefused_feat).view(Bs,-1,4,32,32).contiguous()


        # print('after transformer {}'.format(fused_fea.shape))

        Bs, C, T, H, W = fused_fea.shape
        # point_feats = fused_fea.view(Bs, C, -1).contiguous()

        ##### attention map #######

        x = self.layer1(fused_fea)
        # print('layer1 {}'.format(x.shape))

        x = self.layer2(x)
        # print('layer2 {}'.format(x.shape))

        x = self.layer3(x)
        # print('layer3 {}'.format(x.shape)) # [bs, c, 1,8,8]

        # x = self.layer4(x)
        # print('layer4 {}'.format(x.shape))

        ##### attention map ####### 

        x = self.avgpool(x)
        # print('avgpool {}'.format(x.shape))

        global_feat = x.view(x.size(0), x.size(1), -1).contiguous() # bs, 1024, 1
        # print('view {}'.format(x.shape))
        # npts = point_feats.shape[2]
        # t_feat = torch.cat((point_feats, global_feat.repeat(1, 1, npts)), dim=1) # BS, 1024+128, npoints
        tx = self.relu(self.conv1_t(global_feat))
        tx = self.relu(self.conv2_t(tx))
        tx = self.relu(self.conv3_t(tx))
        tx = self.conv4_t(tx).view(Bs, 3, 1)
        # tx = self.mlp5_t(tx)
        # tx = self.mlp6_t(tx) # B x 3 x 1
        out_tx = tx.contiguous().transpose(2, 1).contiguous() # B x 1 x 3

        rx = F.relu(self.conv1_r(global_feat))
        rx = F.relu(self.conv2_r(rx))
        rx = F.relu(self.conv3_r(rx))
        # rx = self.conv4_r(rx).view(bs, self.num_obj, self.num_rot, 4)
        out_rx = self.conv4_r(rx).view(Bs, 1, 3).contiguous()

        pred_dof = torch.cat((out_tx, out_rx), dim=2).view(Bs, 6).contiguous()

        pred_interframe_of = self.inter_translation_layer(global_feat).view(
            Bs, 1, 3
        ).contiguous().squeeze(1)
        # pred_interframe_of = torch.zeros(Bs,3).cuda()
        
        normalize_dof = False
        dof_means = [-0.94144243, -0.51833163, -0.24158226, -0.37933836,  0.08427333, -0.01649231]
        dof_std = [11.07899716, 11.08303632, 12.56951074,  5.77583911,  5.68897645, 5.8570513]
        # original
        mat = tools.dof2mat_tensor(input_dof=pred_dof, device=device, normalize_dof = normalize_dof, dof_means= dof_means, dof_std = dof_std)
        # print('mat {}'.format(mat.shape))
        
        # print('input_vol {}'.format(input_vol.shape))
        grid = tools.myAffineGrid2(input_tensor=input_vol, input_mat=mat, 
                                    input_spacing=(1, 1, 1), device=device)
        # grid = grid.to(device)
        # print('grid {}'.format(grid.shape))
        vol_resampled = F.grid_sample(input_vol, grid, align_corners=True)
        # print('resample {}'.format(vol_resampled.shape))
        # print('mat_out {}'.format(x.shape))
        slice_id = int(input_vol.shape[2] / 2)
        indices = torch.tensor([slice_id]).to(device)
        out_frame_tensor_full = torch.index_select(vol_resampled, 2, indices)

        return vol_resampled.detach(), pred_dof, out_frame_tensor_full, pred_interframe_of, seg_feat
    
if __name__ == '__main__':
    device = torch.device("cuda")
    net = RegistNetwork(layers=[3, 4, 6, 3]).cuda()
    # print(net)
    vol = Variable(torch.rand(4, 1, 32, 128, 128)).cuda()
    frame = Variable(torch.rand(4, 1, 1, 128, 128)).cuda()
    vol_resampled, pred_dof, out_frame_tensor_full, pred_interframe_of, seg_feat = net(vol, frame,device=device)
    print(vol_resampled.size())
    print(pred_dof.size())
    print(out_frame_tensor_full.size())
    print(pred_interframe_of)
