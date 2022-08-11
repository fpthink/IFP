from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
global idx
idx=-1
global preOrcur
preOrcur=0
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import etw_pytorch_utils as pt_utils
from collections import namedtuple
import torch.nn.functional as F
from torchvision.utils import make_grid

from pointnet2.utils.pointnet2_modules import PointNet2SAModule, PointNet2FPModule, PointnetProposalModule
from pointnet2.utils.pose_dla_dcn import DLASeg
from pointnet2.utils.fuse import Atten_Fusion_Conv,IA_Layer,Fusion_Conv,Similiar_Fusion_Conv,New_Atten_Fusion_Conv
from pointnet2.utils import vgg
from torch.nn.functional import grid_sample

def conv3x3(in_planes, out_planes, stride = 1):

    """3x3 convolution with padding"""

    return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride,padding = 1, bias = False)

def Feature_Gather(feature_map, xy, mode, padding_mode='zeros'):

    """

    :param xy:(B,N,2)  normalize to [-1,1]

    :param feature_map:(B,C,H,W)

    :return:

    """
    # use grid_sample for this.
    # xy(B,N,2)->(B,1,N,2)
    xy = xy.unsqueeze(1)

    interpolate_feature = grid_sample(feature_map, xy, mode=mode, padding_mode=padding_mode)  # (B,C,1,N)
    return interpolate_feature.squeeze(2) # (B,C,N)

class BasicBlock(nn.Module):

    def __init__(self, inplanes, mid,outplanes, stride = 1,down=True,res=False):

        super(BasicBlock, self).__init__()
        self.res=res
        self.conv1 = conv3x3(inplanes, mid, stride)
        self.bn1 = nn.BatchNorm2d(mid)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(mid, mid, stride)
        self.bn2 = nn.BatchNorm2d(mid)
        self.relu2 = nn.ReLU(inplace=True)

        if down:
            self.conv3 = conv3x3(mid, outplanes, stride * 2)
        else:
            self.conv3 = conv3x3(mid, outplanes, stride)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        return out

class Pointnet_Backbone(nn.Module):
    r"""
        PointNet2 with single-scale grouping
        Semantic segmentation network that uses feature propogation layers

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, input_channels=3, use_xyz=True, opt=None):
        super(Pointnet_Backbone, self).__init__()
        self.opt=opt
        self.grad=[]

        self.DLA=DLASeg('dla15', None,
               pretrained=False,
               down_ratio=2,
               final_kernel=1,
               last_level=4,
               head_conv=None)

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointNet2SAModule(
                radius=0.3,
                nsample=32,
                mlp=[input_channels, 32, 32, 64],
                use_xyz=use_xyz,
                use_edge=False
            )
        )
        self.SA_modules.append(
            PointNet2SAModule(
                radius=None,#0.5
                nsample=48,
                mlp=[64, 64, 64, 128],
                use_xyz=use_xyz,
                use_edge=False
            )
        )
        self.SA_modules.append(
            PointNet2SAModule(
                radius=None,#0.7
                nsample=48,
                mlp=[128, 128, 128, 128],
                use_xyz=use_xyz,
                use_edge=False
            )
        )

        self.pre_offsets = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1,stride=1,padding=0, bias=True),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(128,2,kernel_size=1,stride=1,bias=True))

        self.Fuse_modules=nn.ModuleList()

        self.Fuse_modules.append(New_Atten_Fusion_Conv(64,64,64))
        self.Fuse_modules.append(New_Atten_Fusion_Conv(128,128,128))
        self.Fuse_modules.append(New_Atten_Fusion_Conv(128,128,128))

        self.cov_final = nn.Conv1d(128, 128, kernel_size=1)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, pointcloud, numpoints, image=None, xy=None):
        # type: (Pointnet2SSG, torch.cuda.FloatTensor) -> pt_utils.Seq
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        size_range = [512.0, 160.0]

        xy[:, :, 0] = xy[:, :, 0] / (size_range[0] - 1.0) * 2.0 - 1.0
        xy[:, :, 1] = xy[:, :, 1] / (size_range[1] - 1.0) * 2.0 - 1.0  # = xy / (size_range - 1.) * 2 - 1.
        l_xy_cor = [xy]
        img = [image]
        img_down,img_up= self.DLA(img[0])
        img+= img_down
        for i in range(len(self.SA_modules)):
            li_xyz, li_features,li_index = self.SA_modules[i](l_xyz[i], l_features[i], numpoints[i])
            li_index = li_index.long().unsqueeze(-1).repeat(1, 1, 2)
            li_xy_cor = torch.gather(l_xy_cor[i], 1, li_index)

            img_gather_feature = Feature_Gather(img[i+1], li_xy_cor,mode='bilinear', padding_mode='zeros')
            img_offsets = self.pre_offsets(img_up[0])
            pw_img_offsets = Feature_Gather(img_offsets, li_xy_cor,mode='bilinear', padding_mode='zeros')
            li_features=self.Fuse_modules[i](li_features,img_gather_feature)
            l_xy_cor.append(li_xy_cor)
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        return l_xyz[-1], self.cov_final(l_features[-1]),pw_img_offsets

class Pointnet_Tracking(nn.Module):
    r"""
        xorr the search and the template
    """
    def __init__(self, input_channels=3, use_xyz=True, objective=False, opt=None):
        super(Pointnet_Tracking, self).__init__()

        self.backbone_net = Pointnet_Backbone(input_channels, use_xyz, opt=opt)

        self.cosine = nn.CosineSimilarity(dim=1)

        self.mlp = pt_utils.SharedMLP([4+128,128,128,128], bn=True)

        self.FC_layer_cla = (
                pt_utils.Seq(128)
                .conv1d(128, bn=True)
                .conv1d(128, bn=True)
                .conv1d(1, activation=None))
        self.fea_layer = (pt_utils.Seq(128)
                .conv1d(128, bn=True)
                .conv1d(128, activation=None))
        self.vote_layer = (
                pt_utils.Seq(3+3+128)
                .conv1d(128, bn=True)
                .conv1d(128, bn=True)
                .conv1d(3+128, activation=None))
        self.vote_aggregation = PointNet2SAModule(
                radius=0.3,
                nsample=16,
                mlp=[128, 128, 128, 128],
                use_xyz=use_xyz,
                use_edge=False)
        self.num_proposal = 64
        self.FC_proposal = (
                pt_utils.Seq(128)
                .conv1d(128, bn=True)
                .conv1d(128, bn=True)
                .conv1d(3+1+1, activation=None))

    def xcorr(self, x_label, x_object, template_xyz):       

        B = x_object.size(0)
        f = x_object.size(1)
        n1 = x_object.size(2)
        n2 = x_label.size(2)
        final_out_cla = self.cosine(x_object.unsqueeze(-1).expand(B,f,n1,n2), x_label.unsqueeze(2).expand(B,f,n1,n2))

        fusion_feature = torch.cat((final_out_cla.unsqueeze(1),template_xyz.transpose(1, 2).contiguous().unsqueeze(-1).expand(B,3,n1,n2)),dim = 1)

        fusion_feature = torch.cat((fusion_feature,x_object.unsqueeze(-1).expand(B,f,n1,n2)),dim = 1)

        fusion_feature = self.mlp(fusion_feature)

        fusion_feature = F.max_pool2d(fusion_feature, kernel_size=[fusion_feature.size(2), 1])
        fusion_feature = fusion_feature.squeeze(2)
        fusion_feature = self.fea_layer(fusion_feature)

        return fusion_feature

    def forward(self, template, search,template_image,search_image,gt_2d,sample_2d,f,sample_depth,rot,wratio,hratio):
        r"""
            template: B*512*3 or B*512*6
            search: B*1024*3 or B*1024*6
        """
        template_xyz, template_feature,template_img_offsets = self.backbone_net(template, [256, 128, 64],template_image,gt_2d)

        search_xyz, search_feature,search_img_offsets = self.backbone_net(search, [512, 256, 128],search_image,sample_2d)

        fusion_feature = self.xcorr(search_feature, template_feature, template_xyz)

        estimation_cla = self.FC_layer_cla(fusion_feature).squeeze(1)
        scale=torch.cat([wratio,hratio],dim=1).unsqueeze(dim=-1)
        trans_search_img_offsets=(search_img_offsets*2.0/scale)/(f.unsqueeze(dim=1))*(sample_depth.unsqueeze(dim=1))
        trans_search_img_offsets=torch.cat([trans_search_img_offsets,torch.zeros((trans_search_img_offsets.size(0),1,trans_search_img_offsets.size(2)),dtype=torch.float32,device=trans_search_img_offsets.device)],dim=1)
        trans_search_img_offsets=torch.bmm(rot,trans_search_img_offsets)

        fusion_xyz_feature = torch.cat((search_xyz.transpose(1, 2).contiguous(),trans_search_img_offsets,fusion_feature),dim = 1)

        offset = self.vote_layer(fusion_xyz_feature)
        vote_xyz = (fusion_xyz_feature[:,:3,:] + offset[:,:3,:]).transpose(1, 2).contiguous()
        vote_feature = (fusion_xyz_feature[:,6:,:] + offset[:,3:,:]).contiguous()

        center_xyzs, proposal_features,_ = self.vote_aggregation(vote_xyz, vote_feature, self.num_proposal)

        proposal_offsets = self.FC_proposal(proposal_features)

        estimation_boxs = torch.cat((proposal_offsets[:,0:3,:]+center_xyzs.transpose(1, 2).contiguous(),proposal_offsets[:,3:5,:]),dim=1)
        
        return estimation_cla, vote_xyz, estimation_boxs.transpose(1, 2).contiguous(), center_xyzs,template_img_offsets.transpose(1, 2).contiguous(),search_img_offsets.transpose(1, 2).contiguous()