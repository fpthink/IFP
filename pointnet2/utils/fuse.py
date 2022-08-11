import torch
import torch.nn as nn
import torch.nn.functional as F
class Fusion_Conv(nn.Module):
    def __init__(self, inplanes, outplanes):

        super(Fusion_Conv, self).__init__()

        self.conv1 = torch.nn.Conv1d(inplanes, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)

        self.conv2 = torch.nn.Conv1d(outplanes, outplanes, 1)
        self.bn2 = torch.nn.BatchNorm1d(outplanes)

    def forward(self, point_features, img_features):
        #print(point_features.shape, img_features.shape)
        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))
        fusion_features = F.relu(self.bn2(self.conv2(fusion_features)))

        return fusion_features
class Similiar_Fusion_Conv(nn.Module):
    def __init__(self,point_channels, img_channels,outplanes):

        super(Similiar_Fusion_Conv, self).__init__()
        self.ptoi=nn.Conv1d(point_channels,img_channels,1)
        self.itoi=nn.Conv1d(img_channels,img_channels,1)
        self.simi=nn.CosineSimilarity(dim=1)

        self.conv1 = nn.Sequential(nn.Conv1d(img_channels, point_channels, 1),
                                    nn.BatchNorm1d(point_channels),
                                    nn.ReLU())
        self.conv2 = torch.nn.Conv1d(point_channels + point_channels, outplanes, 1)
        self.bn2 = torch.nn.BatchNorm1d(outplanes)

    def forward(self, point_features, img_features):
        point_feas_f=self.ptoi(point_features)
        point_feas_f=point_feas_f/(torch.norm(point_feas_f,p=2,dim=1,keepdim=True)+1e-8)

        img_feas_f=self.itoi(img_features)
        img_feas_f = img_feas_f / (torch.norm(img_feas_f, p=2, dim=1, keepdim=True)+1e-8)

        similiar=self.simi(point_feas_f,img_feas_f)
        similiar=similiar.unsqueeze(dim=1)
        fuse_p=point_feas_f+similiar*img_feas_f
        fuse_p=self.conv1(fuse_p)
        fusion_features = torch.cat([point_features, fuse_p], dim=1)
        fusion_features = F.relu(self.bn2(self.conv2(fusion_features)))
        return fusion_features

class IA_Layer(nn.Module):

    def __init__(self, channels):

        super(IA_Layer, self).__init__()
        self.ic, self.pc = channels
        rc = self.pc // 4
        self.conv1 = nn.Sequential(nn.Conv1d(self.ic, self.pc, 1),
                                    nn.BatchNorm1d(self.pc),
                                    nn.ReLU())
        self.fc1 = nn.Linear(self.ic, rc)
        self.fc2 = nn.Linear(self.pc, rc)
        self.fc3 = nn.Linear(rc, 1)





    def forward(self, img_feas, point_feas):

        batch = img_feas.size(0)
        img_feas_f = img_feas.transpose(1,2).contiguous().view(-1, self.ic) #BCN->BNC->(BN)C
        point_feas_f = point_feas.transpose(1,2).contiguous().view(-1, self.pc) #BCN->BNC->(BN)C'
        # print(img_feas)
        ri = self.fc1(img_feas_f)
        rp = self.fc2(point_feas_f)
        att = torch.sigmoid(self.fc3(torch.tanh(ri + rp))) #BNx1
        att = att.squeeze(1)
        att = att.view(batch, 1, -1) #B1N
        # print(img_feas.size(), att.size())
        img_feas_new = self.conv1(img_feas)
        out = img_feas_new * att

        return out

class Atten_Fusion_Conv(nn.Module):

    def __init__(self, inplanes_I, inplanes_P, outplanes):
        super(Atten_Fusion_Conv, self).__init__()

        self.IA_Layer = IA_Layer(channels = [inplanes_I, inplanes_P])
        # self.conv1 = torch.nn.Conv1d(inplanes_P, outplanes, 1)
        self.conv1 = torch.nn.Conv1d(inplanes_P + inplanes_P, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)


    def forward(self, point_features, img_features):
        # print(point_features.shape, img_features.shape)

        # (B, C1, N)
        img_features =  self.IA_Layer(img_features, point_features)
        #print("img_features:", img_features.shape)

        #fusion_features = img_features + point_features
        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))

        return fusion_features

class New_IA_Layer(nn.Module):

    def __init__(self, channels):

        super(New_IA_Layer, self).__init__()
        self.ic, self.pc = channels
        rc = self.pc // 4
        self.conv1 = nn.Sequential(nn.Conv1d(self.ic, self.pc, 1),
                                    nn.BatchNorm1d(self.pc),
                                    nn.ReLU())
        self.fc1 = nn.Linear(self.ic, rc)
        self.fc2 = nn.Linear(self.pc, rc)
        self.fc3 = nn.Linear(rc, 1)
        self.conv2=nn.Sequential(nn.Conv1d(self.ic,rc, 1),
                                    nn.BatchNorm1d(rc),
                                    nn.ReLU(),
                                    nn.Conv1d(rc,1, 1),
                                 )





    def forward(self, img_feas, point_feas):

        batch = img_feas.size(0)
        img_feas_f = img_feas.transpose(1,2).contiguous().view(-1, self.ic) #BCN->BNC->(BN)C
        point_feas_f = point_feas.transpose(1,2).contiguous().view(-1, self.pc) #BCN->BNC->(BN)C'
        # print(img_feas)
        ri = self.fc1(img_feas_f)
        rp = self.fc2(point_feas_f)
        att = torch.sigmoid(self.fc3(torch.tanh(ri + rp))) #BNx1
        att = att.squeeze(1)
        att = att.view(batch, 1, -1) #B1N
        # print(img_feas.size(), att.size())
        img_feas_new = self.conv1(img_feas)
        G = torch.sigmoid(self.conv2(img_feas))
        out = img_feas_new * (att+G)/2.0

        return out
class New_Atten_Fusion_Conv(nn.Module):

    def __init__(self, inplanes_I, inplanes_P, outplanes):
        super(New_Atten_Fusion_Conv, self).__init__()

        self.IA_Layer = New_IA_Layer(channels = [inplanes_I, inplanes_P])
        # self.conv1 = torch.nn.Conv1d(inplanes_P, outplanes, 1)
        self.conv1 = torch.nn.Conv1d(inplanes_P + inplanes_P, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)


    def forward(self, point_features, img_features):
        # print(point_features.shape, img_features.shape)

        # (B, C1, N)
        img_features =  self.IA_Layer(img_features, point_features)
        #print("img_features:", img_features.shape)

        #fusion_features = img_features + point_features
        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))

        return fusion_features





















