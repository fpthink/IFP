from torch.utils.data import Dataset
from data_classes import PointCloud, Box
from pyquaternion import Quaternion
import numpy as np
import pandas as pd
import os
import torch
import cv2
from tqdm import tqdm
import kitty_utils as utils
from kitty_utils import getModel
from searchspace import KalmanFiltering
import logging
import random
from torchvision.transforms import transforms as T
import glob
from torch.autograd import Variable
import copy

class kittiDataset():

    def __init__(self, path,augment=True,transforms=None,imgsize=(1024,320)):
        self.KITTI_Folder = path
        self.KITTI_velo = os.path.join(self.KITTI_Folder, "velodyne")
        self.KITTI_label = os.path.join(self.KITTI_Folder, "label_02")
        self.KITTI_image= os.path.join(self.KITTI_Folder,'image_02')
        self.augment=augment
        self.transforms=transforms
        self.width=imgsize[0]
        self.height=imgsize[1]

    def getSceneID(self, split):
        if "TRAIN" in split.upper():  # Training SET
            if "TINY" in split.upper():
                sceneID = [0]
            else:
                sceneID = list(range(0, 17))
        elif "VALID" in split.upper():  # Validation Set
            if "TINY" in split.upper():
                sceneID = [18]
            else:
                sceneID = list(range(17, 19))
        elif "TEST" in split.upper():  # Testing Set
            if "TINY" in split.upper():
                sceneID = [19]
            else:
                sceneID = list(range(19, 21))

        else:  # Full Dataset
            sceneID = list(range(21))
        return sceneID

    def getsceneImgNum(self,sceneID):
        list_of_scene = [
            path for path in os.listdir(self.KITTI_image)
            if os.path.isdir(os.path.join(self.KITTI_image, path)) and
               int(path) in sceneID
        ]
        list_of_scene=sorted(list_of_scene)
        sceneImgNum=[]
        list_of_img_path=[]
        for scene in list_of_scene:
            img_dir_path=os.path.join(self.KITTI_image,scene)
            img_path=sorted(glob.glob('%s/*.png' % img_dir_path))
            sceneImgNum.append(len(img_path))
            list_of_img_path+=img_path
        return sceneImgNum,list_of_img_path
    
    def getBBandPC(self, anno):
        calib_path = os.path.join(self.KITTI_Folder, 'calib',
                                  anno['scene'] + ".txt")
        calib = self.read_calib_file(calib_path)
        transf_mat = np.vstack((calib["Tr_velo_cam"], np.array([0, 0, 0, 1])))
        PC, box = self.getPCandBBfromPandas(anno, transf_mat)
        return PC, box,calib

    def getListOfAnno(self, sceneID, category_name="Car"):
        list_of_scene = [
            path for path in os.listdir(self.KITTI_velo)
            if os.path.isdir(os.path.join(self.KITTI_velo, path)) and
            int(path) in sceneID
        ]
        list_of_scene = sorted(list_of_scene)
        print(list_of_scene)
        list_of_tracklet_anno = []
        for scene in list_of_scene:

            label_file = os.path.join(self.KITTI_label, scene + ".txt")
            df = pd.read_csv(
                label_file,
                sep=' ',
                names=[
                    "frame", "track_id", "type", "truncated", "occluded",
                    "alpha", "bbox_left", "bbox_top", "bbox_right",
                    "bbox_bottom", "height", "width", "length", "x", "y", "z",
                    "rotation_y","score"
                ])
            df = df[df["type"] == category_name]
            df.insert(loc=0, column="scene", value=scene)
            for track_id in df.track_id.unique():
                df_tracklet = df[df["track_id"] == track_id]
                df_tracklet = df_tracklet.reset_index(drop=True)
                # tracklet label list[{scene:0000 frame:0 track_id:0,type:car ...},{},....]
                tracklet_anno = [anno for index, anno in df_tracklet.iterrows()]
                list_of_tracklet_anno.append(tracklet_anno)

        return list_of_tracklet_anno

    def getPCandBBfromPandas(self, box, calib):
        center = [box["x"], box["y"]- box["height"] / 2 , box["z"]]
        size = [box["width"], box["length"], box["height"]]
        orientation = Quaternion(
            axis=[0, 1, 0], radians=box["rotation_y"]) * Quaternion(
                axis=[1, 0, 0], radians=np.pi / 2)
        BB = Box(center, size, orientation)

        try:
            # VELODYNE PointCloud
            velodyne_path = os.path.join(self.KITTI_velo, box["scene"],
                                         '{:06}.bin'.format(box["frame"]))
            PC = PointCloud(
                np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4).T)
            PC.transform(calib)
        except :
            # in case the Point cloud is missing
            # (0001/[000177-000180].bin)
            PC = PointCloud(np.array([[0, 0, 0]]).T)

        return PC, BB

    def read_calib_file(self, filepath):
        """Read in a calibration file and parse into a dictionary."""
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                values = line.split()
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    ind = values[0].find(':')
                    if ind != -1:
                        data[values[0][:ind]] = np.array(
                            [float(x) for x in values[1:]]).reshape(3, 4)
                    else:
                        data[values[0]] = np.array(
                            [float(x) for x in values[1:]]).reshape(3, 4)
                except ValueError:
                    data[values[0]] = np.array(
                        [float(x) for x in values[1:]]).reshape(3, 3)
        return data
    
    def get_img_data(self,img_path,width=1024,height=320):
        img=cv2.imread(img_path)

        if img is None:
            raise ValueError('File corrupt {}'.format(img_path))
        shape = img.shape[:2]
        
        if self.augment :
            # SV augmentation by 50%
            fraction = 0.50
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            S = img_hsv[:, :, 1].astype(np.float32)
            V = img_hsv[:, :, 2].astype(np.float32)

            a = (random.random() * 2 - 1) * fraction + 1
            S *= a
            if a > 1:
                np.clip(S, a_min=0, a_max=255, out=S)

            a = (random.random() * 2 - 1) * fraction + 1
            V *= a
            if a > 1:
                np.clip(V, a_min=0, a_max=255, out=V)

            img_hsv[:, :, 1] = S.astype(np.uint8)
            img_hsv[:, :, 2] = V.astype(np.uint8)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

        img=cv2.resize(img,(width,height),interpolation=cv2.INTER_AREA)
        wratio=float(width)/shape[1]
        hratio=float(height)/shape[0]

        img = np.ascontiguousarray(img[:, :, ::-1])
        if self.transforms is not None:
            img = self.transforms(img)
        return img,wratio,hratio
    
    def transform2Dbox(self,wratio=1.0,hratio=1.0,anno=None):
        anno["bbox_left"]=wratio*anno["bbox_left"]
        anno["bbox_top"]=hratio*anno["bbox_top"]
        anno["bbox_right"]=wratio*anno["bbox_right"]
        anno["bbox_bottom"]=hratio*anno["bbox_bottom"]
        anno["bbox_center_x"]=(anno["bbox_left"]+anno["bbox_right"])/2.0
        anno["bbox_center_y"]=(anno["bbox_top"]+anno["bbox_bottom"])/2.0
        anno['wratio']=wratio
        anno['hratio']=hratio

class SiameseDataset(Dataset):

    def __init__(self,
                 input_size,
                 path,
                 split,
                 category_name="Car",
                 regress="GAUSSIAN",
                 offset_BB=0,
                 scale_BB=1.0,
                 augment=True,
                 transforms=None,
                 imgsize=(1024,320)
                 ):

        self.dataset = kittiDataset(path=path,augment=augment,transforms=transforms,imgsize=imgsize)

        self.input_size = input_size
        self.split = split
        self.sceneID = self.dataset.getSceneID(split=split)
        self.getBBandPC = self.dataset.getBBandPC
        self.sceneImgNum,self.list_of_img_path=self.dataset.getsceneImgNum(self.sceneID)

        self.category_name = category_name
        self.regress = regress
        self.list_of_tracklet_anno = self.dataset.getListOfAnno(
            self.sceneID, category_name)
        self.list_of_anno = [
            anno for tracklet_anno in self.list_of_tracklet_anno
            for anno in tracklet_anno
        ]

    def isTiny(self):
        return ("TINY" in self.split.upper())

    def project_to_2d(self,point, calib, inv_align, wratio, hratio):
        N=point.size(0)
        sample_PC=copy.deepcopy(point)
        sample_PC = sample_PC.numpy()
        homo_points=np.vstack((np.transpose(sample_PC),np.ones((1,N))))
        homo_points=inv_align.dot(homo_points)
        
        homo_points=calib.dot(homo_points)
        homo_points[0,:]/=homo_points[2,:]
        homo_points[1,:]/=homo_points[2,:]
        
        xy=homo_points[:2,:]
        xy[0,:]*=wratio
        xy[1,:]*=hratio
        xy=np.transpose(xy)
        xy=np.ascontiguousarray(xy)

        return torch.from_numpy(xy).float()

    def __getitem__(self, index):
        return self.getitem(index)


class SiameseTrain(SiameseDataset):

    def __init__(self,
                 input_size,
                 path,
                 split="",
                 category_name="Car",
                 regress="GAUSSIAN",
                 sigma_Gaussian=1,
                 offset_BB=0,
                 scale_BB=1.0,
                 augment=True,
                 transforms=None,
                 imgsize=(1024,320),
                 train_mode=True,
                 num_candidates_perframe=4):
        super(SiameseTrain,self).__init__(
            input_size=input_size,
            path=path,
            split=split,
            category_name=category_name,
            regress=regress,
            offset_BB=offset_BB,
            scale_BB=scale_BB,
            augment=augment,
            transforms=transforms,
            imgsize=imgsize
        )

        self.train_mode=train_mode
        self.sigma_Gaussian = sigma_Gaussian
        self.offset_BB = offset_BB
        self.scale_BB = scale_BB

        self.num_candidates_perframe = num_candidates_perframe

        logging.info("preloading PC...")
        self.list_of_PCs = [None] * len(self.list_of_anno)
        self.list_of_BBs = [None] * len(self.list_of_anno)
        self.list_of_imgs = [None] * len(self.list_of_img_path)
        self.list_of_wh_ratio = [None] * len(self.list_of_img_path)
        self.list_of_calib = [None] * len(self.list_of_anno)

        for index in tqdm(range(len(self.list_of_img_path))):
            img,wratio,hratio=self.dataset.get_img_data(self.list_of_img_path[index],width=self.dataset.width,height=self.dataset.height)
            self.list_of_imgs[index] = img
            self.list_of_wh_ratio[index]=(wratio,hratio)

        for index in tqdm(range(len(self.list_of_anno))):
            anno = self.list_of_anno[index]
            PC, box,calib = self.getBBandPC(anno)
            new_PC = utils.cropPC(PC, box, offset=10)

            self.list_of_PCs[index] = new_PC
            self.list_of_BBs[index] = box
            self.list_of_calib[index]=calib
            img_index=sum(self.sceneImgNum[0:int(anno["scene"])-self.sceneID[0]])+anno["frame"]
            self.dataset.transform2Dbox(self.list_of_wh_ratio[img_index][0],self.list_of_wh_ratio[img_index][1],anno)

        logging.info("PC preloaded!")

        logging.info("preloading Model..")
        self.model_PC = [None] * len(self.list_of_tracklet_anno)
        for i in tqdm(range(len(self.list_of_tracklet_anno))):
            list_of_anno = self.list_of_tracklet_anno[i]
            cnt = 0
            for anno in list_of_anno:
                anno["model_idx"] = i
                anno["relative_idx"] = cnt
                cnt += 1

        logging.info("Model preloaded!")

    def reset_image_list(self):
        for index in tqdm(range(len(self.list_of_img_path))):
            img,wratio,hratio=self.dataset.get_img_data(self.list_of_img_path[index],width=self.dataset.width,height=self.dataset.height)
            self.list_of_imgs[index] = img
            self.list_of_wh_ratio[index]=(wratio,hratio)

    def __getitem__(self, index):
        return self.getitem(index)

    def getPCandBBfromIndex(self, anno_idx):
        this_PC = self.list_of_PCs[anno_idx]
        this_BB = self.list_of_BBs[anno_idx]
        this_calib=self.list_of_calib[anno_idx]
        return this_PC, this_BB,this_calib

    def getitem(self, index):
        anno_idx = self.getAnnotationIndex(index)
        sample_idx = self.getSearchSpaceIndex(index)

        if sample_idx == 0:
            sample_offsets = np.zeros(4)
        else:
            gaussian = KalmanFiltering(bnd=[1, 1, 0.5, 5])
            sample_offsets = gaussian.sample(1)[0]

        this_anno = self.list_of_anno[anno_idx]

        this_PC, this_BB ,this_calib = self.getPCandBBfromIndex(anno_idx)

        current_img_index = sum(self.sceneImgNum[0:int(this_anno["scene"])-self.sceneID[0]])+this_anno["frame"]
        current_img=self.list_of_imgs[current_img_index]

        sample_BB = utils.getOffsetBB(this_BB, sample_offsets)
        sample_PC, sample_label, sample_reg,sample_depth = utils.cropAndCenterPC_label(
            this_PC,sample_BB, this_BB, sample_offsets, offset=self.offset_BB, scale=self.scale_BB)
        if sample_PC.nbr_points() <= 20 :
            return self.getitem(np.random.randint(0, self.__len__()))
        
        sample_PC, sample_label, sample_reg,sample_depth = utils.regularizePCwithlabel(sample_PC,sample_label,sample_reg,self.input_size,sample_depth,istrain=self.train_mode)
        sample_PC_2d_offsets=utils.generate_img_center_offsets(sample_PC,sample_BB,this_calib,this_BB,this_anno["wratio"],this_anno["hratio"])

        if this_anno["relative_idx"] == 0:
            prev_idx = anno_idx
            fir_idx = anno_idx
        else:
            prev_idx = anno_idx - 1
            fir_idx = anno_idx - this_anno["relative_idx"]
        gt_PC_pre, gt_BB_pre, _ = self.getPCandBBfromIndex(prev_idx)
        gt_PC_fir, gt_BB_fir,_ = self.getPCandBBfromIndex(fir_idx)

        pre_anno = self.list_of_anno[prev_idx]
        pre_img_index = sum(self.sceneImgNum[0:int(pre_anno["scene"])-self.sceneID[0]])+pre_anno["frame"]
        pre_img = self.list_of_imgs[pre_img_index]
        if sample_idx == 0:
            samplegt_offsets = np.zeros(4)
        else:
            samplegt_offsets = np.random.uniform(low=-0.3, high=0.3, size=4)
            samplegt_offsets[3] = samplegt_offsets[3]*5.0
        gt_BB_pre_offset= utils.getOffsetBB(gt_BB_pre, samplegt_offsets)

        gt_PC ,frame_wise_box_nums= getModel([gt_PC_fir,gt_PC_pre], [gt_BB_fir,gt_BB_pre_offset], offset=self.offset_BB, scale=self.scale_BB)

        if gt_PC.nbr_points() <= 20 :
            return self.getitem(np.random.randint(0, self.__len__()))
        gt_PC,template_label = utils.regularizeTemplatePCwithlabel(gt_PC,frame_wise_box_nums,self.input_size,istrain=self.train_mode)
        gt_PC_2d_offsets=utils.generate_img_center_offsets(gt_PC,gt_BB_pre_offset,this_calib,gt_BB_pre,this_anno["wratio"],this_anno["hratio"])

        current_inv_align_matrix=np.hstack((sample_BB.rotation_matrix,sample_BB.center.reshape((3,1))))
        current_inv_align_matrix=np.vstack((current_inv_align_matrix,np.array([0,0,0,1])))

        pre_inv_align_matrix=np.hstack((gt_BB_pre_offset.rotation_matrix,gt_BB_pre_offset.center.reshape((3,1))))
        pre_inv_align_matrix = np.vstack((pre_inv_align_matrix, np.array([0, 0, 0, 1])))

        R_rect=np.hstack((this_calib["R_rect"],np.array([0,0,0]).reshape((3,1))))
        R_rect=np.vstack((R_rect,np.array([0,0,0,1])))
        calib_ref_img=this_calib["P2"].dot(R_rect)
        sample_2d=self.project_to_2d(sample_PC,calib_ref_img,current_inv_align_matrix,this_anno["wratio"],this_anno["hratio"])
        gt_2d=self.project_to_2d(gt_PC,calib_ref_img,pre_inv_align_matrix,this_anno["wratio"],this_anno["hratio"])

        f=this_calib["P2"][0,0]
        f=torch.tensor([f],dtype=torch.float32)
        rot=torch.from_numpy(np.transpose(sample_BB.rotation_matrix.copy())).float()
        wratio=torch.tensor([this_anno["wratio"]],dtype=torch.float32)
        hratio=torch.tensor([this_anno["hratio"]],dtype=torch.float32)

        return sample_PC, sample_label, sample_reg, gt_PC,current_img,pre_img,sample_PC_2d_offsets[:128,:]/2.0,gt_PC_2d_offsets[:64,:]/2.0,sample_2d,gt_2d,template_label,f,sample_depth,rot,wratio,hratio

    def __len__(self):
        nb_anno = len(self.list_of_anno)
        return nb_anno * self.num_candidates_perframe

    def getAnnotationIndex(self, index):
        return int(index / (self.num_candidates_perframe))

    def getSearchSpaceIndex(self, index):
        return int(index % self.num_candidates_perframe)


class SiameseTest(SiameseDataset):
    def __init__(self,
                 input_size,
                 path,
                 split="",
                 category_name="Car",
                 regress="GAUSSIAN",
                 offset_BB=0,
                 scale_BB=1.0,
                 augment=True,
                 transforms=None,
                 imgsize=(512, 160)
                 ):
        super(SiameseTest,self).__init__(
            input_size=input_size,
            path=path,
            split=split,
            category_name=category_name,
            regress=regress,
            offset_BB=offset_BB,
            scale_BB=scale_BB,
            augment=augment,
            transforms=transforms,
            imgsize=imgsize
        )
        self.split = split
        self.offset_BB = offset_BB
        self.scale_BB = scale_BB

    def getitem(self, index):
        list_of_anno = self.list_of_tracklet_anno[index]
        PCs = []
        BBs = []
        IMGS= []
        calibs= []
        for anno in list_of_anno:
            this_PC, this_BB,this_calib = self.getBBandPC(anno)
            fd = this_calib["P2"][0, 0]
            img_path = os.path.join(self.dataset.KITTI_image, anno["scene"],'{:06}.png'.format(anno["frame"]))
            img,wratio,hratio=self.dataset.get_img_data(img_path,self.dataset.width,self.dataset.height)
            PCs.append(this_PC)
            BBs.append(this_BB)
            IMGS.append(img)

        return PCs, BBs, list_of_anno,IMGS,this_calib,wratio,hratio,fd

    def __len__(self):
        return len(self.list_of_tracklet_anno)
