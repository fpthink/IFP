import argparse
import os
import random
import time
import logging
import pdb
from tqdm import tqdm
import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.nn.functional as F
from torchvision.utils import make_grid

from torch.autograd import Variable
from torchvision.transforms import transforms as T
from Dataset import SiameseTrain
from pointnet2.models import Pointnet_Tracking


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=48, help='input batch size')
parser.add_argument('--workers', type=int, default=12, help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default=32, help='number of epochs to train for')
parser.add_argument('--ngpu', type=int, default=2, help='# GPUs')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate at t=0')
parser.add_argument('--data_dir', type=str, default = '/opt/data/common/kitti_tracking/kitti_t_o/training',  help='dataset path')
parser.add_argument('--category_name', type=str, default = 'Car',  help='Object to Track (Car/Pedestrian/Van/Cyclist)')
parser.add_argument('--save_root_dir', type=str, default='./kitti_results/',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model name for training resume')

opt = parser.parse_args()
print (opt)

#torch.cuda.set_device(opt.main_gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
opt.manualSeed = 1
np.random.seed(opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

def lr_lbmd(cur_epoch):
	cur_decay = 1
	for decay_step in [10,20,30,40,50]:
		if cur_epoch >= decay_step:
			cur_decay = cur_decay * 0.5
	return max(cur_decay, 0.00001 / opt.learning_rate)

save_dir = os.path.join(opt.save_root_dir,'{}'.format(opt.category_name))

try:
	os.makedirs(save_dir)
except OSError:
	pass

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
					filename=os.path.join(save_dir, 'train.log'), level=logging.INFO)
logging.info('======================================================')

# 1. Load data
transforms = T.Compose([T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                        ])
train_data = SiameseTrain(
            input_size=1024,
            path= opt.data_dir,
            split='Train',
            category_name=opt.category_name,
            offset_BB=0,
            scale_BB=1.25,
			augment=False,
			transforms=transforms,
            imgsize=(512,160)
)

train_dataloader = torch.utils.data.DataLoader(
    train_data,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers),
    pin_memory=True)

test_data = SiameseTrain(
    input_size=1024,
    path=opt.data_dir,
	    split='Valid',
    category_name=opt.category_name,
    offset_BB=0,
    scale_BB=1.25,
	augment=False,
	transforms=transforms,
	imgsize=(512,160)
)

test_dataloader = torch.utils.data.DataLoader(
    test_data,
    batch_size=int(opt.batchSize/2),
    shuffle=False,
    num_workers=int(opt.workers),
    pin_memory=True)
										  
print('#Train data:', len(train_data), '#Test data:', len(test_data))
print (opt)

# 2. Define model, loss and optimizer
netR = Pointnet_Tracking(input_channels=0, use_xyz=True, opt=opt)
if opt.ngpu > 1:
	netR = torch.nn.DataParallel(netR, range(opt.ngpu))
if opt.model != '':
	netR.load_state_dict(torch.load(os.path.join(save_dir, opt.model)))
	  
netR.cuda()
print(netR)
logging.info(opt.__repr__())
logging.info(netR.__repr__())
criterion_template_img_reg = nn.SmoothL1Loss(reduction='none').cuda()
criterion_search_img_reg = nn.SmoothL1Loss(reduction='none').cuda()
criterion_cla = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0])).cuda()
criterion_reg = nn.SmoothL1Loss(reduction='none').cuda()
criterion_objective = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]), reduction='none').cuda()
criterion_box = nn.SmoothL1Loss(reduction='none').cuda()

optimizer = optim.Adam(netR.parameters(), lr=opt.learning_rate, betas = (0.5, 0.999), eps=1e-06)
scheduler = lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.2)
f = open(os.path.join(save_dir,'loss.txt'), mode='w')

# 3. Training and testing
for epoch in range(opt.nepoch):
	print('======>>>>> Online epoch: #%d, lr=%f <<<<<======' %(epoch, scheduler.get_lr()[0]))
	# 3.1 switch to train mode
	torch.cuda.synchronize()
	netR.train()
	train_mse = 0.0
	timer = time.time()

	batch_correct = 0.0
	batch_search_img_reg = 0.0
	batch_template_img_reg = 0.0
	batch_cla_loss = 0.0
	batch_reg_loss = 0.0
	batch_box_loss = 0.0
	batch_num = 0.0
	batch_iou = 0.0
	batch_true_correct = 0.0
	for i, data in enumerate(tqdm(train_dataloader, 0)):
		if len(data[0]) == 1:
			continue
		torch.cuda.synchronize()
		# 3.1.1 load inputs and targets
		label_point_set, label_cla, label_reg, object_point_set,current_img,pre_img,sample_PC_2d_offsets,gt_PC_2d_offsets,sample_2d,gt_2d,template_label,fd,sample_depth,rot,wratio,hratio = data
		label_cla = Variable(label_cla, requires_grad=False).cuda()
		label_reg = Variable(label_reg, requires_grad=False).cuda()
		object_point_set = Variable(object_point_set, requires_grad=False).cuda()
		label_point_set = Variable(label_point_set, requires_grad=False).cuda()

		current_img=Variable(current_img, requires_grad=False).cuda()
		pre_img=Variable(pre_img, requires_grad=False).cuda()
		sample_PC_2d_offsets=Variable(sample_PC_2d_offsets, requires_grad=False).cuda()
		gt_PC_2d_offsets=Variable(gt_PC_2d_offsets, requires_grad=False).cuda()
		sample_2d=Variable(sample_2d, requires_grad=False).cuda()
		gt_2d=Variable(gt_2d, requires_grad=False).cuda()
		template_label=Variable(template_label, requires_grad=False).cuda()
		fd=Variable(fd, requires_grad=False).cuda()
		sample_depth=Variable(sample_depth, requires_grad=False).cuda()
		rot=Variable(rot, requires_grad=False).cuda()
		wratio=Variable(wratio, requires_grad=False).cuda()
		hratio=Variable(hratio, requires_grad=False).cuda()
		# 3.1.2 compute output
		optimizer.zero_grad()
		estimation_cla, estimation_reg,estimation_box,center_xyz,template_img_offsets,search_img_offsets = netR(object_point_set, label_point_set,pre_img,current_img,gt_2d,sample_2d,fd,sample_depth,rot,wratio,hratio)
		search_img_reg_mask=(search_img_offsets[:,:,0]==0) &(search_img_offsets[:,:,1]==0)
		search_img_reg_mask=~search_img_reg_mask
		search_img_reg_mask = search_img_reg_mask.float() * label_cla
		img_dist = torch.sum(sample_PC_2d_offsets ** 2, dim=-1)
		img_dist = torch.sqrt(img_dist + 1e-6)
		img_dist_mask = img_dist<10.0
		search_img_reg_mask=search_img_reg_mask*img_dist_mask.float()
		loss_search_img_reg=criterion_search_img_reg(search_img_offsets,sample_PC_2d_offsets)
		loss_search_img_reg=(loss_search_img_reg.mean(2)*search_img_reg_mask).sum()/(search_img_reg_mask.sum()+1e-06)

		template_img_reg_mask = (template_img_offsets[:, :, 0] == 0) & (template_img_offsets[:, :, 1] == 0)
		template_img_reg_mask = ~template_img_reg_mask
		template_img_reg_mask = template_img_reg_mask.float() * template_label
		img_dist = torch.sum(gt_PC_2d_offsets ** 2, dim=-1)
		img_dist = torch.sqrt(img_dist + 1e-6)
		template_img_dist_mask = img_dist < 10.0
		template_img_reg_mask = template_img_reg_mask * template_img_dist_mask.float()
		loss_template_img_reg = criterion_template_img_reg(template_img_offsets, gt_PC_2d_offsets)
		loss_template_img_reg = (loss_template_img_reg.mean(2) * template_img_reg_mask).sum() / (template_img_reg_mask.sum() + 1e-06)

		loss_cla = criterion_cla(estimation_cla, label_cla)
		loss_reg = criterion_reg(estimation_reg, label_reg[:,:,0:3])
		loss_reg = (loss_reg.mean(2) * label_cla).sum()/(label_cla.sum()+1e-06)

		dist = torch.sum((center_xyz - label_reg[:,0:64,0:3])**2, dim=-1)
		dist = torch.sqrt(dist+1e-6)
		B = dist.size(0)
		K = 64
		objectness_label = torch.zeros((B,K), dtype=torch.float).cuda()
		objectness_mask = torch.zeros((B,K)).cuda()
		objectness_label[dist<0.3] = 1
		objectness_mask[dist<0.3] = 1
		objectness_mask[dist>0.6] = 1
		loss_objective = criterion_objective(estimation_box[:,:,4], objectness_label)
		loss_objective = torch.sum(loss_objective * objectness_mask)/(torch.sum(objectness_mask)+1e-6)
		loss_box = criterion_box(estimation_box[:,:,0:4],label_reg[:,0:64,:])
		loss_box = (loss_box.mean(2) * objectness_label).sum()/(objectness_label.sum()+1e-6)
		
		loss = 0.0*loss_cla + loss_reg + 1.5*loss_objective + 0.2*loss_box + 0.1*loss_search_img_reg
		# 3.1.3 compute gradient and do SGD step
		loss.backward()
		optimizer.step()

		f.write('epoch:{},batch:{},loss:{:.3f}\n'.format(epoch , i, float(loss)))
		torch.cuda.synchronize()

		# 3.1.4 update training error
		estimation_cla_cpu = estimation_cla.sigmoid().detach().cpu().numpy()
		label_cla_cpu = label_cla.detach().cpu().numpy()
		correct = float(np.sum((estimation_cla_cpu[0:len(label_point_set),:] > 0.4) == label_cla_cpu[0:len(label_point_set),:])) / 128.0
		true_correct = float(np.sum((np.float32(estimation_cla_cpu[0:len(label_point_set),:] > 0.4) + label_cla_cpu[0:len(label_point_set),:]) == 2)/(np.sum(label_cla_cpu[0:len(label_point_set),:])))

		train_mse = train_mse + loss.data*len(label_point_set)
		batch_correct += correct
		batch_search_img_reg+=loss_search_img_reg.data
		batch_template_img_reg+=loss_template_img_reg.data
		batch_cla_loss += loss_cla.data
		batch_reg_loss += loss_reg.data
		batch_box_loss += loss_box.data
		batch_num += len(label_point_set)
		batch_true_correct += true_correct

		if (i+1)%20 == 0:
			print('\n ---- batch: %03d ----' % (i+1))
			print('cla_loss: %f, reg_loss: %f, box_loss: %f, img_search_reg: %f,img_template_reg: %f' % (batch_cla_loss/20,batch_reg_loss/20,batch_box_loss/20,batch_search_img_reg/20,batch_template_img_reg/20))
			print('accuracy: %f' % (batch_correct / float(batch_num)))
			print('true accuracy: %f' % (batch_true_correct / 20))
			batch_correct = 0.0
			batch_search_img_reg=0.0
			batch_template_img_reg=0.0
			batch_cla_loss = 0.0
			batch_reg_loss = 0.0
			batch_box_loss = 0.0
			batch_num = 0.0
			batch_true_correct = 0.0
	scheduler.step(epoch)
	# time taken
	train_mse = train_mse/len(train_data)
	torch.cuda.synchronize()
	timer = time.time() - timer
	timer = timer / len(train_data)
	print('==> time to learn 1 sample = %f (ms)' %(timer*1000))
	torch.save(netR.state_dict(), '%s/netR_%d.pth' % (save_dir, epoch))

	# 3.2 switch to evaluate mode
	torch.cuda.synchronize()
	netR.eval()
	test_cla_loss = 0.0
	test_reg_loss = 0.0
	test_box_loss = 0.0
	test_correct = 0.0
	test_true_correct = 0.0
	test_img_search_reg_loss = 0.0
	test_img_template_reg_loss = 0.0
	timer = time.time()
	with torch.no_grad():
		for i, data in enumerate(tqdm(test_dataloader, 0)):
			torch.cuda.synchronize()
			# 3.2.1 load inputs and targets
			label_point_set, label_cla, label_reg, object_point_set, current_img, pre_img, sample_PC_2d_offsets,gt_PC_2d_offsets, sample_2d, gt_2d ,template_label,fd,sample_depth,rot,wratio,hratio= data
			label_cla = Variable(label_cla, requires_grad=False).cuda()
			label_reg = Variable(label_reg, requires_grad=False).cuda()
			object_point_set = Variable(object_point_set, requires_grad=False).cuda()
			label_point_set = Variable(label_point_set, requires_grad=False).cuda()

			current_img = Variable(current_img, requires_grad=False).cuda()
			pre_img = Variable(pre_img, requires_grad=False).cuda()
			sample_PC_2d_offsets = Variable(sample_PC_2d_offsets, requires_grad=False).cuda()
			gt_PC_2d_offsets = Variable(gt_PC_2d_offsets, requires_grad=False).cuda()
			sample_2d = Variable(sample_2d, requires_grad=False).cuda()
			gt_2d = Variable(gt_2d, requires_grad=False).cuda()
			template_label = Variable(template_label, requires_grad=False).cuda()
			fd = Variable(fd, requires_grad=False).cuda()
			sample_depth = Variable(sample_depth, requires_grad=False).cuda()
			rot = Variable(rot, requires_grad=False).cuda()
			wratio = Variable(wratio, requires_grad=False).cuda()
			hratio = Variable(hratio, requires_grad=False).cuda()
			# 3.2.2 compute output
			estimation_cla, estimation_reg, estimation_box, center_xyz,template_img_offsets,search_img_offsets = netR(object_point_set, label_point_set,pre_img,current_img,gt_2d,sample_2d,fd,sample_depth,rot,wratio,hratio)
			search_img_reg_mask = (search_img_offsets[:, :, 0] == 0) & (search_img_offsets[:, :, 1] == 0)
			search_img_reg_mask = ~search_img_reg_mask
			search_img_reg_mask = search_img_reg_mask.float() * label_cla
			img_dist = torch.sum(sample_PC_2d_offsets ** 2, dim=-1)
			img_dist = torch.sqrt(img_dist + 1e-6)
			img_dist_mask = img_dist < 10.0
			search_img_reg_mask = search_img_reg_mask * img_dist_mask.float()
			loss_search_img_reg = criterion_search_img_reg(search_img_offsets, sample_PC_2d_offsets)
			loss_search_img_reg = (loss_search_img_reg.mean(2) * search_img_reg_mask).sum() / (
						search_img_reg_mask.sum() + 1e-06)

			template_img_reg_mask = (template_img_offsets[:, :, 0] == 0) & (template_img_offsets[:, :, 1] == 0)
			template_img_reg_mask = ~template_img_reg_mask
			template_img_reg_mask = template_img_reg_mask.float() * template_label
			img_dist = torch.sum(gt_PC_2d_offsets ** 2, dim=-1)
			img_dist = torch.sqrt(img_dist + 1e-6)
			template_img_dist_mask = img_dist < 10.0
			template_img_reg_mask = template_img_reg_mask * template_img_dist_mask.float()

			loss_template_img_reg = criterion_template_img_reg(template_img_offsets, gt_PC_2d_offsets)
			loss_template_img_reg = (loss_template_img_reg.mean(2) * template_img_reg_mask).sum() / (
						template_img_reg_mask.sum() + 1e-06)

			loss_cla = criterion_cla(estimation_cla, label_cla)
			loss_reg = criterion_reg(estimation_reg, label_reg[:, :, 0:3])
			loss_reg = (loss_reg.mean(2) * label_cla).sum() / (label_cla.sum() + 1e-06)

			dist = torch.sum((center_xyz - label_reg[:, 0:64, 0:3]) ** 2, dim=-1)
			dist = torch.sqrt(dist + 1e-6)
			B = dist.size(0)
			K = 64
			objectness_label = torch.zeros((B, K), dtype=torch.float).cuda()
			objectness_mask = torch.zeros((B, K)).cuda()
			objectness_label[dist < 0.3] = 1
			objectness_mask[dist < 0.3] = 1
			objectness_mask[dist > 0.6] = 1
			loss_objective = criterion_objective(estimation_box[:, :, 4], objectness_label)
			loss_objective = torch.sum(loss_objective * objectness_mask) / (torch.sum(objectness_mask) + 1e-6)
			loss_box = criterion_box(estimation_box[:, :, 0:4], label_reg[:, 0:64, :])
			loss_box = (loss_box.mean(2) * objectness_label).sum() / (objectness_label.sum() + 1e-06)
			loss = 0.0 * loss_cla + loss_reg + 1.5 * loss_objective + 0.2 * loss_box+0.1*(loss_search_img_reg+0.0*loss_template_img_reg)

			torch.cuda.synchronize()
			test_cla_loss = test_cla_loss + loss_cla.data * len(label_point_set)
			test_reg_loss = test_reg_loss + loss_reg.data * len(label_point_set)
			test_box_loss = test_box_loss + loss_box.data * len(label_point_set)
			test_img_search_reg_loss = test_img_search_reg_loss + loss_search_img_reg.data * len(label_point_set)
			test_img_template_reg_loss = test_img_template_reg_loss + loss_template_img_reg.data * len(label_point_set)
			estimation_cla_cpu = estimation_cla.sigmoid().detach().cpu().numpy()
			label_cla_cpu = label_cla.detach().cpu().numpy()
			correct = float(np.sum(
				(estimation_cla_cpu[0:len(label_point_set), :] > 0.4) == label_cla_cpu[0:len(label_point_set),
				                                                         :])) / 128.0
			true_correct = float(np.sum((np.float32(
				estimation_cla_cpu[0:len(label_point_set), :] > 0.4) + label_cla_cpu[0:len(label_point_set),
			                                                           :]) == 2) / (
				                     np.sum(label_cla_cpu[0:len(label_point_set), :])))
			test_correct += correct
			test_true_correct += true_correct * len(label_point_set)

	# time taken
	torch.cuda.synchronize()
	timer = time.time() - timer
	timer = timer / len(test_data)
	print('==> time to learn 1 sample = %f (ms)' %(timer*1000))
	test_cla_loss = test_cla_loss / len(test_data)
	test_reg_loss = test_reg_loss / len(test_data)
	test_box_loss = test_box_loss / len(test_data)
	test_img_search_reg_loss = test_img_search_reg_loss / len(test_data)
	test_img_template_reg_loss = test_img_template_reg_loss / len(test_data)
	print('cla_loss: %f, reg_loss: %f, box_loss: %f, img_search_reg_loss: %f, img_template_reg_loss: %f, #test_data = %d' %(test_cla_loss,test_reg_loss,test_box_loss,test_img_search_reg_loss,test_img_template_reg_loss,len(test_data)))
	test_correct = test_correct / len(test_data)
	print('mean-correct of 1 sample: %f, #test_data = %d' %(test_correct, len(test_data)))
	test_true_correct = test_true_correct / len(test_data)
	print('true correct of 1 sample: %f' %(test_true_correct))
	# log
	logging.info('Epoch#%d: train error=%e, test error=%e,%e,%e,%e,%e, test correct=%e, %e, lr = %f' %(epoch, train_mse, test_cla_loss,test_reg_loss,test_box_loss,test_img_search_reg_loss,test_img_template_reg_loss, test_correct, test_true_correct, scheduler.get_lr()[0]))
