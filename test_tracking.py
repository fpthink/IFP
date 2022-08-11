import time
import os
import argparse
import random

import numpy as np
from tqdm import tqdm
from torchvision.transforms import transforms as T

import torch

import kitty_utils as utils
import copy
from datetime import datetime

from metrics import AverageMeter, Success, Precision
from metrics import estimateOverlap, estimateAccuracy
from data_classes import PointCloud
from Dataset import SiameseTest

import torch.nn.functional as F
from torch.autograd import Variable

from pointnet2.models import Pointnet_Tracking


def test(loader, model, epoch=-1, shape_aggregation="", reference_BB="", max_iter=-1, IoU_Space=3):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	post_process_time = AverageMeter()
	Success_main = Success()
	Precision_main = Precision()

	Success_batch = Success()
	Precision_batch = Precision()

	# switch to evaluate mode
	model.eval()
	end = time.time()

	dataset = loader.dataset
	batch_num = 0
	total_frame = 0
	with tqdm(enumerate(loader), total=len(loader.dataset.list_of_tracklet_anno)) as t:
		for batch in loader:
			batch_num = batch_num + 1
			# measure data loading time
			# data_time.update((time.time() - end))
			for PCs, BBs, list_of_anno, IMGS, calib, wratio, hratio, fd in batch:  # tracklet
				results_BBs = []
				for i, _ in enumerate(PCs):
					this_anno = list_of_anno[i]
					this_BB = BBs[i]
					this_PC = PCs[i]
					this_IMG = IMGS[i]

					# INITIAL FRAME
					if i == 0:

						box = BBs[i]
						results_BBs.append(box)
						model_PC,_= utils.getModel([this_PC], [this_BB], offset=dataset.offset_BB,
						                          scale=dataset.scale_BB)
						data_time.update((time.time() - end))
						end = time.time()
					else:
						previous_BB = BBs[i - 1]
						previous_IMG=IMGS[i-1]
						# DEFINE REFERENCE BB
						if ("previous_result".upper() in reference_BB.upper()):
							ref_BB = results_BBs[-1]
						elif ("previous_gt".upper() in reference_BB.upper()):
							ref_BB = previous_BB
							# ref_BB = utils.getOffsetBB(this_BB,np.array([-1,1,1]))
						elif ("current_gt".upper() in reference_BB.upper()):
							ref_BB = this_BB

						candidate_PC, candidate_label, candidate_reg, new_ref_box, new_this_box,candidate_depth = utils.cropAndCenterPC_label_test(
							this_PC,
							ref_BB, this_BB,
							offset=dataset.offset_BB,
							scale=dataset.scale_BB)

						candidate_PCs, candidate_labels, candidate_reg,candidate_depth = utils.regularizePCwithlabel(candidate_PC,
						                                                                             candidate_label,
						                                                                             candidate_reg,
						                                                                             dataset.input_size,
						                                                                             candidate_depth,
						                                                                             istrain=False)


						# AGGREGATION: IO vs ONLY0 vs ONLYI vs ALL
						if ("firstandprevious".upper() in shape_aggregation.upper()):
							model_PC,_ = utils.getModel([PCs[0], PCs[i - 1]], [results_BBs[0], results_BBs[i - 1]],
							                          offset=dataset.offset_BB, scale=dataset.scale_BB)
						elif ("first".upper() in shape_aggregation.upper()):
							model_PC,_ = utils.getModel([PCs[0]], [results_BBs[0]], offset=dataset.offset_BB,
							                          scale=dataset.scale_BB)
						elif ("previous".upper() in shape_aggregation.upper()):
							model_PC,_ = utils.getModel([PCs[i - 1]], [results_BBs[i - 1]], offset=dataset.offset_BB,
							                          scale=dataset.scale_BB)
						elif ("all".upper() in shape_aggregation.upper()):
							model_PC,_ = utils.getModel(PCs[:i], results_BBs, offset=dataset.offset_BB,
							                          scale=dataset.scale_BB)
						else:
							model_PC,_ = utils.getModel(PCs[:i], results_BBs, offset=dataset.offset_BB,
							                          scale=dataset.scale_BB)

						model_PC_torch = utils.regularizePC(model_PC, dataset.input_size, istrain=False)

						current_inv_align_matrix = np.hstack(
							(ref_BB.rotation_matrix, ref_BB.center.reshape((3, 1))))
						current_inv_align_matrix = np.vstack((current_inv_align_matrix, np.array([0, 0, 0, 1])))

						pre_inv_align_matrix = np.hstack(
							(results_BBs[i - 1].rotation_matrix, results_BBs[i - 1].center.reshape((3, 1))))
						pre_inv_align_matrix = np.vstack((pre_inv_align_matrix, np.array([0, 0, 0, 1])))

						R_rect = np.hstack((calib["R_rect"], np.array([0, 0, 0]).reshape((3, 1))))
						R_rect = np.vstack((R_rect, np.array([0, 0, 0, 1])))
						calib_ref_img = calib["P2"].dot(R_rect)
						candidate_2d = dataset.project_to_2d(candidate_PCs, calib_ref_img, current_inv_align_matrix,
						                               wratio, hratio)
						model_2d = dataset.project_to_2d(model_PC_torch, calib_ref_img, pre_inv_align_matrix, wratio,
						                           hratio)

						candidate_PCs_torch = candidate_PCs.unsqueeze(0)
						candidate_depth_torch=candidate_depth.unsqueeze(0)
						FD=torch.tensor([fd],dtype=torch.float32).unsqueeze(0)
						rot=torch.from_numpy(np.transpose(ref_BB.rotation_matrix.copy())).float().unsqueeze(0)
						model_PC_torch=model_PC_torch.unsqueeze(0)
						this_IMG=this_IMG.unsqueeze(0)
						previous_IMG=previous_IMG.unsqueeze(0)
						candidate_2d=candidate_2d.unsqueeze(0)
						model_2d=model_2d.unsqueeze(0)
						Wratio=torch.tensor([wratio],dtype=torch.float32).unsqueeze(0)
						Hratio=torch.tensor([hratio],dtype=torch.float32).unsqueeze(0)

						model_PC_torch = Variable(model_PC_torch, requires_grad=False).cuda()
						candidate_PCs_torch = Variable(candidate_PCs_torch, requires_grad=False).cuda()
						candidate_depth_torch = Variable(candidate_depth_torch, requires_grad=False).cuda()
						FD = Variable(FD, requires_grad=False).cuda()
						rot = Variable(rot, requires_grad=False).cuda()
						Wratio = Variable(Wratio, requires_grad=False).cuda()
						Hratio = Variable(Hratio, requires_grad=False).cuda()

						this_IMG = Variable(this_IMG, requires_grad=False).cuda()
						previous_IMG = Variable(previous_IMG, requires_grad=False).cuda()
						candidate_2d = Variable(candidate_2d, requires_grad=False).cuda()
						model_2d = Variable(model_2d, requires_grad=False).cuda()

						data_time.update((time.time() - end))
						end = time.time()
						
						estimation_cla, estimation_reg, estimation_box, center_xyz, _, _ = model(model_PC_torch,
						                                                                   candidate_PCs_torch,
						                                                                   previous_IMG,
						                                                                   this_IMG,
						                                                                   model_2d,
						                                                                   candidate_2d,
						                                                                   FD,
						                                                                   candidate_depth_torch,
						                                                                   rot,
						                                                                   Wratio,
						                                                                   Hratio
						                                                                   )
						batch_time.update(time.time() - end)
						end = time.time()

						estimation_boxs_cpu = estimation_box.squeeze(0).detach().cpu()
						score=torch.sigmoid(estimation_boxs_cpu[:,4])
						estimation_boxs_cpu=estimation_boxs_cpu.numpy()
						box_idx = estimation_boxs_cpu[:, 4].argmax()
						estimation_box_cpu = estimation_boxs_cpu[box_idx, 0:4]
						score = score[box_idx]

						box = utils.getOffsetBBtest(ref_BB, estimation_box_cpu)
						results_BBs.append(box)

					# estimate overlap/accuracy fro current sample
					this_overlap = estimateOverlap(BBs[i], results_BBs[-1], dim=IoU_Space)
					this_accuracy = estimateAccuracy(BBs[i], results_BBs[-1], dim=IoU_Space)

					Success_main.add_overlap(this_overlap)
					Precision_main.add_accuracy(this_accuracy)
					Success_batch.add_overlap(this_overlap)
					Precision_batch.add_accuracy(this_accuracy)

					# measure elapsed time
					post_process_time.update(time.time() - end)
					end = time.time()

					if Success_main.count >= max_iter and max_iter >= 0:
						return Success_main.average, Precision_main.average
  
				total_frame += len(PCs)
				t.update(1)
				t.set_description('forward:{:3.0f}ms, '.format(1000*batch_time.avg) +
                                  'pre:{:3.0f}ms, '.format(1000*data_time.avg) +
                                  'post:{:3.0f}ms, '.format(1000*post_process_time.avg) +
                                  'Cur Succ/Prec: ' +
								  '{:.1f}/'.format(Success_batch.average) +
								  '{:.1f}'.format(Precision_batch.average)+
				                  ', Total Succ/Prec: ' +
				                  '{:.1f}/'.format(Success_main.average) +
				                  '{:.1f}'.format(Precision_main.average)+
                      			  ', cur_frames:{}'.format(len(PCs))+
								  ', total_frames:{}'.format(total_frame)
                           )
				Success_batch.reset()
				Precision_batch.reset()

	return Success_main.average, Precision_main.average


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--ngpu', type=int, default=1, help='# GPUs')
	parser.add_argument('--save_root_dir', type=str, default='kitti_results/Car', help='output folder')
	parser.add_argument('--data_dir', type=str, default='/opt/data/common/kitti_tracking/kitti_t_o/training',
	                    help='dataset path')
	parser.add_argument('--model', type=str, default='netR_30.pth', help='model name for training resume')
	parser.add_argument('--category_name', type=str, default='Car', help='Object to Track (Car/Pedestrian/Van/Cyclist)')
	parser.add_argument('--shape_aggregation', required=False, type=str, default="firstandprevious",
	                    help='Aggregation of shapes (first/previous/firstandprevious/all)')
	parser.add_argument('--reference_BB', required=False, type=str, default="previous_result",
	                    help='previous_result/previous_gt/current_gt')
	parser.add_argument('--IoU_Space', required=False, type=int, default=3, help='IoUBox vs IoUBEV (2 vs 3)')

	args = parser.parse_args()
 
	print(args)
 
	os.environ["CUDA_VISIBLE_DEVICES"] = '0'
	args.manualSeed = 1
	random.seed(args.manualSeed)
	torch.manual_seed(args.manualSeed)

	netR = Pointnet_Tracking(input_channels=0, use_xyz=True, opt=args)
	if args.ngpu > 1:
		netR = torch.nn.DataParallel(netR, range(args.ngpu))
	if args.model != '' and args.ngpu > 1:
		netR.load_state_dict(torch.load(os.path.join(args.save_root_dir, args.model)))
	elif args.model != '' and args.ngpu <= 1:
		state_dict_ = torch.load(os.path.join(args.save_root_dir, args.model),
									map_location=lambda storage, loc: storage)
		print('loaded {}'.format(os.path.join(args.save_root_dir, args.model)))
		state_dict = {}
		for k in state_dict_:
			if k.startswith('module') and not k.startswith('module_list'):
				state_dict[k[7:]] = state_dict_[k]
			else:
				state_dict[k] = state_dict_[k]
		netR.load_state_dict(state_dict)
	netR.cuda()
	torch.cuda.synchronize()
	# Car/Pedestrian/Van/Cyclist
	transforms = T.Compose([T.ToTensor(),
							T.Normalize(mean=[0.485, 0.456, 0.406],
										std=[0.229, 0.224, 0.225])
							])
	dataset_Test = SiameseTest(
		input_size=1024,
		path=args.data_dir,
		split='Test',
		category_name=args.category_name,
		offset_BB=0,
		scale_BB=1.25,
		augment=False,
		transforms=transforms,
		imgsize=(512, 160)
	)

	test_loader = torch.utils.data.DataLoader(
		dataset_Test,
		collate_fn=lambda x: x,
		batch_size=1,
		shuffle=False,
		num_workers=8,
		pin_memory=True)

	# one pass
	Succ, Prec = test(
				test_loader,
				netR,
				shape_aggregation=args.shape_aggregation,
				reference_BB=args.reference_BB,
				IoU_Space=args.IoU_Space,
				)
	print("mean Succ/Prec {:.2f}/{:.2f}".format(Succ, Prec))
