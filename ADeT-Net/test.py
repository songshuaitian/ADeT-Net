import json
import os
import argparse

import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.parallel import DataContainer
from pytorch_msssim import ssim
from collections import OrderedDict

from torchvision.utils import save_image

from datasets import build_dataset, build_dataloader
from utils import AverageMeter, write_img, chw_to_hwc, parse_options
from downstream.yolov5.models.common import DetectMultiBackend
from downstream.yolov5.utils.torch_utils import select_device
from downstream.RADepth import networks
from downstream.SegFormer.tools.test import segmodel_built
parser = argparse.ArgumentParser()
parser.add_argument('--project_root', default='', type=str, help='Root path of the whole project')
parser.add_argument('--model', default='model-s', type=str, help='model name')
parser.add_argument('--premodel', default='premodel-s', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--result_dir', default='./results/', type=str, help='path to results saving')
parser.add_argument('--dataset', default='model', type=str, help='dataset name')
parser.add_argument('--exp', default='best', type=str, help='experiment setting')
parser.add_argument('--hazy', default='hazy', type=str, help='experiment setting')
parser.add_argument('--gpu', default='0', type=str, help='GPUs used for training')


args = parser.parse_args()
args.data_od = f"{args.project_root}/downstream/yolov5/data/coco.yaml"
args.weight_od = f"{args.project_root}/downstream/yolov5/best.pt"
args.seg_config = f"{args.project_root}/downstream/SegFormer/local_configs/segformer/B5/segformer.b5.640x640.ade.160k.py"
args.seg_checkpoint = f"{args.project_root}/downstream/SegFormer/cpk/segformer.b5.640x640.ade.160k.pth"
args.load_weights_folder = f"{args.project_root}/downstream/RADepth/models/RA-Depth/"
args.splits_dir_depth = f"{args.project_root}/downstream/RADepth/splits/eigen"
args.text_feature = f"{args.project_root}/text/text128.json"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def single(save_dir):
	state_dict = torch.load(save_dir)['state_dict']
	new_state_dict = OrderedDict()

	for k, v in state_dict.items():
		name = k[7:]
		new_state_dict[name] = v

	return new_state_dict
def load_weights_for_model(save_dir, model):
	state_dict = torch.load(save_dir)['state_dict']
	model_state_dict = model.state_dict()
	new_state_dict = OrderedDict()
	for k, v in state_dict.items():
		name = k[7:]
		if name in model_state_dict:
			new_state_dict[name] = v
		else:
			print(f"Layer {name} not found in model, skipping.")

	model.load_state_dict(new_state_dict, strict=False)

	return model

def test(test_loader, All_model,Text_feature, result_dir,task):
	PSNR = AverageMeter()
	SSIM = AverageMeter()

	torch.cuda.empty_cache()

	network.eval()
	if task is 'od':
		os.makedirs(os.path.join(result_dir, 'images','val2017'), exist_ok=True)
	else:
		os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)
	f_result = open(os.path.join(result_dir, 'results.csv'), 'w')

	for idx, batch in enumerate(test_loader):
		source_img = batch['lq'].cuda()
		target = batch['gt'].cuda()
		if task is 'depth':
			folder = batch['folder'][0]
			filename = os.path.basename(batch['lq_path'][0])
			os.makedirs(os.path.join(result_dir, 'imgs', folder, 'image_02', 'data'), exist_ok=True)
		else:
			filename = os.path.basename(batch['lq_path'][0])


		if task == 'seg':
			with torch.no_grad():

				B, C, H, W = source_img.shape
				pre_output = All_model['pre_network'](source_img)
				img_metas = [[{'ori_shape': (H, W, C),
							   'img_shape': (H, W, C),
							   'pad_shape': (H, W, C),
							   'flip': False,
							   'flip_direction': 'horizontal'}]]
				pre_output_list = [pre_output]
				pre_batch_seg = {
					'img': pre_output_list,
					'img_metas': DataContainer(img_metas).data
				}
				pre_outputs_seg = All_model['seg_model'](return_loss=False, **pre_batch_seg)
				Text_feature['seg'] = Text_feature['seg'].cuda()
				pre_outputs_seg = np.stack(pre_outputs_seg, axis=0)
				pre_outputs_seg = torch.tensor(pre_outputs_seg, dtype=torch.float32)
				pre_outputs_seg = pre_outputs_seg.cuda()
				pre_outputs_seg = pre_outputs_seg.unsqueeze(1)
				pre_outputs_seg = F.interpolate(pre_outputs_seg, size=(H, W), mode='bilinear',
												align_corners=False)

				output = All_model['network'](source_img, pre_output, pre_outputs_seg, Text_feature['seg'],
											  task='seg').clamp_(-1, 1)
				psnr_val = 10 * torch.log10(1 / F.mse_loss(output, target)).item()

				_, _, H, W = output.size()
				down_ratio = max(1, round(min(H, W) / 256))
				ssim_val = ssim(F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))),
								F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))),
								data_range=1, size_average=False).item()
		elif task == 'od':
			with torch.no_grad():

				pre_output_object = All_model['pre_network'](source_img)
				output_object_backbone = All_model['model_od_backbone'](pre_output_object)

				Text_feature['od'] = Text_feature['od'].cuda()
				output = All_model['network'](source_img, pre_output_object, output_object_backbone,Text_feature['od'], task='od').clamp_(-1, 1)
				psnr_val = 10 * torch.log10(1 / F.mse_loss(output, target)).item()

				_, _, H, W = output.size()
				down_ratio = max(1, round(min(H, W) / 256))
				ssim_val = ssim(F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))),
								F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))),
								data_range=1, size_average=False).item()
		else:
			with torch.no_grad():

				pre_output_depth_dehaze = All_model['pre_network'](source_img)
				_, _, original_height, original_width = pre_output_depth_dehaze.shape
				pre_output_depth_dehaze1 = F.interpolate(pre_output_depth_dehaze, size=(192, 640), mode='bilinear',
														 align_corners=False)

				pre_output_depth = All_model['depth_decoder'](All_model['depth_encoder'](pre_output_depth_dehaze1))
				pre_output_depth = pre_output_depth[("disp", 0)]
				pre_output_depth = torch.nn.functional.interpolate(
					pre_output_depth, (original_height, original_width), mode="bilinear", align_corners=False)

				Text_feature['depth'] = Text_feature['depth'].cuda()

				output = All_model['network'](source_img, pre_output_depth_dehaze, pre_output_depth,
											  Text_feature['depth'], task='depth').clamp_(-1, 1)
				psnr_val = 10 * torch.log10(1 / F.mse_loss(output, target)).item()

				_, _, H, W = output.size()
				down_ratio = max(1, round(min(H, W) / 256))  # Zhou Wang
				ssim_val = ssim(F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))),
								F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))),
								data_range=1, size_average=False).item()
		PSNR.update(psnr_val)
		SSIM.update(ssim_val)

		print('Test: [{0}]\t'
			  'PSNR: {psnr.val:.02f} ({psnr.avg:.02f})\t'
			  'SSIM: {ssim.val:.04f} ({ssim.avg:.04f})'
			  .format(idx, psnr=PSNR, ssim=SSIM))

		f_result.write('%s,%.02f,%.03f\n'%(filename, psnr_val, ssim_val))



		if task == 'od':
			save_image(output, os.path.join(result_dir, 'images', 'val2017', filename))
		elif task == 'depth':
			save_image(output, os.path.join(result_dir, 'imgs', folder, 'image_02', 'data', filename))
		else:
			save_image(output, os.path.join(result_dir, 'imgs', filename))



	f_result.close()

	os.rename(os.path.join(result_dir, 'results.csv'), 
			  os.path.join(result_dir, '%.02f | %.04f.csv'%(PSNR.avg, SSIM.avg)))


if __name__ == '__main__':
	network = eval(args.model.replace('-', '_'))()
	network.cuda()
	pre_network = eval(args.premodel.replace('-', '_'))()
	pre_network.cuda()
	saved_model_dir = os.path.join(args.save_dir, args.exp, 'model.pth')
	saved_model_dir_pre = os.path.join(args.save_dir, args.exp, 'pre_model.pth')
	root_path = ''#Current project path
	opt, _ = parse_options(root_path, is_train=False)
	if os.path.exists(saved_model_dir):
		print('==> Start testing, current model name: ' + args.model)
		network.load_state_dict(single(saved_model_dir))
	else:
		print('==> No existing trained model!')
		exit(0)
	if os.path.exists(saved_model_dir):
		print('==> Start testing, current premodel name: ' + args.model)
		pre_network = load_weights_for_model(saved_model_dir, pre_network)

	values_list = []
	with open(args.text_feature, 'r') as f:
		for line in f:
			data = json.loads(line)
			values = data['features'][0]['layers'][0]['values']
			values_list.append(values)
	Text_feature = {
		'seg': values_list[0],
		'od': values_list[1],
		'depth': values_list[2]

	}
	Text_feature = {
		'seg': torch.tensor(values_list[0], dtype=torch.float).unsqueeze(0),  # [1, 128]
		'od': torch.tensor(values_list[1], dtype=torch.float).unsqueeze(0),  # [1, 128]
		'depth': torch.tensor(values_list[2], dtype=torch.float).unsqueeze(0)  # [1, 128]
	}

	device = select_device(args.gpu, batch_size=1)

	# object network
	model_od = DetectMultiBackend(args.weight_od, device=device, dnn=False, data=args.data_od, fp16=False)
	ckpt_object = torch.load(args.weight_od, map_location='cpu')
	model_od.hyp = ckpt_object.get('opt').get('hyp')
	model_od_backbone = nn.Sequential(*list(model_od.model.model.children())[:5])

	#depth network
	encoder_dict = torch.load(os.path.join(args.load_weights_folder, "encoder.pth"))
	depth_encoder = networks.hrnet18(False)
	depth_decoder = networks.DepthDecoder_MSF(depth_encoder.num_ch_enc, [0], num_output_channels=1)
	model_dict = depth_encoder.state_dict()
	depth_encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
	depth_decoder.load_state_dict(torch.load(os.path.join(args.load_weights_folder, "depth.pth")))
	depth_encoder.cuda()
	depth_encoder.eval()
	depth_decoder.cuda()
	depth_decoder.eval()

	# seg network
	seg_cfg = mmcv.Config.fromfile(args.seg_config)
	seg_model = segmodel_built(seg_cfg, args.seg_checkpoint, map_location='cpu')
	seg_model.eval()
	seg_model.cuda()
	All_model = {
		'network': network,
		'pre_network': pre_network,
		'seg_model': seg_model,
		'model_od': model_od,
		'model_od_backbone': model_od_backbone,
		'depth_encoder': depth_encoder,
		'depth_decoder': depth_decoder,
	}




	#seg data
	val_seg_opt = opt['datasets']['val']
	test_seg_set = build_dataset(val_seg_opt)
	test_seg_loader = build_dataloader(
		test_seg_set, val_seg_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
	result_dir_seg = os.path.join(args.result_dir, args.dataset, args.model,'seg')

	#object data
	val_object_opt = opt['datasets']['val_object']
	test_object_set = build_dataset(val_object_opt)
	test_object_loader = build_dataloader(
		test_object_set, val_object_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
	result_dir_od = os.path.join(args.result_dir, args.dataset, args.model, 'od')

	#depth data
	val_depth_opt = opt['datasets']['val_depth']
	test_depth_set = build_dataset(val_depth_opt)
	test_depth_loader = build_dataloader(
		test_depth_set, val_depth_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None,
		seed=opt['manual_seed'])
	result_dir_depth = os.path.join(args.result_dir, args.dataset, args.model,'depth')


	print('seg test is start!')
	test(test_seg_loader, All_model, Text_feature,result_dir_seg,task='seg')

	print('od test is start!')
	test(test_object_loader, All_model, Text_feature,result_dir_od,task='od')
	print('depth test is start!')
	test(test_depth_loader, All_model,Text_feature, result_dir_depth,task='depth')