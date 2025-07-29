import itertools
import os
import argparse
import json
import shutil
import sys
from pathlib import Path
import yaml
import math
import mmcv
import numpy as np
import cv2
import PIL.Image as pil
import torch.multiprocessing as mp
from mmcv.runner import load_checkpoint
import itertools
from mmcv.parallel import DataContainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm
from setproctitle import setproctitle
import dehazing_model
from downstream.SegFormer.tools.test import segmodel_built
from datasets import build_dataset, build_dataloader
from datasets.data_sampler import EnlargedSampler
from downstream.RADepth.layers import get_smooth_loss
from downstream.RADepth.utils import readlines
from downstream.yolov5.utils.general import colorstr
from downstream.yolov5.utils.torch_utils import select_device
from pytorch_ssim import ssim
from utils import AverageMeter,ContrastLoss_vgg
from utils.options import parse_options
from downstream.yolov5.utils.loss import ComputeLoss

from downstream.yolov5.utils.dataloaders import create_dataloader
from downstream.yolov5.models.common import DetectMultiBackend
from downstream.RADepth import networks
from downstream.RADepth.datasets import KITTIRAWDataset
# from yolov5.models.common import DetectMultiBackend
import torchvision.utils as vutils

sys.path.append(os.path.join(os.path.dirname(__file__), 'downstream', 'yolov5'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'downstream', 'RADepth'))


parser = argparse.ArgumentParser()

parser.add_argument('--project_root', default='', type=str, help='Root path of the whole project')

parser.add_argument('--model', default='model-s', type=str, help='model name')
parser.add_argument('--premodel', default='premodel-s', type=str, help='model name')
parser.add_argument('--num_workers', default=2, type=int, help='number of workers')
parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--result_dir', type=str, help='path to result dataset')

# downstream yolov5
parser.add_argument("--data_od", type=str, help="dataset.yaml path")
parser.add_argument("--weight_od", type=str, help="cpk path")
parser.add_argument("--task_od_train_data", type=str, default="train")
parser.add_argument("--task_od_val_data", type=str, default="val")

# downstream segmentation
parser.add_argument('--seg_config', type=str, help='seg_config file path')
parser.add_argument('--seg_checkpoint', type=str, help='seg checkpoint file')

# downstream depth
parser.add_argument("--load_weights_folder", type=str, help="depth model to load")
parser.add_argument("--splits_dir_depth", type=str, help="path to the training data of depth task")
parser.add_argument("--depth_height", type=int, default=192, help="input image height of depth task")
parser.add_argument("--depth_width", type=int, default=640, help="input image width of depth task")

# text
parser.add_argument('--text_feature', type=str, help='text feature json path')

# logging and misc
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--dataset', default='ADE', type=str, help='dataset name')
parser.add_argument('--exp', default='model', type=str, help='experiment setting')
parser.add_argument('--exp_save', default='dehazing_datasets_best', type=str, help='experiment setting')
parser.add_argument('--cpk', default='dehazing_datasets_every', type=str, help='experiment setting')
parser.add_argument('--gpu', default='0', type=str, help='GPUs used for training')

args = parser.parse_args()


args.result_dir = f"{args.project_root}/seg_reslut/"
args.data_od = f"{args.project_root}/downstream/yolov5/data/coco.yaml"
args.weight_od = f"{args.project_root}/downstream/yolov5/best.pt"
args.seg_config = f"{args.project_root}/downstream/SegFormer/local_configs/segformer/B5/segformer.b5.640x640.ade.160k.py"
args.seg_checkpoint = f"{args.project_root}/downstream/SegFormer/cpk/segformer.b5.640x640.ade.160k.pth"
args.load_weights_folder = f"{args.project_root}/downstream/RADepth/models/RA-Depth/"
args.splits_dir_depth = f"{args.project_root}/downstream/RADepth/splits/eigen"
args.text_feature = f"{args.project_root}/text/text128.json"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

beta_1 = 0.1
beta_2 = 0.3 #beta_2 > beta_1

def train(train_loader, daloader_od,dataloader_depth,All_model,All_loss,Text_feature, optimizer,scaler):

    losses = AverageMeter()
    torch.cuda.empty_cache()
    All_model['network'].train()
    seg_loader = itertools.cycle(train_loader)
    depth_loaer = itertools.cycle(dataloader_depth)
    object_loader = iter(daloader_od)

    for object_img_gt, object_img_lq, targets, _, _ in object_loader:
        '''
            dehazing and seg task!!!
        '''
        batch = next(seg_loader)
        source_img = batch['lq'].cuda()
        target_img = batch['gt'].cuda()
        seg_ann = batch['seg_ann'].cuda()
        pre_output = All_model['pre_network'](source_img)
        loss_h_seg = All_loss['criterion'](pre_output, target_img)
        loss_e_seg = All_loss['criterion'](source_img, target_img)
        img_metas = [[{'ori_shape': (256, 256, 3),
                       'img_shape': (256, 256, 3),
                       'pad_shape': (256, 256, 3),
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
        output = All_model['network'](source_img,pre_output,pre_outputs_seg,Text_feature['seg'],task='seg')
        '''
            seg task!!!!!
        '''
        output_list = [output]


        batch_seg = {
            'img': output_list,
            'img_metas': DataContainer(img_metas).data
        }


        outputs_seg = All_model['seg_model'](return_loss=False, **batch_seg)
        outputs = np.stack(outputs_seg, axis=0)
        outputs = torch.tensor(outputs, dtype=torch.float32)
        outputs = outputs.cuda()
        outputs = outputs.unsqueeze(1)
        outputs = outputs.repeat(1, 151, 1, 1)
        seg_ann = seg_ann.long()
        entropy_loss = All_loss['criterion_entropy'](outputs, seg_ann)
        loss_cr = All_loss['criterion_cr'](output, target_img, source_img)
        loss = All_loss['criterion'](output, target_img)
        loss_p_seg = loss
        '''
            dehazing and depth task !!!
        '''
        data_depth = next(depth_loaer)
        depth_img = data_depth[("color_MiS", 0, 0)].cuda()
        depth_hazy_img = data_depth[("img_lq_MiS", 0, 0)].cuda()
        pre_output_depth_dehaze = All_model['pre_network'](depth_hazy_img)
        loss_h_depth = All_loss['criterion'](pre_output_depth_dehaze, depth_img)
        loss_e_depth = All_loss['criterion'](depth_hazy_img, depth_img)
        pre_output_depth = All_model['depth_decoder'](All_model['depth_encoder'](pre_output_depth_dehaze))
        pre_output_depth = pre_output_depth[("disp", 0)]
        Text_feature['depth'] = Text_feature['depth'].cuda()
        output_depth_dehaze = All_model['network'](depth_hazy_img,pre_output_depth_dehaze,pre_output_depth,Text_feature['depth'],task='depth')

        '''
            depth task
        '''
        output_depth = All_model['depth_decoder'](All_model['depth_encoder'](output_depth_dehaze))  # 去雾图像输入到深度估计网络的结果
        disp_output_depth = output_depth[("disp", 0)]
        target_depth = All_model['depth_decoder'](All_model['depth_encoder'](depth_img))
        disp_target_depth = target_depth[("disp", 0)]
        mean_disp_output_depth = disp_output_depth.mean(2, True).mean(3, True)
        norm_disp_output_depth = disp_output_depth / (mean_disp_output_depth + 1e-7)
        smooth_loss_MiS = All_loss['get_smooth_loss'](norm_disp_output_depth, depth_img) + All_loss['criterion'](disp_output_depth,disp_target_depth)
        dehaze_depth_loss1 = All_loss['criterion'](output_depth_dehaze, depth_img)
        loss_p_depth = dehaze_depth_loss1
        dehaze_depth_loss_cr = All_loss['criterion_cr'](output_depth_dehaze, depth_img, depth_hazy_img)

        '''
            dehazing and object task !!!
        '''
        object_img_lq = object_img_lq.float()
        object_img_lq = object_img_lq.cuda()
        object_img_gt = object_img_gt.float().cuda()
        targets = targets.cuda()
        object_img_lq = object_img_lq.to(torch.float32) / 255.0
        object_img_gt = object_img_gt.to(torch.float32) / 255.0
        pre_output_object = All_model['pre_network'](object_img_lq)
        loss_h_od = All_loss['criterion'](pre_output_object, object_img_gt)
        loss_e_od = All_loss['criterion'](object_img_lq,object_img_gt)
        output_object_backbone = All_model['model_od_backbone'](pre_output_object)
        Text_feature['od'] = Text_feature['od'].cuda()
        output_object = All_model['network'](object_img_lq,pre_output_object,output_object_backbone,Text_feature['od'],task='od')
        pred, output_object1 = All_model['model_od'](output_object)
        loss_cr_object = All_loss['criterion_cr'](output_object, object_img_gt, object_img_lq)
        loss_object_1 = All_loss['criterion'](output_object, object_img_gt)
        loss_computeLoss = All_loss['ComputeLoss'](output_object1, targets)[0]
        loss_p_od = loss_object_1
        loss_p = loss_p_seg + loss_p_od + loss_p_depth
        loss_h = loss_h_od + loss_h_depth + loss_h_seg
        loss_e = loss_e_od + loss_e_seg + loss_e_depth
        loss_ivc = max(loss_p - loss_h + beta_1, torch.tensor(0.).cuda()) + max(loss_p - loss_e + beta_2,torch.tensor(0.).cuda())

        loss = loss + loss_object_1+dehaze_depth_loss1 +0.1*( loss_cr_object + dehaze_depth_loss_cr+loss_cr)+0.01*( entropy_loss + loss_computeLoss + smooth_loss_MiS)

        loss = loss + loss_ivc
        losses.update(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    return losses.avg



'''
    test three data!!!
'''
def valid(val_loader,val_object_loader,val_depth_loader ,All_model,Text_feature):

    def evaluate(loader,task='seg'):
        PSNR = AverageMeter()
        SSIM = AverageMeter()
        for batch in loader:
            source_img = batch['lq'].cuda()
            target_img = batch['gt'].cuda()
            if task == 'seg':
                with torch.no_grad():
                    B,C,H,W = source_img.shape
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

                    output = All_model['network'](source_img, pre_output, pre_outputs_seg, Text_feature['seg'], task='seg')
            elif task == 'od':
                with torch.no_grad():
                    pre_output_object = All_model['pre_network'](source_img)
                    output_object_backbone = All_model['model_od_backbone'](pre_output_object)
                    Text_feature['od'] = Text_feature['od'].cuda()
                    output = All_model['network'](source_img, pre_output_object, output_object_backbone,
                                                         Text_feature['od'], task='od')
            else:
                with torch.no_grad():
                    pre_output_depth_dehaze = All_model['pre_network'](source_img)
                    _,_,original_height,original_width = pre_output_depth_dehaze.shape
                    pre_output_depth_dehaze1 = F.interpolate(pre_output_depth_dehaze, size=(192, 640), mode='bilinear', align_corners=False)
                    pre_output_depth = All_model['depth_decoder'](All_model['depth_encoder'](pre_output_depth_dehaze1))
                    pre_output_depth = pre_output_depth[("disp", 0)]
                    pre_output_depth = torch.nn.functional.interpolate(
                        pre_output_depth, (original_height, original_width), mode="bilinear", align_corners=False)
                    Text_feature['depth'] = Text_feature['depth'].cuda()
                    output = All_model['network'](source_img, pre_output_depth_dehaze, pre_output_depth,Text_feature['depth'], task='depth')
            mse_loss = F.mse_loss(output, target_img, reduction='none').mean((1, 2, 3))
            psnr = 10 * torch.log10(1 / mse_loss).mean()
            PSNR.update(psnr.item(), source_img.size(0))

            ssim_value = ssim(output, target_img, size_average=True)
            SSIM.update(ssim_value.item(), source_img.size(0))

        return PSNR.avg, SSIM.avg

    torch.cuda.empty_cache()
    network.eval()
    val_psnr, val_ssim = evaluate(val_loader,task='seg')
    val_object_psnr, val_object_ssim = evaluate(val_object_loader,task='od')
    val_depth_psnr, val_depth_ssim = evaluate(val_depth_loader,task='depth')
    avg_psnr_three = (val_psnr * len(val_loader.dataset) + val_object_psnr * len(val_object_loader.dataset) + val_depth_psnr * len(val_depth_loader.dataset)) / (len(val_loader.dataset) + len(val_object_loader.dataset) + len(val_depth_loader.dataset))
    avg_ssim_three =  (val_ssim * len(val_loader.dataset) + val_object_ssim * len(val_object_loader.dataset) + val_depth_ssim * len(val_depth_loader.dataset)) / (len(val_loader.dataset) + len(val_object_loader.dataset) + len(val_depth_loader.dataset))
    return {
        'val_psnr': val_psnr,
        'val_ssim': val_ssim,
        'val_object_psnr': val_object_psnr,
        'val_object_ssim': val_object_ssim,
        'val_depth_psnr': val_depth_psnr,
        'val_depth_ssim': val_depth_ssim,
        'avg_psnr': avg_psnr_three,
        'avg_ssim': avg_ssim_three
    }



def create_train_val_dataloader(opt):

    train_loader, val_loaders = None, []
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':

            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))


        elif phase == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        elif phase == 'val_object':
            val_object_set = build_dataset(dataset_opt)
            val_object_loader = build_dataloader(
                val_object_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        elif phase == 'val_depth':
            val_depth_set = build_dataset(dataset_opt)
            val_depth_loader = build_dataloader(
                val_depth_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')
    return train_loader, train_sampler, val_loader,val_object_loader,val_depth_loader, total_epochs, total_iters

if __name__ == '__main__':
    seg_cfg = mmcv.Config.fromfile(args.seg_config)
    setting_filename = os.path.join('configs', args.exp, args.model + '.json')
    if not os.path.exists(setting_filename):
        setting_filename = os.path.join('configs', args.exp, 'default.json')
    with open(setting_filename, 'r') as f:
        setting = json.load(f)

    root_path = args.project_root
    opt, args_configs1 = parse_options(root_path, is_train=True)
    network = eval(args.model.replace('-', '_'))()
    pre_network = eval(args.premodel.replace('-', '_'))()
    device = select_device(args.gpu, batch_size=opt['datasets']['train']["batch_size_per_gpu"])
    network = nn.DataParallel(network).cuda()
    pre_network = nn.DataParallel(pre_network).cuda()

    '''
        downstream network!!!
    '''
    #object network
    model_od = DetectMultiBackend(args.weight_od,device=device,dnn=False,data=args.data_od,fp16=False)
    ckpt_object = torch.load(args.weight_od,map_location='cpu')
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

    #seg network
    seg_model = segmodel_built(seg_cfg,args.seg_checkpoint,map_location='cpu')
    seg_model.eval()
    seg_model.cuda()


    save_dir = os.path.join(args.save_dir, args.exp_save)
    save_dir1 = os.path.join(args.save_dir, args.cpk)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir1, exist_ok=True)



    criterion = nn.L1Loss()
    criterion_cr = ContrastLoss_vgg()
    criterion_entropy = nn.CrossEntropyLoss(ignore_index=150)
    ComputeLoss = ComputeLoss(model_od)
    if setting['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'])
    elif setting['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])
    else:
        raise Exception("ERROR: unsupported optimizer")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'],
                                                           eta_min=setting['lr'] * 1e-2)

    scaler = GradScaler()


    '''
        load pre_network cpt
    '''
    if os.path.exists(os.path.join(save_dir, 'pre_model.pth')):
        pre_checkpoint = torch.load(os.path.join(save_dir, 'pre_model.pth'))
        pre_network.load_state_dict(pre_checkpoint['state_dict'])



    if os.path.exists(os.path.join(save_dir,'model.pth')):
        print('==> Continue training from existing model')
        checkpoint = torch.load(os.path.join(save_dir, 'model.pth'))
        network.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        best_psnr = checkpoint['best_psnr']


        old_checkpoint = torch.load(os.path.join(save_dir, 'pre_model.pth'))
        old_state_dict = old_checkpoint['state_dict']

        current_state_dict = network.state_dict()

        matched_keys = [k for k, v in old_state_dict.items() if
                        k in current_state_dict and current_state_dict[k].shape == v.shape]


        for name, param in network.named_parameters():
            if name in matched_keys:
                param.requires_grad = False
                print(f"Freezing param: {name}")
        print('==>best_psnr:',best_psnr)
    else:
        print('==> Starting training, current model name: ' + args.model)
        checkpoint = torch.load(os.path.join(save_dir, 'model.pth'))
        state_dict_a = checkpoint['state_dict']
        state_dict_b = network.state_dict()
        matched_dict = {k: v for k, v in state_dict_a.items() if k in state_dict_b and state_dict_b[k].shape == v.shape}
        network.load_state_dict(matched_dict, strict=False)
        for name, param in network.named_parameters():
            if name in matched_dict:
                param.requires_grad = False
        best_psnr = 0
        start_epoch = 0

    '''
        seg dataloader and test dataloader↓
    '''
    result = create_train_val_dataloader(opt)
    train_loader, train_sampler, val_loader,val_object_loader,val_depth_loader, total_epochs, total_iters = result


    '''
        object dataloader↓
    '''

    with open(args.data_od, 'r') as file:
        data_od = yaml.safe_load(file)
    daloader_od = create_dataloader(
        opt,
        opt['datasets']['train']['dataroot_object'],
        opt['datasets']['train']['gt_size'],
        opt['datasets']['train']["batch_size_per_gpu"],
        model_od.stride,
        single_cls=False,
        pad=0.5,
        rect=True,
        workers=2,
        prefix=colorstr(f"{args.task_od_train_data}: "),
    )[0]


    filenames_depth = readlines(os.path.join(args.splits_dir_depth, "train_files5000.txt"))
    dataset_depth =  KITTIRAWDataset(opt,opt['datasets']['train']['dataroot_depth_task'],filenames_depth,args.depth_height,args.depth_width,[0],4,is_train=True,img_ext='.png',depth_img_ext='.npy')
    dataloader_depth = DataLoader(dataset_depth, opt['datasets']['train']["batch_size_per_gpu"], shuffle=False,
                                  num_workers=2,
                                  pin_memory=True, drop_last=False,persistent_workers=True)


    All_model={
        'network':network,
        'pre_network':pre_network,
        'seg_model':seg_model,
        'model_od':model_od,
        'model_od_backbone':model_od_backbone,
        'depth_encoder':depth_encoder,
        'depth_decoder':depth_decoder,
    }
    All_loss={
        'criterion':criterion,
        'criterion_cr':criterion_cr,
        'criterion_entropy':criterion_entropy,
        'ComputeLoss':ComputeLoss,
        'get_smooth_loss':get_smooth_loss,

    }
    values_list = []
    with open(args.text_feature,'r') as f:
        for line in f:
            data = json.loads(line)
            values = data['features'][0]['layers'][0]['values']
            values_list.append(values)
    Text_feature ={
        'seg':values_list[0],
        'od':values_list[1],
        'depth':values_list[2]

    }
    Text_feature = {
        'seg': torch.tensor(values_list[0], dtype=torch.float).unsqueeze(0),  # [1, 128]
        'od': torch.tensor(values_list[1], dtype=torch.float).unsqueeze(0),  # [1, 128]
        'depth': torch.tensor(values_list[2], dtype=torch.float).unsqueeze(0)  # [1, 128]
    }

    for epoch in tqdm(range(start_epoch, setting['epochs'] + 1), initial=start_epoch, total=setting['epochs'] + 1):
        loss = train(train_loader,daloader_od,dataloader_depth, All_model,All_loss,Text_feature,optimizer,scaler)

        scheduler.step()

        if epoch % setting['eval_freq'] == 0:
            result_valid = valid(val_loader,val_object_loader,val_depth_loader, All_model,Text_feature)

            if result_valid['avg_psnr'] > best_psnr:
                best_psnr = result_valid['avg_psnr']
                torch.save({'state_dict': network.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': scheduler.state_dict(),
                            'scaler': scaler.state_dict(),
                            'epoch': epoch,
                            'best_psnr':best_psnr
                            },
                           os.path.join(save_dir, 'model.pth'.format(epoch)))
            torch.save({'state_dict': network.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': scheduler.state_dict(),
                        'scaler': scaler.state_dict(),
                        'epoch': epoch,
                        'best_psnr':result_valid['avg_psnr']
                        },
                        os.path.join(save_dir1, 'model{}_{}.pth'.format(epoch,format(result_valid['avg_psnr'],'.3f'))))


            os.makedirs('./checkpoint', exist_ok=True)
            with open('./checkpoint/loss-1.txt', 'a') as file:
                file.write('Epoch [{}/{}], Loss: {:.4f}'
                           .format(epoch + 1, setting['epochs'], loss))
                file.write('Best PSNR: {:.4f}\n'.format(best_psnr))
                file.write('Val PSNR: {:.4f}\n'.format(result_valid['avg_psnr']))
                file.write('Val SSIM: {:.4f}\n'.format(result_valid['avg_ssim']))
                file.write('\n')
            print('seg_psnr:',result_valid['val_psnr'],'seg_ssim:',result_valid['val_ssim'])
            print('object_psnr:',result_valid['val_object_psnr'],'object_ssim:',result_valid['val_object_ssim'])
            print('depth_psnr:',result_valid['val_depth_psnr'],'depth_ssim:',result_valid['val_depth_ssim'])
            print('loss:', loss,'best_psnr:', best_psnr,'avg_psnr:',result_valid['avg_psnr'],'avg_ssim:',result_valid['avg_ssim'])


