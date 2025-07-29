import os
import cv2
import random
import numpy as np
from torch.utils import data as data
from scipy import ndimage
import scipy
import scipy.stats as ss
from scipy.interpolate import interp2d
from scipy.linalg import orth
import mmcv
from utils.transforms import augment, paired_random_crop
from utils import FileClient, img2tensor
from utils.registry import DATASET_REGISTRY

from utils.data_util import make_dataset

def uint2single(img):
    return np.float32(img/255.)

def single2uint(img):
    return np.uint8((img.clip(0, 1)*255.).round())

def random_resize(img, scale_factor=1.):
    return cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)



'''
    train or test ↓↓↓
'''
@DATASET_REGISTRY.register()
class HazeOnlineDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(HazeOnlineDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder = opt['dataroot_gt']
        self.depth_folder = opt['dataroot_depth']
        self.seg_label = opt['dataroot_seg_ann']
        self.gt_paths = make_dataset(self.gt_folder)
        self.seg_label_paths = make_dataset(self.seg_label)
        self.depth_paths = make_dataset(self.depth_folder)
        self.beta_range = opt['beta_range']
        self.A_range = opt['A_range']
        self.color_p = opt['color_p']
        self.color_range = opt['color_range']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)


        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.gt_paths[index]

        img_gt = cv2.imread(gt_path).astype(np.float32) / 255.0

        depth_path = os.path.join(self.depth_folder, gt_path.split('/')[-1].split('.')[0] + '.npy')

        img_depth = np.load(depth_path)

        img_depth = 1 / img_depth
        gt_seg_maps = []
        seg_map = os.path.join(self.seg_label, gt_path.split('/')[-1].split('.')[0] + '.png')

        gt_seg_map = mmcv.imread(
            seg_map, flag='unchanged', backend='pillow')

        img_depth = (img_depth - img_depth.min()) / (img_depth.max() - img_depth.min())

        beta = np.random.rand(1) * (self.beta_range[1] - self.beta_range[0]) + self.beta_range[0]
        t = np.exp(-(1- img_depth) * 2.0 * beta)
        t = t[:, :, np.newaxis]
        A = np.random.rand(1) * (self.A_range[1] - self.A_range[0]) + self.A_range[0]
        if np.random.rand(1) < self.color_p:
            A_random = np.random.rand(3) * (self.color_range[1] - self.color_range[0]) + self.color_range[0]
            A = A + A_random


        img_lq = img_gt.copy()

        img_lq = img_lq * t + A * (1 - t)


        if img_gt.shape[-1] > 3:

            img_gt = img_gt[:, :, :3]
            img_lq = img_lq[:, :, :3]


        # augmentation for training
        if self.opt['phase'] == 'train':
            input_gt_size = np.min(img_gt.shape[:2])
            input_lq_size = np.min(img_lq.shape[:2])
            scale = input_gt_size // input_lq_size
            gt_size = self.opt['gt_size']

            if self.opt['use_resize_crop']:
                # random resize
                if input_gt_size > gt_size:
                    input_gt_random_size = random.randint(gt_size, input_gt_size)
                    input_gt_random_size = input_gt_random_size - input_gt_random_size % scale # make sure divisible by scale
                    resize_factor = input_gt_random_size / input_gt_size

                else:
                    resize_factor = (gt_size+1) / input_gt_size
                img_gt = random_resize(img_gt, resize_factor)
                img_lq = random_resize(img_lq, resize_factor)
                gt_seg_map = random_resize(gt_seg_map,resize_factor)
                t = random_resize(t, resize_factor)


                # random crop
                img_gt, img_lq, gt_seg_map= paired_random_crop(img_gt, img_lq,gt_seg_map,gt_size, input_gt_size // input_lq_size,
                                               gt_path)


            # flip, rotation
            img_gt, img_lq,gt_seg_map = augment([img_gt, img_lq,gt_seg_map], self.opt['use_flip'],
                                     self.opt['use_rot'])


        if self.opt['phase'] != 'train':
            crop_eval_size = self.opt.get('crop_eval_size', None)#  None
            if crop_eval_size:
                input_gt_size = img_gt.shape[0]
                input_lq_size = img_lq.shape[0]
                scale = input_gt_size // input_lq_size
                img_gt, img_lq,gt_seg_map = paired_random_crop(img_gt, img_lq,gt_seg_map, crop_eval_size, input_gt_size // input_lq_size,
                                               gt_path)


        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)


        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': gt_path,
            'gt_path': gt_path,
            'seg_ann': gt_seg_map
        }

    def __len__(self):
        return len(self.gt_paths)




'''
            test dataloader↓↓↓
'''

@DATASET_REGISTRY.register()
class HazeOnlineDataset_test_seg(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(HazeOnlineDataset_test_seg, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder = opt['dataroot_gt']
        self.depth_folder = opt['dataroot_depth']

        self.gt_paths = make_dataset(self.gt_folder)

        self.depth_paths = make_dataset(self.depth_folder)
        self.beta_range = opt['beta_range']
        self.A_range = opt['A_range']
        self.color_p = opt['color_p']
        self.color_range = opt['color_range']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)



        # Load gt and lq images. Dimension order: HWC; channel order: BGR;

        gt_path = self.gt_paths[index]

        img_gt = cv2.imread(gt_path).astype(np.float32) / 255.0

        depth_path = os.path.join(self.depth_folder, gt_path.split('/')[-1].split('.')[0] + '.npy')

        img_depth = np.load(depth_path)
        img_depth = 1 / img_depth

        img_depth = (img_depth - img_depth.min()) / (img_depth.max() - img_depth.min())

        beta = np.random.rand(1) * (self.beta_range[1] - self.beta_range[0]) + self.beta_range[0]
        t = np.exp(-(1 - img_depth) * 2.0 * beta)

        t = t[:, :, np.newaxis]

        A = np.random.rand(1) * (self.A_range[1] - self.A_range[0]) + self.A_range[0]
        if np.random.rand(1) < self.color_p:
            A_random = np.random.rand(3) * (self.color_range[1] - self.color_range[0]) + self.color_range[0]
            A = A + A_random

        img_lq = img_gt.copy()

        # add haze
        img_lq = img_lq * t + A * (1 - t)

        if img_gt.shape[-1] > 3:
            img_gt = img_gt[:, :, :3]
            img_lq = img_lq[:, :, :3]

        # augmentation for training
        if self.opt['phase'] == 'train':
            input_gt_size = np.min(img_gt.shape[:2])
            input_lq_size = np.min(img_lq.shape[:2])
            scale = input_gt_size // input_lq_size
            gt_size = self.opt['gt_size']

            if self.opt['use_resize_crop']:
                # random resize
                if input_gt_size > gt_size:
                    input_gt_random_size = random.randint(gt_size, input_gt_size)
                    input_gt_random_size = input_gt_random_size - input_gt_random_size % scale  # make sure divisible by scale
                    resize_factor = input_gt_random_size / input_gt_size
                else:
                    resize_factor = (gt_size + 1) / input_gt_size
                img_gt = random_resize(img_gt, resize_factor)
                img_lq = random_resize(img_lq, resize_factor)
                t = random_resize(t, resize_factor)

                # random crop
                img_gt, img_lq, = paired_random_crop(img_gt, img_lq, gt_size, input_gt_size // input_lq_size,
                                                     gt_path)

            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'],
                                     self.opt['use_rot'])

        if self.opt['phase'] != 'train':
            crop_eval_size = self.opt.get('crop_eval_size', None)  # None

            if crop_eval_size:
                input_gt_size = img_gt.shape[0]
                input_lq_size = img_lq.shape[0]
                scale = input_gt_size // input_lq_size
                img_gt, img_lq = paired_random_crop(img_gt, img_lq, crop_eval_size, input_gt_size // input_lq_size,
                                                    gt_path)

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': gt_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.gt_paths)

@DATASET_REGISTRY.register()
class HazeOnlineDataset_test_object(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(HazeOnlineDataset_test_object, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder = opt['dataroot_gt']
        self.depth_folder = opt['dataroot_depth']

        self.gt_paths = make_dataset(self.gt_folder)

        self.depth_paths = make_dataset(self.depth_folder)
        self.beta_range = opt['beta_range']
        self.A_range = opt['A_range']
        self.color_p = opt['color_p']
        self.color_range = opt['color_range']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)


        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.gt_paths[index]

        img_gt = cv2.imread(gt_path).astype(np.float32) / 255.0

        depth_path = os.path.join(self.depth_folder, gt_path.split('/')[-1].split('.')[0] + '.npy')
        img_depth = np.load(depth_path)
        img_depth = 1 / img_depth
        img_depth = (img_depth - img_depth.min()) / (img_depth.max() - img_depth.min())

        beta = np.random.rand(1) * (self.beta_range[1] - self.beta_range[0]) + self.beta_range[0]
        t = np.exp(-(1 - img_depth) * 2.0 * beta)
        t = t[:, :, np.newaxis]
        A = np.random.rand(1) * (self.A_range[1] - self.A_range[0]) + self.A_range[0]
        if np.random.rand(1) < self.color_p:
            A_random = np.random.rand(3) * (self.color_range[1] - self.color_range[0]) + self.color_range[0]
            A = A + A_random

        img_lq = img_gt.copy()

        # add haze

        img_lq = img_lq * t + A * (1 - t)


        if img_gt.shape[-1] > 3:
            img_gt = img_gt[:, :, :3]
            img_lq = img_lq[:, :, :3]

        # augmentation for training
        if self.opt['phase'] == 'train':
            input_gt_size = np.min(img_gt.shape[:2])
            input_lq_size = np.min(img_lq.shape[:2])
            scale = input_gt_size // input_lq_size
            gt_size = self.opt['gt_size']

            if self.opt['use_resize_crop']:
                # random resize
                if input_gt_size > gt_size:
                    input_gt_random_size = random.randint(gt_size, input_gt_size)
                    input_gt_random_size = input_gt_random_size - input_gt_random_size % scale  # make sure divisible by scale
                    resize_factor = input_gt_random_size / input_gt_size
                else:
                    resize_factor = (gt_size + 1) / input_gt_size
                img_gt = random_resize(img_gt, resize_factor)
                img_lq = random_resize(img_lq, resize_factor)
                t = random_resize(t, resize_factor)

                # random crop
                img_gt, img_lq, = paired_random_crop(img_gt, img_lq, gt_size, input_gt_size // input_lq_size,
                                                     gt_path)

            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'],
                                     self.opt['use_rot'])

        if self.opt['phase'] != 'train':
            crop_eval_size = self.opt.get('crop_eval_size', None)  # None

            if crop_eval_size:
                input_gt_size = img_gt.shape[0]
                input_lq_size = img_lq.shape[0]
                scale = input_gt_size // input_lq_size
                img_gt, img_lq = paired_random_crop(img_gt, img_lq, crop_eval_size, input_gt_size // input_lq_size,
                                                    gt_path)

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': gt_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.gt_paths)

@DATASET_REGISTRY.register()
class HazeOnlineDataset_test_depth(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Path to the text file with image information.
            beta_range (list): Range for beta values.
            A_range (list): Range for atmospheric light values.
            color_p (float): Probability for color adjustment.
            color_range (list): Range for random color adjustment.
    """

    def __init__(self, opt):
        super(HazeOnlineDataset_test_depth, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        # Read GT and depth image information from the test.txt file
        self.txt_path = opt['dataroot_gt']
        self.root_dir = os.path.dirname(self.txt_path)
        self.data_info = self._load_info(self.txt_path)

        self.beta_range = opt['beta_range']
        self.A_range = opt['A_range']
        self.color_p = opt['color_p']
        self.color_range = opt['color_range']

    def _load_info(self, file_path):
        """Load image information from the test.txt file."""
        data_info = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    folder, frame, lr = line.strip().split()
                    data_info.append((folder, frame, lr))
        return data_info

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # Get image information
        folder, frame, lr = self.data_info[index]

        # Determine the image folder and file name
        if lr == 'l':
            image_folder = 'image_02'
            depth_folder = 'imagedepth_02'
        elif lr == 'r':
            image_folder = 'image_03'
            depth_folder = 'imagedepth_03'
        else:
            raise ValueError(f"Invalid view: {lr}")

        # Construct the full paths
        gt_path = os.path.join(self.root_dir, folder, image_folder, 'data', f'{frame}.png')
        depth_path = os.path.join(self.root_dir, folder, depth_folder, f'{frame}.npy')

        # Load GT image and depth map
        img_gt = cv2.imread(gt_path).astype(np.float32) / 255.0
        img_depth = np.load(depth_path)
        img_depth = 1 / img_depth
        img_depth = (img_depth - img_depth.min()) / (img_depth.max() - img_depth.min())

        # Generate haze effects
        beta = np.random.rand(1) * (self.beta_range[1] - self.beta_range[0]) + self.beta_range[0]
        t = np.exp(-(1 - img_depth) * 2.0 * beta)
        t = t[:, :, np.newaxis]
        A = np.random.rand(1) * (self.A_range[1] - self.A_range[0]) + self.A_range[0]
        if np.random.rand(1) < self.color_p:
            A_random = np.random.rand(3) * (self.color_range[1] - self.color_range[0]) + self.color_range[0]
            A = A + A_random

        img_lq = img_gt * t + A * (1 - t)

        if img_gt.shape[-1] > 3:
            img_gt = img_gt[:, :, :3]
            img_lq = img_lq[:, :, :3]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': gt_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.data_info)

