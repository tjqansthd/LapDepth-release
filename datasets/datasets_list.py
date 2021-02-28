import torch.utils.data as data
from PIL import Image
import numpy as np
from imageio import imread
import random
import torch
import time
import cv2
from PIL import ImageFile
from transform_list import RandomCropNumpy,EnhancedCompose,RandomColor,RandomHorizontalFlip,ArrayToTensorNumpy,Normalize
from torchvision import transforms
ImageFile.LOAD_TRUNCATED_IMAGES = True

def _is_pil_image(img):
    return isinstance(img, Image.Image)

class MyDataset(data.Dataset):
    def __init__(self, args, train=True, return_filename = False):
        self.use_dense_depth = args.use_dense_depth
        if train is True:
            if args.dataset == 'KITTI':
                self.datafile = args.trainfile_kitti
                self.angle_range = (-1, 1)
                self.depth_scale = 256.0
            elif args.dataset == 'NYU':
                self.datafile = args.trainfile_nyu
                self.angle_range = (-2.5, 2.5)
                self.depth_scale = 1000.0
                args.height = 416
                args.width = 544
        else:
            if args.dataset == 'KITTI':
                self.datafile = args.testfile_kitti
                self.depth_scale = 256.0
            elif args.dataset == 'NYU':
                self.datafile = args.testfile_nyu
                self.depth_scale = 1000.0
                args.height = 416
                args.width = 544
        self.train = train
        self.transform = Transformer(args)
        self.args = args
        self.return_filename = return_filename
        with open(self.datafile, 'r') as f:
            self.fileset = f.readlines()
        self.fileset = sorted(self.fileset)

    def __getitem__(self, index):
        divided_file = self.fileset[index].split()
        if self.args.dataset == 'KITTI':
            date = divided_file[0].split('/')[0] + '/'

        # Opening image files.   rgb: input color image, gt: sparse depth map
        rgb_file = self.args.data_path + '/' + divided_file[0]
        rgb = Image.open(rgb_file)
        gt = False
        gt_dense = False
        if (self.train is False):
            divided_file_ = divided_file[0].split('/')
            if self.args.dataset == 'KITTI':
                filename = divided_file_[1] + '_' + divided_file_[4]
            else:
                filename = divided_file_[0] + '_' + divided_file_[1]
            
            if self.args.dataset == 'KITTI':
                # Considering missing gt in Eigen split
                if divided_file[1] != 'None':
                    gt_file = self.args.data_path + '/data_depth_annotated/' + divided_file[1]
                    gt = Image.open(gt_file)
                    if self.use_dense_depth is True:
                        gt_dense_file = self.args.data_path + '/data_depth_annotated/' + divided_file[2]
                        gt_dense = Image.open(gt_dense_file)
                else:
                    pass
            elif self.args.dataset == 'NYU':
                gt_file = self.args.data_path + '/' + divided_file[1]
                gt = Image.open(gt_file)
                if self.use_dense_depth is True:
                    gt_dense_file = self.args.data_path + '/' + divided_file[2]
                    gt_dense = Image.open(gt_dense_file)
        else:
            angle = np.random.uniform(self.angle_range[0], self.angle_range[1])
            if self.args.dataset == 'KITTI':
                gt_file = self.args.data_path + '/data_depth_annotated/' + divided_file[1]
                if self.use_dense_depth is True:
                    gt_dense_file = self.args.data_path + '/data_depth_annotated/' + divided_file[2]
            elif self.args.dataset == 'NYU':
                gt_file = self.args.data_path + '/' + divided_file[1]
                if self.use_dense_depth is True:
                    gt_dense_file = self.args.data_path + '/' + divided_file[2]
            
            gt = Image.open(gt_file)            
            rgb = rgb.rotate(angle, resample=Image.BILINEAR)
            gt = gt.rotate(angle, resample=Image.NEAREST)
            if self.use_dense_depth is True:
                gt_dense = Image.open(gt_dense_file) 
                gt_dense = gt_dense.rotate(angle, resample=Image.NEAREST)

        # cropping in size that can be divided by 16
        if self.args.dataset == 'KITTI':
            h = rgb.height
            w = rgb.width
            bound_left = (w - 1216)//2
            bound_right = bound_left + 1216
            bound_top = h - 352
            bound_bottom = bound_top + 352
        elif self.args.dataset == 'NYU':
            if self.train is True:
                bound_left = 43
                bound_right = 608
                bound_top = 45
                bound_bottom = 472
            else:
                bound_left = 0
                bound_right = 640
                bound_top = 0
                bound_bottom = 480
        # crop and normalize 0 to 1 ==>  rgb range:(0,1),  depth range: (0, max_depth)
        
        if (self.args.dataset == 'NYU' and (self.train is False) and (self.return_filename is False)):
            rgb = rgb.crop((40,42,616,474))
        else:
            rgb = rgb.crop((bound_left,bound_top,bound_right,bound_bottom))
            
        rgb = np.asarray(rgb, dtype=np.float32)/255.0

        if _is_pil_image(gt):
            gt = gt.crop((bound_left,bound_top,bound_right,bound_bottom))
            gt = (np.asarray(gt, dtype=np.float32))/self.depth_scale
            gt = np.expand_dims(gt, axis=2)
            gt = np.clip(gt, 0, self.args.max_depth)
        if self.use_dense_depth is True:
            if _is_pil_image(gt_dense):
                gt_dense = gt_dense.crop((bound_left,bound_top,bound_right,bound_bottom))
                gt_dense = (np.asarray(gt_dense, dtype=np.float32))/self.depth_scale
                gt_dense = np.expand_dims(gt_dense, axis=2)
                gt_dense = np.clip(gt_dense, 0, self.args.max_depth)
                gt_dense = gt_dense * (gt.max()/gt_dense.max())

        rgb, gt, gt_dense = self.transform([rgb] + [gt] + [gt_dense], self.train)

        if self.return_filename is True:
            return rgb, gt, gt_dense, filename
        else:
            return rgb, gt, gt_dense

    def __len__(self):
        return len(self.fileset)

class Transformer(object):
    def __init__(self, args):
        if args.dataset == 'KITTI':
            self.train_transform = EnhancedCompose([
                RandomCropNumpy((args.height,args.width)),
                RandomHorizontalFlip(),
                [RandomColor(multiplier_range=(0.9, 1.1)), None, None],
                ArrayToTensorNumpy(),
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None, None]
            ])
            self.test_transform = EnhancedCompose([
                ArrayToTensorNumpy(),
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None, None]
            ])
        elif args.dataset == 'NYU':
            self.train_transform = EnhancedCompose([
                RandomCropNumpy((args.height,args.width)),
                RandomHorizontalFlip(),
                [RandomColor(multiplier_range=(0.8, 1.2),brightness_mult_range=(0.75, 1.25)), None, None],
                ArrayToTensorNumpy(),
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None, None]
            ])
            self.test_transform = EnhancedCompose([
                ArrayToTensorNumpy(),
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None, None]
            ])
    def __call__(self, images, train=True):
        if train is True:
            return self.train_transform(images)
        else:
            return self.test_transform(images)
