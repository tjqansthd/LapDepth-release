# -*- coding: utf-8 -*-
import argparse
import time
import csv
import math
import numpy as np
import os

import torch
from torch.autograd import Variable
from torchvision import transforms, datasets
import cv2

import torch.backends.cudnn as cudnn
import torch.distributed as dist

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from datasets.datasets_list import MyDataset, Transformer

from itertools import cycle
from tqdm import tqdm
import imageio
import imageio.core.util
import itertools
from path import Path

import matplotlib.pyplot as plt
from PIL import Image

from utils import *
from logger import TermLogger, AverageMeter
from tensorboardX import SummaryWriter
from trainer import validate
from model import *

parser = argparse.ArgumentParser(description='Laplacian Depth Residual Network training on KITTI',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Directory setting 
parser.add_argument('--models_list_dir', type=str, default='')
parser.add_argument('--result_dir', type=str, default='')
parser.add_argument('--model_dir',type=str)
parser.add_argument('--trainfile_kitti', type=str, default = "./datasets/eigen_train_files_with_gt_dense.txt")
parser.add_argument('--testfile_kitti', type=str, default = "./datasets/eigen_test_files_with_gt_dense.txt")
parser.add_argument('--trainfile_nyu', type=str, default = "./datasets/nyudepthv2_train_files_with_gt_dense.txt")
parser.add_argument('--testfile_nyu', type=str, default = "./datasets/nyudepthv2_test_files_with_gt_dense.txt")
parser.add_argument('--data_path', type=str, default = "./datasets/KITTI")
parser.add_argument('--use_dense_depth', action='store_true', help='using dense depth data for gradient loss')

# Dataloader setting
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epoch_size', default=0, type=int, metavar='N', help='manual epoch size (will match dataset size if not set)')
parser.add_argument('--epochs', default=0, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', default=0, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--batch_size', default=24, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--dataset', type=str, default = "KITTI")

# Logging setting
parser.add_argument('--print-freq', default=100, type=int, metavar='N', help='print frequency')
parser.add_argument('--log-metric', default='_LRDN_evaluation.csv', metavar='PATH', help='csv where to save validation metric value')
parser.add_argument('--val_in_train', action='store_true', help='validation process in training')

# Model setting
parser.add_argument('--encoder', type=str, default = "ResNext101")
parser.add_argument('--norm', type=str, default = "BN")
parser.add_argument('--act', type=str, default = "ReLU")
parser.add_argument('--height', type=int, default = 352)
parser.add_argument('--width', type=int, default = 704)
parser.add_argument('--max_depth', default=80.0, type=float, metavar='MaxVal', help='max value of depth')
parser.add_argument('--lv6', action='store_true', help='use lv6 Laplacian decoder')

# Evaluation setting
parser.add_argument('--evaluate', action='store_true', help='evaluate score')
parser.add_argument('--multi_test', action='store_true', help='test all of model in the dir')
parser.add_argument('--img_save', action='store_true', help='will save test set image')
parser.add_argument('--cap', default=80.0, type=float, metavar='MaxVal', help='cap setting for kitti eval')

# GPU parallel process setting
parser.add_argument('--gpu_num', type=str, default = "0,1,2,3", help='force available gpu index')
parser.add_argument('--rank',                      type=int,   help='node rank for distributed training', default=0)

def silence_imageio_warning(*args, **kwargs):
    pass

imageio.core.util._precision_warn = silence_imageio_warning

def main():
    args = parser.parse_args() 
    print("=> No Distributed Training")
    print('=> Index of using GPU: ',args.gpu_num)
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu_num
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
    
    torch.manual_seed(args.seed)

    if args.evaluate is True:
        save_path = save_path_formatter(args, parser)
        args.save_path = 'checkpoints'/save_path
        print("=> information will be saved in {}".format(args.save_path))
        args.save_path.makedirs_p()
        training_writer = SummaryWriter(args.save_path)    
    
    ######################   Data loading part    ##########################
    if args.dataset == 'KITTI':
        args.max_depth = 80.0
    elif args.dataset == 'NYU':
        args.max_depth = 10.0

    if args.result_dir == '':
        args.result_dir = './' + args.dataset + '_Eval_results'
    args.log_metric = args.dataset + '_' + args.encoder + args.log_metric
    
    test_set = MyDataset(args, train=False)
    print("=> Dataset: ",args.dataset)
    print("=> Data height: {}, width: {} ".format(args.height, args.width))
    print('=> test  samples_num: {}  '.format(len(test_set)))

    test_sampler = None

    val_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=test_sampler)
    
    cudnn.benchmark = True
    ###########################################################################
    
    ###################### setting model list #################################
    if args.multi_test is True:
        print("=> all of model tested")
        models_list_dir = Path(args.models_list_dir)
        models_list = sorted(models_list_dir.files('*.pkl'))
    else:
        print("=> just one model tested")
        models_list = [args.model_dir]


    ###################### setting Network part ###################
    print("=> creating model")
    Model = LDRN(args)

    num_params_encoder = 0
    num_params_decoder = 0
    for p in Model.encoder.parameters():
        num_params_encoder += p.numel()
    for p in Model.decoder.parameters():
        num_params_decoder += p.numel()
    print("===============================================")
    print("model encoder parameters: ", num_params_encoder)
    print("model decoder parameters: ", num_params_decoder)
    print("Total parameters: {}".format(num_params_encoder + num_params_decoder))
    print("===============================================")
    Model = Model.cuda()
    Model = torch.nn.DataParallel(Model)

    if args.evaluate is True:
        ############################ data log #######################################
        logger = TermLogger(n_epochs=args.epochs, train_size=min(len(val_loader), args.epoch_size), valid_size=len(val_loader))
        with open(args.save_path/args.log_metric, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            if args.dataset == 'KITTI':
                writer.writerow(['Filename','Abs_diff', 'Abs_rel','Sq_rel','a1','a2','a3','RMSE','RMSE_log'])
            elif args.dataset == 'Make3D':
                writer.writerow(['Filename','Abs_diff', 'Abs_rel','log10','rmse'])
            elif args.dataset == 'NYU':
                writer.writerow(['Filename','Abs_diff', 'Abs_rel','log10','a1','a2','a3','RMSE','RMSE_log'])
        ########################### Evaluating part #################################
        test_model = Model

        print("Model Initialized")


        test_len = len(models_list)
        print("=> Length of model list: ",test_len)

        for i in range(test_len):
            filename = models_list[i].split('/')[-1]
            logger.reset_valid_bar()
            test_model.load_state_dict(torch.load(models_list[i],map_location='cuda:0'))
            #test_model.load_state_dict(torch.load(models_list[i]))
            test_model.eval()
            if args.dataset == 'KITTI':
                errors, error_names = validate(args, val_loader, test_model, logger,'KITTI')
            elif args.dataset == 'NYU':
                errors, error_names = validate(args, val_loader, test_model, logger,'NYU')
            for error, name in zip(errors, error_names):
                training_writer.add_scalar(name, error, 0)
            logger.valid_writer.write(' * model: {}'.format(models_list[i]))
            print("")
            error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names[0:len(error_names)], errors[0:len(errors)]))
            logger.valid_writer.write(' * Avg {}'.format(error_string))
            print("")
            logger.valid_bar.finish()
            with open(args.save_path/args.log_metric, 'a') as csvfile:
                writer = csv.writer(csvfile, delimiter='\t')
                writer.writerow(['%s'%filename]+['%.4f'%(errors[k]) for k in range(len(errors))])

        print(args.dataset," valdiation finish")
        ##  Test
        
        if args.img_save is False :
            print("--only Test mode finish--")
            return
    else:
        test_model = Model
        test_model.load_state_dict(torch.load(models_list[0],map_location='cuda:0'))
        #test_model.load_state_dict(torch.load(models_list[0]))
        test_model.eval()
        print("=> No validation")


    test_set = MyDataset(args, train=False, return_filename=True)
    test_sampler = None
    val_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=test_sampler)


    if args.img_save is True:
        cmap = plt.cm.jet
        print("=> img save start")
        for idx, (rgb_data, gt_data, gt_dense, filename) in enumerate(val_loader):
            if gt_data.ndim != 4 and gt_data[0] == False:
                continue
            img_H = gt_data.shape[2]
            img_W = gt_data.shape[3]
            gt_data = Variable(gt_data.cuda())
            input_img = Variable(rgb_data.cuda())
            gt_data = gt_data.clamp(0, args.max_depth)
            if args.use_dense_depth is True:
                gt_dense = Variable(gt_dense.cuda())
                gt_dense = gt_dense.clamp(0, args.max_depth)

            input_img_flip = torch.flip(input_img,[3])
            with torch.no_grad():
                _,final_depth = test_model(input_img)
                _,final_depth_flip = test_model(input_img_flip)
            final_depth_flip = torch.flip(final_depth_flip,[3])
            final_depth = 0.5*(final_depth + final_depth_flip)
            
            final_depth = final_depth.clamp(0,args.max_depth)
            d_min = min(final_depth.min(), gt_data.min())
            d_max = max(final_depth.max(), gt_data.max())
            
            d_min = d_min.cpu().detach().numpy().astype(np.float32)
            d_max = d_max.cpu().detach().numpy().astype(np.float32)

            filename = filename[0]
            img_arr = [final_depth,final_depth, final_depth, gt_data, rgb_data, gt_dense,gt_dense,gt_dense]
            folder_name_list = ['/output_depth', '/output_depth_cmap_gray', '/output_depth_cmap_jet','/ground_truth','/input_rgb','/dense_gt','/dense_gt_cmap_gray','/dense_gt_cmap_jet']
            img_name_list = ['/'+filename, '/cmap_gray_'+filename,'/cmap_jet_'+filename,'/gt_'+filename,'/rgb_'+filename,'/gt_dense_'+filename,'/gt_dense_cmap_gray_'+filename,'/gt_dense_cmap_jet_'+filename]
            if args.use_dense_depth is False:
                img_arr = img_arr[:5]
                folder_name_list = folder_name_list[:5]
                img_name_list = img_name_list[:5]

            folder_iter = cycle(folder_name_list)
            img_name_iter = cycle(img_name_list)
            for img in img_arr:
                folder_name = next(folder_iter)
                img_name = next(img_name_iter)
                if folder_name == '/output_depth_cmap_gray' or folder_name =='/dense_gt_cmap_gray':
                    if args.dataset == 'NYU':
                        img = img*1000.0
                        img = img.cpu().detach().numpy().astype(np.uint16)
                        img_org = img.copy()
                    else:
                        img = img*256.0
                        img = img.cpu().detach().numpy().astype(np.uint16)
                        img_org = img.copy()
                elif folder_name == '/output_depth_cmap_jet' or folder_name == '/dense_gt_cmap_jet':
                    img_org = img
                else:
                    img = (img/img.max())*255.0
                    img_org = img.cpu().detach().numpy().astype(np.float32)
                result_dir = args.result_dir + folder_name
                for t in range(img_org.shape[0]):
                    img = img_org[t]
                    if folder_name == '/output_depth_cmap_jet' or folder_name == '/dense_gt_cmap_jet':
                        img_ = np.squeeze(img.cpu().numpy().astype(np.float32))
                        img_ = ((img_ - d_min)/(d_max - d_min))
                        img_ = cmap(img_)[:,:,:3]*255
                    else:
                        if img.shape[0] == 3:
                            img_ = np.empty([img_H,img_W,3]).astype(img.dtype)
                            '''
                            img_[:,:,2] = img[0,:,:]
                            img_[:,:,1] = img[1,:,:]
                            img_[:,:,0] = img[2,:,:]        # for BGR
                            '''
                            img_ = img.transpose(1,2,0)     # for RGB
                        elif img.shape[0] == 1:
                            img_ = np.ones([img_H,img_W]).astype(img.dtype)
                            img_[:,:] = img[0,:,:]
                    if not os.path.exists(result_dir):
                        os.makedirs(result_dir)
                    if folder_name == '/output_depth_cmap_gray' or folder_name =='/dense_gt_cmap_gray':
                        plt.imsave(result_dir + img_name ,np.log10(img_),cmap='Greys')
                    elif folder_name == '/output_depth_cmap_jet' or folder_name =='/dense_gt_cmap_jet':
                        img_ = Image.fromarray(img_.astype('uint8'))
                        img_.save(result_dir + img_name)
                    else:
                        imageio.imwrite(result_dir + img_name,img_)
            if (idx+1)%10 == 0:
                print(idx+1,"th image is processed..")
        print("--Test image save finish--")
    return

if __name__ == "__main__":
    main()