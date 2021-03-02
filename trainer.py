import numpy as np
import random
from utils import *
from logger import TermLogger, AverageMeter
from tensorboardX import SummaryWriter
import time
from calculate_error import *
import torch
from torch.autograd import Variable
from torchvision.utils import save_image
import csv
import os
import imageio
from tqdm import tqdm
from path import Path
import warnings
warnings.filterwarnings(action='ignore')

def validate(args, val_loader, model, logger, dataset = 'KITTI'):
    ##global device
    batch_time = AverageMeter()
    if dataset == 'KITTI':
        error_names = ['abs_diff', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3','rmse','rmse_log']
    elif dataset == 'NYU':
        error_names = ['abs_diff', 'abs_rel', 'log10', 'a1', 'a2', 'a3','rmse','rmse_log']
    elif dataset == 'Make3D':
        error_names = ['abs_diff', 'abs_rel', 'ave_log10', 'rmse']

    errors = AverageMeter(i=len(error_names))

    # switch to evaluate mode
    model.eval()
    end = time.time()
    logger.valid_bar.update(0)
    for i, (rgb_data, gt_data, _) in enumerate(val_loader):
        if gt_data.ndim != 4 and gt_data[0] == False:
            continue
        end = time.time()
        rgb_data = rgb_data.cuda()
        gt_data = gt_data.cuda()

        # compute output
        input_img = rgb_data
        input_img_flip = torch.flip(input_img,[3])
        with torch.no_grad():
            _, output_depth = model(input_img)
            batch_time.update(time.time() - end)
            _, output_depth_flip = model(input_img_flip)
            output_depth_flip = torch.flip(output_depth_flip,[3])
            output_depth = 0.5*(output_depth + output_depth_flip)
            
        if dataset == 'KITTI':
            err_result = compute_errors(gt_data, output_depth,crop=True, cap=args.cap)
        elif dataset == 'NYU':
            err_result = compute_errors_NYU(gt_data, output_depth,crop=True)
        elif dataset == 'Make3D':
            err_result = compute_errors_Make3D(depth, output_depth)

        errors.update(err_result)
        # measure elapsed time
        logger.valid_bar.update(i+1)
        if i % 10 == 0:
            logger.valid_writer.write('valid: Time {} Abs Error {:.4f} ({:.4f})'.format(batch_time, errors.val[0], errors.avg[0]))

    logger.valid_bar.update(len(val_loader))

    return errors.avg,error_names

def validate_in_test(args, val_loader, model, logger, dataset = 'KITTI'):
    
    # switch to evaluate mode
    model.eval()

    ##global device
    if dataset == 'KITTI':
        error_names = ['abs_diff', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3','rmse','rmse_log']
    elif dataset == 'NYU':
        error_names = ['abs_diff', 'abs_rel', 'log10', 'a1', 'a2', 'a3','rmse','rmse_log']
    elif dataset == 'Make3D':
        error_names = ['abs_diff', 'abs_rel', 'ave_log10', 'rmse']
    errors = AverageMeter(i=len(error_names))
    # order: abs_diff, abs_rel, sq_rel, a1, a2, a3, rmse, rmse_log

    for i, (rgb_data, gt_data, _) in enumerate(val_loader):
        if gt_data.ndim != 4 and gt_data[0] == False:
            continue
        rgb_data = rgb_data.cuda()
        gt_data = gt_data.cuda()

        # compute output
        with torch.no_grad():
            _, output_depth = model(rgb_data)
        
        if dataset == 'KITTI':
            err_result = compute_errors(gt_data, output_depth,crop=True, cap=args.cap)
        elif dataset == 'NYU':
            err_result = compute_errors_NYU(gt_data, output_depth,crop=True)
        elif dataset == 'Make3D':
            err_result = compute_errors_Make3D(depth, output_depth)
        errors.update(err_result)
        if i == 101:
            break
    a1 = errors.avg[3]
    rmse_loss = errors.avg[6]

    # turn back to train mode
    model.train()
    return a1, rmse_loss

def train_net(args,model, optimizer, dataset_loader,val_loader, n_epochs,logger):
    num = 0
    model_num = 0    
    
    data_iter = iter(dataset_loader)
    rgb_fixed, depth_fixed, _ = next(data_iter)
    depth_fixed = depth_fixed.cuda()
    
    save_dir = './' + args.dataset + '_LDRN_' + args.encoder + '_epoch' + str(n_epochs+5)
    
    if (args.rank == 0):
        print("Training for %d epochs..." % (n_epochs+5))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    

    H = args.height
    W = args.width

    test_loss_dir = Path(args.save_path)
    test_loss_dir_rmse = str(test_loss_dir/'test_rmse_list.txt')
    test_loss_dir = str(test_loss_dir/'test_loss_list.txt')
    train_loss_dir = Path(args.save_path)
    train_loss_dir_rmse = str(train_loss_dir/'train_rmse_list.txt')
    a1_acc_dir = str(train_loss_dir/'a1_acc_list.txt')
    train_loss_dir = str(train_loss_dir/'train_loss_list.txt')
    loss_pdf = "train_loss.pdf"
    rmse_pdf = "train_rmse.pdf"
    a1_pdf = "train_a1.pdf"        
    
    if args.dataset == "KITTI":
        # create mask for gradient loss
        y1_c,y2_c = int(0.40810811 * depth_fixed.size(2)), int(0.99189189 * depth_fixed.size(2))
        x1_c,x2_c = int(0.03594771 * depth_fixed.size(3)), int(0.96405229 * depth_fixed.size(3))    ### Crop used by Garg ECCV 2016 
        y1,y2 = int(0.3324324 * H), int(0.99189189 * H)
        if (args.rank == 0):
            print(" - valid y range: %d ~ %d"%(y1,y2))
        crop_mask = depth_fixed != depth_fixed
        crop_mask[:,:,y1:y2,:] = 1
        crop_mask_a1 = depth_fixed != depth_fixed
        crop_mask_a1[:,:,y1_c:y2_c,x1_c:x2_c] = 1
    else:
        crop_mask = None

    loss_list = []
    rmse_list = []
    train_loss_list = []
    train_rmse_list = []
    a1_acc_list = []
    num_cnt = 0
    train_loss_cnt = 0

    n_iter = 0
    iter_per_epoch = len(dataset_loader)
    base_lr = args.lr
    end_lr = args.end_lr
    total_iter = n_epochs * iter_per_epoch
    ################ train mode ####################
    model.train()
    ################################################
    for epoch in tqdm(range(n_epochs+5)):
        dataset_loader.sampler.set_epoch(epoch)
        random.seed(epoch)
        np.random.seed(epoch)               # numpy 관련 무작위 고정
        torch.manual_seed(epoch)            # cpu 연산 무작위 고정
        torch.cuda.manual_seed(epoch)       # gpu 연산 무작위 고정
        torch.cuda.manual_seed_all(epoch)   # 멀티 gpu 연산 무작위 고정
        ####################################### one epoch training #############################################
        for i, (rgb_data, gt_data, gt_dense) in enumerate(dataset_loader):

            # get the inputs
            inputs = rgb_data
            depths = gt_data
            inputs = inputs.cuda()
            depths = depths.cuda()
            inputs, depths = Variable(inputs), Variable(depths)
            if args.use_dense_depth is True:
                dense_depths = gt_dense
                dense_depths = dense_depths.cuda()
                dense_depths = Variable(dense_depths)
            
            '''Network loss'''
            # Feed-forward pass
            d_res_list, outputs = model(inputs)
            if args.lv6 is True:
                [lap6_pred, lap5_pred, lap4_pred, lap3_pred, lap2_pred, lap1_pred] = d_res_list
            else:
                [lap5_pred, lap4_pred, lap3_pred, lap2_pred, lap1_pred] = d_res_list
            ##################################### Valid mask definition ####################################
            # masking valied area
            valid_mask, final_mask = make_mask(depths, crop_mask, args.dataset)

            valid_out = outputs[valid_mask]
            valid_gt_sparse = depths[valid_mask]

            ###################################### scale invariant loss #####################################
            scale_inv_loss = scale_invariant_loss(valid_out, valid_gt_sparse)
            
            ###################################### gradient loss ############################################
            grad_epoch = 15 if args.dataset == 'KITTI' else 20
            if args.use_dense_depth is True:
                if epoch < grad_epoch:
                    gradient_loss = torch.tensor(0.).cuda()
                else:
                    gradient_loss = imgrad_loss(outputs, dense_depths, final_mask)
                    gradient_loss = 0.1*gradient_loss
            else:
                gradient_loss = torch.tensor(0.).cuda()

            loss = scale_inv_loss + gradient_loss

            # zero the parameter gradients and backward & optimize
            optimizer.zero_grad()
            loss.backward()

            if n_iter == total_iter:
                current_lr = end_lr
            else:
                current_lr = (base_lr - end_lr) * (1 - n_iter / total_iter) ** 0.5 + end_lr
                n_iter += 1

            optimizer.param_groups[0]['lr'] = current_lr
            optimizer.param_groups[1]['lr'] = current_lr
            optimizer.step()
            
            if ((i+1) % (iter_per_epoch//2) == 0) and (args.rank == 0):
                torch.save(model.state_dict(), save_dir+'/epoch_%02d_loss_%.4f_1.pkl' %(model_num+1,loss))
            if ((i+1) % args.print_freq == 0) and (args.rank == 0):
                print("epoch: %d,  %d/%d"%(epoch+1,i+1,args.epoch_size))
                print("[%6d/%6d]  total: %.5f, gradient: %.5f, scale_inv: %.5f"%(n_iter, total_iter, loss.item(),gradient_loss.item(),scale_inv_loss.item()))
                total_loss = loss.item()                    
                rmse_loss = (torch.sqrt(torch.pow(valid_out.detach()-valid_gt_sparse,2))).mean()
                rmse_loss = rmse_loss.item()
                train_loss_cnt = train_loss_cnt + 1
                train_plot(args.save_path,total_loss, rmse_loss, train_loss_list, train_rmse_list, train_loss_dir,train_loss_dir_rmse,loss_pdf, rmse_pdf, train_loss_cnt,True)
                
                if args.val_in_train is True:
                    print("=> validate...")
                    a1_acc, rmse_test_loss = validate_in_test(args, val_loader, model, logger, args.dataset)
                    validate_plot(args.save_path,a1_acc, a1_acc_list, a1_acc_dir,a1_pdf, train_loss_cnt,True)         

        if (args.rank == 0):
            print("=> learning decay... current lr: %.6f"%(current_lr))
            torch.save(model.state_dict(), save_dir+'/epoch_%02d_loss_%.4f_2.pkl' %(model_num+1,loss))
        model_num = model_num + 1
    
    return loss


if __name__ == "__main__":
    main()
