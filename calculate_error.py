from __future__ import division
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
import numpy as np


def compute_errors(gt_sparse, pred, crop=True, cap=80.0):
    abs_diff, abs_rel, sq_rel, a1, a2, a3,rmse_tot,rmse_log_tot = 0,0,0,0,0,0,0,0
    batch_size = gt_sparse.size(0)
    '''
    crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    construct a mask of False values, with the same size as target
    and then set to True values inside the crop
    '''
    
    for current_gt_sparse, current_pred in zip(gt_sparse, pred):
        h = current_gt_sparse.shape[1]
        w = current_gt_sparse.shape[2]
        if crop:
            crop_mask = current_gt_sparse != current_gt_sparse
            crop_mask = crop_mask[0,:,:]
            y1,y2 = int(0.40810811 * h), int(0.99189189 * h)
            x1,x2 = int(0.03594771 * w), int(0.96405229 * w)    ### Crop used by Garg ECCV 2016
            '''
            y1,y2 = int(0.3324324 * pred.size(2)), int(0.91351351 * pred.size(2))
            x1,x2 = int(0.0359477 * pred.size(3)), int(0.96405229 * pred.size(3))     ### Crop used by Godard CVPR 2017
            '''
            crop_mask[y1:y2,x1:x2] = 1

        current_gt_sparse = current_gt_sparse[0,:,:]
        current_pred = current_pred[0,:,:]
        
        
        valid = (current_gt_sparse < cap)&(current_gt_sparse>1e-3)
        if crop:
            valid = valid & crop_mask

        valid_gt = current_gt_sparse[valid].clamp(1e-3, cap)
        valid_pred = current_pred[valid]
        valid_pred = valid_pred.clamp(1e-3,cap)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()
        rmse = (valid_gt - valid_pred) ** 2
        rmse_tot += torch.sqrt(torch.mean(rmse))
        rmse_log = (torch.log(valid_gt) - torch.log(valid_pred)) ** 2
        rmse_log_tot += torch.sqrt(torch.mean(rmse_log))
        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)

    return [metric.item() / batch_size for metric in [abs_diff, abs_rel, sq_rel, a1, a2, a3,rmse_tot,rmse_log_tot]]

def compute_errors_NYU(gt, pred, crop=True):
    abs_diff, abs_rel, log10, a1, a2, a3,rmse_tot,rmse_log_tot = 0,0,0,0,0,0,0,0
    batch_size = gt.size(0)
    if crop:
        crop_mask = gt[0] != gt[0]
        crop_mask = crop_mask[0,:,:]
        crop_mask[45:471, 41:601] = 1

    for sparse_gt, pred in zip(gt, pred):
        sparse_gt = sparse_gt[0,:,:]
        pred = pred[0,:,:]
        h,w = sparse_gt.shape
        pred_uncropped = torch.zeros((h, w), dtype=torch.float32).cuda()
        pred_uncropped[42:474, 40:616] = pred
        pred = pred_uncropped

        valid = (sparse_gt < 10)&(sparse_gt > 1e-3)
        if crop:
            valid = valid & crop_mask
        valid_gt = sparse_gt[valid].clamp(1e-3, 10)
        valid_pred = pred[valid]
        valid_pred = valid_pred.clamp(1e-3,10)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()
        rmse = (valid_gt - valid_pred) ** 2
        rmse_tot += torch.sqrt(torch.mean(rmse))
        rmse_log = (torch.log(valid_gt) - torch.log(valid_pred)) ** 2
        rmse_log_tot += torch.sqrt(torch.mean(rmse_log))
        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        log10 += torch.mean(torch.abs(torch.log10(valid_gt)-torch.log10(valid_pred)))

    return [metric.item() / batch_size for metric in [abs_diff, abs_rel, log10, a1, a2, a3,rmse_tot,rmse_log_tot]]
