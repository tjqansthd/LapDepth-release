from option import args, parser
import csv
import numpy as np
import torch
from torchvision import transforms, datasets
import torch.backends.cudnn as cudnn

#################### Distributed learning setting #######################
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
#########################################################################

import torch.optim as optim
import torch.nn as nn
import torch.utils.data

from datasets.datasets_list import MyDataset, Transformer

import os
from utils import *

from logger import TermLogger, AverageMeter
from trainer import validate, train_net
from model import LDRN

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    args.multigpu = False
    if args.distributed:
        args.multigpu = True
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                 world_size=args.world_size, rank=args.rank)
        args.batch_size = int(args.batch_size/ngpus_per_node)
        args.workers = int((args.num_workers + ngpus_per_node - 1)/ngpus_per_node)
        print("==> gpu:",args.gpu,", rank:",args.rank,", batch_size:",args.batch_size,", workers:",args.workers)
        torch.cuda.set_device(args.gpu)
    elif args.gpu is None:
        print("==> DataParallel Training")
        args.multigpu = True
        os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu_num
    else:
        print("==> Single GPU Training")
        torch.cuda.set_device(args.gpu)

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
        
    save_path = save_path_formatter(args, parser)
    args.save_path = 'checkpoints'/save_path
    if (args.rank == 0):
        print('=> number of GPU: ',args.gpu_num)
        print("=> information will be saved in {}".format(args.save_path))
    args.save_path.makedirs_p()
    torch.manual_seed(args.seed)

    ##############################    Data loading part    ################################
    if args.dataset == 'KITTI':
        args.max_depth = 80.0
    elif args.dataset == 'NYU':
        args.max_depth = 10.0

    train_set = MyDataset(args, train=True)
    test_set = MyDataset(args, train=False)

    if (args.rank == 0):
        print("=> Dataset: ",args.dataset)
        print("=> Data height: {}, width: {} ".format(args.height, args.width))
        print('=> train samples_num: {}  '.format(len(train_set)))
        print('=> test  samples_num: {}  '.format(len(test_set)))

    train_sampler = None
    test_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_set, shuffle=False)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=test_sampler)

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)
    cudnn.benchmark = True
    #########################################################################################

    ###################### Setting Network, Loss, Optimizer part ###################
    if (args.rank == 0):
        print("=> creating model")
    Model = LDRN(args)
    ############################### Number of model parameters ##############################
    num_params_encoder = 0
    num_params_decoder = 0
    for p in Model.encoder.parameters():
        num_params_encoder += p.numel()
    for p in Model.decoder.parameters():
        num_params_decoder += p.numel()
    if (args.rank == 0):
        print("===============================================")
        print("model encoder parameters: ", num_params_encoder)
        print("model decoder parameters: ", num_params_decoder)
        print("Total parameters: {}".format(num_params_encoder + num_params_decoder))
        trainable_params = sum([np.prod(p.shape) for p in Model.parameters() if p.requires_grad])
        print("Total trainable parameters: {}".format(trainable_params))
        print("===============================================")
    ############################### apex distributed package wrapping ########################
    if args.distributed:
        if args.norm == 'BN':
            Model = nn.SyncBatchNorm.convert_sync_batchnorm(Model)
            if (args.rank == 0):
                print("=> use SyncBatchNorm")
        Model = Model.cuda(args.gpu)
        Model = torch.nn.parallel.DistributedDataParallel(Model, device_ids=[args.gpu], output_device=args.gpu,
                                                           find_unused_parameters=True)
        print("=> Model Initialized on GPU: {} - Distributed Traning".format(args.gpu))
        enc_param = Model.module.encoder.parameters()
        dec_param = Model.module.decoder.parameters()
    elif args.gpu is None:
        Model = Model.cuda()
        Model = torch.nn.DataParallel(Model)
        print("=> Model Initialized - DataParallel")
        enc_param = Model.module.encoder.parameters()
        dec_param = Model.module.decoder.parameters()
    else:
        Model = Model.cuda(args.gpu)
        print("=> Model Initialized on GPU: {} - Single GPU training".format(args.gpu))
        enc_param = Model.encoder.parameters()
        dec_param = Model.decoder.parameters()
    
    ###########################################################################################

    ################################ pretrained model loading #################################
    if args.model_dir != '':
        #Model.load_state_dict(torch.load(args.model_dir,map_location='cuda:'+args.gpu_num))
        Model.load_state_dict(torch.load(args.model_dir))
        if (args.rank == 0):
            print('=> pretrained model is created')
    #############################################################################################


    ############################## optimizer and criterion setting ##############################
    optimizer = torch.optim.AdamW([{'params': Model.module.encoder.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
                                   {'params': Model.module.decoder.parameters(), 'weight_decay': 0, 'lr': args.lr}], eps=args.adam_eps)
    ##############################################################################################
    logger = None

    ####################################### Training part ##########################################

    if (args.rank == 0):
        print("training start!")

    loss = train_net(args, Model, optimizer, train_loader,val_loader, args.epochs,logger)

    if (args.rank == 0):
        print("training is finished")

if __name__ == '__main__':
    args.batch_size_dist = args.batch_size
    args.num_threads = args.workers
    args.world_size = 1
    args.rank = 0
    nodes = "127.0.0.1"
    ngpus_per_node = torch.cuda.device_count()
    args.num_workers = args.workers
    args.ngpus_per_node = ngpus_per_node

    if args.distributed:
        print("==> Distributed Training")
        mp.set_start_method('forkserver')

        print("==> Initial rank: ",args.rank)
        port = np.random.randint(10000, 10030)
        args.dist_url = 'tcp://{}:{}'.format(nodes, port)
        print("==> dist_url: ",args.dist_url)
        args.dist_backend = 'nccl'
        args.gpu = None
        args.workers = 9
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs = ngpus_per_node, args = (ngpus_per_node, args))
    else:
        if ngpus_per_node == 1:
            args.gpu = 0
        main_worker(args.gpu, ngpus_per_node, args)




