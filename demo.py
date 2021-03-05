import torch
import numpy as np
from model import LDRN
import glob
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import os
import imageio

parser = argparse.ArgumentParser(description='Laplacian Depth Residual Network training on KITTI',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Directory setting 

parser.add_argument('--model_dir',type=str, default = '')
parser.add_argument('--img_dir', type=str, default = None)
parser.add_argument('--img_folder_dir', type=str, default= None)

# Dataloader setting
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')

# Model setting
parser.add_argument('--encoder', type=str, default = "ResNext101")
parser.add_argument('--pretrained', type=str, default = "KITTI")
parser.add_argument('--norm', type=str, default = "BN")
parser.add_argument('--n_Group', type=int, default = 32)
parser.add_argument('--reduction', type=int, default = 16)
parser.add_argument('--act', type=str, default = "ReLU")
parser.add_argument('--max_depth', default=80.0, type=float, metavar='MaxVal', help='max value of depth')
parser.add_argument('--lv6', action='store_true', help='use lv6 Laplacian decoder')

# GPU setting
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--gpu_num', type=str, default = "0,1,2,3", help='force available gpu index')
parser.add_argument('--rank', type=int,   help='node rank for distributed training', default=0)


args = parser.parse_args()

assert (args.img_dir is not None) or (args.img_folder_dir is not None), "Expected name of input image file or folder"

if args.cuda and torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu_num
    cudnn.benchmark = True
    print('=> on CUDA')
else:
    print('=> on CPU')

if args.pretrained == 'KITTI':
    args.max_depth = 80.0
elif args.pretrained == 'NYU':
    args.max_depth = 10.0

print('=> loading model..')
Model = LDRN(args)
if args.cuda and torch.cuda.is_available():
    Model = Model.cuda()
Model = torch.nn.DataParallel(Model)
assert (args.model_dir != ''), "Expected pretrained model directory"
Model.load_state_dict(torch.load(args.model_dir))
Model.eval()

if args.img_dir is not None:
    if args.img_dir[-1] == '/':
        args.img_dir = args.img_dir[:-1]
    img_list = [args.img_dir]
    result_filelist = ['./out_' + args.img_dir.split('/')[-1]]
elif args.img_folder_dir is not None:
    if args.img_folder_dir[-1] == '/':
        args.img_folder_dir = args.img_folder_dir[:-1]
    png_img_list = glob.glob(args.img_folder_dir + '/*.png')
    jpg_img_list = glob.glob(args.img_folder_dir + '/*.jpg')
    img_list = png_img_list + jpg_img_list
    img_list = sorted(img_list)
    result_folder = './out_' + args.img_folder_dir.split('/')[-1]
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    result_filelist = []
    for file in img_list:
        result_filename = result_folder + '/out_' + file.split('/')[-1]
        result_filelist.append(result_filename)

print("=> process..")
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

for i, img_file in enumerate(img_list):
    img = Image.open(img_file)
    img = np.asarray(img, dtype=np.float32)/255.0
    if img.ndim == 2:
        img = np.expand_dims(img,2)
        img = np.repeat(img,3,2)
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).float()
    img = normalize(img)
    if args.cuda and torch.cuda.is_available():
        img = img.cuda()
    
    _, org_h, org_w = img.shape

    # new height and width setting which can be divided by 16
    img = img.unsqueeze(0)

    if args.pretrained == 'KITTI':
        new_h = 352
        new_w = org_w * (352.0/org_h)
        new_w = int((new_w//16)*16)
        img = F.interpolate(img, (new_h, new_w), mode='bilinear')
    elif args.pretrained == 'NYU':
        new_h = 432
        new_w = org_w * (432.0/org_h)
        new_w = int((new_w//16)*16)
        img = F.interpolate(img, (new_h, new_w), mode='bilinear')

    # depth prediction
    #with torch.no_grad():
    #    _, out = Model(img)

    img_flip = torch.flip(img,[3])
    with torch.no_grad():
        _, out = Model(img)
        _, out_flip = Model(img_flip)
        out_flip = torch.flip(out_flip,[3])
        out = 0.5*(out + out_flip)

    if new_h > org_h:
        out = F.interpolate(out, (org_h, org_w), mode='bilinear')
    out = out[0,0]
    
    if args.pretrained == 'KITTI':
        out = out[int(out.shape[0]*0.18):,:]
        out = out*256.0
    elif args.pretrained == 'NYU':
        out = out*1000.0
    out = out.cpu().detach().numpy().astype(np.uint16)
    out = (out/out.max())*255.0
    result_filename = result_filelist[i]
    plt.imsave(result_filename ,np.log10(out), cmap='plasma_r')
    if (i+1)%10 == 0:
        print("=>",i+1,"th image is processed..")

print("=> Done.")


