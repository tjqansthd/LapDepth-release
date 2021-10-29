# LapDepth-release
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/monocular-depth-estimation-using-laplacian/monocular-depth-estimation-on-kitti-eigen)](https://paperswithcode.com/sota/monocular-depth-estimation-on-kitti-eigen?p=monocular-depth-estimation-using-laplacian)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/monocular-depth-estimation-using-laplacian/monocular-depth-estimation-on-nyu-depth-v2)](https://paperswithcode.com/sota/monocular-depth-estimation-on-nyu-depth-v2?p=monocular-depth-estimation-using-laplacian)

This repository is a Pytorch implementation of the paper [**"Monocular Depth Estimation Using Laplacian Pyramid-Based Depth Residuals"**](https://ieeexplore.ieee.org/document/9316778)

<p align="center"><img src='https://github.com/tjqansthd/LapDepth-release/blob/master/example/FIG1.png' width=800></p>  

Minsoo Song, Seokjae Lim, and [Wonjun Kim*](https://sites.google.com/view/dcvl)  
IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)

## Video presentation
[![Screenshot](https://img.youtube.com/vi/mbK4ysUX8oo/maxresdefault.jpg)](https://youtu.be/mbK4ysUX8oo)

## Requirements

* Python >= 3.7
* Pytorch >= 1.6.0
* Ubuntu 16.04
* CUDA 9.2
* cuDNN (if CUDA available)

some other packages: geffnet, path, IPython, blessings, progressbar

## Pretrained models
You can download pre-trained model
* [Trained with KITTI](https://drive.google.com/file/d/10Fsw3KbhiKj-rRkoIesghSPCn84TYY5P/view?usp=sharing)
   - batch 16, SyncBatchNorm, data loss

   |  cap  |  a1   |  a2   |  a3   | Abs Rel | Sq Rel | RMSE  | RMSE log |
   | :---: | :---: | :---: | :---: | :-----: | :----: | :---: | :------: |
   | 0-80m | 0.965 | 0.995 | 0.999 |  0.059  | 0.201  | 2.397 |  0.090   |

   |  cap  |  a1   |  a2   |  a3   | Abs Rel | Sq Rel | RMSE  | RMSE log |
   | :---: | :---: | :---: | :---: | :-----: | :----: | :---: | :------: |
   | 0-50m | 0.970 | 0.996 | 0.999 |  0.057  | 0.155  | 1.788 |  0.085   |
* [Trained with KITTI](https://drive.google.com/file/d/1Xxxo3Zw4kVwBw43i6akQqjX2Go8vdP4H/view?usp=sharing)
   - batch 16, GroupNorm, data loss + gradient loss

   |  cap  |  a1   |  a2   |  a3   | Abs Rel | Sq Rel | RMSE  | RMSE log |
   | :---: | :---: | :---: | :---: | :-----: | :----: | :---: | :------: |
   | 0-80m | 0.961 | 0.994 | 0.999 |  0.059  | 0.209  | 2.489 |  0.091   |

   |  cap  |  a1   |  a2   |  a3   | Abs Rel | Sq Rel | RMSE  | RMSE log |
   | :---: | :---: | :---: | :---: | :-----: | :----: | :---: | :------: |
   | 0-50m | 0.968 | 0.996 | 0.999 |  0.057  | 0.155  | 1.807 |  0.085   |
* [Trained with NYU Depth V2](https://drive.google.com/file/d/13WyHCmQINyzprCerkOBT_Pf_W-PbWzBi/view?usp=sharing)
   - batch 16, SyncBatchNorm, data loss

   |  cap  |  a1   |  a2   |  a3   | Abs Rel | log10 | RMSE  | RMSE log |
   | :---: | :---: | :---: | :---: | :-----: | :---: | :---: | :------: |
   | 0-10m | 0.895 | 0.983 | 0.996 |  0.105  | 0.045 | 0.384 |  0.135   |

## Demo images (Single Test Image Prediction)
Make sure you download the pre-trained model and placed it in the './pretrained/' directory before running the demo.  
Demo Command Line:
```bash
############### Example of argument usage #####################
## Running demo using a specified image (jpg or png)
python demo.py --model_dir ./pretrained/LDRN_KITTI_ResNext101_pretrained_data.pkl --img_dir ./your/file/path/filename --pretrained KITTI --cuda --gpu_num 0
python demo.py --model_dir ./pretrained/LDRN_NYU_ResNext101_pretrained_data.pkl --img_dir ./your/file/path/filename --pretrained NYU --cuda --gpu_num 0
# output image name => 'out_' + filename

## Running demo using a whole folder of images
python demo.py --model_dir ./pretrained/LDRN_KITTI_ResNext101_pretrained_data.pkl --img_folder_dir ./your/folder/path/folder_name --pretrained KITTI --cuda --gpu_num 0
# output folder name => 'out_' + folder_name
```
 If you are using a model pre-trained from KITTI, insert **'`--pretrained KITTI`'** command  
 (in the case of NYU, **'`--pretrained NYU`'**).  
 If you run the demo on GPU, insert **'`--cuda`'**.  
 **'`--gpu_num`'** argument is an index list of your available GPUs you want to use (e.g., 0,1,2,3).  
 ex) If you want to activate only the 3rd gpu out of 4 gpus, insert **'`--gpu_num 2`'**  

## Dataset Preparation
We referred to [BTS](https://github.com/cogaplex-bts/bts) in the data preparation process.

### KITTI
**1. [Official ground truth](http://www.cvlibs.net/download.php?file=data_depth_annotated.zip)**  
   * Download official KITTI ground truth on the link and make KITTI dataset directory.
```bash
    $ cd ./datasets
    $ mkdir KITTI && cd KITTI
    $ mv ~/Downloads/data_depth_annotated.zip ./datasets/KITTI
    $ unzip data_depth_annotated.zip
```
**2. Raw dataset**  
   * Construct raw KITTI dataset using following commands.
```bash
    $ mv ./datasets/kitti_archives_to_download.txt ./datasets/KITTI
    $ cd ./datasets/KITTI
    $ aria2c -x 16 -i ./kitti_archives_to_download.txt
    $ parallel unzip ::: *.zip
```
**3. Dense g.t dataset**  
   We take an inpainting method from [DenseDepth](https://github.com/ialhashim/DenseDepth) to get dense g.t for gradient loss.  
   (You can train our model using only data loss without gradient loss, then you don't need dense g.t)  
   Corresponding inpainted results from **'`./datasets/KITTI/data_depth_annotated/2011_xx_xx_drive_xxxx_sync/proj_depth/groundtruth/image_02`'** are should be saved in **'`./datasets/KITTI/data_depth_annotated/2011_xx_xx_drive_xxxx_sync/dense_gt/image_02`'**.  
KITTI data structures are should be organized as below:                           

    |-- datasets
      |-- KITTI
         |-- data_depth_annotated  
            |-- 2011_xx_xx_drive_xxxx_sync
               |-- proj_depth  
                  |-- groundtruth            # official G.T folder
            |-- ... (all drives of all days in the raw KITTI)  
         |-- 2011_09_26                      # raw RGB data folder  
            |-- 2011_09_26_drive_xxxx_sync
         |-- 2011_09_29
         |-- ... (all days in the raw KITTI)  


### NYU Depth V2
**1. Training set**  
    Make NYU dataset directory
```bash
    $ cd ./datasets
    $ mkdir NYU_Depth_V2 && cd NYU_Depth_V2
```
* Constructing training data using following steps :
    * Download Raw NYU Depth V2 dataset (450GB) from this **[Link](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_raw.zip).**  
    * Extract the raw dataset into '`./datasets/NYU_Depth_V2`'  
    (It should make **'`./datasets/NYU_Depth_V2/raw/....`'**).  
    * Run './datasets/sync_project_frames_multi_threads.m' to get synchronized data. (need Matlab)  
    (It shoud make **'`./datasets/NYU_Depth_V2/sync/....`'**).  
* Or, you can directly download whole 'sync' folder from our Google drive **[Link](https://drive.google.com/file/d/106oW6C7dfLHQYCNXZw9pn9q61ewNIZV1/view?usp=sharing)** into **'`./datasets/NYU_Depth_V2/`'**

**2. Testing set**  
    Download official nyu_depth_v2_labeled.mat and extract image files from the mat file.
```bash
    $ cd ./datasets
    ## Download official labled NYU_Depth_V2 mat file
    $ wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
    ## Extract image files from the mat file
    $ python extract_official_train_test_set_from_mat.py nyu_depth_v2_labeled.mat splits.mat ./NYU_Depth_V2/official_splits/
```

## Evaluation
Make sure you download the pre-trained model and placed it in the './pretrained/' directory before running the evaluation code.
* Evaluation Command Line:
```bash
# Running evaluation using a pre-trained models
## KITTI
python eval.py --model_dir ./pretrained/LDRN_KITTI_ResNext101_pretrained_data.pkl --evaluate --batch_size 1 --dataset KITTI --data_path ./datasets/KITTI --gpu_num 0
## NYU Depth V2
python eval.py --model_dir ./pretrained/LDRN_NYU_ResNext101_pretrained_data.pkl --evaluate --batch_size 1 --dataset NYU --data_path --data_path ./datasets/NYU_Depth_V2/official_splits/test --gpu_num 0

### if you want to save image files from results, insert `--img_save` command
### if you have dense g.t files, insert `--img_save` with `--use_dense_depth` command
```

## Training
LDRN (Laplacian Depth Residual Network) training
* Training Command Line:

```bash
# KITTI 
python train.py --distributed --batch_size 16 --dataset KITTI --data_path ./datasets/KITTI --gpu_num 0,1,2,3
# NYU
python train.py --distributed --batch_size 16 --dataset NYU --data_path ./datasets/NYU_Depth_V2/sync --epochs 30 --gpu_num 0,1,2,3 
## if you want to train using gradient loss, insert `--use_dense_depth` command
## if you don't want distributed training, remove `--distributed` command
```
  **'`--gpu_num`'** argument is an index list of your available GPUs you want to use (e.g., 0,1,2,3).  
  ex) If you want to activate only the 3rd gpu out of 4 gpus, insert **'`--gpu_num 2`'**

## Reference
When using this code in your research, please cite the following paper:  

M. Song, S. Lim and W. Kim, **"Monocular Depth Estimation Using Laplacian Pyramid-Based Depth Residuals,"** in **IEEE Transactions on Circuits and Systems for Video Technology**, vol. 31, no. 11, pp. 4381–4393, Nov. 2021.

```
@ARTICLE{9316778,
  author={M. {Song} and S. {Lim} and W. {Kim}},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Monocular Depth Estimation Using Laplacian Pyramid-Based Depth Residuals},
  month={Nov.},
  year={2021},
  volume={31},
  number={11},
  pages={4381-4393},
  doi={10.1109/TCSVT.2021.3049869}}
```

