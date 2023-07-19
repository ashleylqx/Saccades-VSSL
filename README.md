# Saccade-VSSL


## Overview
This repository implements 'Self-supervised Video Representation Learning via Capturing Semantic Changes Indicated by Saccades (TCSVT2023)'.

Qiuxia Lai, Ailing Zeng, Ye Wang, Lihong Cao, Yu Li, Qiang Xu.


## Requirements
- python  3.7.12
- torch 1.13.0+cu116
- torchvision 0.14.0+cu116
- cudatoolkit 11.6.0
- tensorboard
- tensorboardX
- accimage  ([official github](https://github.com/pytorch/accimage)) 
- faiss-gpu ([official github](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)) 
- Pillow
- opencv
- scipy
- tqdm


## Weight Download
Our models are pre-trained on UCF101 split 1 with RGB data only.
Linear probe and fine-tuning are performed on UCF101 and HMDB51.

- Pre-trained models can be downloaded from [this link](https://pan.baidu.com/s/1NlB4vP-XzzTu7OB5EMDfPQ?pwd=8yhn).
- Linear probe models can be downloaded from [this link](https://pan.baidu.com/s/16ipH16MTX1D7xioDeFMK5A?pwd=fhdy).
- Finetune models can be downloaded from [this link](https://pan.baidu.com/s/1FODLhCW5gy8R12TnbZf_WQ?pwd=bm6g).


<!--
Baidu Disk: <> password:`pswd`
-->

## Dataset Preparation

The datasets are arranged as follows, where <base_path> is defined in `config.py`.

Note that all the `split` folders are available at `data/<dataset_name>/`.

```markdown
<base_path>/DataSets/
    |---UCF101/
    |   |---jpegs_256/
    |   |   |---<video1>/
    |   |   |   |---XXX.jpg
    |   |   |   |--- ...
    |   |   |---<video2>/
    |   |   |--- ...
    |   |---split/
    |   |   |---ClassInd.txt
    |   |   |---trainlist01.txt
    |   |   |---testlist01.txt
    |   |   |--- ...
    |   |---scan_idx_7/
    |   |   |---v_ApplyEyeMakeup_g01_c01.txt
    |   |   |---v_ApplyEyeMakeup_g01_c02.txt
    |   |   |---v_ApplyEyeMakeup_g01_c03.txt
    |   |   |--- ...
    |
    |---HMDB51/
    |   |---jpegs_256/
    |   |   |   |---<video1>/
    |   |   |   |   |---XXX.jpg
    |   |   |   |   |--- ...
    |   |   |   |---<video2>/
    |   |   |   |--- ...
    |   |---split/
    |   |   |---ClassInd.txt
    |   |   |---trainlist01.txt
    |   |   |---testlist01.txt
    |   |   |--- ...
            
```

### UCF101
UCF101 with scanpath data can be downloaded from [this BaiduDisk link](https://pan.baidu.com/s/18i_--D0KuMIc36oaMKcGsA?pwd=2g2q). 
Our files are obtained from this [repo](https://github.com/feichtenhofer/twostreamfusion). 
**Official files** can be downloaded from [this link](http://crcv.ucf.edu/data/UCF101.php).

- Download three splits of the `zip` file. Unzip together, and got folder `jpegs_256`. Put it into `<base_path>\DataSets\UCF101`.
- Download `scan_idx_7.zip` file. Unzip and Put it into  `<base_path>\DataSets\UCF101`.

The scanpath data is generated using [G-Eymol (TPAMI 2019)](https://github.com/dariozanca/G-Eymol/tree/master).


### HMDB51
HMDB51 can be downloaded from  [this BaiduDisk link](https://pan.baidu.com/s/1L4LP_Lwg8kNzKAekVTYy3Q?pwd=f2f8). 
Our files are obtained from this [repo](https://github.com/feichtenhofer/twostreamfusion).
**Official files**  can be downloaded from [this link](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/).

- Download the `zip` file. Unzip together, and got folder `jpegs_256`. Put it into `<base_path>\DataSets\HMDB51`.


## Pre-training

```commandline
# R3D
python train_ssl_full.py --fs_warmup_epoch 241 --cluster_freq 5 --num_clusters 1500 1500 1500 --dataset=ucf101 --batch_size 16 --save_freq 60 --lr_ratio 0.001 --epochs 300 --learning_rate 0.1 --lr_decay_epochs 90 180 240 --gpu-id 2 \
      --focus_num 3 --grid_num 7 --deduplicate True --margin_h 12

# R21D
python train_ssl_full.py --fs_warmup_epoch 301 --cluster_freq 5 --num_clusters 1500 1500 1500 --dataset=ucf101 --batch_size 8 --save_freq 60 --lr_ratio 0.001 --epochs 360 --learning_rate 0.1 --lr_decay_epochs 90 180 240 --gpu-id 3 \
      --model r21d --pro_p 4 --focus_num 3 --grid_num 7 --deduplicate True --margin_h 12      
```

## Video retrieval

```commandline
# R3D
python train_ssl_full.py --fs_warmup_epoch 241 --cluster_freq 5 --num_clusters 1500 1500 1500 --dataset=ucf101 --batch_size 16 --save_freq 60 --lr_ratio 0.001 --epochs 300 --learning_rate 0.0008 --lr_decay_epochs 90 180 240 --gpu-id 1 \
      --evaluate --focus_num 3 --grid_num 7 --deduplicate True --margin_h 12 --model_name <model_name> --resume <your_best_ckpt>.pth
# R21D
python train_ssl_full.py --fs_warmup_epoch 301 --cluster_freq 5 --num_clusters 1500 1500 1500 --dataset=ucf101 --batch_size 8 --save_freq 60 --lr_ratio 0.001 --epochs 360 --learning_rate 0.0008 --lr_decay_epochs 90 180 240 --gpu-id 0 \
      --evaluate --model r21d --pro_p 4 --focus_num 3 --grid_num 7 --deduplicate True --margin_h 12 --model_name <model_name> --resume <your_best_ckpt>.pth
```

## Downstream Tasks
### Linear probe
```commandline
# ucf101
python downstream.py --finetune False --dropout 0.7 --seed 42 --model_postfix _dp_0_7_s42 --lr 0.1 --ft_lr 0.1 --epochs 200 --lr_decay_epochs 60 120 160 --lr_decay_rate 0.1 --batch_size 32 \
      --focus_init <model_name>/<your_best_ckpt>.pth --gpu-id 0
# hmdb51
python downstream.py --dataset hmdb51 --finetune False --dropout 0.7 --seed 42 --model_postfix _dp_0_7_s42_hmdb --lr 0.1 --ft_lr 0.1 --epochs 200 --lr_decay_epochs 60 120 160 --lr_decay_rate 0.1 --batch_size 32 \
      --focus_init <model_name>/<your_best_ckpt>.pth --gpu-id 1
```

### Finetuning
```commandline
python downstream.py --finetune True --dropout 0.7 --lr 0.1 --ft_lr 0.1 --epochs 200 --lr_decay_epochs 60 120 160 --lr_decay_rate 0.1 --batch_size 32 \
      --final_bn False --final_norm False --focus_init <model_name>/<your_best_ckpt>.pth --gpu-id 0
```


## Citation
If you find this repository useful, please consider citing the following reference.
```
@ARTICLE{lai2023self,
    title={Self-supervised video representation learning via capturing semantic changes indicated by saccades},
    author={Qiuxia Lai and Ailing Zeng and Ye Wang and Lihong Cao and Yu Li and Qiang Xu},
    journal={IEEE Trans. on Circuits and Systems for Video Technology},
    year={2023}
}
```

## Contact

Qiuxia Lai: ashleylqx`at`gmail.com | qxlai`at`cuc.edu.cn

