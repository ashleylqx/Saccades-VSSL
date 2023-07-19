"""Finetune 3D CNN."""
import os
import argparse
import time
import random
from tqdm import tqdm
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from lib.utils import AverageMeter, adjust_learning_rate
import lib.augmentation as A
from config import *

from datasets.ucf101_coclr import UCF101Dataset
from models.r3d import R3DNet
from models.r21d import R2Plus1DNet

import ast



def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().data.item()

    return n_correct_elems / batch_size


def load_pretrained_weights(ckpt_path):
    """load pretrained weights and adjust params name."""
    adjusted_weights = {}
    pretrained_weights = torch.load(ckpt_path)
    for name, params in pretrained_weights.items():
        if 'base_network' in name:
            name = name[name.find('.')+1:]
            adjusted_weights[name] = params
    return adjusted_weights


def train(args, model, criterion, optimizer, train_dataloader, epoch):
    torch.set_grad_enabled(True)
    model.train()

    losses = AverageMeter()
    accuracies = AverageMeter()
    bar = tqdm(train_dataloader)
    for idx, data in enumerate(bar):
        # get inputs
        inputs, targets, _ = data

        inputs = inputs.cuda()
        targets = targets.to(inputs.device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward and backward
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        acc = calculate_accuracy(outputs, targets)
        losses.update(loss.data.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        bar.set_description('Train:[{0:3d}/{1:3d}][{2:4d}/{3:4d}]|'
              'Loss {loss.val:.4f} ({loss.avg:.4f})|'
              'Acc {acc.val:.3f} ({acc.avg:.3f})|'
              .format(
                epoch, args.epochs, idx + 1, len(train_dataloader),
                loss=losses, acc=accuracies))
    return losses.avg, accuracies.avg

def test(args, model, criterion, test_dataloader, epoch):
    torch.set_grad_enabled(False)
    model.eval()

    accuracies = AverageMeter()
    losses = AverageMeter()

    if args.modality == 'res':
        print("[Warning]: using residual frames as input")

    # total_loss = 0.0
    bar = tqdm(test_dataloader)
    for idx, data in enumerate(bar):
        # get inputs
        sampled_clips, targets, _ = data

        sampled_clips = sampled_clips.cuda()
        targets = targets.to(sampled_clips.device)
        outputs = []
        for clips in sampled_clips:
            inputs = clips.cuda()
            # forward
            o = model(inputs)
            o = torch.mean(o, dim=0)
            outputs.append(o)
        outputs = torch.stack(outputs)
        loss = criterion(outputs, targets[:, 0])
        # compute loss and acc
        losses.update(loss.data.item(), sampled_clips.size(0))
        acc = calculate_accuracy(outputs, targets[:, 0])
        accuracies.update(acc, sampled_clips.size(0))
        bar.set_description('Test: [{0:3d}/{1:3d}][{2:4d}/{3:4d}], Loss {loss.val:.4f} ({loss.avg:.4f})|'
                            'ACC {acc.val:.3f} ({acc.avg:.3f})'.format(
                             epoch, args.epochs, idx + 1, len(test_dataloader), loss=losses, acc=accuracies))
    return losses.avg, accuracies.avg


def parse_args():
    parser = argparse.ArgumentParser(description='Finetune 3D CNN from pretrained weights')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model', type=str, default='r3d', help='c3d/r3d/r21d')
    parser.add_argument('--return_conv', type=ast.literal_eval, default=False)


    parser.add_argument('--dataset', type=str, default='ucf101', help='ucf101/hmdb51')
    parser.add_argument('--eval_dataset', type=str, default='ucf101', choices=['ucf101', 'hmdb51'])
    parser.add_argument('--split', type=str, default='1', help='dataset split')
    parser.add_argument('--clip_len', type=int, default=16, help='clip length')
    parser.add_argument('--crop_size', type=int, default=112, help='number of frames in a clip')
    parser.add_argument('--img_dim', type=int, default=196, help='number of frames in a clip')
    parser.add_argument('--bottom_area', type=float, default=0.175, help='number of frames in a clip')
    parser.add_argument('--flip_consist', type=ast.literal_eval, default=True)
    parser.add_argument('--crop_consist', type=ast.literal_eval, default=True)
    parser.add_argument('--jitter_consist', type=ast.literal_eval, default=True)


    parser.add_argument('--gpu-id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--ft_lr', type=float, default=1e-3, help='finetune learning rate')
    parser.add_argument('--lr_decay_epochs', type=int, nargs='+', default=[45, 90, 125, 160],
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='decay rate for learning rate')
    parser.add_argument('--momentum', type=float, default=9e-1, help='momentum')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--cos', type=ast.literal_eval, default=False, help='whether to use cos anealing')
    parser.add_argument('--opt_type', type=str, default='sgd', help='optimizer type')
    parser.add_argument('--model_name', type=str, default='', help='model name')
    parser.add_argument('--model_path', type=str, default='./ckpt/', help='path to save model')
    parser.add_argument('--best_ckpt', default='best.pth', type=str, help='checkpoint path')
    parser.add_argument('--model_postfix', default='', type=str,
                        help='postfix of model name (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--focus_init', default='', type=str, metavar='PATH',
                        help='path to focus model checkpoint for initlization (default: none)')

    parser.add_argument('--dropout', type=float, default=0.9, help='dropout ratio in classifier')
    parser.add_argument('--finetune', type=ast.literal_eval, default=True, help='True: finetune; False: linear probe')

    parser.add_argument('--epochs', type=int, default=150, help='number of total epochs to run')
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--test_batch_size', type=int, default=32, help='mini-batch size')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--seed', type=int, default=632, help='seed for initializing training.')
    parser.add_argument('--modality', default='rgb', type=str, help='modality from [rgb, res]')
    args = parser.parse_args()

    if args.finetune: # for training the entire network
        args.final_bn = False
        args.final_norm = False
        args.use_dropout = True
    else: # for linear probe
        args.final_bn = True
        args.final_norm = True
        args.use_dropout = False

    return args

args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

def main(args):
    gpu_num = torch.cuda.device_count()

    ########### model ##############
    if args.dataset == 'ucf101':
        class_num = 101
    elif args.dataset == 'hmdb51':
        class_num = 51

    if args.model == 'r3d':
        model = R3DNet(layer_sizes=(1, 1, 1, 1), with_classifier=True, num_classes=class_num,
                       use_dropout=args.use_dropout, dropout=args.dropout,
                       use_l2_norm=args.final_norm, use_final_bn=args.final_bn)
    elif args.model == 'r21d':
        model = R2Plus1DNet(layer_sizes=(1, 1, 1, 1), with_classifier=True, num_classes=class_num,
                            use_dropout=args.use_dropout, dropout=args.dropout,
                            use_l2_norm=args.final_norm, use_final_bn=args.final_bn)

    model = torch.nn.DataParallel(model).cuda()

    if not args.model_name:
        args.model_name = '{}_{}_bs{}_cls_{}'.format(args.model, args.modality, args.batch_size * gpu_num, time.strftime('%m%d'))
        if args.finetune:
            args.model_name = args.model_name + '_ft'
        else:
            args.model_name = args.model_name + '_lp'

    args.model_name = args.model_name + args.model_postfix
    print(args.model_name)

    args.model_folder = os.path.join(args.model_path, args.model_name)
    if not os.path.isdir(args.model_folder) and args.mode == 'train':
        os.makedirs(args.model_folder)

    print(vars(args))

    # Uncomment to fix all parameters for reproducibility
    seed = args.seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_transforms = transforms.Compose([
        A.RandomSizedCrop(size=args.crop_size, consistent=args.crop_consist, seq_len=args.clip_len,
                          bottom_area=args.bottom_area),
        transforms.RandomApply([
            A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=1.0, consistent=args.jitter_consist, seq_len=args.clip_len)
        ], p=0.8),
        transforms.RandomApply([A.GaussianBlur([.1, 2.], seq_len=args.clip_len)], p=0.5),
        A.RandomHorizontalFlip(consistent=args.flip_consist, seq_len=args.clip_len),
        A.ToTensor(),
    ])

    test_transforms = transforms.Compose([
        A.CenterCrop(size=(args.img_dim, args.img_dim)),
        A.Scale(size=(args.crop_size, args.crop_size)),
        A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.3, consistent=True),
        A.ToTensor()])

    if args.dataset == 'ucf101':
        train_dataset = UCF101Dataset(UCF101_PATH, args.clip_len, args.split, True, train_transforms)
        test_dataset = UCF101Dataset(UCF101_PATH, args.clip_len, args.split, False, test_transforms)
    elif args.dataset == 'hmdb51':
        train_dataset = UCF101Dataset(HMDB51_PATH, args.clip_len, args.split, True, train_transforms)
        test_dataset = UCF101Dataset(HMDB51_PATH, args.clip_len, args.split, False, test_transforms)

    gpu_num = torch.cuda.device_count()
    print('gpu num:', gpu_num)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size*gpu_num, shuffle=True,
                                num_workers=args.workers, pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size * gpu_num, shuffle=False,
                                 num_workers=args.workers, pin_memory=True)

    ### loss funciton, optimizer and scheduler ###
    criterion = nn.CrossEntropyLoss()

    if not args.finetune:
        for name, param in model.named_parameters():
            if not 'linear' in name and 'final_bn' not in name:
                param.requires_grad = False

        print('\n===========Check Grad============')
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.requires_grad)
        print('=================================\n')

    if args.opt_type=='sgd':
        optimizer = torch.optim.SGD([{'params': [param for name, param in model.named_parameters() if 'linear' not in name]},
                                    {'params': [param for name, param in model.named_parameters() if 'linear' in name], 'lr': args.ft_lr}],
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.wd)
        print('Using SGD ...')
    elif args.opt_type=='adam':
        optimizer = torch.optim.Adam([{'params': [param for name, param in model.named_parameters() if 'linear' not in name]},
                                    {'params': [param for name, param in model.named_parameters() if 'linear' in name], 'lr': args.ft_lr}],
                                     lr=args.lr,
                                     weight_decay=args.wd)

    args.start_epoch = 1
    if args.focus_init:  # memory_curr is NOT ready, self.cluster_result=None -> depends on fs_start_epoch
        init_ckpt_path = os.path.join(args.model_path, args.focus_init)
        if os.path.isfile(init_ckpt_path):
            print("=> loading checkpoint '{}'".format(args.focus_init))
            checkpoint = torch.load(init_ckpt_path, map_location='cpu')
            pre_state_dict = checkpoint['model']
            cur_state_dict = model.state_dict()
            cnt = 0
            for k in list(cur_state_dict.keys()):
                if 'linear' not in k and 'final_bn' not in k:
                    cur_state_dict[k] = pre_state_dict[k]
                    cnt += 1
            model.load_state_dict(cur_state_dict, strict=False)
            print('Initializing {} params ...'.format(cnt))
            print("=> loaded focus init checkpoint '{}'...".format(args.focus_init))
            del checkpoint
        else:
            print("=> no focus init checkpoint found at '{}'".format(args.focus_init))
            exit(1)

    # optionally resume from a checkpoint
    title = args.dataset + '-' + args.model_name
    prev_best_val_acc = 0
    if args.resume:
        ckpt_path = os.path.join(args.model_folder, args.resume)
        if os.path.isfile(ckpt_path):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

            best_path = os.path.join(args.model_folder, args.best_ckpt)
            best_ckpt = torch.load(best_path, map_location='cpu')
            prev_best_val_acc = best_ckpt['val_acc']

            del checkpoint
            del best_ckpt

            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            exit(1)
    else:
        print("=> training from scratch ...")

    cur_file = os.path.join(args.model_folder, 'ckpt.pth')
    best_file = os.path.join(args.model_folder, args.best_ckpt)
    if args.mode == 'train':
        best_epoch = args.start_epoch
        for epoch in range(args.start_epoch, args.epochs):
            adjust_learning_rate(optimizer, epoch, args)
            print('LR: %f, LR_ft: %f' % (optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']))

            train_loss, train_acc = train(args, model, criterion, optimizer, train_dataloader, epoch)
            val_loss, val_acc = test(args, model, criterion, test_dataloader, epoch)

            # save model
            state = {'opt': args,
                     'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'epoch': epoch,
                     'val_loss': val_loss,
                     'val_acc': val_acc,
                     }
            torch.save(state, cur_file)
            # save model for the best val
            if val_acc > prev_best_val_acc:
                print('==> Saving best...')
                prev_best_val_acc = val_acc
                shutil.copyfile(cur_file, best_file)
                best_epoch = epoch

            print('[BEST] epoch: {}, acc: {:.3f}'.format(best_epoch, prev_best_val_acc))

    print(args.model_name)

    model.load_state_dict(torch.load(best_file)['model'])
    print('Final testing epoch %d...'%torch.load(best_file)['epoch'])
    with torch.no_grad():
        test(args, model, criterion, test_dataloader, torch.load(best_file)['epoch'])


if __name__ == '__main__':
    main(args)


