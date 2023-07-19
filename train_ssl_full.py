import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms

import os
import argparse
import time
from tqdm import tqdm
import shutil

from lib.NCEAverage import NCEAverage_pcl
from lib.NCECriterion import NCESoftmaxLoss
from lib.utils import AverageMeter, adjust_learning_rate, accuracy
import lib.transforms as T
import lib.augmentation as A

from datasets.ucf101_coclr import HMDBDataset_Retrieval, UCF101Dataset_Retrieval, \
    UCF101Dataset_Saccade
from models.r21d import R2Plus1DNet_Saccade
from models.r3d import R3DNet_Saccade

from torch.utils.data import DataLoader

import random
import numpy as np
import ast

from config import *

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--cluster_freq', type=int, default=1, help='cluster frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use') # 8
    parser.add_argument('--epochs', type=int, default=360, help='number of training epochs')

    # Device options
    parser.add_argument('--gpu-id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')

    # optimization
    parser.add_argument('--opt_type', type=str, default='sgd', help='optimizer type')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
    parser.add_argument('--pred_loss', type=str, default='l1', help='loss for prediction')
    parser.add_argument('--lr_ratio', type=float, default=0.001, help='learning rate ratio of proj layer')
    parser.add_argument('--lr_decay_epochs', type=int, nargs='+', default=[90, 80, 240], help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='decay rate for learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--cos', type=ast.literal_eval, default=False, help='whether to use cos anealing')

    # resume path
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--focus_init', default='', type=str, metavar='PATH',
                        help='path to focus model checkpoint for initlization (default: none)')
    parser.add_argument('--model_postfix', default='', type=str,
                        help='postfix of model name (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--analyze', action='store_true',
                        help='analyze model complexity regarding FLOPs and #params.')

    # model definition
    parser.add_argument('--model', type=str, default='r3d', choices=['r3d', 'r21d'])
    parser.add_argument('--nce_k', type=int, default=1024)
    parser.add_argument('--nce_t', type=float, default=0.07)
    parser.add_argument('--nce_m', type=float, default=0.5)
    parser.add_argument('--conv_level', type=int, default=5, help='level of conv features for prediction')
    parser.add_argument('--return_conv', type=ast.literal_eval, default=False)
    parser.add_argument('--proj_dim', type=int, default=512, help='dim of projection head output')
    parser.add_argument('--f_req_clust', type=int, default=None)
    parser.add_argument('--num_clusters', type=int, nargs='+', default=[500, 1000, 1500], help='where to decay lr, can be a list')
    parser.add_argument('--fs_warmup_epoch', type=int, default=1)
    parser.add_argument('--pcl', type=int, default=5, choices=[5])
    parser.add_argument('--pro_p', type=float, default=1.0, help='power of cosine distance')
    parser.add_argument('--pro_clamp_value', type=float, default=0.0, help='clamp value of cosine distance')

    # dataset
    parser.add_argument('--dataset', type=str, default='ucf101', choices=['ucf101', 'hmdb51'])
    parser.add_argument('--eval_dataset', type=str, default='ucf101', choices=['ucf101', 'hmdb51'])
    parser.add_argument('--split', type=str, default='1', choices=['1', '2', '3'])
    parser.add_argument('--clip_len', type=int, default=16, help='number of frames in a clip')
    parser.add_argument('--crop_size', type=int, default=112, help='number of frames in a clip')
    parser.add_argument('--img_dim', type=int, default=196, help='number of frames in a clip')
    parser.add_argument('--bottom_area', type=float, default=0.175, help='number of frames in a clip')
    parser.add_argument('--flip_consist', type=ast.literal_eval, default=True)
    parser.add_argument('--crop_consist', type=ast.literal_eval, default=True)
    parser.add_argument('--jitter_consist', type=ast.literal_eval, default=True)
    parser.add_argument('--grid_num', type=int, default=7, help='number of grid for mask')
    parser.add_argument('--deduplicate', type=ast.literal_eval, default=False, help='for scan')
    parser.add_argument('--margin_h', type=int, default=0, help='margin of the fixation masks')

    # specify folder
    parser.add_argument('--model_name', type=str, default='', help='model name')
    parser.add_argument('--model_path', type=str, default='./ckpt/', help='path to save model')

    # add new views
    parser.add_argument('--use_focus', type=ast.literal_eval, default=True)
    parser.add_argument('--neg', type=str, default='repeat', choices=['repeat', 'shuffle'])
    parser.add_argument('--seed', type=int, default=632)

    # focus related params
    parser.add_argument('--sample_num', type=int, default=1)
    parser.add_argument('--focus_level', type=int, default=0, choices=list(range(-1, 6)))
    parser.add_argument('--focus_num', type=int, default=3)

    # retrieve related params
    parser.add_argument('--eval_retrieve', type=ast.literal_eval, default=True)
    parser.add_argument('--retrieve', type=ast.literal_eval, default=False)
    parser.add_argument('--r_sample_num', type=int, default=10)
    parser.add_argument('--r_batch_size', type=int, default=8, help='retrieval batch_size')
    parser.add_argument('--r_return_conv', type=ast.literal_eval, default=True)



    opt = parser.parse_args()

    if opt.focus_level == -1 or opt.focus_num==1:
        opt.use_focus = False

    if opt.return_conv:
        opt.feat_dim = 9216
    else:
        opt.feat_dim = 512

    return opt


def set_model(args, n_data):
    # set the model
    if args.model == 'r3d':
        model = R3DNet_Saccade(layer_sizes=(1, 1, 1, 1), with_classifier=False, return_conv=args.return_conv,
                               focus_level=args.focus_level,
                               conv_level=args.conv_level, sample_num=args.sample_num, focus_num=args.focus_num,
                               pro_p=args.pro_p, pro_clamp_value=args.pro_clamp_value)

    elif args.model == 'r21d':
        model = R2Plus1DNet_Saccade(layer_sizes=(1, 1, 1, 1), with_classifier=False, return_conv=args.return_conv,
                                    focus_level=args.focus_level,
                                    conv_level=args.conv_level, sample_num=args.sample_num, focus_num=args.focus_num,
                                    pro_p=args.pro_p, pro_clamp_value=args.pro_clamp_value)


    contrast = NCEAverage_pcl(args.feat_dim, n_data, args.nce_k, args.nce_t, args.nce_m, args.focus_num, proj_dim=args.proj_dim)

    criterion = NCESoftmaxLoss()
    criterion_p = nn.CrossEntropyLoss()

    if args.pred_loss =='l1':
        criterion_pred = nn.L1Loss()
        args.pred_sign = 1.0
    elif args.pred_loss =='l2':
        criterion_pred = nn.MSELoss()
        args.pred_sign = 1.0
    elif args.pred_loss =='cos':
        criterion_pred = nn.CosineSimilarity(dim=1)
        args.pred_sign = -1.0

    # GPU mode
    model = torch.nn.DataParallel(model).cuda()
    contrast = torch.nn.DataParallel(contrast).cuda()
    criterion = criterion.cuda()
    criterion_p = criterion_p.cuda()
    cudnn.benchmark = True

    return model, contrast, criterion, criterion_p, criterion_pred


def set_optimizer_mp(args, model, contrast):
    # return optimizer
    if args.opt_type=='sgd':
        optimizer = torch.optim.SGD([{"params":model.parameters()},
                                     {"params":contrast.parameters(), "lr":args.learning_rate*args.lr_ratio}],
                                    lr=args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        print('Using SGD ...')
    elif args.opt_type=='adam':
        optimizer = torch.optim.Adam([{"params": model.parameters()},
                                      {"params": contrast.parameters(), "lr": args.learning_rate * args.lr_ratio}],
                                     lr=args.learning_rate,
                                     weight_decay=args.weight_decay)
        print('Using Adam ...')
    return optimizer


def train(epoch, train_loader, model, contrast, criterion, criterion_p, criterion_pred, optimizer, opt, transforms_cuda):
    """
    one epoch training
    """
    model.train()
    contrast.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    loss_c_meter = AverageMeter()
    loss_p_meter = AverageMeter()
    loss_total_meter = AverageMeter()
    prob_meter = AverageMeter()
    acc_proto = AverageMeter()

    bar = tqdm(train_loader)
    end = time.time()
    for idx, (inputs, f_maps, _, index, f_index) in enumerate(bar):
        data_time.update(time.time() - end)

        bsz = inputs.size(0)
        inputs = inputs.float().cuda()
        f_maps = f_maps.to(inputs.device)
        index = index.to(inputs.device)
        f_index = f_index.to(inputs.device)

        # ===================forward=====================

        # reshape inputs; for dim 0, v1 clips comes first, then, v2 clips, then, v3 clips, ...
        inputs = inputs.reshape((-1, 3, opt.clip_len, opt.crop_size, opt.crop_size))
        inputs = transforms_cuda(inputs)
        f_maps = f_maps.reshape((-1, 1, opt.crop_size, opt.crop_size))

        feat, feat_pred_1, feat_pred_2, feat_tgt_1, feat_tgt_2 = model(inputs, f_maps)

        out, out_proto, target_proto = \
            contrast(feat, index.view(-1)*torch.tensor(opt.focus_num).to(inputs.device)+f_index.view(-1))
                       # cluster_result=cluster_result)
        # out_l, out_ab, size (bs, nce_k+1, 1)


        loss_c = criterion(out)
        loss_p1 = criterion_pred(feat_pred_1, feat_tgt_1.detach()).mean()
        loss_p2 = criterion_pred(feat_pred_2, feat_tgt_2.detach()).mean()
        loss_p = opt.pred_sign * (loss_p1 + loss_p2) / 2.  # input, target
        loss = loss_c + loss_p
        prob = out[:, 0].mean()

        # ===================meters=====================
        loss_c_meter.update(loss_c.item(), bsz)
        loss_p_meter.update(loss_p.item(), bsz)
        loss_meter.update(loss.item(), bsz)
        prob_meter.update(prob.item(), bsz)

        if out_proto is not None:
            loss_proto = 0
            for proto_out, proto_target in zip(out_proto, target_proto):
                loss_proto += criterion_p(proto_out, proto_target)
                accp = accuracy(proto_out, proto_target)[0]
                acc_proto.update(accp[0], bsz)

            # average loss across all sets of prototypes
            loss_proto /= len(args.num_clusters)
            loss += loss_proto

        loss_total_meter.update(loss.item(), bsz)

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



        batch_time.update(time.time() - end)
        end = time.time()

        bar.set_description('Train: [{0}/{1}][{2}/{3}]|'
                        'BS {batch_size}|SN {sample_num}|'
                        'l {loss.val:.3f} ({loss.avg:.3f})|'
                        'l_c {loss_c.val:.3f} ({loss_c.avg:.3f})|'
                        'l_p {loss_p.val:.3f} ({loss_p.avg:.3f})|'
                        'l_t {loss_total.val:.3f} ({loss_total.avg:.3f})|'
                        'ac_p {acc_proto.val:.3f} ({acc_proto.avg:.3f})|'
                        'prob {prob.val:.3f} ({prob.avg:.3f})|'.format(
        epoch, opt.epochs, idx + 1, len(train_loader), batch_size=bsz, sample_num=opt.sample_num,
        loss_total = loss_total_meter, acc_proto = acc_proto,
        loss=loss_meter, loss_c=loss_c_meter, loss_p=loss_p_meter, prob=prob_meter))

    return loss_meter.avg, prob_meter.avg, loss_total_meter.avg, acc_proto.avg


args = parse_option()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

def main(args):
    if not torch.cuda.is_available():
        raise Exception('Only support GPU mode')

    gpu_num = torch.cuda.device_count()
    if not args.model_name:
        args.model_name = 'scan{}_{}_mgh{}_v2_f{}_{}_fl{}_fsnum{}_bs{}_{}'.format(
            args.grid_num, args.deduplicate, args.margin_h,
            args.focus_num, args.model,
            args.focus_level,
            args.sample_num, args.batch_size * gpu_num, time.strftime('%m%d'))

    args.model_name = args.model_name + args.model_postfix

    args.model_folder = os.path.join(args.model_path, args.model_name)
    if not os.path.isdir(args.model_folder):
        os.makedirs(args.model_folder)

    # parse the args
    print(vars(args))

    # Fix all parameters for reproducibility
    seed = args.seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #'''

    ''' Data '''
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
    transform_train_cuda = transforms.Compose([
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225], channel=1)])


    trainset = UCF101Dataset_Saccade(UCF101_PATH, transforms_=train_transforms, sample_num=args.sample_num, focus_num=args.focus_num,
                                     split=args.split, num=args.grid_num, dedup=args.deduplicate, mg_h=args.margin_h)

    print('gpu num:', gpu_num)
    train_loader = DataLoader(trainset, batch_size=args.batch_size*gpu_num, shuffle=True,
                                        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    n_data = trainset.__len__()

    # prepare dataloders for retrieval
    if args.eval_retrieve:
        train_transforms_r = transforms.Compose([
                A.CenterCrop(size=(args.img_dim, args.img_dim)),
                A.Scale(size=(args.crop_size, args.crop_size)),
                A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.3, consistent=True),
                A.ToTensor()])

        if args.eval_dataset == 'ucf101':
            train_dataset_r = UCF101Dataset_Retrieval(UCF101_PATH, clip_len=16, transforms_=train_transforms_r, train=True,
                                                      sample_num=args.r_sample_num, focus_num=args.focus_num, retrieve=True,
                                                      split=args.split, num=args.grid_num, dedup=args.deduplicate, mg_h=args.margin_h)

        elif args.eval_dataset == 'hmdb51':
            train_dataset_r = HMDBDataset_Retrieval(HMDB51_PATH, clip_len=16, transforms_=train_transforms_r,
                                                    train=True, sample_num=args.r_sample_num, focus_num=5, retrieve=True,
                                                    split=args.split)

        train_dataloader_r = DataLoader(train_dataset_r, batch_size=args.r_batch_size*gpu_num, shuffle=False,
                                      num_workers=args.num_workers, pin_memory=False, drop_last=True)

        test_transforms_r = transforms.Compose([
                A.CenterCrop(size=(args.img_dim, args.img_dim)),
                A.Scale(size=(args.crop_size, args.crop_size)),
                A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.3, consistent=True),
                A.ToTensor()])

        if args.eval_dataset == 'ucf101':
            test_dataset_r = UCF101Dataset_Retrieval(UCF101_PATH, clip_len=16, transforms_=test_transforms_r, train=False,
                                                     sample_num=args.r_sample_num, focus_num=args.focus_num, retrieve=True,
                                                     split=args.split, num=args.grid_num, dedup=args.deduplicate, mg_h=args.margin_h)

        elif args.eval_dataset == 'hmdb51':
            test_dataset_r = HMDBDataset_Retrieval(HMDB51_PATH, clip_len=16, transforms_=test_transforms_r,
                                                   train=False,
                                                   sample_num=args.r_sample_num, focus_num=5, retrieve=True,
                                                   split=args.split)

        test_dataloader_r = DataLoader(test_dataset_r, batch_size=args.r_batch_size*gpu_num, shuffle=False,
                                     num_workers=args.num_workers, pin_memory=False, drop_last=True)

    # prepare dataloaders for updating memory_mi
    train_transforms_u = transforms.Compose([
            A.CenterCrop(size=(args.img_dim, args.img_dim)),
            A.Scale(size=(args.crop_size, args.crop_size)),
            A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.3, consistent=True),
            A.ToTensor()])
    train_dataset_u = UCF101Dataset_Retrieval(UCF101_PATH, clip_len=args.clip_len,
                                              transforms_=train_transforms_u, train=True,
                                              sample_num=args.sample_num, focus_num=args.focus_num,
                                              retrieve=False, split=args.split, num=args.grid_num, dedup=args.deduplicate, mg_h=args.margin_h)

    train_loader_clust = DataLoader(train_dataset_u, batch_size=args.batch_size * gpu_num, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=False, drop_last=False)
    # set the model
    model, contrast, criterion, criterion_p, criterion_pred = set_model(args, n_data)

    # set the optimizer
    optimizer = set_optimizer_mp(args, model, contrast)

    args.start_epoch = 1
    if args.focus_init:  # memory_curr is NOT ready, self.cluster_result=None -> depends on fs_start_epoch
        init_ckpt_path = os.path.join(args.model_path, args.focus_init)
        if os.path.isfile(init_ckpt_path):
            print("=> loading checkpoint '{}'".format(args.focus_init))
            checkpoint = torch.load(init_ckpt_path, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            assert args.focus_level == checkpoint['f_level']
            assert args.focus_num == checkpoint['f_num']
            model.load_state_dict(checkpoint['model'])
            # optimizer.load_state_dict(checkpoint['optimizer']) # comment this to allow using new lr
            contrast.load_state_dict(checkpoint['contrast'], strict=False)
            print("=> loaded focus init checkpoint '{}'...".format(args.focus_init))
            del checkpoint
        else:
            print("=> no focus init checkpoint found at '{}'".format(args.focus_init))
            exit(1)

    # optionally resume from a checkpoint
    if args.resume:  # memory_curr is ready, self.cluster_result=None
        ckpt_path = os.path.join(args.model_folder, args.resume)
        if os.path.isfile(ckpt_path):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            # args.fs_warmup_epoch = checkpoint['fs_warmup_epoch']
            args.focus_level = checkpoint['f_level']
            args.focus_num = checkpoint['f_num']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            contrast.load_state_dict(checkpoint['contrast'], strict=False)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            if args.start_epoch > args.fs_warmup_epoch:
                with torch.no_grad():
                    contrast.module.update_clust(args.num_clusters)

            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            exit(1)
    else:
        print("=> training from scratch ...")

    if args.evaluate:
        with torch.no_grad():
            test_retrieval(model, contrast, train_dataloader_r, test_dataloader_r, args, transform_train_cuda)
        return

    cur_file = os.path.join(args.model_folder, 'ckpt.pth')

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):
        if args.epochs-epoch==59:
            args.save_freq=10
        adjust_learning_rate(optimizer, epoch, args)
        print('LR: %f, LR_c: %f' % (optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']))

        if epoch >= args.fs_warmup_epoch and epoch % args.cluster_freq == 0 and epoch < args.epochs:
            with torch.no_grad():
                clustering_feat(model, contrast, train_loader_clust, args, transform_train_cuda)

        loss, prob, loss_total, acc_proto = train(epoch, train_loader, model, contrast, criterion, criterion_p, criterion_pred, optimizer,
                                                            args, transform_train_cuda)

        # save model
        state = {
            'opt': args,
            'model': model.state_dict(),
            'contrast': contrast.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'fs_warmup_epoch': args.fs_warmup_epoch,
            'f_level': args.focus_level,
            'f_num': args.focus_num,
        }
        torch.save(state, cur_file)
        del state
        if epoch % args.save_freq == 0 or epoch==10 or epoch==args.save_freq//2:
            print('==> Saving...')
            save_file = os.path.join(args.model_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            shutil.copyfile(cur_file, save_file)

            if args.eval_retrieve:
                with torch.no_grad():
                    test_retrieval(model, contrast, train_dataloader_r, test_dataloader_r, args, transform_train_cuda)

        torch.cuda.empty_cache()

    print(args.model_name)


def test_retrieval(model, contrast, train_dataloader, test_dataloader, args, transforms_cuda):
    if args.eval_dataset == 'ucf101':
        class_num = 101
    elif args.eval_dataset == 'hmdb51':
        class_num = 51

    model.eval()
    contrast.eval()

    model.module.focus_level = -1
    model.module.sample_num = args.r_sample_num

    # ===== extract training features =====
    features = []
    classes = []
    for data in train_dataloader:
        sampled_clips, f_maps, idxs, _, _ = data
        clips = sampled_clips.reshape((-1, 3, args.clip_len, args.crop_size, args.crop_size))
        inputs = clips.cuda()
        inputs = transforms_cuda(inputs)
        f_maps = f_maps.reshape((-1, 1, args.crop_size, args.crop_size)).to(inputs.device)

        outputs, _, _, _, _ = model(inputs, f_maps)
        outputs = contrast(outputs, None, None, mode='eval')
        if args.r_sample_num > 1:
            outputs = outputs.reshape((-1, args.r_sample_num, outputs.size(1)))
            outputs = torch.mean(outputs, dim=1)
            idxs = idxs[:, 0]

        features.append(outputs.cpu().detach().numpy().tolist())
        classes.append(idxs.cpu().detach().numpy().tolist())


    X_train = np.array(features).reshape(-1, outputs.size(1))
    y_train = np.array(classes).reshape(-1)


    # ===== extract testing features =====
    features = []
    classes = []
    for data in test_dataloader:
        sampled_clips, f_maps, idxs, _, _ = data

        clips = sampled_clips.reshape((-1, 3, args.clip_len, args.crop_size, args.crop_size))
        inputs = clips.cuda()
        inputs = transforms_cuda(inputs)
        f_maps = f_maps.reshape((-1, 1, args.crop_size, args.crop_size)).to(inputs.device)
        # forward
        outputs, _, _, _, _ = model(inputs, f_maps)
        outputs = contrast(outputs, None, None, mode='eval')
        # perform mean among all samples before saving
        if args.r_sample_num > 1:
            outputs = outputs.reshape((-1, args.r_sample_num, outputs.size(1)))
            outputs = torch.mean(outputs, dim=1)
            idxs = idxs[:, 0]

        features.append(outputs.cpu().detach().numpy().tolist())
        classes.append(idxs.cpu().detach().numpy().tolist())

    X_test = np.array(features).reshape(-1, outputs.size(1))
    y_test = np.array(classes).reshape(-1)

    model.module.focus_level = args.focus_level
    model.module.sample_num = args.sample_num * 2 * args.focus_num

    del features
    del classes

    train_feature = torch.tensor(X_train).cuda()
    test_feature = torch.tensor(X_test).to(train_feature.device)

    y_train = torch.tensor(y_train).to(train_feature.device)
    y_test = torch.tensor(y_test).to(train_feature.device)

    ks = [1, 5, 10, 20, 50]
    topk_correct = {k: 0 for k in ks}

    # normalize
    test_feature = F.normalize(test_feature, p=2, dim=1)
    train_feature = F.normalize(train_feature, p=2, dim=1)

    # dot product
    sim = test_feature.matmul(train_feature.t())

    topk_result = []
    top_k_stat = {k:None for k in ks}
    print('----- feature_proj dim %d -----' % test_feature.size(-1))
    for k in ks:
        topkval, topkidx = torch.topk(sim, k, dim=1)
        result = torch.any(y_train[topkidx] == y_test.unsqueeze(1), dim=1)
        topk_result.append(result.unsqueeze(0))
        class_statistics = {idx: result[y_test==idx].float().mean().item() for idx in range(class_num)}
        top_k_stat[k] = class_statistics
        acc = result.float().mean().item()
        topk_correct[k] = acc
        print('Top-%d acc = %.4f' % (k, acc))




# similar to update_mem_labels in coclr_mi_v3.py
def clustering_feat(model, contrast, train_dataloader, args, transforms_cuda):

    model.eval()
    contrast.eval()
    model.module.conv_level = -1
    # ===== extract training features to update memory_mi =====
    print('Updating memory_curr using feat_proj ...')

    for data in tqdm(train_dataloader):
        sampled_clips, f_maps, idxs, index, f_index = data
        clips = sampled_clips.reshape((-1, 3, args.clip_len, args.crop_size, args.crop_size))
        inputs = clips.cuda()
        inputs = transforms_cuda(inputs)
        f_maps = f_maps.reshape((-1, 1, args.crop_size, args.crop_size)).to(inputs.device)
        index = index.to(inputs.device)
        f_index = f_index.to(inputs.device)

        outputs, _, _, _, _ = model(inputs, f_maps)
        y = index.view(-1) * torch.tensor(args.focus_num).to(inputs.device) + f_index.view(-1)
        outputs = contrast(outputs, None, None, mode='eval')
        contrast.module.memory_curr.index_copy_(0, y, outputs)

    model.module.conv_level = args.conv_level

    # ======= perform clustering =======
    contrast.module.update_clust(args.num_clusters)


if __name__ == '__main__':
    main(args)
