# Adjust cropping for torch 1.7.1
import os
import random
import sys
sys.path.append("..")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


from config import *

def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.cuda.manual_seed_all(seed)  # all gpus

def image_to_np(image):
  image_np = np.empty([image.channels, image.height, image.width], dtype=np.uint8)
  image.copyto(image_np)
  # image_np = np.transpose(image_np, (1,2,0)) # [c,h,w] -> [h,w,c]
  # print('image_np:', image_np.max(), image_np.min())
  return image_np

def readim(image_name):
  # read image
  # img_data = accimage.Image(image_name)
  img_data = Image.open(image_name)
  # img_data = image_to_np(img_data) # RGB
  return img_data

def load_from_frames(foldername, framenames, start_index, tuple_len, clip_len, interval):
  clip_tuple = []
  for i in range(tuple_len):
      one_clip = []
      for j in range(clip_len):
          im_name = os.path.join(foldername, framenames[start_index + i * (tuple_len + interval) + j])
          im_data = readim(im_name)
          one_clip.append(im_data)
      #one_clip_arr = np.array(one_clip)
      clip_tuple.append(one_clip)
  return clip_tuple

def load_one_clip(foldername, framenames, start_index, clip_len, intv=1):
    one_clip = []
    for i in range(clip_len):
        im_name = os.path.join(foldername, framenames[start_index + i*intv])
        im_data = readim(im_name)
        one_clip.append(im_data)
    # return np.array(one_clip)
    return one_clip


class UCF101Dataset(Dataset):
    """UCF101 dataset for recognition. The class index start from 0.
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
        test_sample_num： number of clips sampled from a video. 1 for clip accuracy.
    """
    def __init__(self, root_dir=UCF101_PATH, clip_len=16, split='1', train=True, transforms_=None, test_sample_num=10):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.split = split
        self.train = train
        self.transforms_ = transforms_
        self.test_sample_num = test_sample_num
        # self.toPIL = transforms.ToPILImage()
        self.toTensor = transforms.ToTensor()
        class_idx_path = os.path.join(root_dir, 'split', 'classInd.txt')
        self.class_idx2label = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(0)[1]
        self.class_label2idx = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(1)[0]

        if self.train:
            train_split_path = os.path.join(root_dir, 'split', 'trainlist0' + self.split + '.txt') #+ '_rgbflow.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split_path = os.path.join(root_dir, 'split', 'testlist0' + self.split + '.txt') #'_rgbflow.txt')
            self.test_split = pd.read_csv(test_split_path, header=None)[0]
        print('Use split'+ self.split)

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            clip (tensor): [channel x time x height x width]
            class_idx (tensor): class index, [0-100]
        """
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]

        class_idx = self.class_label2idx[videoname[:videoname.find('/')]] - 1 # add - 1 because it is range [1,101] which should be [0, 100]

        vid = videoname.split(' ')[0]
        vid = vid[:-4].split('/')[1]
        # #''' # already modified the txt files
        # # to avoid void folder because different names: HandStandPushups vs HandstandPushups
        # vids = vid.split('_')
        # if vids[1] == 'HandStandPushups':
        #     vid = vids[0] + '_HandstandPushups_' + vids[2] + '_' + vids[3]
        # #'''

        rgb_folder = os.path.join(self.root_dir, 'jpegs_256/', vid)

        framenames = [f for f in os.listdir(rgb_folder) if f.endswith('.jpg')]
        framenames.sort()
        length = len(framenames) - 1
        # length = len(framenames) - 2
        # if length < 16:
        if length < self.clip_len:
            print(vid, length)
            print('\n')
            raise

        # random select a clip for train
        if self.train:
            clip_start = random.randint(0, length - self.clip_len)
            clip = load_one_clip(rgb_folder, framenames, clip_start, self.clip_len)

            if self.transforms_:
                clip = self.transforms_(clip)  # list of [C, H, W]
                clip = torch.stack(clip, 1)  # [C, T, H, W]; equivalent to torch.stack(clip).permute([1, 0, 2, 3])

            else:
                # (T, H, W, C)
                clip = [image_to_np(img) for img in clip]
                clip = torch.tensor(np.array(clip))

            return clip, torch.tensor(int(class_idx)), idx
        # sample several clips for test
        else:
            all_clips = []
            all_idx = []
            for i in np.linspace(self.clip_len/2, length-self.clip_len/2, self.test_sample_num):
                clip_start = int(i - self.clip_len/2)
                #clip = videodata[clip_start: clip_start + self.clip_len]
                clip = load_one_clip(rgb_folder, framenames, clip_start, self.clip_len)

                if self.transforms_:
                    clip = self.transforms_(clip)  # list of [C, H, W]
                    clip = torch.stack(clip, 1)  # [C, T, H, W]; equivalent to torch.stack(clip).permute([1, 0, 2, 3])

                else:
                    # (T, H, W, C)
                    clip = [image_to_np(img) for img in clip]
                    clip = torch.tensor(np.array(clip))

                # transform_ is not None (test_num, C, T, H, W) or (test_num, T, H, W, C)
                all_clips.append(clip)
                all_idx.append(torch.tensor(int(class_idx)))

            # return torch.stack(all_clips), torch.stack(all_u_clips), torch.stack(all_v_clips), torch.tensor(int(class_idx)), idx
            return torch.stack(all_clips), torch.stack(all_idx), idx


class HMDBDataset_Retrieval(Dataset):
    """UCF101 dataset for recognition. The class index start from 0.
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
        test_sample_num： number of clips sampled from a video. 1 for clip accuracy.
    """
    def __init__(self, root_dir=HMDB51_PATH, clip_len=16, split='1', train=True, transforms_=None, sample_num=1,
                 focus_num=5, img_h=128, img_w=171, crop_size=112, retrieve=False):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.split = split
        self.train = train
        self.transforms_ = transforms_
        self.sample_num = sample_num * focus_num
        self.toPIL = transforms.ToPILImage()
        self.retrieve = retrieve
        if retrieve:
            self.fCrop = transforms.CenterCrop(crop_size)
        else:
            self.fCrop = transforms.RandomCrop(crop_size)
        self.toTensor = transforms.ToTensor()

        class_idx_path = os.path.join(root_dir, 'split', 'classInd.txt')
        self.class_idx2label = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(0)[1]
        self.class_label2idx = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(1)[0]

        self.focus_num = focus_num
        assert focus_num == 5

        '''If based on 7x7'''
        # f_maps_list = []
        # seps_x = torch.arange(0, 7, 3)
        # seps_y = torch.arange(0, 7, 3)
        # for i in range(len(seps_x)-1):
        #     for j in range(len(seps_y)-1):
        #         tmp = torch.zeros((7,7))
        #         tmp[seps_x[i]:seps_x[i]+4, seps_y[j]:seps_y[j]+4] = 1
        #         f_maps_list.append(tmp)
        # tmp = torch.zeros((7,7))
        # tmp[2:5, 2:5] = 1
        # f_maps_list.append(tmp)
        # self.F_MAPS = torch.stack(f_maps_list)
        # assert self.focus_num == self.F_MAPS.size(0)
        '''If based on 128x171'''
        # img_h, img_w = 128, 171
        seps_h = img_h // 2
        seps_w = img_w // 2
        F_MAPS = torch.zeros((self.focus_num, img_h, img_w))
        F_MAPS[0, :seps_h, :seps_w] = 1
        F_MAPS[1, :seps_h, seps_w:] = 1
        F_MAPS[2, seps_h:, :seps_w] = 1
        F_MAPS[3, seps_h:, seps_w:] = 1
        F_MAPS[4, seps_h // 2:-(seps_h // 2), seps_w // 2:-(seps_w // 2)] = 1
        self.F_MAPS = F_MAPS

        if self.train:
            train_split_path = os.path.join(root_dir, 'split', 'trainlist0' + self.split + '.txt')  # + '_rgbflow.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split_path = os.path.join(root_dir, 'split', 'testlist0' + self.split + '.txt')  # '_rgbflow.txt')
            self.test_split = pd.read_csv(test_split_path, header=None)[0]
        print('Use split' + self.split)

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            clip (tensor): [channel x time x height x width]
            class_idx (tensor): class index, [0-100]
        """
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]

        class_idx = self.class_label2idx[videoname[:videoname.find(
            '/')]] - 1  # add - 1 because it is range [1,101] which should be [0, 100]

        vid = videoname.split(' ')[0]
        vid = vid[:-4].split('/')[1]
        # #''' # already modified the txt files
        # # to avoid void folder because different names: HandStandPushups vs HandstandPushups
        # vids = vid.split('_')
        # if vids[1] == 'HandStandPushups':
        #     vid = vids[0] + '_HandstandPushups_' + vids[2] + '_' + vids[3]
        # #'''

        rgb_folder = os.path.join(self.root_dir, 'jpegs_256/', vid)

        framenames = [f for f in os.listdir(rgb_folder) if f.endswith('.jpg')]
        framenames.sort()
        length = len(framenames) - 1

        if length < self.clip_len:
            print(vid, length)
            print('\n')
            raise

        # random select clips for train
        all_clips = []
        all_idx = []
        all_v_idx = []

        # random select focus_maps
        # focus_idx = np.random.randint(self.focus_num, size=self.sample_num)
        # all_focus_map = self.F_MAPS.index_select(0, torch.tensor(focus_idx).long())
        focus_idx = []
        all_focus_map_cropped = []

        all_focus_map = []
        for fidx in range(self.sample_num//self.focus_num):
            focus_index = np.random.permutation(np.arange(0, self.focus_num))
            all_focus_map.append(self.F_MAPS[focus_index])
            focus_idx.append(focus_index)
        all_focus_map = torch.cat(all_focus_map, dim=0)
        focus_idx = np.concatenate(focus_idx, axis=0)

        if self.retrieve:
            focus_idx = np.random.randint(self.focus_num, size=self.sample_num//self.focus_num)
            all_focus_map = self.F_MAPS.index_select(0, torch.tensor(focus_idx))
            for cidx, c_value in enumerate(
                    np.linspace(self.clip_len / 2, length - self.clip_len / 2, self.sample_num//self.focus_num)):
                clip_start = int(c_value - self.clip_len / 2)
                # clip = videodata[clip_start: clip_start + self.clip_len]
                clip = load_one_clip(rgb_folder, framenames, clip_start, self.clip_len)
                focus_map = all_focus_map[cidx]
                if self.transforms_:
                    focus_map_resize = self.toPIL(focus_map).resize(clip[0].size)
                    clip.append(focus_map_resize)
                    clip = self.transforms_(clip)  # list of [C, H, W]

                    focus_map_cropped = clip[-1]
                    clip = torch.stack(clip[:-1], 1)  # [C, T, H, W]; equivalent to torch.stack(clip).permute([1, 0, 2, 3])

                else:
                    # (T, H, W, C)
                    clip = [image_to_np(img) for img in clip]
                    clip = torch.tensor(np.array(clip))
                    focus_map_cropped = torch.tensor(focus_map)

                # transform_ is not None (test_num, C, T, H, W) or (test_num, T, H, W, C)
                all_clips.append(clip)
                all_focus_map_cropped.append(focus_map_cropped)  # (test_num, T x C X H x W)
                all_idx.append(torch.tensor(int(class_idx)))
                all_v_idx.append(torch.tensor(idx))

                # print('clip:', clip[-1].max(), clip[-1].min(), 'map:', focus_map.max(), focus_map.min())
                # print('trans_clip:', frame.max(), frame.min(),
                #       'map_crop:', focus_map_cropped.max(), focus_map_cropped.min())

        else:
            for cidx in range(self.sample_num):
                clip_start = random.randint(0, length - self.clip_len)
                # clip = videodata[clip_start: clip_start + self.clip_len]
                clip = load_one_clip(rgb_folder, framenames, clip_start, self.clip_len)
                focus_map = all_focus_map[cidx]
                if self.transforms_:
                    focus_map_resize = self.toPIL(focus_map).resize(clip[0].size)
                    clip.append(focus_map_resize)
                    clip = self.transforms_(clip)  # list of [C, H, W]

                    focus_map_cropped = clip[-1]
                    clip = torch.stack(clip[:-1], 1)  # [C, T, H, W]; equivalent to torch.stack(clip).permute([1, 0, 2, 3])

                else:
                    # (T, H, W, C)
                    clip = [image_to_np(img) for img in clip]
                    clip = torch.tensor(np.array(clip))
                    focus_map_cropped = torch.tensor(focus_map)

                # transform_ is not None (test_num, C, T, H, W) or (test_num, T, H, W, C)
                all_clips.append(clip)
                all_focus_map_cropped.append(focus_map_cropped)  # (test_num, T x C X H x W)
                all_idx.append(torch.tensor(int(class_idx)))
                all_v_idx.append(torch.tensor(idx))

        return torch.stack(all_clips), torch.stack(all_focus_map_cropped), \
               torch.stack(all_idx), \
               torch.stack(all_v_idx), \
               torch.tensor(focus_idx)

    def gaus2d(self, mx=0, my=0, sx=1, sy=1):
        return 1. / (2. * np.pi * sx * sy) * torch.exp(
            -((self.x - mx) ** 2. / (2. * sx ** 2.) + (self.y - my) ** 2. / (2. * sy ** 2.)))


class UCF101Dataset_Retrieval(Dataset):
    """UCF101 dataset for recognition. The class index start from 0.
    Using saliency maps to generate focus_index
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
        test_sample_num： number of clips sampled from a video. 1 for clip accuracy.
    """

    def __init__(self, root_dir=UCF101_PATH, clip_len=16, split='1', train=True, transforms_=None, sample_num=1,
                 focus_num=5, img_h=128, img_w=171, crop_size=112, retrieve=False, num=7, dedup=False, mg_h=0):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.split = split
        self.train = train
        self.transforms_ = transforms_
        self.sample_num = sample_num * focus_num
        self.toPIL = transforms.ToPILImage()
        self.retrieve = retrieve
        self.dedup = dedup # whether deduplicate
        self.scan_dir = os.path.join(self.root_dir, 'scan_idx_{}'.format(num))
        if retrieve:
            self.fCrop = transforms.CenterCrop(crop_size)
        else:
            self.fCrop = transforms.RandomCrop(crop_size)
        self.toTensor = transforms.ToTensor()

        class_idx_path = os.path.join(root_dir, 'split', 'classInd.txt')
        self.class_idx2label = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(0)[1]
        self.class_label2idx = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(1)[0]

        self.focus_num = focus_num
        self.focus_comp_idx = random.sample(list(np.arange(num**2)), self.focus_num)

        '''If based on 128x171'''
        # img_h, img_w = 128, 171
        # num = 7
        mg_w = np.round(mg_h * 1.0 * img_w / img_h).astype('int')
        delta_h = img_h // num
        delta_w = img_w // num
        # for 25% mask
        mask_half_h = img_h // 4
        mask_half_w = img_w // 4

        F_MAPS = torch.zeros((num ** 2, img_h, img_w))
        for h_idx in range(num):
            center_h = h_idx * delta_h + delta_h // 2
            # top = max(0, center_h - mask_half_h)
            # bottom = center_h + mask_half_h  # auto
            top = max(0, center_h - mask_half_h - mg_h // 2)
            bottom = center_h + mask_half_h + mg_h // 2  # auto
            for w_idx in range(num):
                center_w = w_idx * delta_w + delta_w // 2
                # generate mask
                idx = h_idx * num + w_idx
                # left = max(0, center_w - mask_half_w)
                # right = center_w + mask_half_w  # auto
                left = max(0, center_w - mask_half_w - mg_w // 2)
                right = center_w + mask_half_w + mg_w // 2  # auto
                F_MAPS[idx][top:bottom, left:right] = 1
        self.F_MAPS = F_MAPS

        if self.train:
            train_split_path = os.path.join(root_dir, 'split', 'trainlist0' + self.split + '.txt')  # + '_rgbflow.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split_path = os.path.join(root_dir, 'split', 'testlist0' + self.split + '.txt')  # '_rgbflow.txt')
            self.test_split = pd.read_csv(test_split_path, header=None)[0]
        print('Use split' + self.split)

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            clip (tensor): [channel x time x height x width]
            class_idx (tensor): class index, [0-100]
        """
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]

        class_idx = self.class_label2idx[videoname[:videoname.find('/')]] - 1  # add - 1 because it is range [1,101] which should be [0, 100]

        vid = videoname.split(' ')[0]
        vid = vid[:-4].split('/')[1]
        # #''' # already modified the txt files
        # # to avoid void folder because different names: HandStandPushups vs HandstandPushups
        # vids = vid.split('_')
        # if vids[1] == 'HandStandPushups':
        #     vid = vids[0] + '_HandstandPushups_' + vids[2] + '_' + vids[3]
        # #'''

        findices = np.loadtxt(os.path.join(self.scan_dir, '{}.txt'.format(vid))).astype('int')
        if self.dedup: # if deduplicate
            tmp = []
            tmp_all = []
            cnt = 0
            while cnt < min(findices.shape[0], self.focus_num):
                if findices[cnt][0] in tmp:
                    cnt += 1
                    continue
                tmp.append(findices[cnt][0])
                tmp_all.append(findices[cnt])
                cnt += 1
            findices = np.array(tmp_all)
        else:
            findices = findices[:self.focus_num]

        rgb_folder = os.path.join(self.root_dir, 'jpegs_256/', vid)

        framenames = [f for f in os.listdir(rgb_folder) if f.endswith('.jpg')]
        framenames.sort()
        length = len(framenames) - 1

        if length < self.clip_len:
            print(vid, length)
            print('\n')
            raise

        # complete to self.focus_num fixations
        if findices.shape[0] < self.focus_num:
            comp_num = self.focus_num - findices.shape[0]
            comp_indices = np.zeros((comp_num, 3))
            comp_startidx = np.random.randint(low=1, high=length - self.clip_len, size=(comp_num,)) # [low, high)
            comp_indices[:, 0] = self.focus_comp_idx[:comp_num].copy()
            comp_indices[:, 1] = comp_startidx.copy()
            findices = np.concatenate([findices, comp_indices], axis=0).astype('int')
            assert findices.shape[0] == self.focus_num

        # random select clips for train
        all_clips = []
        all_idx = []
        all_v_idx = []
        all_focus_map_cropped = []
        focus_idx = []

        if self.retrieve:
            for cidx, c_value in enumerate(
                    np.linspace(self.clip_len / 2, length - self.clip_len / 2, self.sample_num//self.focus_num)):
                clip_start = int(c_value - self.clip_len / 2)
                # clip = videodata[clip_start: clip_start + self.clip_len]
                clip = load_one_clip(rgb_folder, framenames, clip_start, self.clip_len)
                focus_map = self.F_MAPS[0] # fidx does not matter in retrieval
                focus_idx.append(0)
                if self.transforms_:
                    focus_map_resize = self.toPIL(focus_map).resize(clip[0].size)
                    clip.append(focus_map_resize)
                    clip = self.transforms_(clip)  # list of [C, H, W]

                    focus_map_cropped = clip[-1]
                    clip = torch.stack(clip[:-1], 1)  # [C, T, H, W]; equivalent to torch.stack(clip).permute([1, 0, 2, 3])

                else:
                    # (T, H, W, C)
                    clip = [image_to_np(img) for img in clip]
                    clip = torch.tensor(np.array(clip))
                    focus_map_cropped = torch.tensor(focus_map)

                # transform_ is not None (test_num, C, T, H, W) or (test_num, T, H, W, C)
                all_clips.append(clip)
                all_focus_map_cropped.append(focus_map_cropped)  # (test_num, T x C X H x W)
                all_idx.append(torch.tensor(int(class_idx)))
                all_v_idx.append(torch.tensor(idx))

                # print('clip:', clip[-1].max(), clip[-1].min(), 'map:', focus_map.max(), focus_map.min())
                # print('trans_clip:', frame.max(), frame.min(),
                #       'map_crop:', focus_map_cropped.max(), focus_map_cropped.min())

        else:
            for cidx in range(self.sample_num):
                clip_start = findices[cidx][1]
                clip_start = min(clip_start, length - self.clip_len)
                # clip = videodata[clip_start: clip_start + self.clip_len]
                clip = load_one_clip(rgb_folder, framenames, clip_start, self.clip_len)
                # focus_map = all_focus_map[cidx]
                # *** generate index from scanpath ***
                focus_map = self.F_MAPS[findices[cidx][0]]
                focus_idx.append(cidx) # range from 0 to self.focus_num-1
                if self.transforms_:
                    focus_map_resize = self.toPIL(focus_map).resize(clip[0].size)
                    clip.append(focus_map_resize)
                    clip = self.transforms_(clip)  # list of [C, H, W]

                    focus_map_cropped = clip[-1]
                    clip = torch.stack(clip[:-1], 1)  # [C, T, H, W]; equivalent to torch.stack(clip).permute([1, 0, 2, 3])

                else:
                    # (T, H, W, C)
                    clip = [image_to_np(img) for img in clip]
                    clip = torch.tensor(np.array(clip))
                    focus_map_cropped = torch.tensor(focus_map)

                # transform_ is not None (test_num, C, T, H, W) or (test_num, T, H, W, C)
                all_clips.append(clip)
                all_focus_map_cropped.append(focus_map_cropped)  # (test_num, T x C X H x W)
                all_idx.append(torch.tensor(int(class_idx)))
                all_v_idx.append(torch.tensor(idx))

        return torch.stack(all_clips), torch.stack(all_focus_map_cropped), \
               torch.stack(all_idx), \
               torch.stack(all_v_idx), \
               torch.tensor(focus_idx)

    def gaus2d(self, mx=0, my=0, sx=1, sy=1):
        return 1. / (2. * np.pi * sx * sy) * torch.exp(
            -((self.x - mx) ** 2. / (2. * sx ** 2.) + (self.y - my) ** 2. / (2. * sy ** 2.)))


class UCF101Dataset_Saccade(Dataset):
    """UCF101 dataset for recognition. The class index start from 0.
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
        test_sample_num： number of clips sampled from a video. 1 for clip accuracy.
    """
    def __init__(self, root_dir=UCF101_PATH, clip_len=16, split='1', train=True, transforms_=None, sample_num=1, f_sigma_div=3.0,
                 focus_num=5, img_h=128, img_w=171, crop_size=112, retrieve=False, num=7, dedup=False, mg_h=0):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.split = split
        self.train = train
        self.transforms_ = transforms_
        self.sample_num = sample_num * 2 * focus_num # even number clips for each video
        self.toPIL = transforms.ToPILImage()
        self.retrieve = retrieve
        self.dedup = dedup  # whether deduplicate
        self.scan_dir = os.path.join(self.root_dir, 'scan_idx_{}'.format(num))
        if retrieve:
            self.fCrop = transforms.CenterCrop(crop_size)
        else:
            self.fCrop = transforms.RandomCrop(crop_size)
        self.toTensor = transforms.ToTensor()

        class_idx_path = os.path.join(root_dir, 'split', 'classInd.txt')
        self.class_idx2label = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(0)[1]
        self.class_label2idx = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(1)[0]

        self.focus_num = focus_num
        self.focus_comp_idx = random.sample(list(np.arange(num**2)), self.focus_num)

        '''If based on 7x7'''
        # img_h, img_w = 128, 171
        # num = 7
        mg_w = np.round(mg_h * 1.0 * img_w / img_h).astype('int')
        delta_h = img_h // num
        delta_w = img_w // num
        # for 25% mask
        mask_half_h = img_h // 4
        mask_half_w = img_w // 4

        F_MAPS = torch.zeros((num ** 2, img_h, img_w))
        for h_idx in range(num):
            center_h = h_idx * delta_h + delta_h // 2
            # top = max(0, center_h - mask_half_h)
            # bottom = center_h + mask_half_h  # auto
            top = max(0, center_h - mask_half_h - mg_h // 2)
            bottom = center_h + mask_half_h + mg_h // 2  # auto
            for w_idx in range(num):
                center_w = w_idx * delta_w + delta_w // 2
                # generate mask
                idx = h_idx * num + w_idx
                # left = max(0, center_w - mask_half_w)
                # right = center_w + mask_half_w  # auto
                left = max(0, center_w - mask_half_w - mg_w // 2)
                right = center_w + mask_half_w + mg_w // 2  # auto
                F_MAPS[idx][top:bottom, left:right] = 1
        self.F_MAPS = F_MAPS

        if self.train:
            train_split_path = os.path.join(root_dir, 'split', 'trainlist0' + self.split + '.txt') #+ '_rgbflow.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split_path = os.path.join(root_dir, 'split', 'testlist0' + self.split + '.txt') #'_rgbflow.txt')
            self.test_split = pd.read_csv(test_split_path, header=None)[0]
        print('Use split'+ self.split)

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            clip (tensor): [channel x time x height x width]
            class_idx (tensor): class index, [0-100]
        """
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]

        class_idx = self.class_label2idx[videoname[:videoname.find('/')]] - 1 # add - 1 because it is range [1,101] which should be [0, 100]

        vid = videoname.split(' ')[0]
        vid = vid[:-4].split('/')[1]
        # #''' # already modified the txt files
        # # to avoid void folder because different names: HandStandPushups vs HandstandPushups
        # vids = vid.split('_')
        # if vids[1] == 'HandStandPushups':
        #     vid = vids[0] + '_HandstandPushups_' + vids[2] + '_' + vids[3]
        # #'''

        findices = np.loadtxt(os.path.join(self.scan_dir, '{}.txt'.format(vid))).astype('int')
        if self.dedup:  # if deduplicate
            tmp = []
            tmp_all = []
            cnt = 0
            while cnt < min(findices.shape[0], self.focus_num):
                if findices[cnt][0] in tmp:
                    cnt += 1
                    continue
                tmp.append(findices[cnt][0])
                tmp_all.append(findices[cnt])
                cnt += 1
            findices = np.array(tmp_all)
        else:
            findices = findices[:self.focus_num]

        rgb_folder = os.path.join(self.root_dir, 'jpegs_256/', vid)

        framenames = [f for f in os.listdir(rgb_folder) if f.endswith('.jpg')]
        framenames.sort()
        length = len(framenames) - 1

        if length < self.clip_len:
            print(vid, length)
            print('\n')
            raise

        # complete to self.focus_num fixations
        if findices.shape[0] < self.focus_num:
            comp_num = self.focus_num - findices.shape[0]
            comp_indices = np.zeros((comp_num, 3))
            comp_startidx = np.random.randint(low=1, high=length - self.clip_len, size=(comp_num,))  # [low, high)
            comp_indices[:, 0] = self.focus_comp_idx[:comp_num].copy()
            comp_indices[:, 1] = comp_startidx.copy()
            comp_indices[:, 2] = comp_startidx.copy() + self.clip_len
            findices = np.concatenate([findices, comp_indices], axis=0).astype('int')
            assert findices.shape[0] == self.focus_num

        # random select a clip for train; sample several clips for test
        all_clips = []
        all_idx = []
        all_v_idx = []
        focus_idx = []
        all_focus_map_cropped = []

        if self.retrieve:
            for cidx, c_value in enumerate(np.linspace(self.clip_len / 2, length - self.clip_len / 2, self.sample_num//self.focus_num//2 )):
                clip_start = int(c_value - self.clip_len / 2)
                # clip = videodata[clip_start: clip_start + self.clip_len]
                clip = load_one_clip(rgb_folder, framenames, clip_start, self.clip_len)
                focus_map = self.F_MAPS[0]  # fidx does not matter in retrieval
                focus_idx.append(0)
                if self.transforms_:
                    focus_map_resize = self.toPIL(focus_map).resize(clip[0].size)
                    clip.append(focus_map_resize)
                    clip = self.transforms_(clip)  # list of [C, H, W]

                    focus_map_cropped = clip[-1]
                    clip = torch.stack(clip[:-1], 1)  # [C, T, H, W]; equivalent to torch.stack(clip).permute([1, 0, 2, 3])

                else:
                    # (T, H, W, C)
                    clip = [image_to_np(img) for img in clip]
                    clip = torch.tensor(np.array(clip))
                    focus_map_cropped = torch.tensor(focus_map)

                # transform_ is not None (test_num, C, T, H, W) or (test_num, T, H, W, C)
                all_clips.append(clip)
                all_focus_map_cropped.append(focus_map_cropped)  # (test_num, T x C X H x W)
                all_idx.append(torch.tensor(int(class_idx)))
                all_v_idx.append(torch.tensor(idx))
        else:
            clip_start_indices = []
            clip_offsets = []
            for cidx in range(self.sample_num):
                if cidx < self.sample_num//2:
                    clip_start = findices[cidx][1]
                    clip_end = findices[cidx][2]
                    clip_offset = (clip_end-clip_start+1)//2
                    clip_start = min(clip_start, length - self.clip_len - clip_offset)
                    # clip_start = min(clip_start, length - self.clip_len - self.clip_len//2)
                    clip_start_indices.append(clip_start)
                    clip_offsets.append(clip_offset)
                    # clip = videodata[clip_start: clip_start + self.clip_len]
                    clip = load_one_clip(rgb_folder, framenames, clip_start, self.clip_len)
                    # focus_map = all_focus_map[cidx]
                    # *** generate index from scanpath ***
                    focus_map = self.F_MAPS[findices[cidx][0]]
                    focus_idx.append(cidx)  # range from 0 to self.focus_num-1
                else:
                    clip_start = clip_start_indices[cidx-self.sample_num//2] + clip_offsets[cidx-self.sample_num//2]
                    # clip_start = clip_start_indices[cidx-self.sample_num//2] + self.clip_len//2
                    clip = load_one_clip(rgb_folder, framenames, clip_start, self.clip_len)
                    fidx = findices[cidx-self.sample_num//2][0]
                    focus_map = self.F_MAPS[fidx]
                    focus_idx.append(cidx-self.sample_num//2)

                if self.transforms_:
                    focus_map_resize = self.toPIL(focus_map).resize(clip[0].size)
                    clip.append(focus_map_resize)
                    clip = self.transforms_(clip)  # list of [C, H, W]

                    focus_map_cropped = clip[-1]
                    clip = torch.stack(clip[:-1], 1)  # [C, T, H, W]; equivalent to torch.stack(clip).permute([1, 0, 2, 3])

                else:
                    # (T, H, W, C)
                    clip = [image_to_np(img) for img in clip]
                    clip = torch.tensor(np.array(clip))
                    focus_map_cropped = torch.tensor(focus_map)

                # transform_ is not None (test_num, C, T, H, W) or (test_num, T, H, W, C)
                all_clips.append(clip)
                all_focus_map_cropped.append(focus_map_cropped)  # (test_num, T x C X H x W)
                all_idx.append(torch.tensor(int(class_idx)))
                all_v_idx.append(torch.tensor(idx))

        # # handle all_focus_map_cropped, focus_idx -> not right, different data aug for map
        # all_focus_map_cropped_half = torch.stack(all_focus_map_cropped[:self.sample_num//2], dim=0)
        # all_focus_map_cropped = all_focus_map_cropped_half.repeat(2,1,1)
        # focus_idx_half = focus_idx[:self.sample_num//2]
        # focus_idx = np.tile(focus_idx_half, 2)

        return torch.stack(all_clips), torch.stack(all_focus_map_cropped), torch.stack(all_idx), torch.stack(all_v_idx), torch.tensor(focus_idx)

    def gaus2d(self, mx=0, my=0, sx=1, sy=1):
        return 1. / (2. * np.pi * sx * sy) * torch.exp(
            -((self.x - mx) ** 2. / (2. * sx ** 2.) + (self.y - my) ** 2. / (2. * sy ** 2.)))
