import os
import random
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from pathlib import Path

import numpy as np
from src.datasets.realesrgan import RealESRGAN_degradation

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



class PairedSROnlineTxtDataset(torch.utils.data.Dataset):
    def __init__(self, split=None, args=None):
        super().__init__()

        self.args = args
        self.split = split
        if split == 'train':
            self.degradation = RealESRGAN_degradation(args.deg_file_path, device='cpu')
            self.crop_preproc = transforms.Compose([
                transforms.RandomCrop((args.resolution_ori, args.resolution_ori)),
                transforms.Resize((args.resolution_tgt, args.resolution_tgt)),
                # transforms.RandomHorizontalFlip(),
            ])
            with open(args.dataset_txt_paths, 'r') as f:
                self.gt_list = [line.strip() for line in f.readlines()]
            if args.train_folder_lr is not None:
                assert args.dataset_txt_paths_lr is not None
                with open(args.dataset_txt_paths_lr, 'r') as f:
                    self.lr_list = [line.strip() for line in f.readlines()]

            if args.highquality_dataset_txt_paths is not None:
                with open(args.highquality_dataset_txt_paths, 'r') as f:
                    self.hq_gt_list = [line.strip() for line in f.readlines()]

        elif split == 'test':
            self.input_folder = os.path.join(args.dataset_test_folder, "LR")
            self.output_folder = os.path.join(args.dataset_test_folder, "HR")
            self.lr_list = []
            self.gt_list = []
            lr_names = os.listdir(os.path.join(self.input_folder))
            gt_names = os.listdir(os.path.join(self.output_folder))
            assert len(lr_names) == len(gt_names)
            for i in range(len(lr_names)):
                self.lr_list.append(os.path.join(self.input_folder, lr_names[i]))
                self.gt_list.append(os.path.join(self.output_folder,gt_names[i]))
            self.crop_preproc = transforms.Compose([
                transforms.RandomCrop((args.resolution_ori, args.resolution_ori)),
                transforms.Resize((args.resolution_tgt, args.resolution_tgt)),
            ])
            assert len(self.lr_list) == len(self.gt_list)

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):

        if self.split == 'train':
            if self.args.highquality_dataset_txt_paths is not None:
                if np.random.uniform() < self.args.prob:
                    gt_img = Image.open(self.gt_list[idx]).convert('RGB')
                else:
                    idx = random.sample(range(0, len(self.hq_gt_list)), 1)
                    gt_img = Image.open(self.hq_gt_list[idx[0]]).convert('RGB')
            else:
                gt_img = Image.open(self.gt_list[idx]).convert('RGB')

            # 確保 train 圖片大小至少為 resolution_ori * resolution_ori
            w, h = gt_img.size
            if w < self.args.resolution_ori or h < self.args.resolution_ori:
                gt_img = gt_img.resize((max(w, self.args.resolution_ori), max(h, self.args.resolution_ori)))
            gt_img = self.crop_preproc(gt_img)

            if self.args.train_folder_lr is None: # 正常從 GT 產生 LR 的過程
                output_t, img_t = self.degradation.degrade_process(np.asarray(gt_img)/255., resize_bak=True) # GT, LR
            else: # 直接使用 LR 資料夾的圖片
                output_t = gt_img
                img_t = Image.open(self.lr_list[idx[0]]).convert('RGB')
                img_t = img_t.resize(output_t.size, Image.BICUBIC) # 確保 img_t 與 output_t 大小相同
                
                # 轉成 Tensor
                output_t = F.to_tensor(output_t)
                img_t = F.to_tensor(img_t)

            output_t, img_t = output_t.squeeze(0), img_t.squeeze(0)
            # input images scaled to -1,1
            img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
            # output images scaled to -1,1
            output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

            example = {}
            # example["prompt"] = caption
            example["neg_prompt"] = self.args.neg_prompt_csd
            example["null_prompt"] = ""
            example["output_pixel_values"] = output_t # GT
            example["conditioning_pixel_values"] = img_t # LR

            return example
            
        elif self.split == 'test':
            input_img = Image.open(self.lr_list[idx]).convert('RGB')
            output_img = Image.open(self.gt_list[idx]).convert('RGB')
            img_t = self.crop_preproc(input_img)
            output_t = self.crop_preproc(output_img)
            # input images scaled to -1, 1
            img_t = F.to_tensor(img_t)
            img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
            # output images scaled to -1,1
            output_t = F.to_tensor(output_t)
            output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

            example = {}
            example["neg_prompt"] = self.args.neg_prompt_csd
            example["null_prompt"] = ""
            example["output_pixel_values"] = output_t
            example["conditioning_pixel_values"] = img_t
            example["base_name"] = os.path.basename(self.lr_list[idx])

            return example
