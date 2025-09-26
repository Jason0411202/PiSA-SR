"""
本程式用於生成 HC18 dataset 的 pair data (GT 與 LR)

在專案根目錄下執行:
src/datasets/for_generate_dataset/for_generate_dataset.sh
來使用此檔案
"""

import os
import argparse
import numpy as np
from PIL import Image
import torch
import glob
import shutil
import random

from ..realesrgan import RealESRGAN_degradation
from basicsr.utils import tensor2img


def generate_pair_data(args):
    # 如果有給 degradation file，就初始化 RealESRGAN degradation
    realesrgan_degradation = None
    if args.degradation_file:
        realesrgan_degradation = RealESRGAN_degradation(args.degradation_file, device='cpu')

    # 取得所有輸入圖片
    if os.path.isdir(args.dataset_dir):
        image_names = sorted(
            glob.glob(f'{args.dataset_dir}/*.png') +
            glob.glob(f'{args.dataset_dir}/*.jpg')
        )
    else:
        image_names = [args.dataset_dir]
    image_names = [name for name in image_names if "Annotation" not in os.path.basename(name)]

    # 建立輸出資料夾
    lr_image_path = os.path.join(args.output_dir, 'LR')
    gt_image_path = os.path.join(args.output_dir, 'GT')

    if os.path.exists(lr_image_path):
        shutil.rmtree(lr_image_path)
    os.makedirs(lr_image_path, exist_ok=True)

    if os.path.exists(gt_image_path):
        shutil.rmtree(gt_image_path)
    os.makedirs(gt_image_path, exist_ok=True)

    print(f'There are {len(image_names)} images to process.')

    for idx, image_name in enumerate(image_names, start=1):
        if not os.path.exists(image_name):
            print(f"[警告] 找不到檔案 {image_name}，略過。")
            continue

        # 載入圖片
        input_image = Image.open(image_name).convert('RGB')
        if args.gt_width is not None and args.gt_height is not None: # 必要時先 resize GT
            input_image = input_image.resize((args.gt_width, args.gt_height), Image.BICUBIC)
        bname = os.path.basename(image_name)

        # Step 1. 製作 GT
        if args.degradation_file is None:
            # 直接用 input image 當 GT
            gt_image = input_image
        else:
            # degradation file → 同時產生 GT 與 LR
            gt_image, lr_image = realesrgan_degradation.degrade_process(np.asarray(input_image)/255., resize_bak=False)
            gt_image = torch.clamp((gt_image * 255.0).round(), 0, 255)
            gt_image = tensor2img(gt_image, out_type=np.uint8, min_max=(0, 255))
            gt_image = Image.fromarray(gt_image)

        # 保證 GT 大小為 8×upscale 的倍數
        gt_width = gt_image.width - gt_image.width % (8 * args.upscale)
        gt_height = gt_image.height - gt_image.height % (8 * args.upscale)
        gt_image = gt_image.resize((gt_width, gt_height), Image.LANCZOS)
        gt_image.save(os.path.join(gt_image_path, bname))

        # Step 2. 產生 LR
        if args.degradation_file is None:
            # bicubic downsample
            lr_image = gt_image.resize(
                (gt_image.size[0] // args.upscale, gt_image.size[1] // args.upscale),
                Image.BICUBIC
            )
        else:
            lr_image = torch.clamp((lr_image * 255.0).round(), 0, 255)
            lr_image = tensor2img(lr_image, out_type=np.uint8, min_max=(0, 255))
            lr_image = Image.fromarray(lr_image)
            lr_image = lr_image.resize(
                (gt_image.size[0] // args.upscale, gt_image.size[1] // args.upscale),
                Image.BICUBIC
            )

        lr_image.save(os.path.join(lr_image_path, bname))

        print(f"[{idx}/{len(image_names)}] Processed {bname} → GT:{gt_image.size}, LR:{lr_image.size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default="../HC18/training_set", help="path to the dataset (train/test folder)")
    parser.add_argument('--output_dir', type=str, default="src/datasets/for_generate_dataset/outputs/bicubic_4x", help="path to save GT and LR images")
    parser.add_argument('--upscale', type=int, default=4, help="upscaling factor")
    parser.add_argument('--gt_width', type=int, default=None, help="resized GT width")
    parser.add_argument('--gt_height', type=int, default=None, help="resized GT height")
    parser.add_argument('--degradation_file', type=str, default=None, help="degradation file path (if None, use bicubic downsampling)")

    args = parser.parse_args()

    # 固定隨機種子
    myseed = 8888
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
    random.seed(myseed)

    # 生成 pair data
    generate_pair_data(args)
