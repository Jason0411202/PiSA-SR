import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F

from pisasr import PiSASR_eval
from src.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

import glob

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import wandb
import datetime
import shutil
from src.my_utils.utils import compute_fid
from src.datasets.realesrgan import RealESRGAN_degradation
from src.my_utils.metric import IQA_Evaluator
from basicsr.utils import tensor2img
import random

run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # 計算執行程式時的當前時間

def pisa_sr(args):
    # Initialize the model
    model = PiSASR_eval(args)
    model.set_eval()

    if args.degradation_file:
        deg_file_path = args.degradation_file
        realesrgan_degradation = RealESRGAN_degradation(deg_file_path, device='cpu')

    # Get all LR input images
    if os.path.isdir(args.input_lr_image):
        lr_image_names = sorted(
            glob.glob(f'{args.input_lr_image}/*.png') + 
            glob.glob(f'{args.input_lr_image}/*.jpg')
    )
    else:
        lr_image_names = [args.input_lr_image]

    # Get all GT input images
    if os.path.isdir(args.input_gt_image):
        gt_image_names = sorted(
            glob.glob(f'{args.input_gt_image}/*.png') + 
            glob.glob(f'{args.input_gt_image}/*.jpg')
    )
    else:
        gt_image_names = [args.input_gt_image]

    print(f'There are {len(lr_image_names)} LR images and {len(gt_image_names)} GT images to be processed.')

    image_names = list(zip(lr_image_names, gt_image_names))

    # Make the output directory
    lr_image_path = os.path.join(args.output_dir, 'LR')
    hr_image_path = os.path.join(args.output_dir, 'HR')
    gt_image_path = os.path.join(args.output_dir, 'GT')

    if os.path.exists(lr_image_path):
        shutil.rmtree(lr_image_path)   # 先刪掉整個資料夾
    os.makedirs(lr_image_path, exist_ok=True) # 將 GT 中的圖片 downsampling 後，變成 LR 並存在這裡

    if os.path.exists(hr_image_path):
        shutil.rmtree(hr_image_path)   # 先刪掉整個資料夾
    os.makedirs(hr_image_path, exist_ok=True) # LR 經過模型處理後，變成 HR後會存在這裡

    if os.path.exists(gt_image_path):
        shutil.rmtree(gt_image_path)   # 先刪掉整個資料夾
    os.makedirs(gt_image_path, exist_ok=True) # 原始圖片經過 resize 後會存在這裡
    print(f'There are {len(image_names)} images.')


    time_records = []

    # 紀錄 PSNR 與 SSIM
    psnr_scores = []
    ssim_scores = []
    for idx, (lr_image_name, gt_image_name) in enumerate(image_names, start=1):
        if idx > args.max_inference_imgs_num:
            print(f"已達到最大推論圖片數量 {args.max_inference_imgs_num}，停止處理。")
            break
        # 檢查檔案是否存在
        if not os.path.exists(lr_image_name) or not os.path.exists(gt_image_name):
            print(f"[警告] 找不到檔案 {lr_image_name} 或 {gt_image_name}，略過。")
            continue

        # Ensure the input image is a multiple of 8
        lr_image = Image.open(lr_image_name).convert('RGB')
        gt_image = Image.open(gt_image_name).convert('RGB')
        print("gt_image.size: ", gt_image.size, "lr_image.size: ", lr_image.size)

        bname = os.path.basename(lr_image_name)
        lr_image.save(os.path.join(lr_image_path, bname))
        gt_image.save(os.path.join(gt_image_path, bname))

        # 將 LR 先強行放大成 GT 的大小, 再由模型修復 artifact
        lr_image_upsample = lr_image.resize(gt_image.size, Image.LANCZOS)
        new_width = lr_image_upsample.width - lr_image_upsample.width % 8 # 確保是 8 的倍數
        new_height = lr_image_upsample.height - lr_image_upsample.height % 8 # 確保是 8 的倍數
        lr_image_upsample = lr_image_upsample.resize((new_width, new_height), Image.LANCZOS)

        # Get caption (you can add the text prompt here)
        validation_prompt = ''

        # Translate the image
        with torch.no_grad():
            c_t = F.to_tensor(lr_image_upsample).unsqueeze(0).cuda() * 2 - 1
            inference_time, output_image = model(args.default, c_t, prompt=validation_prompt)

        print(f"Inference time: {inference_time:.4f} seconds")
        time_records.append(inference_time)

        output_image = output_image * 0.5 + 0.5
        output_image = torch.clip(output_image, 0, 1)
        output_pil = transforms.ToPILImage()(output_image[0].cpu())

        # bug 修復，先前的版本誤拿 GT 去修復顏色
        if args.align_method == 'adain':
            output_pil = adain_color_fix(target=output_pil, source=lr_image_upsample)
        elif args.align_method == 'wavelet':
            output_pil = wavelet_color_fix(target=output_pil, source=lr_image_upsample)

        output_pil.save(os.path.join(hr_image_path, bname))
        if idx < 5:
            wandb.log({
                "HR": [wandb.Image(output_pil, caption=f"HR-{bname}")]
            })
        else:
            break

    # Calculate the average inference time, excluding the first few for stabilization
    if len(time_records) > 3:
        average_time = np.mean(time_records[3:])
    else:
        average_time = np.mean(time_records)
    print(f"Average inference time: {average_time:.4f} seconds")

    wandb.log({"Average Inference Time": average_time})
    
    iqa_evaluator = IQA_Evaluator(hr_dir=hr_image_path, gt_dir=gt_image_path)
    iqa_evaluator.evaluate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_lr_image', type=str, default='src/datasets/for_generate_dataset/outputs/bicubic_4x/test_set/LR', help="path to the input image") # LR 圖片資料夾
    parser.add_argument('--input_gt_image', type=str, default='src/datasets/for_generate_dataset/outputs/bicubic_4x/test_set/GT', help="path to the input image") # GT 圖片資料夾
    parser.add_argument('--output_dir', '-o', type=str, default='experiments/test', help="the directory to save the output")
    parser.add_argument("--pretrained_model_path", type=str, default='preset/models/sd-2.1-base')
    parser.add_argument('--pretrained_path', type=str, default='preset/models/pisa_sr.pkl', help="path to a model state dict to be used")
    parser.add_argument('--seed', type=int, default=42, help="Random seed to be used")
    parser.add_argument("--process_size", type=int, default=64) # 輸入的圖片邊長會被至少調整至 process_size // upscale (處理太小的模型用的)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default="adain")
    parser.add_argument("--degradation_file", type=str, default=None)
    parser.add_argument("--lambda_pix", default=1.0, type=float, help="the scale for pixel-level enhancement")
    parser.add_argument("--lambda_sem", default=1.0, type=float, help="the scale for sementic-level enhancements")
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=256) 
    parser.add_argument("--latent_tiled_overlap", type=int, default=64) 
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--wandb_project_name", type=str, default="test_pisasr")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--use_residual_in_training", default=True, type=bool) # 是否在訓練時使用殘差學習 (預設 True)
    parser.add_argument("--default",  action="store_true", help="use default or adjustale setting?") 
    parser.add_argument("--max_inference_imgs_num",  default=500, type=int, help="max inference images number to evaluate PSNR, SSIM, FID") 

    args = parser.parse_args()

    # 初始化 wandb
    wandb.init(
        project=args.wandb_project_name,
        name=args.wandb_run_name + "_" + run_timestamp,
        config=vars(args)
    )

    myseed = 8888
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
    random.seed(myseed)

    # Call the processing function
    pisa_sr(args)
