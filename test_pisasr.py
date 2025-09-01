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


def pisa_sr(args):
    # Initialize the model
    model = PiSASR_eval(args)
    model.set_eval()

    # Get all input images
    if os.path.isdir(args.input_image):
        image_names = sorted(glob.glob(f'{args.input_image}/*.png'))
    else:
        image_names = [args.input_image]

    # Make the output directory
    lr_image_path = os.path.join(args.output_dir, 'LR')
    hr_image_path = os.path.join(args.output_dir, 'HR')
    gt_image_path = os.path.join(args.output_dir, 'GT')
    os.makedirs(lr_image_path, exist_ok=True) # 將 GT 中的圖片 downsampling 後，變成 LR 並存在這裡
    os.makedirs(hr_image_path, exist_ok=True) # LR 經過模型處理後，變成 HR後會存在這裡
    os.makedirs(gt_image_path, exist_ok=True) # 原始圖片經過 resize 後會存在這裡
    print(f'There are {len(image_names)} images.')

    time_records = []

    # 紀錄 PSNR 與 SSIM
    psnr_scores = []
    ssim_scores = []
    for image_name in image_names:
        # 檢查檔案是否存在
        if not os.path.exists(image_name):
            print(f"[警告] 找不到檔案 {image_name}，略過。")
            continue

        # Ensure the input image is a multiple of 8
        input_image = Image.open(image_name).convert('RGB')
        ori_width, ori_height = input_image.size
        rscale = args.upscale
        resize_flag = False

        if ori_width < args.process_size // rscale or ori_height < args.process_size // rscale:
            scale = (args.process_size // rscale) / min(ori_width, ori_height)
            input_image = input_image.resize((int(scale * ori_width), int(scale * ori_height)))
            resize_flag = True

        bname = os.path.basename(image_name)
        # Step 1. 製作 Ground Truth (GT) → 512x512
        gt_image = input_image.resize((512, 512), Image.LANCZOS)
        gt_image.save(os.path.join(gt_image_path, bname))

        # Step 2. Degradation. Downsample 成 128x128 (模擬低解析度輸入)
        input_image = gt_image.resize((128, 128), Image.BICUBIC)
        input_image.save(os.path.join(lr_image_path, bname))

        # 強行放大成 512x512 (模擬低解析度輸入), 再由模型修復 artifact
        input_image = input_image.resize((input_image.size[0] * rscale, input_image.size[1] * rscale))
        new_width = input_image.width - input_image.width % 8
        new_height = input_image.height - input_image.height % 8
        input_image = input_image.resize((new_width, new_height), Image.LANCZOS)

        # Get caption (you can add the text prompt here)
        validation_prompt = ''

        # Translate the image
        with torch.no_grad():
            c_t = F.to_tensor(input_image).unsqueeze(0).cuda() * 2 - 1
            inference_time, output_image = model(args.default, c_t, prompt=validation_prompt)

        print(f"Inference time: {inference_time:.4f} seconds")
        time_records.append(inference_time)

        output_image = output_image * 0.5 + 0.5
        output_image = torch.clip(output_image, 0, 1)
        output_pil = transforms.ToPILImage()(output_image[0].cpu())

        if args.align_method == 'adain':
            output_pil = adain_color_fix(target=output_pil, source=input_image)
        elif args.align_method == 'wavelet':
            output_pil = wavelet_color_fix(target=output_pil, source=input_image)

        # 計算 PSNR 與 SSIM（用 NumPy array）
        output_np = np.array(output_pil)
        gt_np = np.array(gt_image)
        # 安全檢查：確保大小一致
        if output_np.shape != gt_np.shape:
            print(f"[警告] 輸出與 GT 尺寸不一致，略過計算 ({bname})")
            print("output_np.shape: ", output_np.shape, "gt_np.shape: ", gt_np.shape)
            
        else:
            psnr_val = compare_psnr(gt_np, output_np, data_range=255)
            ssim_val = compare_ssim(gt_np, output_np, channel_axis=-1, data_range=255)

            psnr_scores.append(psnr_val)
            ssim_scores.append(ssim_val)

            print(f"PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")

        if resize_flag:
            output_pil = output_pil.resize((int(args.upscale * ori_width), int(args.upscale * ori_height)))
        output_pil.save(os.path.join(hr_image_path, bname))

    # Calculate the average inference time, excluding the first few for stabilization
    if len(time_records) > 3:
        average_time = np.mean(time_records[3:])
    else:
        average_time = np.mean(time_records)
    print(f"Average inference time: {average_time:.4f} seconds")

    # 計算平均 PSNR / SSIM
    if len(psnr_scores) > 0:
        avg_psnr = np.mean(psnr_scores)
        avg_ssim = np.mean(ssim_scores)
        print(f"Average PSNR: {avg_psnr:.2f}, Average SSIM: {avg_ssim:.4f}")
    else:
        print("[警告] 沒有成功計算任何 PSNR/SSIM")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', '-i', type=str, default='preset/test_datasets', help="path to the input image")
    parser.add_argument('--output_dir', '-o', type=str, default='experiments/test', help="the directory to save the output")
    parser.add_argument("--pretrained_model_path", type=str, default='preset/models/stable-diffusion-2-1-base')
    parser.add_argument('--pretrained_path', type=str, default='preset/models/pisa_sr.pkl', help="path to a model state dict to be used")
    parser.add_argument('--seed', type=int, default=42, help="Random seed to be used")
    parser.add_argument("--process_size", type=int, default=512) # 輸入的圖片邊長會被至少調整至 process_size // upscale (處理太小的模型用的)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default="adain")
    parser.add_argument("--lambda_pix", default=1.0, type=float, help="the scale for pixel-level enhancement")
    parser.add_argument("--lambda_sem", default=1.0, type=float, help="the scale for sementic-level enhancements")
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=96) 
    parser.add_argument("--latent_tiled_overlap", type=int, default=32) 
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--default",  action="store_true", help="use default or adjustale setting?") 

    args = parser.parse_args()

    # Call the processing function
    pisa_sr(args)
