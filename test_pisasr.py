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
from src.my_utils.metric import IQA_Evaluator
import random

from src.models.de_net import DEResNet

run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # 計算執行程式時的當前時間

def pisa_sr(args):
    # Initialize the model
    model = PiSASR_eval(args)
    model.set_eval()

    # 載入 degradation condition pretrained 神經網路
    net_de = DEResNet(num_in_ch=3, num_degradation=2)
    net_de.load_model(args.de_net_path)
    net_de = net_de.cuda()
    net_de.eval()

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

    # 產生 LR-GT 的 pair data, 並以 list[(lr_path, gt_path)] 的形式儲存
    gt_map = {}
    for gt in gt_image_names:
        gt_map[os.path.basename(gt)] = gt
    image_names = []
    for lr in lr_image_names:
        b = os.path.basename(lr)
        if b in gt_map:
            image_names.append((lr, gt_map[b]))
        else:
            print("No matching GT for:", b)
    print(f'There are {len(image_names)} image pairs to be processed.')

    # Make the output directory
    lr_image_path = os.path.join(args.output_dir, 'LR')
    hr_image_path = os.path.join(args.output_dir, 'HR')
    gt_image_path = os.path.join(args.output_dir, 'GT')

    if os.path.exists(lr_image_path):
        shutil.rmtree(lr_image_path)   # 先刪掉整個資料夾
    os.makedirs(lr_image_path, exist_ok=True) # 讀入的 LR 會存在這裡

    if os.path.exists(hr_image_path):
        shutil.rmtree(hr_image_path)   # 先刪掉整個資料夾
    os.makedirs(hr_image_path, exist_ok=True) # LR 經過模型處理後，變成 HR後會存在這裡

    if os.path.exists(gt_image_path):
        shutil.rmtree(gt_image_path)   # 先刪掉整個資料夾
    os.makedirs(gt_image_path, exist_ok=True) # 原始圖片經過 resize 後會存在這裡
    print(f'There are {len(image_names)} images.')


    time_records = []

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

        # 將 LR 先強行放大成 GT 的大小, 再由模型修復 artifact
        lr_image_upsample = lr_image.resize(gt_image.size, Image.LANCZOS)
        new_width = lr_image_upsample.width - lr_image_upsample.width % 8 # 確保是 8 的倍數
        new_height = lr_image_upsample.height - lr_image_upsample.height % 8 # 確保是 8 的倍數
        lr_image_upsample = lr_image_upsample.resize((new_width, new_height), Image.LANCZOS)

        # Get caption (you can add the text prompt here)
        validation_prompt = ''

        # Inference, 透過模型將 LR 圖片轉成 HR 圖片
        with torch.no_grad():
            c_t = F.to_tensor(lr_image_upsample).unsqueeze(0).cuda() * 2 - 1

            deg_score = None
            if args.enable_deg_condition == "True": # 若啟用 degradation condition encoder, 則取得 LR 的 degradation score
                with torch.no_grad():
                    deg_score = net_de(c_t)
                    deg_score = deg_score.to(device=next(model.parameters()).device,
                                 dtype=next(model.parameters()).dtype)
            inference_time, output_image = model(args.default, c_t, deg_score, prompt=validation_prompt)

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

        lr_image_upsample.save(os.path.join(lr_image_path, bname))
        output_pil.save(os.path.join(hr_image_path, bname))
        gt_image.save(os.path.join(gt_image_path, bname))
        if idx < 5:
            wandb.log({
                "LR": [wandb.Image(lr_image_upsample, caption=f"LR-{bname}")],
                "HR": [wandb.Image(output_pil, caption=f"HR-{bname}")],
                "GT": [wandb.Image(gt_image, caption=f"GT-{bname}")]
            })

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
    parser.add_argument('--output_dir', '-o', type=str, default='experiments/test', help="the directory to save the output") # 輸出圖片資料夾, 包含 LR, HR, GT 三個子資料夾
    parser.add_argument("--pretrained_model_path", type=str, default='preset/models/sd-2.1-base') # 預訓練完成的 Stable Diffusion 模型
    parser.add_argument('--pretrained_path', type=str, default='preset/models/pisa_sr.pkl', help="path to a model state dict to be used") # 預訓練完成的 PiSA-SR 模型
    parser.add_argument('--seed', type=int, default=42, help="Random seed to be used") # random seed
    parser.add_argument("--process_size", type=int, default=64) # 輸入的圖片邊長會被至少調整至 process_size // upscale (處理太小的圖片用的)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default="adain") # 拿 LR 修正 HR 的顏色
    parser.add_argument("--lambda_pix", default=1.0, type=float, help="the scale for pixel-level enhancement") # 作者論文參數, 用於控制輸出的影像更偏向 Pixel-level
    parser.add_argument("--lambda_sem", default=1.0, type=float, help="the scale for semantic-level enhancements") # 作者論文參數, 用於控制輸出的影像更偏向 Semantic-level
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224) # tiled VAE 參數, 用於處理大圖
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024) # tiled VAE 參數, 用於處理大圖
    parser.add_argument("--latent_tiled_size", type=int, default=256) # tiled diffusion 參數, 用於處理大圖
    parser.add_argument("--latent_tiled_overlap", type=int, default=64) # tiled diffusion 參數, 用於處理大圖
    parser.add_argument("--mixed_precision", type=str, default="fp16") # 設定 float 經度以減少記憶體用量
    parser.add_argument("--wandb_project_name", type=str, default="test_pisasr") # 設定負責 log 的 wandb 專案名稱
    parser.add_argument("--wandb_run_name", type=str, default="") # 設定負責 log 的 wandb run 名稱
    parser.add_argument("--use_residual_in_training", type=str, default="True", choices=["True", "False"]) # 是否在訓練時使用殘差學習 (預設 True)
    parser.add_argument("--default",  action="store_true", help="use default or adjustable setting?")
    parser.add_argument("--max_inference_imgs_num",  default=500, type=int, help="max inference images number to evaluate PSNR, SSIM, FID") # 最大 inference 圖片數量

    # degradation condition 相關參數
    parser.add_argument("--enable_deg_condition", type=str, default="False", choices=["True", "False"], help="Whether to enable degradation condition.") # 是否啟用 degradation condition (預設 False)
    parser.add_argument("--de_net_path", type=str, default="src/de_net_pretrain_model/de_net.pth", help="Path to the pretrained degradation estimation network.")

    args = parser.parse_args()

    # 初始化 wandb
    wandb.init(
        project=args.wandb_project_name,
        name=args.wandb_run_name + "_" + run_timestamp,
        config=vars(args)
    )

    myseed = args.seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
    random.seed(myseed)

    # Call the processing function
    pisa_sr(args)
