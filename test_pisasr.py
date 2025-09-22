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
from basicsr.utils import tensor2img

run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # 計算執行程式時的當前時間

def pisa_sr(args):
    # Initialize the model
    model = PiSASR_eval(args)
    model.set_eval()

    if args.degradation_file:
        deg_file_path = args.degradation_file
    realesrgan_degradation = RealESRGAN_degradation(deg_file_path, device='cpu')

    # Get all input images
    if os.path.isdir(args.input_image):
        image_names = sorted(
            glob.glob(f'{args.input_image}/*.png') + 
            glob.glob(f'{args.input_image}/*.jpg')
    )
    else:
        image_names = [args.input_image]

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
    for idx, image_name in enumerate(image_names, start=1):
        if idx > args.max_inference_imgs_num:
            print(f"已達到最大推論圖片數量 {args.max_inference_imgs_num}，停止處理。")
            break
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

        # Step 1. 製作 Ground Truth (GT)
        if args.degradation_file == None: # 如果不給 degradation_file, 表示 bicubic_4x, 就直接用 input image 當 GT
            gt_image = input_image
        else: # 用的 degradation 來一次性產生 GT 與 LR (因為可能會有水平翻轉等問題, 因此 GT 也會動到)
            gt_image, lr_image = realesrgan_degradation.degrade_process(np.asarray(input_image)/255., resize_bak=False)
            gt_image = torch.clamp((gt_image * 255.0).round(), 0, 255) # 映射回 0-255
            gt_image = tensor2img(gt_image, out_type=np.uint8, min_max=(0, 255)) # tensor to numpy
            gt_image = Image.fromarray(gt_image) # numpy to PIL

        gt_width = gt_image.width - gt_image.width % (8 * args.upscale) # 確保是 8 * args.upscale 的倍數
        gt_height = gt_image.height - gt_image.height % (8 * args.upscale) # 確保是 8 * args.upscale 的倍數
        gt_image = gt_image.resize((gt_width, gt_height), Image.LANCZOS)
        gt_image.save(os.path.join(gt_image_path, bname))
        if idx < 5:
            wandb.log({
                "GT": [wandb.Image(gt_image, caption=f"GT-{bname}")],
        })

        # Step 2. Degradation. 將 GT Downsample 成 LR
        if args.degradation_file == None: # 如果不給 degradation_file, 表示 bicubic_4x, 就直接 downsample
            lr_image = gt_image.resize((gt_image.size[0] // args.upscale, gt_image.size[1] // args.upscale), Image.BICUBIC)
        else: # 因為已經在上一步產生 downsampled 的 LR 了, 這裡就只需要映射回 0-255 即可
            lr_image = torch.clamp((lr_image * 255.0).round(), 0, 255) # 映射回 0-255
            lr_image = tensor2img(lr_image, out_type=np.uint8, min_max=(0, 255)) # tensor to numpy
            lr_image = Image.fromarray(lr_image) # numpy to PIL
            lr_image = lr_image.resize((gt_image.size[0] // args.upscale, gt_image.size[1] // args.upscale), Image.BICUBIC)

        lr_image.save(os.path.join(lr_image_path, bname))
        if idx < 5:
            wandb.log({
                "LR": [wandb.Image(lr_image, caption=f"LR-{bname}")],
            })

        print("input_image.size: ", input_image.size, "gt_image.size: ", gt_image.size, "lr_image.size: ", lr_image.size)

        # 將 LR 先強行放大成 rscale 倍, 再由模型修復 artifact
        lr_image_upsample = lr_image.resize((lr_image.size[0] * rscale, lr_image.size[1] * rscale))
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
        if idx < 5:
            wandb.log({
                "HR": [wandb.Image(output_pil, caption=f"HR-{bname}")]
            })

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
        wandb.log({
            "Average PSNR": avg_psnr,
            "Average SSIM": avg_ssim,
            "Average Inference Time": average_time
        })
    else:
        print("[警告] 沒有成功計算任何 PSNR/SSIM")

    # 計算 FID
    try:
        fid_value = compute_fid(real_path=gt_image_path, fake_path=hr_image_path)
        wandb.log({"FID": fid_value})
    except Exception as e:
        logging.error(f"Failed to compute FID: {e}\n{traceback.format_exc()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', '-i', type=str, default='../HC18/test_set', help="path to the input image") # 為了取得 pair data 來計算 PSNR 與 SSIM, 輸入從原始程式中的 LR 改成直接輸入 GT, 程式會自動 degradation 來產生 LR
    parser.add_argument('--output_dir', '-o', type=str, default='experiments/test', help="the directory to save the output")
    parser.add_argument("--pretrained_model_path", type=str, default='preset/models/sd-2.1-base')
    parser.add_argument('--pretrained_path', type=str, default='preset/models/pisa_sr.pkl', help="path to a model state dict to be used")
    parser.add_argument('--seed', type=int, default=42, help="Random seed to be used")
    parser.add_argument("--process_size", type=int, default=64) # 輸入的圖片邊長會被至少調整至 process_size // upscale (處理太小的模型用的)
    parser.add_argument("--upscale", type=int, default=4)
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
