# 實驗 2
# 測試時，嘗試不同的 test 參數 lambda_pix 及 lambda_sem 的比例，來驗證哪種組合最能衝高 PSNR 跟 SSIM

# 1. lambda_pix=1, lambda_sem=1 (對照組)
# 2. lambda_pix=1, lambda_sem=0
# 3. lambda_pix=1.5, lambda_sem=0
# 4. lambda_pix=2, lambda_sem=0

## 1. lambda_pix=1, lambda_sem=1 (對照組)
CUDA_VISIBLE_DEVICES=4 \
python test_pisasr.py \
  --pretrained_model_path preset/models/sd-2.1-base \
  --pretrained_path experiments/train-pisasr/checkpoints/model_1001.pkl \
  --process_size 64 \
  --upscale 4 \
  --input_image ../HC18/test_set \
  --output_dir experiments/HC18/exp2/1/test \
  --lambda_pix 1.0 \
  --lambda_sem 1.0 \
  --wandb_run_name "exp2_1_lambda_pix1.0_lambda_sem1.0 (test)"

## 2. lambda_pix=1, lambda_sem=0
CUDA_VISIBLE_DEVICES=4 \
python test_pisasr.py \
  --pretrained_model_path preset/models/sd-2.1-base \
  --pretrained_path experiments/train-pisasr/checkpoints/model_1001.pkl \
  --process_size 64 \
  --upscale 4 \
  --input_image ../HC18/test_set \
  --output_dir experiments/HC18/exp2/2/test \
  --lambda_pix 1.0 \
  --lambda_sem 0.0 \
  --wandb_run_name "exp2_2_lambda_pix1.0_lambda_sem0.0 (test)"

## 3. lambda_pix=1.5, lambda_sem=0
CUDA_VISIBLE_DEVICES=4 \
python test_pisasr.py \
  --pretrained_model_path preset/models/sd-2.1-base \
  --pretrained_path experiments/train-pisasr/checkpoints/model_1001.pkl \
  --process_size 64 \
  --upscale 4 \
  --input_image ../HC18/test_set \
  --output_dir experiments/HC18/exp2/3/test \
  --lambda_pix 1.5 \
  --lambda_sem 0.0 \
  --wandb_run_name "exp2_3_lambda_pix1.5_lambda_sem0.0 (test)"

## 4. lambda_pix=2, lambda_sem=0
CUDA_VISIBLE_DEVICES=4 \
python test_pisasr.py \
  --pretrained_model_path preset/models/sd-2.1-base \
  --pretrained_path experiments/train-pisasr/checkpoints/model_1001.pkl \
  --process_size 64 \
  --upscale 4 \
  --input_image ../HC18/test_set \
  --output_dir experiments/HC18/exp2/4/test \
  --lambda_pix 2.0 \
  --lambda_sem 0.0 \
  --wandb_run_name "exp2_4_lambda_pix2.0_lambda_sem0.0 (test)"
