# 實驗 2
# 首先使用作者的預設參數在 HC18 資料集上進行 finetune。測試時，嘗試不同的 test 參數 ( lambda_pix 及 lambda_sem 的比例)，來驗證哪種組合最能衝高 PSNR 跟 SSIM

# 1. lambda_pix=0.0, lambda_sem=1.0
# 2. lambda_pix=0.2, lambda_sem=1.0
# 3. lambda_pix=0.5, lambda_sem=1.0
# 4. lambda_pix=0.8, lambda_sem=1.0
# 5. lambda_pix=1.0, lambda_sem=1.0
# 6. lambda_pix=1.0, lambda_sem=0.0
# 7. lambda_pix=1.0, lambda_sem=0.2
# 8. lambda_pix=1.0, lambda_sem=0.5
# 9. lambda_pix=1.0, lambda_sem=0.8


# 1. lambda_pix=0.0, lambda_sem=1.0
CUDA_VISIBLE_DEVICES=4 \
python test_pisasr.py \
  --pretrained_model_path preset/models/sd-2.1-base \
  --pretrained_path experiments/HC18/exp0/1/train/checkpoints/model_1001.pkl \
  --process_size 64 \
  --upscale 4 \
  --input_image ../HC18/test_set \
  --output_dir experiments/HC18/exp2/1/test \
  --lambda_pix 0.0 \
  --lambda_sem 1.0 \
  --wandb_project_name "pisasr-exp2" \
  --wandb_run_name "exp2_1_lambda_pix0.0_lambda_sem1.0 (test)"

# 2. lambda_pix=0.2, lambda_sem=1.0
CUDA_VISIBLE_DEVICES=4 \
python test_pisasr.py \
  --pretrained_model_path preset/models/sd-2.1-base \
  --pretrained_path experiments/HC18/exp0/1/train/checkpoints/model_1001.pkl \
  --process_size 64 \
  --upscale 4 \
  --input_image ../HC18/test_set \
  --output_dir experiments/HC18/exp2/2/test \
  --lambda_pix 0.2 \
  --lambda_sem 1.0 \
  --wandb_project_name "pisasr-exp2" \
  --wandb_run_name "exp2_2_lambda_pix0.2_lambda_sem1.0 (test)"

# 3. lambda_pix=0.5, lambda_sem=1.0
CUDA_VISIBLE_DEVICES=4 \
python test_pisasr.py \
  --pretrained_model_path preset/models/sd-2.1-base \
  --pretrained_path experiments/HC18/exp0/1/train/checkpoints/model_1001.pkl \
  --process_size 64 \
  --upscale 4 \
  --input_image ../HC18/test_set \
  --output_dir experiments/HC18/exp2/3/test \
  --lambda_pix 0.5 \
  --lambda_sem 1.0 \
  --wandb_project_name "pisasr-exp2" \
  --wandb_run_name "exp2_3_lambda_pix0.5_lambda_sem1.0 (test)"

# 4. lambda_pix=0.8, lambda_sem=1.0
CUDA_VISIBLE_DEVICES=4 \
python test_pisasr.py \
  --pretrained_model_path preset/models/sd-2.1-base \
  --pretrained_path experiments/HC18/exp0/1/train/checkpoints/model_1001.pkl \
  --process_size 64 \
  --upscale 4 \
  --input_image ../HC18/test_set \
  --output_dir experiments/HC18/exp2/4/test \
  --lambda_pix 0.8 \
  --lambda_sem 1.0 \
  -wandb_project_name "pisasr-exp2" \
  --wandb_run_name "exp2_4_lambda_pix0.8_lambda_sem1.0 (test)"

# 5. lambda_pix=1.0, lambda_sem=1.0
CUDA_VISIBLE_DEVICES=4 \
python test_pisasr.py \
  --pretrained_model_path preset/models/sd-2.1-base \
  --pretrained_path experiments/HC18/exp0/1/train/checkpoints/model_1001.pkl \
  --process_size 64 \
  --upscale 4 \
  --input_image ../HC18/test_set \
  --output_dir experiments/HC18/exp2/5/test \
  --lambda_pix 1.0 \
  --lambda_sem 1.0 \
  -wandb_project_name "pisasr-exp2" \
  --wandb_run_name "exp2_5_lambda_pix1.0_lambda_sem1.0 (test)"

# 6. lambda_pix=1.0, lambda_sem=0.0
CUDA_VISIBLE_DEVICES=4 \
python test_pisasr.py \
  --pretrained_model_path preset/models/sd-2.1-base \
  --pretrained_path experiments/HC18/exp0/1/train/checkpoints/model_1001.pkl \
  --process_size 64 \
  --upscale 4 \
  --input_image ../HC18/test_set \
  --output_dir experiments/HC18/exp2/6/test \
  --lambda_pix 1.0 \
  --lambda_sem 0.0 \
  -wandb_project_name "pisasr-exp2" \
  --wandb_run_name "exp2_6_lambda_pix1.0_lambda_sem0.0 (test)"

# 7. lambda_pix=1.0, lambda_sem=0.2
CUDA_VISIBLE_DEVICES=4 \
python test_pisasr.py \
  --pretrained_model_path preset/models/sd-2.1-base \
  --pretrained_path experiments/HC18/exp0/1/train/checkpoints/model_1001.pkl \
  --process_size 64 \
  --upscale 4 \
  --input_image ../HC18/test_set \
  --output_dir experiments/HC18/exp2/7/test \
  --lambda_pix 1.0 \
  --lambda_sem 0.2 \
  -wandb_project_name "pisasr-exp2" \
  --wandb_run_name "exp2_7_lambda_pix1.0_lambda_sem0.2 (test)"

# 8. lambda_pix=1.0, lambda_sem=0.5
CUDA_VISIBLE_DEVICES=4 \
python test_pisasr.py \
  --pretrained_model_path preset/models/sd-2.1-base \
  --pretrained_path experiments/HC18/exp0/1/train/checkpoints/model_1001.pkl \
  --process_size 64 \
  --upscale 4 \
  --input_image ../HC18/test_set \
  --output_dir experiments/HC18/exp2/8/test \
  --lambda_pix 1.0 \
  --lambda_sem 0.5 \
  -wandb_project_name "pisasr-exp2" \
  --wandb_run_name "exp2_8_lambda_pix1.0_lambda_sem0.5 (test)"

# 9. lambda_pix=1.0, lambda_sem=0.8
CUDA_VISIBLE_DEVICES=4 \
python test_pisasr.py \
  --pretrained_model_path preset/models/sd-2.1-base \
  --pretrained_path experiments/HC18/exp0/1/train/checkpoints/model_1001.pkl \
  --process_size 64 \
  --upscale 4 \
  --input_image ../HC18/test_set \
  --output_dir experiments/HC18/exp2/9/test \
  --lambda_pix 1.0 \
  --lambda_sem 0.8 \
  -wandb_project_name "pisasr-exp2" \
  --wandb_run_name "exp2_9_lambda_pix1.0_lambda_sem0.8 (test)"
