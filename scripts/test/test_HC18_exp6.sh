# 實驗 6
# 首先使用作者的預設參數在 HC18 資料集上進行 finetune。測試時，嘗試不同的 test 參數 ( lambda_pix 及 lambda_sem 的比例)，來驗證哪種組合最能衝高 PSNR 跟 SSIM
# 與實驗 2 不同的是，在移除殘差的情況下進行測試，以驗證殘差對於最終結果的影響

# 1. lambda_pix=0.0, lambda_sem=1.0
# 2. lambda_pix=0.2, lambda_sem=1.0
# 3. lambda_pix=0.5, lambda_sem=1.0
# 4. lambda_pix=0.8, lambda_sem=1.0
# 5. lambda_pix=1.0, lambda_sem=1.0
# 6. lambda_pix=1.0, lambda_sem=0.0
# 7. lambda_pix=1.0, lambda_sem=0.2
# 8. lambda_pix=1.0, lambda_sem=0.5
# 9. lambda_pix=1.0, lambda_sem=0.8


## train
CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch train_pisasr.py \
    --pretrained_model_path="preset/models/sd-2.1-base" \
    --pretrained_model_path_csd="preset/models/sd-2.1-base" \
    --train_folder="../HC18/training_set" \
    --dataset_txt_paths="preset/gt_path.txt" \
    --highquality_dataset_txt_paths="preset/gt_path.txt" \
    --learning_rate=5e-5 \
    --train_batch_size=1 \
    --prob=0.0 \
    --gradient_accumulation_steps=4 \
    --enable_xformers_memory_efficient_attention --checkpointing_steps 500 \
    --seed 123 \
    --output_dir="experiments/HC18/exp6/1/train" \
    --cfg_csd 7.5 \
    --timesteps1 1 \
    --lambda_lpips=2.0 \
    --lambda_l2=1.0 \
    --lambda_csd=1.0 \
    --pix_steps=4000 \
    --lora_rank_unet_pix=4 \
    --lora_rank_unet_sem=4 \
    --min_dm_step_ratio=0.02 \
    --max_dm_step_ratio=0.5 \
    --null_text_ratio=0.5 \
    --align_method="adain" \
    --deg_file_path="params.yml" \
    --tracker_project_name "PiSASR" \
    --is_module True \
    --resolution_ori=128 \
    --use_residual_in_training False \
    --wandb_project_name "pisasr-exp6" \
    --wandb_run_name "exp6_no_residual (train)"


# --------------------- TEST ---------------------

# 1. lambda_pix=0.0, lambda_sem=1.0
CUDA_VISIBLE_DEVICES=4 \
python test_pisasr.py \
  --pretrained_model_path preset/models/sd-2.1-base \
  --pretrained_path experiments/HC18/exp6/1/train/checkpoints/model_1001.pkl \
  --process_size 64 \
  --upscale 4 \
  --input_image ../HC18/test_set \
  --output_dir experiments/HC18/exp6/1/test \
  --lambda_pix 0.0 \
  --lambda_sem 1.0 \
  --wandb_project_name "pisasr-exp6" \
  --wandb_run_name "exp6_1_no_residual_lambda_pix0.0_lambda_sem1.0 (test)"

# 2. lambda_pix=0.2, lambda_sem=1.0
CUDA_VISIBLE_DEVICES=4 \
python test_pisasr.py \
  --pretrained_model_path preset/models/sd-2.1-base \
  --pretrained_path experiments/HC18/exp6/1/train/checkpoints/model_1001.pkl \
  --process_size 64 \
  --upscale 4 \
  --input_image ../HC18/test_set \
  --output_dir experiments/HC18/exp6/2/test \
  --lambda_pix 0.2 \
  --lambda_sem 1.0 \
  --wandb_project_name "pisasr-exp6" \
  --wandb_run_name "exp6_2_no_residual_lambda_pix0.2_lambda_sem1.0 (test)"

# 3. lambda_pix=0.5, lambda_sem=1.0
CUDA_VISIBLE_DEVICES=4 \
python test_pisasr.py \
  --pretrained_model_path preset/models/sd-2.1-base \
  --pretrained_path experiments/HC18/exp6/1/train/checkpoints/model_1001.pkl \
  --process_size 64 \
  --upscale 4 \
  --input_image ../HC18/test_set \
  --output_dir experiments/HC18/exp6/3/test \
  --lambda_pix 0.5 \
  --lambda_sem 1.0 \
  --wandb_project_name "pisasr-exp6" \
  --wandb_run_name "exp6_3_no_residual_lambda_pix0.5_lambda_sem1.0 (test)"

# 4. lambda_pix=0.8, lambda_sem=1.0
CUDA_VISIBLE_DEVICES=4 \
python test_pisasr.py \
  --pretrained_model_path preset/models/sd-2.1-base \
  --pretrained_path experiments/HC18/exp6/1/train/checkpoints/model_1001.pkl \
  --process_size 64 \
  --upscale 4 \
  --input_image ../HC18/test_set \
  --output_dir experiments/HC18/exp6/4/test \
  --lambda_pix 0.8 \
  --lambda_sem 1.0 \
  --wandb_project_name "pisasr-exp6" \
  --wandb_run_name "exp6_4_no_residual_lambda_pix0.8_lambda_sem1.0 (test)"

# 5. lambda_pix=1.0, lambda_sem=1.0
CUDA_VISIBLE_DEVICES=4 \
python test_pisasr.py \
  --pretrained_model_path preset/models/sd-2.1-base \
  --pretrained_path experiments/HC18/exp6/1/train/checkpoints/model_1001.pkl \
  --process_size 64 \
  --upscale 4 \
  --input_image ../HC18/test_set \
  --output_dir experiments/HC18/exp6/5/test \
  --lambda_pix 1.0 \
  --lambda_sem 1.0 \
  --wandb_project_name "pisasr-exp6" \
  --wandb_run_name "exp6_5_no_residual_lambda_pix1.0_lambda_sem1.0 (test)"

# 6. lambda_pix=1.0, lambda_sem=0.0
CUDA_VISIBLE_DEVICES=4 \
python test_pisasr.py \
  --pretrained_model_path preset/models/sd-2.1-base \
  --pretrained_path experiments/HC18/exp6/1/train/checkpoints/model_1001.pkl \
  --process_size 64 \
  --upscale 4 \
  --input_image ../HC18/test_set \
  --output_dir experiments/HC18/exp6/6/test \
  --lambda_pix 1.0 \
  --lambda_sem 0.0 \
  --wandb_project_name "pisasr-exp6" \
  --wandb_run_name "exp6_6_no_residual_lambda_pix1.0_lambda_sem0.0 (test)"

# 7. lambda_pix=1.0, lambda_sem=0.2
CUDA_VISIBLE_DEVICES=4 \
python test_pisasr.py \
  --pretrained_model_path preset/models/sd-2.1-base \
  --pretrained_path experiments/HC18/exp6/1/train/checkpoints/model_1001.pkl \
  --process_size 64 \
  --upscale 4 \
  --input_image ../HC18/test_set \
  --output_dir experiments/HC18/exp6/7/test \
  --lambda_pix 1.0 \
  --lambda_sem 0.2 \
  --wandb_project_name "pisasr-exp6" \
  --wandb_run_name "exp6_7_no_residual_lambda_pix1.0_lambda_sem0.2 (test)"

# 8. lambda_pix=1.0, lambda_sem=0.5
CUDA_VISIBLE_DEVICES=4 \
python test_pisasr.py \
  --pretrained_model_path preset/models/sd-2.1-base \
  --pretrained_path experiments/HC18/exp6/1/train/checkpoints/model_1001.pkl \
  --process_size 64 \
  --upscale 4 \
  --input_image ../HC18/test_set \
  --output_dir experiments/HC18/exp6/8/test \
  --lambda_pix 1.0 \
  --lambda_sem 0.5 \
  --wandb_project_name "pisasr-exp6" \
  --wandb_run_name "exp6_8_no_residual_lambda_pix1.0_lambda_sem0.5 (test)"

# 9. lambda_pix=1.0, lambda_sem=0.8
CUDA_VISIBLE_DEVICES=4 \
python test_pisasr.py \
  --pretrained_model_path preset/models/sd-2.1-base \
  --pretrained_path experiments/HC18/exp6/1/train/checkpoints/model_1001.pkl \
  --process_size 64 \
  --upscale 4 \
  --input_image ../HC18/test_set \
  --output_dir experiments/HC18/exp6/9/test \
  --lambda_pix 1.0 \
  --lambda_sem 0.8 \
  --wandb_project_name "pisasr-exp6" \
  --wandb_run_name "exp6_9_no_residual_lambda_pix1.0_lambda_sem0.8 (test)"
