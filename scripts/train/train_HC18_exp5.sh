# 實驗 5
# 使用不同程度的 degradation 設定檔 (degradation_0.yml ~ degradation_3.yml)
# 測試時統一僅用 bicubic downsampling


########################################
# 0. deg_file_path="degradation_0.yml" (無 degradation)
CUDA_VISIBLE_DEVICES="6,7" accelerate launch train_pisasr.py \
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
    --output_dir="experiments/HC18/exp5/0/train" \
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
    --deg_file_path="degradation_0.yml" \
    --tracker_project_name "PiSASR" \
    --is_module True \
    --resolution_ori=128 \
    --wandb_project_name "pisasr-exp5" \
    --wandb_run_name "exp5_deg0 (train)"

## 0. test
CUDA_VISIBLE_DEVICES=7 \
python test_pisasr.py \
  --pretrained_model_path preset/models/sd-2.1-base \
  --pretrained_path experiments/HC18/exp5/0/train/checkpoints/model_1001.pkl \
  --process_size 64 \
  --upscale 4 \
  --input_image ../HC18/test_set \
  --output_dir experiments/HC18/exp5/0/test \
  --lambda_pix 1.0 \
  --lambda_sem 1.0 \
  --wandb_project_name "pisasr-exp5" \
  --wandb_run_name "exp5_deg0 (test)"


########################################
# 1. deg_file_path="degradation_1.yml" (輕度 degradation)
CUDA_VISIBLE_DEVICES="6,7" accelerate launch train_pisasr.py \
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
    --output_dir="experiments/HC18/exp5/1/train" \
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
    --deg_file_path="degradation_1.yml" \
    --tracker_project_name "PiSASR" \
    --is_module True \
    --resolution_ori=128 \
    --wandb_project_name "pisasr-exp5" \
    --wandb_run_name "exp5_deg1 (train)"

## 1. test
CUDA_VISIBLE_DEVICES=7 \
python test_pisasr.py \
  --pretrained_model_path preset/models/sd-2.1-base \
  --pretrained_path experiments/HC18/exp5/1/train/checkpoints/model_1001.pkl \
  --process_size 64 \
  --upscale 4 \
  --input_image ../HC18/test_set \
  --output_dir experiments/HC18/exp5/1/test \
  --lambda_pix 1.0 \
  --lambda_sem 1.0 \
  --wandb_project_name "pisasr-exp5" \
  --wandb_run_name "exp5_deg1 (test)"


########################################
# 2. deg_file_path="degradation_2.yml" (中度 degradation)
CUDA_VISIBLE_DEVICES="6,7" accelerate launch train_pisasr.py \
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
    --output_dir="experiments/HC18/exp5/2/train" \
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
    --deg_file_path="degradation_2.yml" \
    --tracker_project_name "PiSASR" \
    --is_module True \
    --resolution_ori=128 \
    --wandb_project_name "pisasr-exp5" \
    --wandb_run_name "exp5_deg2 (train)"

## 2. test
CUDA_VISIBLE_DEVICES=7 \
python test_pisasr.py \
  --pretrained_model_path preset/models/sd-2.1-base \
  --pretrained_path experiments/HC18/exp5/2/train/checkpoints/model_1001.pkl \
  --process_size 64 \
  --upscale 4 \
  --input_image ../HC18/test_set \
  --output_dir experiments/HC18/exp5/2/test \
  --lambda_pix 1.0 \
  --lambda_sem 1.0 \
  --wandb_project_name "pisasr-exp5" \
  --wandb_run_name "exp5_deg2 (test)"


########################################
# 3. deg_file_path="degradation_3.yml" (重度 degradation)
CUDA_VISIBLE_DEVICES="6,7" accelerate launch train_pisasr.py \
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
    --output_dir="experiments/HC18/exp5/3/train" \
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
    --deg_file_path="degradation_3.yml" \
    --tracker_project_name "PiSASR" \
    --is_module True \
    --resolution_ori=128 \
    --wandb_project_name "pisasr-exp5" \
    --wandb_run_name "exp5_deg3 (train)"

## 3. test
CUDA_VISIBLE_DEVICES=7 \
python test_pisasr.py \
  --pretrained_model_path preset/models/sd-2.1-base \
  --pretrained_path experiments/HC18/exp5/3/train/checkpoints/model_1001.pkl \
  --process_size 64 \
  --upscale 4 \
  --input_image ../HC18/test_set \
  --output_dir experiments/HC18/exp5/3/test \
  --lambda_pix 1.0 \
  --lambda_sem 1.0 \
  --wandb_project_name "pisasr-exp5" \
  --wandb_run_name "exp5_deg3 (test)"
