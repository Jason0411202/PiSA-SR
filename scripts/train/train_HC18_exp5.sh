# 實驗 5
# 目前的 training 是使用 real-esrgan 的 degradation
# 由於測試時僅用簡單的 bicubic，因此我認為訓練時不需要那麼強的 degradation
# 訓練時，實驗 training degradation 使用強 (real-esrgan) 或僅 bicubic 比較好

########################################
# 1. deg_file_path="bicubic_4x.yml" (僅 bicubic 4x in training)
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
    --deg_file_path="bicubic_4x.yml" \
    --tracker_project_name "PiSASR" \
    --is_module True \
    --resolution_ori=128 \
    --wandb_run_name "exp5_deg_bicubic4x (train)"

## 1. test
CUDA_VISIBLE_DEVICES=4 \
python test_pisasr.py \
  --pretrained_model_path preset/models/sd-2.1-base \
  --pretrained_path experiments/HC18/exp5/1/train/checkpoints/model_1001.pkl \
  --process_size 64 \
  --upscale 4 \
  --input_image ../HC18/test_set \
  --output_dir experiments/HC18/exp5/1/test \
  --lambda_pix 1.0 \
  --lambda_sem 1.0 \
  --wandb_run_name "exp5_deg_bicubic4x (test)"


########################################
# 2. deg_file_path="params.yml" (對照組：real-esrgan degradation)
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
    --deg_file_path="params.yml" \
    --tracker_project_name "PiSASR" \
    --is_module True \
    --resolution_ori=128 \
    --wandb_run_name "exp5_deg_params (train)"

## 2. test
CUDA_VISIBLE_DEVICES=4 \
python test_pisasr.py \
  --pretrained_model_path preset/models/sd-2.1-base \
  --pretrained_path experiments/HC18/exp5/2/train/checkpoints/model_1001.pkl \
  --process_size 64 \
  --upscale 4 \
  --input_image ../HC18/test_set \
  --output_dir experiments/HC18/exp5/2/test \
  --lambda_pix 1.0 \
  --lambda_sem 1.0 \
  --wandb_run_name "exp5_deg_params (test)"
