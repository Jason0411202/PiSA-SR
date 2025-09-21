# 實驗 7
# 目的：比較訓練時是否使用殘差 (x_denoised = encoded_control - model_pred) "在 test degradation 嚴重時" 對結果的影響
# pisasr.py 中包含的 `x_denoised = encoded_control - model_pred` 便是殘差邏輯

TRAIN_DEVICES="1,2"
TEST_DEVICE="1"

# 不使用殘差進行訓練
# Training
CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} accelerate launch train_pisasr.py \
    --train_folder="../HC18/training_set" \
    --output_dir="experiments/HC18/exp7/1/train" \
    --wandb_project_name "pisasr-exp7" \
    --wandb_run_name "exp7_no_residual (train)" \
    --use_residual_in_training False

# Testing
CUDA_VISIBLE_DEVICES=${TEST_DEVICE} \
python test_pisasr.py \
    --pretrained_path experiments/HC18/exp7/1/train/checkpoints/model_1001.pkl \
    --output_dir experiments/HC18/exp7/1/test \
    --degradation_type "realesrgan_4x" \
    --wandb_project_name "pisasr-exp7" \
    --wandb_run_name "exp7_no_residual (test)" \
    --use_residual_in_training False

# 使用殘差進行訓練 (對照組)
# Training
CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} accelerate launch train_pisasr.py \
    --train_folder="../HC18/training_set" \
    --output_dir="experiments/HC18/exp7/2/train" \
    --wandb_project_name "pisasr-exp7" \
    --wandb_run_name "exp7_with_residual (train)" \
    --use_residual_in_training True

# Testing
CUDA_VISIBLE_DEVICES=${TEST_DEVICE} \
python test_pisasr.py \
    --pretrained_path experiments/HC18/exp7/2/train/checkpoints/model_1001.pkl \
    --output_dir experiments/HC18/exp7/2/test \
    --degradation_type "realesrgan_4x" \
    --wandb_project_name "pisasr-exp7" \
    --wandb_run_name "exp7_with_residual (test)" \
    --use_residual_in_training True
