# 實驗 3
# 目的：比較訓練時是否使用殘差 (x_denoised = encoded_control - model_pred) 對結果的影響
# pisasr.py 中包含的 `x_denoised = encoded_control - model_pred` 便是殘差邏輯

TRAIN_DEVICES="0,7"
TEST_DEVICE="0"

# 不使用殘差進行訓練
# Training
CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} accelerate launch train_pisasr.py \
    --train_folder="../HC18/training_set" \
    --output_dir="experiments/HC18/exp3/1/train" \
    --wandb_project_name "pisasr-exp3" \
    --wandb_run_name "exp3_no_residual (train)" \
    --use_residual_in_training False

# Testing
CUDA_VISIBLE_DEVICES=${TEST_DEVICE} \
python test_pisasr.py \
    --pretrained_path experiments/HC18/exp3/1/train/checkpoints/model_1001.pkl \
    --output_dir experiments/HC18/exp3/1/test \
    --wandb_project_name "pisasr-exp3" \
    --wandb_run_name "exp3_no_residual (test)"

# 使用殘差進行訓練 (對照組)
# Training
CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} accelerate launch train_pisasr.py \
    --train_folder="../HC18/training_set" \
    --output_dir="experiments/HC18/exp3/2/train" \
    --wandb_project_name "pisasr-exp3" \
    --wandb_run_name "exp3_with_residual (train)" \
    --use_residual_in_training True

# Testing
CUDA_VISIBLE_DEVICES=${TEST_DEVICE} \
python test_pisasr.py \
    --pretrained_path experiments/HC18/exp3/2/train/checkpoints/model_1001.pkl \
    --output_dir experiments/HC18/exp3/2/test \
    --wandb_project_name "pisasr-exp3" \
    --wandb_run_name "exp3_with_residual (test)"
