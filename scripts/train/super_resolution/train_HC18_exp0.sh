# output_dir 統一為 experiments/資料集名稱/第幾個實驗/本實驗中的第幾個 run/{train or test 結果}

# 最基本的 baseline
# 使用 HC18 資料集, 預設參數進行訓練

TRAIN_DEVICES="6,7"
TEST_DEVICE="6"

# # Training
# CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} accelerate launch train_pisasr.py \
#     --train_folder="../HC18/training_set" \
#     --output_dir="experiments/HC18/exp0/1/train" \
#     --wandb_project_name "pisasr-exp0" \
#     --wandb_run_name "exp0 (train)"

# Testing
CUDA_VISIBLE_DEVICES=${TEST_DEVICE} \
python test_pisasr.py \
    --pretrained_path experiments/HC18/exp0/1/train/checkpoints/model_1001.pkl \
    --output_dir experiments/HC18/exp0/1/test \
    --wandb_project_name "pisasr-exp0" \
    --wandb_run_name "exp0 (test)"