# output_dir 統一為 experiments/資料集名稱/第幾個實驗/本實驗中的第幾個 run/{train or test 結果}

# 最基本的 baseline
# 使用 HC18 資料集, 預設參數進行訓練

TRAIN_DEVICES="4,5"
TEST_DEVICE="4"

# Training
CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} accelerate launch train_pisasr.py \
    --train_folder="../HC18/training_set" \
    --output_dir="experiments/HC18/best/1/train" \
    --wandb_project_name "HC18-best" \
    --wandb_run_name "best-1-train" \
    --pix_steps 50000 \
    --max_train_steps 31000 \
    --deg_file_path="degradation_1.yml"

# Testing
CUDA_VISIBLE_DEVICES=${TEST_DEVICE} \
python test_pisasr.py \
    --pretrained_path "experiments/HC18/best/1/train/checkpoints/model_30001.pkl" \
    --output_dir "experiments/HC18/best/1/test" \
    --wandb_project_name "HC18-best" \
    --wandb_run_name "best-1-test" \
    --lambda_pix 1.0 \
    --lambda_sem 0.0
