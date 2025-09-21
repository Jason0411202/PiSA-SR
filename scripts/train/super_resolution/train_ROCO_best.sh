# output_dir 統一為 experiments/資料集名稱/第幾個實驗/本實驗中的第幾個 run/{train or test 結果}

# 最基本的 baseline
# 使用 ROCO 資料集, 預設參數進行訓練

# 由於訓練失敗 (PIL.UnidentifiedImageError: cannot identify image file '/home/youkai/roco_dataset/all_data/train/radiology/images/PMC4240561_MA-68-291-g002.jpg')
# 目前只能使用 20001 步的模型進行測試

TRAIN_DEVICES="1,7"
TEST_DEVICE="1"

# # Training
# CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} accelerate launch train_pisasr.py \
#     --train_folder="../roco_dataset/all_data/train/radiology/images" \
#     --output_dir="experiments/ROCO/best/1/train" \
#     --wandb_project_name "ROCO-best" \
#     --wandb_run_name "best-ROCO-train" \
#     --pix_steps 50000 \
#     --max_train_steps 31000 \
#     --deg_file_path="degradation_1.yml"

# Testing on ROCO test set
CUDA_VISIBLE_DEVICES=${TEST_DEVICE} \
python test_pisasr.py \
    --input_image "../roco_dataset/all_data/test/radiology/images" \
    --pretrained_path "experiments/ROCO/best/1/train/checkpoints/model_20001.pkl" \
    --output_dir "experiments/ROCO/best/1/test" \
    --wandb_project_name "ROCO-best" \
    --wandb_run_name "best-ROCO-test" \
    --lambda_pix 1.0 \
    --lambda_sem 0.0

# Testing on HC18 test set
CUDA_VISIBLE_DEVICES=${TEST_DEVICE} \
python test_pisasr.py \
    --pretrained_path "experiments/ROCO/best/1/train/checkpoints/model_20001.pkl" \
    --output_dir "experiments/ROCO/best/1/test" \
    --wandb_project_name "ROCO-best" \
    --wandb_run_name "best-HC18-test" \
    --lambda_pix 1.0 \
    --lambda_sem 0.0
