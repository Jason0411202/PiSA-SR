# =========================================================
# exp8: HC18 denoising 任務 (best)
# 嘗試找出最佳參數組合並測試
# =========================================================

TRAIN_DEVICES="6,7"
TEST_DEVICE="6"

CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} accelerate launch train_pisasr.py \
    --train_folder="../HC18/training_set" \
    --output_dir="experiments/HC18/denoise_best/1/train" \
    --wandb_project_name "HC18-denoise-best" \
    --wandb_run_name "denoise-best-1-train" \
    --pix_steps 50000 \
    --max_train_steps 31000 \
    --deg_file_path="gaussian_noise_30.yml"

# Testing
CUDA_VISIBLE_DEVICES=${TEST_DEVICE} \
python test_pisasr.py \
    --pretrained_path "experiments/HC18/denoise_best/1/train/checkpoints/model_1001.pkl" \
    --output_dir "experiments/HC18/denoise_best/1/test" \
    --upscale 1 \
    --wandb_project_name "HC18-denoise-best" \
    --wandb_run_name "denoise-best-1-test" \
    --degradation_file "gaussian_noise_30.yml" \
    --lambda_pix 1.0 \
    --lambda_sem 0.0
