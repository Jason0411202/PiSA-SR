# =========================================================
# exp8: HC18 denoising 任務
# Gaussian noise range: 15, 30, 60, 90
# =========================================================

TRAIN_DEVICES="0,1"
TEST_DEVICE="0"

# -------------------------
# exp8/1: Gaussian noise level 15
# -------------------------
CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} accelerate launch train_pisasr.py \
    --train_folder="../HC18/training_set" \
    --output_dir="experiments/HC18/exp8/1/train" \
    --wandb_project_name "pisasr-exp8" \
    --wandb_run_name "exp8-denoising-noise_range_15 (train)" \
    --pix_steps 1001 \
    --deg_file_path="gaussian_noise_15.yml" \

# Testing
CUDA_VISIBLE_DEVICES=${TEST_DEVICE} \
python test_pisasr.py \
    --pretrained_path "experiments/HC18/exp8/1/train/checkpoints/model_1001.pkl" \
    --output_dir "experiments/HC18/exp8/1/test" \
    --upscale 1 \
    --wandb_project_name "pisasr-exp8" \
    --wandb_run_name "exp8-denoising-noise_range_15 (test)" \
    --degradation_file "gaussian_noise_15.yml" \
    --lambda_pix 1.0 \
    --lambda_sem 0.0


# -------------------------
# exp8/2: Gaussian noise level 30
# -------------------------
CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} accelerate launch train_pisasr.py \
    --train_folder="../HC18/training_set" \
    --output_dir="experiments/HC18/exp8/2/train" \
    --wandb_project_name "pisasr-exp8" \
    --wandb_run_name "exp8-denoising-noise_range_30 (train)" \
    --pix_steps 1001 \
    --deg_file_path="gaussian_noise_30.yml" \

# Testing
CUDA_VISIBLE_DEVICES=${TEST_DEVICE} \
python test_pisasr.py \
    --pretrained_path "experiments/HC18/exp8/2/train/checkpoints/model_1001.pkl" \
    --output_dir "experiments/HC18/exp8/2/test" \
    --upscale 1 \
    --wandb_project_name "pisasr-exp8" \
    --wandb_run_name "exp8-denoising-noise_range_30 (test)" \
    --degradation_file "gaussian_noise_30.yml" \
    --lambda_pix 1.0 \
    --lambda_sem 0.0


# -------------------------
# exp8/3: Gaussian noise level 60
# -------------------------
CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} accelerate launch train_pisasr.py \
    --train_folder="../HC18/training_set" \
    --output_dir="experiments/HC18/exp8/3/train" \
    --wandb_project_name "pisasr-exp8" \
    --wandb_run_name "exp8-denoising-noise_range_60 (train)" \
    --pix_steps 1001 \
    --deg_file_path="gaussian_noise_60.yml" \

# Testing
CUDA_VISIBLE_DEVICES=${TEST_DEVICE} \
python test_pisasr.py \
    --pretrained_path "experiments/HC18/exp8/3/train/checkpoints/model_1001.pkl" \
    --output_dir "experiments/HC18/exp8/3/test" \
    --upscale 1 \
    --wandb_project_name "pisasr-exp8" \
    --wandb_run_name "exp8-denoising-noise_range_60 (test)" \
    --degradation_file "gaussian_noise_60.yml" \
    --lambda_pix 1.0 \
    --lambda_sem 0.0


# -------------------------
# exp8/4: Gaussian noise level 90
# -------------------------
CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} accelerate launch train_pisasr.py \
    --train_folder="../HC18/training_set" \
    --output_dir="experiments/HC18/exp8/4/train" \
    --wandb_project_name "pisasr-exp8" \
    --wandb_run_name "exp8-denoising-noise_range_90 (train)" \
    --pix_steps 1001 \
    --deg_file_path="gaussian_noise_90.yml" \

# Testing
CUDA_VISIBLE_DEVICES=${TEST_DEVICE} \
python test_pisasr.py \
    --pretrained_path "experiments/HC18/exp8/4/train/checkpoints/model_1001.pkl" \
    --output_dir "experiments/HC18/exp8/4/test" \
    --upscale 1 \
    --wandb_project_name "pisasr-exp8" \
    --wandb_run_name "exp8-denoising-noise_range_90 (test)" \
    --degradation_file "gaussian_noise_90.yml" \
    --lambda_pix 1.0 \
    --lambda_sem 0.0
