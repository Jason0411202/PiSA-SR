# =========================================================
# exp9: HC18 denoising 任務
# Gaussian blur range: [0.5, 1.0]
# =========================================================

TRAIN_DEVICES="4,5"
TEST_DEVICE="4"

# -------------------------
# exp9/1: Gaussian blur level 0.5
# -------------------------
CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} accelerate launch train_pisasr.py \
    --train_folder="../HC18/training_set" \
    --output_dir="experiments/HC18/exp9/1/train" \
    --wandb_project_name "pisasr-exp9" \
    --wandb_run_name "exp9-denoising-blur_range_0.5 (train)" \
    --pix_steps 1001 \
    --deg_file_path="gaussian_blur_0.5.yml" \

# Testing
CUDA_VISIBLE_DEVICES=${TEST_DEVICE} \
python test_pisasr.py \
    --pretrained_path "experiments/HC18/exp9/1/train/checkpoints/model_1001.pkl" \
    --output_dir "experiments/HC18/exp9/1/test" \
    --upscale 1 \
    --wandb_project_name "pisasr-exp9" \
    --wandb_run_name "exp9-denoising-blur_range_0.5 (test)" \
    --degradation_file "gaussian_blur_0.5.yml" \
    --lambda_pix 1.0 \
    --lambda_sem 0.0


# -------------------------
# exp9/2: Gaussian blur level 1.0
# -------------------------
CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} accelerate launch train_pisasr.py \
    --train_folder="../HC18/training_set" \
    --output_dir="experiments/HC18/exp9/2/train" \
    --wandb_project_name "pisasr-exp9" \
    --wandb_run_name "exp9-denoising-blur_range_1.0 (train)" \
    --pix_steps 1001 \
    --deg_file_path="gaussian_blur_1.0.yml" \

# Testing
CUDA_VISIBLE_DEVICES=${TEST_DEVICE} \
python test_pisasr.py \
    --pretrained_path "experiments/HC18/exp9/2/train/checkpoints/model_1001.pkl" \
    --output_dir "experiments/HC18/exp9/2/test" \
    --upscale 1 \
    --wandb_project_name "pisasr-exp9" \
    --wandb_run_name "exp9-denoising-blur_range_1.0 (test)" \
    --degradation_file "gaussian_blur_1.0.yml" \
    --lambda_pix 1.0 \
    --lambda_sem 0.0


# -------------------------
# exp9/3: Gaussian blur level 2.0
# -------------------------
CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} accelerate launch train_pisasr.py \
    --train_folder="../HC18/training_set" \
    --output_dir="experiments/HC18/exp9/3/train" \
    --wandb_project_name "pisasr-exp9" \
    --wandb_run_name "exp9-denoising-blur_range_2.0 (train)" \
    --pix_steps 1001 \
    --deg_file_path="gaussian_blur_2.0.yml" \

# Testing
CUDA_VISIBLE_DEVICES=${TEST_DEVICE} \
python test_pisasr.py \
    --pretrained_path "experiments/HC18/exp9/3/train/checkpoints/model_1001.pkl" \
    --output_dir "experiments/HC18/exp9/3/test" \
    --upscale 1 \
    --wandb_project_name "pisasr-exp9" \
    --wandb_run_name "exp9-denoising-blur_range_2.0 (test)" \
    --degradation_file "gaussian_blur_2.0.yml" \
    --lambda_pix 1.0 \
    --lambda_sem 0.0


# -------------------------
# exp9/4: Gaussian blur level 3.0
# -------------------------
CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} accelerate launch train_pisasr.py \
    --train_folder="../HC18/training_set" \
    --output_dir="experiments/HC18/exp9/4/train" \
    --wandb_project_name "pisasr-exp9" \
    --wandb_run_name "exp9-denoising-blur_range_3.0 (train)" \
    --pix_steps 1001 \
    --deg_file_path="gaussian_blur_3.0.yml" \

# Testing
CUDA_VISIBLE_DEVICES=${TEST_DEVICE} \
python test_pisasr.py \
    --pretrained_path "experiments/HC18/exp9/4/train/checkpoints/model_1001.pkl" \
    --output_dir "experiments/HC18/exp9/4/test" \
    --upscale 1 \
    --wandb_project_name "pisasr-exp9" \
    --wandb_run_name "exp9-denoising-blur_range_3.0 (test)" \
    --degradation_file "gaussian_blur_3.0.yml" \
    --lambda_pix 1.0 \
    --lambda_sem 0.0

