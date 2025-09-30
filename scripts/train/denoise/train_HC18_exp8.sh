# =========================================================
# exp8: HC18 denoising 任務
# Gaussian noise range: 15, 30, 60, 90
# =========================================================

TRAIN_DEVICES="5,7"
TEST_DEVICE="5"
PROJECT_NAME="pisasr-exp8"
EXP="exp8"

# Gaussian noise levels
noise_levels=(15 30 60 90)

for i in "${!noise_levels[@]}"; do
  noise=${noise_levels[$i]}
  exp_id=$((i + 1))  # 1~4 唯一編號

  echo ">>> [Training] $EXP - Gaussian noise level $noise (exp_id=$exp_id)"
  CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} accelerate launch train_pisasr.py \
      --train_folder="../HC18/training_set" \
      --output_dir="experiments/HC18/${EXP}/${exp_id}/train" \
      --wandb_project_name "${PROJECT_NAME}" \
      --wandb_run_name "${EXP}-denoising-noise_range_${noise} (train)" \
      --deg_file_path="gaussian_noise_${noise}.yml"

  echo ">>> [Testing] $EXP - Gaussian noise level $noise (exp_id=$exp_id)"
  CUDA_VISIBLE_DEVICES=${TEST_DEVICE} \
  python test_pisasr.py \
      --pretrained_path "experiments/HC18/${EXP}/${exp_id}/train/checkpoints/model_1001.pkl" \
      --output_dir "experiments/HC18/${EXP}/${exp_id}/test" \
      --wandb_project_name "${PROJECT_NAME}" \
      --wandb_run_name "${EXP}-denoising-noise_range_${noise} (test)" \
      --degradation_file "gaussian_noise_${noise}.yml"
done
