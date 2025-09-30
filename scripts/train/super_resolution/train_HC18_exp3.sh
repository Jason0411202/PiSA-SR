# 實驗三
# 目的：比較訓練時是否使用殘差 (x_denoised = encoded_control - model_pred) 對結果的影響
# pisasr.py 中包含的 `x_denoised = encoded_control - model_pred` 便是殘差邏輯

TRAIN_DEVICES="0,1"
TEST_DEVICE="0"
PROJECT_NAME="pisasr-exp3"
EXP="exp3"

# 是否使用殘差
use_residual_in_training=("False" "True")

for i in "${!use_residual_in_training[@]}"; do
  residual=${use_residual_in_training[$i]}
  exp_id=$((i + 1))

  echo ">>> [Training] $EXP (Residual=$residual, exp_id=$exp_id)"
  CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} accelerate launch train_pisasr.py \
      --train_folder="../HC18/training_set" \
      --output_dir="experiments/HC18/${EXP}/${exp_id}/train" \
      --wandb_project_name "${PROJECT_NAME}" \
      --wandb_run_name "${EXP}_residual${residual} (train)" \
      --use_residual_in_training ${residual}

  echo ">>> [Testing] $EXP (Residual=$residual, exp_id=$exp_id)"
  CUDA_VISIBLE_DEVICES=${TEST_DEVICE} \
  python test_pisasr.py \
      --pretrained_path experiments/HC18/${EXP}/${exp_id}/train/checkpoints/model_1001.pkl \
      --output_dir experiments/HC18/${EXP}/${exp_id}/test \
      --wandb_project_name "${PROJECT_NAME}" \
      --wandb_run_name "${EXP}_residual${residual} (test)" \
      --use_residual_in_training ${residual}
done
