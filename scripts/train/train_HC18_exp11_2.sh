# 實驗十一之二
# 在 gaussian_noise_10_30 degradation 上, 測試不同 GAN loss 強度造成的影響
# GAN loss 只作用於 semantic-level 的 LoRA 的訓練, pixel-level 的 LoRA 則不受影響

TRAIN_DEVICES="3,7"
TEST_DEVICE="3"
PROJECT_NAME="pisasr-exp11-2"
EXP="exp11-2"

degradations=("gaussian_noise_10_30")
enable_gan_loss=("True")
lambda_gans=("0" "0.001" "0.01" "0.1" "1.0" "10.0")  # GAN loss 強度
model_steps="1001"

for degradation in "${degradations[@]}"; do
  for gan_loss in "${enable_gan_loss[@]}"; do
    for lambda_gan in "${lambda_gans[@]}"; do
      safe_lambda=$(echo ${lambda_gan} | sed 's/\./_/g')
      run_name="${EXP}_${degradation}_lambda_gan${safe_lambda}"

      echo ">>> [Training] ${run_name}"
      CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} accelerate launch train_pisasr.py \
          --train_folder="src/datasets/for_generate_dataset/outputs/${degradation}/training_set/GT" \
          --train_folder_lr="src/datasets/for_generate_dataset/outputs/${degradation}/training_set/LR" \
          --output_dir="experiments/HC18/${EXP}/${run_name}/train" \
          --enable_gan_loss=${gan_loss} \
          --lambda_gan=${lambda_gan} \
          --wandb_project_name "${PROJECT_NAME}" \
          --wandb_run_name "${run_name} (train)"

      echo ">>> [Testing] ${run_name}"
      CUDA_VISIBLE_DEVICES=${TEST_DEVICE} \
      python test_pisasr.py \
          --input_gt_image src/datasets/for_generate_dataset/outputs/${degradation}/test_set/GT \
          --input_lr_image src/datasets/for_generate_dataset/outputs/${degradation}/test_set/LR \
          --pretrained_path "experiments/HC18/${EXP}/${run_name}/train/checkpoints/model_${model_steps}.pkl" \
          --output_dir "experiments/HC18/${EXP}/${run_name}/test" \
          --wandb_project_name "${PROJECT_NAME}" \
          --wandb_run_name "${run_name} (test)"

    done
  done
done
