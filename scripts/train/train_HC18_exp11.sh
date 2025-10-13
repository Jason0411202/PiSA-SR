# 實驗十一
# 在不同 degradation 上, 測試引入 GAN loss 造成的影響
# GAN loss 只作用於 semantic-level 的 LoRA 的訓練, pixel-level 的 LoRA 則不受影響

TRAIN_DEVICES="5,6,7"
TEST_DEVICE="5"
PROJECT_NAME="pisasr-exp11"
EXP="exp11"

# degradations (順序固定)
degradations=("gaussian_noise_10_30" "bicubic_4x" "gaussian_blur" "complex" )
enable_gan_loss=("True" "False")  # 是否使用 GAN loss
model_steps="1001"  # 使用的模型訓練步數

for i in "${!degradations[@]}"; do
  for j in "${!enable_gan_loss[@]}"; do
    degradation=${degradations[$i]}
    gan_loss=${enable_gan_loss[$j]}
    exp_id=$((i*2 + j + 1))   # 保證 1~8 唯一編號

    echo ">>> [Training] $EXP - $degradation (GAN=$gan_loss, exp_id=$exp_id)"
    CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} accelerate launch train_pisasr.py \
        --train_folder="src/datasets/for_generate_dataset/outputs/${degradation}/training_set/GT" \
        --train_folder_lr="src/datasets/for_generate_dataset/outputs/${degradation}/training_set/LR" \
        --output_dir="experiments/HC18/${EXP}/${exp_id}/train" \
        --enable_gan_loss=${gan_loss} \
        --wandb_project_name "${PROJECT_NAME}" \
        --wandb_run_name "${EXP}_${degradation}_enable_gan_loss${gan_loss} (train)"

    echo ">>> [Testing] $EXP - $degradation (GAN=$gan_loss, exp_id=$exp_id)"
    CUDA_VISIBLE_DEVICES=${TEST_DEVICE} \
    python test_pisasr.py \
        --input_gt_image src/datasets/for_generate_dataset/outputs/${degradation}/test_set/GT \
        --input_lr_image src/datasets/for_generate_dataset/outputs/${degradation}/test_set/LR \
        --pretrained_path experiments/HC18/${EXP}/${exp_id}/train/checkpoints/model_${model_steps}.pkl \
        --output_dir experiments/HC18/${EXP}/${exp_id}/test \
        --wandb_project_name "${PROJECT_NAME}" \
        --wandb_run_name "${EXP}_${degradation}_enable_gan_loss${gan_loss} (test)"
  done
done