# 實驗十一之一
# 測試實驗十一加入 GAN loss 時，不同訓練 steps (1001, 3001, ..., 30001) 的模型權重表現

TRAIN_DEVICES="1,6"
TEST_DEVICE="1"
PROJECT_NAME="pisasr-exp11-1"
EXP="exp11-1"

# degradations (順序固定)
degradations=("bicubic_4x" "gaussian_noise_10_30")
enable_gan_loss=("True" "False")  # 是否使用 GAN loss

for i in "${!degradations[@]}"; do
  for j in "${!enable_gan_loss[@]}"; do
    degradation=${degradations[$i]}
    gan_loss=${enable_gan_loss[$j]}
    exp_id=$((i*2 + j + 1))   # 保證唯一編號

    echo ">>> [Training] $EXP - $degradation (GAN=$gan_loss, exp_id=$exp_id)"
    CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} accelerate launch train_pisasr.py \
        --train_folder="src/datasets/for_generate_dataset/outputs/${degradation}/training_set/GT" \
        --train_folder_lr="src/datasets/for_generate_dataset/outputs/${degradation}/training_set/LR" \
        --output_dir="experiments/HC18/${EXP}/${exp_id}/train" \
        --enable_gan_loss=${gan_loss} \
        --pix_steps 15000 \
        --max_train_steps 31000 \
        --wandb_project_name "${PROJECT_NAME}" \
        --wandb_run_name "${EXP}_${degradation}_enable_gan_loss${gan_loss} (train)"

    echo ">>> [Testing models for different checkpoints]"
    for step in $(seq 1001 1000 30001); do
      model_path="experiments/HC18/${EXP}/${exp_id}/train/checkpoints/model_${step}.pkl"
      if [ -f "$model_path" ]; then
        echo ">>> [Testing] $EXP - $degradation (GAN=$gan_loss, step=$step)"
        CUDA_VISIBLE_DEVICES=${TEST_DEVICE} \
        python test_pisasr.py \
            --input_gt_image src/datasets/for_generate_dataset/outputs/${degradation}/test_set/GT \
            --input_lr_image src/datasets/for_generate_dataset/outputs/${degradation}/test_set/LR \
            --pretrained_path "$model_path" \
            --output_dir "experiments/HC18/${EXP}/${exp_id}/test/step_${step}" \
            --wandb_project_name "${PROJECT_NAME}" \
            --wandb_run_name "${EXP}_${degradation}_enable_gan_loss${gan_loss}_step${step} (test)"
      else
        echo ">>> [Warning] Model file not found: $model_path"
      fi
    done
  done
done
