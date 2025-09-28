# 實驗十
# output_dir 統一為 experiments/資料集名稱/第幾個實驗/本實驗中的第幾個 run/{train or test 結果}
# 使用 HC18 資料集, 且為了與 SeeSR 做比較, dataset 的 setting 相同
# 共測試 4 種 degradation, 分別為 bicubic_4x, gaussian_noise, gaussian_blur, complex

TRAIN_DEVICES="2,6"
TEST_DEVICE="2"
PROJECT_NAME="pisasr-exp10"
EXP="exp10"

# degradations (順序固定)
degradations=("bicubic_4x" "gaussian_noise" "gaussian_blur" "complex")

for i in "${!degradations[@]}"; do
  degradation=${degradations[$i]}
  exp_id=$((i+1))   # 編號從 1 開始

  echo ">>> [Training] $EXP - $degradation (exp_id=$exp_id)"
  CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} accelerate launch train_pisasr.py \
      --train_folder="src/datasets/for_generate_dataset/outputs/${degradation}/training_set/GT" \
      --train_folder_lr="src/datasets/for_generate_dataset/outputs/${degradation}/training_set/LR" \
      --output_dir="experiments/HC18/${EXP}/${exp_id}/train" \
      --pix_steps 15000 \
      --max_train_steps 31000 \
      --wandb_project_name "${PROJECT_NAME}" \
      --wandb_run_name "${EXP}_${degradation} (train)"

  echo ">>> [Testing] $EXP - $degradation (exp_id=$exp_id)"
  CUDA_VISIBLE_DEVICES=${TEST_DEVICE} \
  python test_pisasr.py \
      --input_gt_image src/datasets/for_generate_dataset/outputs/${degradation}/test_set/GT \
      --input_lr_image src/datasets/for_generate_dataset/outputs/${degradation}/test_set/LR \
      --pretrained_path experiments/HC18/${EXP}/${exp_id}/train/checkpoints/model_30001.pkl \
      --output_dir experiments/HC18/${EXP}/${exp_id}/test \
      --wandb_project_name "${PROJECT_NAME}" \
      --wandb_run_name "${EXP}_${degradation} (test)"
done
