# 實驗十三: 實驗加入 degradation condition 後的效果

TRAIN_DEVICES="2,3"
TEST_DEVICE="2"
PROJECT_NAME="pisasr-exp13"
EXP="exp13"

degradations=("complex")
enable_deg_condition=("True" "False")
model_steps="1001"

for degradation in "${degradations[@]}"; do
  for deg_opt in "${enable_deg_condition[@]}"; do
    run_name="${EXP}_${degradation}_deg_opt=${deg_opt}"

    echo ">>> [Training] ${run_name}"
    CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} accelerate launch train_pisasr.py \
        --train_folder="src/datasets/for_generate_dataset/outputs/${degradation}/training_set/GT" \
        --train_folder_lr="src/datasets/for_generate_dataset/outputs/${degradation}/training_set/LR" \
        --output_dir="experiments/HC18/${EXP}/${run_name}/train" \
        --enable_deg_condition "${deg_opt}" \
        --wandb_project_name "${PROJECT_NAME}" \
        --wandb_run_name "${run_name} (train)"

    echo ">>> [Testing] ${run_name}"
    CUDA_VISIBLE_DEVICES=${TEST_DEVICE} \
    python test_pisasr.py \
        --input_gt_image src/datasets/for_generate_dataset/outputs/${degradation}/test_set/GT \
        --input_lr_image src/datasets/for_generate_dataset/outputs/${degradation}/test_set/LR \
        --pretrained_path "experiments/HC18/${EXP}/${run_name}/train/checkpoints/model_${model_steps}.pkl" \
        --output_dir "experiments/HC18/${EXP}/${run_name}/test" \
        --enable_deg_condition "${deg_opt}" \
        --wandb_project_name "${PROJECT_NAME}" \
        --wandb_run_name "${run_name} (test)"

  done
done
