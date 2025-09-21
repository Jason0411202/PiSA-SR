# 實驗 5
# 使用不同程度的 degradation 設定檔 (degradation_0.yml ~ degradation_3.yml)
# 測試時統一僅用 bicubic downsampling

TRAIN_DEVICES="1,2"
TEST_DEVICE="1"

# 1. deg_file_path="degradation_1.yml" (輕度 degradation)
CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} accelerate launch train_pisasr.py \
    --train_folder="../HC18/training_set" \
    --output_dir="experiments/HC18/exp5/1/train" \
    --wandb_project_name "pisasr-exp5" \
    --wandb_run_name "exp5_deg1 (train)" \
    --deg_file_path="degradation_1.yml"

CUDA_VISIBLE_DEVICES=${TEST_DEVICE} \
python test_pisasr.py \
  --pretrained_path experiments/HC18/exp5/1/train/checkpoints/model_1001.pkl \
  --output_dir experiments/HC18/exp5/1/test \
  --wandb_project_name "pisasr-exp5" \
  --wandb_run_name "exp5_deg1 (test)"

# 2. deg_file_path="degradation_2.yml" (中度 degradation)
CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} accelerate launch train_pisasr.py \
    --train_folder="../HC18/training_set" \
    --output_dir="experiments/HC18/exp5/2/train" \
    --wandb_project_name "pisasr-exp5" \
    --wandb_run_name "exp5_deg2 (train)" \
    --deg_file_path="degradation_2.yml"

CUDA_VISIBLE_DEVICES=${TEST_DEVICE} \
python test_pisasr.py \
  --pretrained_path experiments/HC18/exp5/2/train/checkpoints/model_1001.pkl \
  --output_dir experiments/HC18/exp5/2/test \
  --wandb_project_name "pisasr-exp5" \
  --wandb_run_name "exp5_deg2 (test)"

# 3. deg_file_path="degradation_3.yml" (對照組, 重度 degradation)
CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} accelerate launch train_pisasr.py \
    --train_folder="../HC18/training_set" \
    --output_dir="experiments/HC18/exp5/3/train" \
    --wandb_project_name "pisasr-exp5" \
    --wandb_run_name "exp5_deg3 (train)" \
    --deg_file_path="degradation_3.yml"

CUDA_VISIBLE_DEVICES=${TEST_DEVICE} \
python test_pisasr.py \
  --pretrained_path experiments/HC18/exp5/3/train/checkpoints/model_1001.pkl \
  --output_dir experiments/HC18/exp5/3/test \
  --wandb_project_name "pisasr-exp5" \
  --wandb_run_name "exp5_deg3 (test)"


