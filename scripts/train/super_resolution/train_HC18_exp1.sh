# 實驗 1
# 訓練時，確定是否要保留 semantic LoRA

TRAIN_DEVICES="3,4"
TEST_DEVICE="3"

# 不訓練 semantic LoRA, 也就是 pix_steps 超過預測的總訓練步數 (1000 步)
# Training
CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} accelerate launch train_pisasr.py \
    --train_folder="../HC18/training_set" \
    --output_dir="experiments/HC18/exp1/1/train" \
    --wandb_project_name "pisasr-exp1" \
    --wandb_run_name "exp1_no_sem (train)" \
    --pix_steps 5000

# Testing
CUDA_VISIBLE_DEVICES=${TEST_DEVICE} \
python test_pisasr.py \
    --pretrained_path experiments/HC18/exp1/1/train/checkpoints/model_1001.pkl \
    --output_dir experiments/HC18/exp1/1/test \
    --wandb_project_name "pisasr-exp1" \
    --wandb_run_name "exp1_no_sem (test)"

# 訓練 semantic LoRA (對照組)
# Training
CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} accelerate launch train_pisasr.py \
    --train_folder="../HC18/training_set" \
    --output_dir="experiments/HC18/exp1/2/train" \
    --wandb_project_name "pisasr-exp1" \
    --wandb_run_name "exp1_with_sem (train)"

# Testing
CUDA_VISIBLE_DEVICES=${TEST_DEVICE} \
python test_pisasr.py \
    --pretrained_path experiments/HC18/exp1/2/train/checkpoints/model_1001.pkl \
    --output_dir experiments/HC18/exp1/2/test \
    --wandb_project_name "pisasr-exp1" \
    --wandb_run_name "exp1_with_sem (test)"
