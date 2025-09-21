# 實驗 6
# 首先使用作者的預設參數 (但不使用 residual) 在 HC18 資料集上進行 finetune。測試時，嘗試不同的 test 參數 ( lambda_pix 及 lambda_sem 的比例)，來驗證哪種組合最能衝高 PSNR 跟 SSIM


# 1. lambda_pix=0.0, lambda_sem=1.0
# 2. lambda_pix=0.2, lambda_sem=1.0
# 3. lambda_pix=0.5, lambda_sem=1.0
# 4. lambda_pix=0.8, lambda_sem=1.0
# 5. lambda_pix=1.0, lambda_sem=1.0
# 6. lambda_pix=1.0, lambda_sem=0.0
# 7. lambda_pix=1.0, lambda_sem=0.2
# 8. lambda_pix=1.0, lambda_sem=0.5
# 9. lambda_pix=1.0, lambda_sem=0.8

TRAIN_DEVICES="2,3"
TEST_DEVICE="2"

CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} accelerate launch train_pisasr.py \
    --train_folder="../HC18/training_set" \
    --output_dir="experiments/HC18/exp6/1/train" \
    --wandb_project_name "pisasr-exp6" \
    --wandb_run_name "exp6 (train)" \
    --use_residual_in_training False

# 實驗 6 - 測試不同 lambda 組合

# 1. lambda_pix=0.0, lambda_sem=1.0
CUDA_VISIBLE_DEVICES=${TEST_DEVICE} \
python test_pisasr.py \
  --pretrained_path experiments/HC18/exp6/1/train/checkpoints/model_1001.pkl \
  --output_dir experiments/HC18/exp6/1/test_lambda_0.0_1.0 \
  --wandb_project_name "pisasr-exp6" \
  --wandb_run_name "exp6 (lambda_pix=0.0, lambda_sem=1.0)" \
  --lambda_pix 0.0 \
  --lambda_sem 1.0 \
  --use_residual_in_training False

# 2. lambda_pix=0.2, lambda_sem=1.0
CUDA_VISIBLE_DEVICES=${TEST_DEVICE} \
python test_pisasr.py \
  --pretrained_path experiments/HC18/exp6/1/train/checkpoints/model_1001.pkl \
  --output_dir experiments/HC18/exp6/1/test_lambda_0.2_1.0 \
  --wandb_project_name "pisasr-exp6" \
  --wandb_run_name "exp6 (lambda_pix=0.2, lambda_sem=1.0)" \
  --lambda_pix 0.2 \
  --lambda_sem 1.0 \
  --use_residual_in_training False

# 3. lambda_pix=0.5, lambda_sem=1.0
CUDA_VISIBLE_DEVICES=${TEST_DEVICE} \
python test_pisasr.py \
  --pretrained_path experiments/HC18/exp6/1/train/checkpoints/model_1001.pkl \
  --output_dir experiments/HC18/exp6/1/test_lambda_0.5_1.0 \
  --wandb_project_name "pisasr-exp6" \
  --wandb_run_name "exp6 (lambda_pix=0.5, lambda_sem=1.0)" \
  --lambda_pix 0.5 \
  --lambda_sem 1.0 \
  --use_residual_in_training False

# 4. lambda_pix=0.8, lambda_sem=1.0
CUDA_VISIBLE_DEVICES=${TEST_DEVICE} \
python test_pisasr.py \
  --pretrained_path experiments/HC18/exp6/1/train/checkpoints/model_1001.pkl \
  --output_dir experiments/HC18/exp6/1/test_lambda_0.8_1.0 \
  --wandb_project_name "pisasr-exp6" \
  --wandb_run_name "exp6 (lambda_pix=0.8, lambda_sem=1.0)" \
  --lambda_pix 0.8 \
  --lambda_sem 1.0 \
  --use_residual_in_training False

# 5. lambda_pix=1.0, lambda_sem=1.0 (對照組)
CUDA_VISIBLE_DEVICES=${TEST_DEVICE} \
python test_pisasr.py \
  --pretrained_path experiments/HC18/exp6/1/train/checkpoints/model_1001.pkl \
  --output_dir experiments/HC18/exp6/1/test_lambda_1.0_1.0 \
  --wandb_project_name "pisasr-exp6" \
  --wandb_run_name "exp6 (lambda_pix=1.0, lambda_sem=1.0)" \
  --lambda_pix 1.0 \
  --lambda_sem 1.0 \
  --use_residual_in_training False

# 6. lambda_pix=1.0, lambda_sem=0.0
CUDA_VISIBLE_DEVICES=${TEST_DEVICE} \
python test_pisasr.py \
  --pretrained_path experiments/HC18/exp6/1/train/checkpoints/model_1001.pkl \
  --output_dir experiments/HC18/exp6/1/test_lambda_1.0_0.0 \
  --wandb_project_name "pisasr-exp6" \
  --wandb_run_name "exp6 (lambda_pix=1.0, lambda_sem=0.0)" \
  --lambda_pix 1.0 \
  --lambda_sem 0.0 \
  --use_residual_in_training False

# 7. lambda_pix=1.0, lambda_sem=0.2
CUDA_VISIBLE_DEVICES=${TEST_DEVICE} \
python test_pisasr.py \
  --pretrained_path experiments/HC18/exp6/1/train/checkpoints/model_1001.pkl \
  --output_dir experiments/HC18/exp6/1/test_lambda_1.0_0.2 \
  --wandb_project_name "pisasr-exp6" \
  --wandb_run_name "exp6 (lambda_pix=1.0, lambda_sem=0.2)" \
  --lambda_pix 1.0 \
  --lambda_sem 0.2 \
  --use_residual_in_training False

# 8. lambda_pix=1.0, lambda_sem=0.5
CUDA_VISIBLE_DEVICES=${TEST_DEVICE} \
python test_pisasr.py \
  --pretrained_path experiments/HC18/exp6/1/train/checkpoints/model_1001.pkl \
  --output_dir experiments/HC18/exp6/1/test_lambda_1.0_0.5 \
  --wandb_project_name "pisasr-exp6" \
  --wandb_run_name "exp6 (lambda_pix=1.0, lambda_sem=0.5)" \
  --lambda_pix 1.0 \
  --lambda_sem 0.5 \
  --use_residual_in_training False

# 9. lambda_pix=1.0, lambda_sem=0.8
CUDA_VISIBLE_DEVICES=${TEST_DEVICE} \
python test_pisasr.py \
  --pretrained_path experiments/HC18/exp6/1/train/checkpoints/model_1001.pkl \
  --output_dir experiments/HC18/exp6/1/test_lambda_1.0_0.8 \
  --wandb_project_name "pisasr-exp6" \
  --wandb_run_name "exp6 (lambda_pix=1.0, lambda_sem=0.8)" \
  --lambda_pix 1.0 \
  --lambda_sem 0.8 \
  --use_residual_in_training False