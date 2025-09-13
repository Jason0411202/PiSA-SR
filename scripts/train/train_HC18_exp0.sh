# 使用 HC18 資料集, 預設參數進行訓練
# training step 寫在 training_utils.py 中

# 注意: pix_steps 有 bug

CUDA_VISIBLE_DEVICES="1,2" accelerate launch train_pisasr.py \
    --train_folder="../HC18/training_set" \
    --output_dir="experiments/HC18/exp0/1/train" \
    --wandb_project_name "pisasr-exp0" \
    --wandb_run_name "exp0 (train)"