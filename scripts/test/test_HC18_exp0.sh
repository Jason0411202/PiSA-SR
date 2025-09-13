# output_dir 統一為 experiments/資料集名稱/第幾個實驗/本實驗中的第幾個 run/{train or test 結果}

# 載入 exp0 train 的 pretrained weight, 使用預設參數進行 inference
# 最基本的 baseline

CUDA_VISIBLE_DEVICES=4 \
python test_pisasr.py \
  --pretrained_path experiments/HC18/exp0/1/train/checkpoints/model_1001.pkl \
  --output_dir experiments/HC18/exp0/1/test \
  --wandb_project_name "pisasr-exp0" \
  --wandb_run_name "exp0 (test)"