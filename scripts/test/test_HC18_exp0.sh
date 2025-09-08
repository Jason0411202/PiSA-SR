# output_dir 統一為 experiments/資料集名稱/第幾個實驗/本實驗中的第幾個 run/{train or test 結果}

# 載入作者的 pretrained weight, 使用預設參數進行 inference
# 最基本的 baseline

CUDA_VISIBLE_DEVICES=4 \
python test_pisasr.py \
  --pretrained_model_path preset/models/sd-2.1-base \
  --pretrained_path preset/models/pisa_sr.pkl \
  --process_size 64 \
  --upscale 4 \
  --input_image ../HC18/test_set \
  --output_dir experiments/HC18/exp0/1/test \
  --lambda_pix 1.0 \
  --lambda_sem 1.0 \
  --wandb_run_name "exp0 (test)"