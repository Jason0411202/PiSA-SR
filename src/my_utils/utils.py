from pathlib import Path
from typing import List
import torch_fidelity
import glob
import os
import torch

def write_image_paths(folder_path: str, txt_path: str, exts: List[str] = None) -> None:
    """
    將資料夾內所有符合副檔名的影像路徑寫入 txt 檔案。

    Args:
        folder_path (str): 要搜尋的資料夾。
        txt_path (str): 輸出 txt 的路徑。
        exts (List[str], optional): 要搜尋的副檔名清單，預設為 [".png"]。
    """
    folder = Path(folder_path)
    txt_file = Path(txt_path)
    exts = exts or [".png"]

    # 確保輸出資料夾存在
    txt_file.parent.mkdir(parents=True, exist_ok=True)

    # 遞迴搜尋符合副檔名的檔案
    image_files = [
        str(p.resolve())
        for p in folder.rglob("*")
        if p.suffix.lower() in exts and "annotation" not in p.stem.lower() # 特判, 過濾掉 HC18 dataset 中包含 _Annotation 的那些標註檔
    ]

    # 寫入檔案
    with txt_file.open("w", encoding="utf-8") as f:
        f.write("\n".join(image_files))

    print(f"已生成 {txt_file}，共 {len(image_files)} 筆影像路徑")

def compute_fid(real_path, fake_path):
    """
    使用 torch-fidelity 計算 FID
    參數:
      - real_path, fake_path: 兩個圖片資料夾路徑
    回傳:
      - fid (float)
    """
    print("Computing FID...")
    print(f"  real_path: {real_path}")
    print(f"  fake_path: {fake_path}")

    # 確認資料夾有圖
    patterns = ['*.png', '*.jpg', '*.jpeg']
    n_real = sum(len(glob.glob(os.path.join(real_path, p))) for p in patterns)
    n_fake = sum(len(glob.glob(os.path.join(fake_path, p))) for p in patterns)
    if n_real == 0:
        print("Not found any real images in", real_path)
        return None
    if n_fake == 0:
        print("Not found any generated images in", fake_path)
        return None
    print(f"  Found {n_real} real images and {n_fake} generated images.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # torch-fidelity
    if torch_fidelity is not None:
        try:
            print("Trying torch-fidelity.calculate_metrics(...)")
            metrics = torch_fidelity.calculate_metrics(
                input1=real_path,
                input2=fake_path,
                cuda=device,
                isc=False,
                fid=True,
                batch_size=1,
                verbose=True
            )
            fid_val = metrics.get('frechet_inception_distance')
            if fid_val is not None:
                fid_val = float(fid_val)
                print(f"torch-fidelity result: {fid_val}")
                return fid_val
            else:
                print("無法解析 torch_fidelity 的分析結果")
                print("metrics: ", metrics)
        except Exception as e:
            print(f"torch-fidelity failed: {e}")
    else:
        print("torch-fidelity not installed")

    return None


if __name__ == "__main__":
    train_folder = "../../../HC18/training_set"  # 資料集資料夾
    txt_output = "preset/gt_path.txt"                                # 輸出的 txt
    write_image_paths(train_folder, txt_output, exts=[".png", ".jpg"])

    real_path = "../../experiments/HC18/exp0/1/test/GT"
    fake_path = "../../experiments/HC18/exp0/1/test/HR"
    fid_value = compute_fid(real_path, fake_path)
    if fid_value is not None:
        print(f"FID: {fid_value}")
