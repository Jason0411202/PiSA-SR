from pathlib import Path
from typing import List


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


if __name__ == "__main__":
    train_folder = "../../../HC18/training_set"  # 資料集資料夾
    txt_output = "preset/gt_path.txt"                                # 輸出的 txt
    write_image_paths(train_folder, txt_output, exts=[".png", ".jpg"])
