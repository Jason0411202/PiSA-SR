import torch, os, glob, pyiqa
from argparse import ArgumentParser
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import wandb


class IQA_Evaluator:
    def __init__(self, hr_dir: str, gt_dir: str, init_wanb: bool = False):
        """
        初始化 IQA evaluator
        Args:
            hr_dir (str): HR 圖片資料夾
            gt_dir (str): GT 圖片資料夾
        """
        if init_wanb == True:
            wandb.init(project="IQA_Evaluator", name="IQA_Evaluator")

        self.hr_dir = hr_dir
        self.gt_dir = gt_dir
        self.device = torch.device("cuda")

        self.psnr = pyiqa.create_metric("psnr", test_y_channel=True, color_space="ycbcr", device=self.device)
        self.ssim = pyiqa.create_metric("ssim", test_y_channel=True, color_space="ycbcr", device=self.device)
        self.lpips = pyiqa.create_metric("lpips", device=self.device)
        self.dists = pyiqa.create_metric("dists", device=self.device)
        self.fid = pyiqa.create_metric("fid", device=self.device)
        self.niqe = pyiqa.create_metric("niqe", device=self.device)
        self.maniqa = pyiqa.create_metric("maniqa-pipal", device=self.device)
        self.clipiqa = pyiqa.create_metric("clipiqa", device=self.device)
        self.musiq = pyiqa.create_metric("musiq", device=self.device)

        self.test_SR_paths = list(sorted(glob.glob(os.path.join(self.gt_dir, "*"))))
        self.test_HR_paths = list(sorted(glob.glob(os.path.join(self.hr_dir, "*"))))

    def evaluate(self):
        """
        評估 IQA 指標，將結果印出並存於 wandb 中
        """

        metrics = {"psnr": [], "ssim": [], "lpips": [], "dists": [], "niqe": [], "maniqa": [], "musiq": [], "clipiqa": []}

        for i, (SR_path, HR_path) in tqdm(enumerate(zip(self.test_SR_paths, self.test_HR_paths))):
            SR = Image.open(SR_path).convert("RGB")
            SR = transforms.ToTensor()(SR).to(self.device).unsqueeze(0)
            HR = Image.open(HR_path).convert("RGB")
            HR = transforms.ToTensor()(HR).to(self.device).unsqueeze(0)
            metrics["psnr"].append(self.psnr(SR, HR).item())
            metrics["ssim"].append(self.ssim(SR, HR).item())
            metrics["lpips"].append(self.lpips(SR, HR).item())
            metrics["dists"].append(self.dists(SR, HR).item())
            metrics["niqe"].append(self.niqe(SR).item())
            metrics["maniqa"].append(self.maniqa(SR).item())
            metrics["clipiqa"].append(self.clipiqa(SR).item())
            metrics["musiq"].append(self.musiq(SR).item())

        for k in metrics.keys():
            metrics[k] = np.mean(metrics[k])

        metrics["fid"] = self.fid(self.gt_dir, self.hr_dir)

        for k, v in metrics.items():
            if k == "niqe":
                print(k, f"{v:.3g}")
                wandb.log({k: v})
            elif k == "fid":
                print(k, f"{v:.5g}")
                wandb.log({k: v})
            else:
                print(k, f"{v:.4g}")
                wandb.log({k: v})