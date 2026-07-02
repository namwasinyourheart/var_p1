import numpy as np
from skimage.metrics import structural_similarity as ssim_skimage
import lpips as lpips_lib
import torch


def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    img1_f = img1.astype(np.float64)
    img2_f = img2.astype(np.float64)
    return ssim_skimage(img1_f, img2_f, channel_axis=-1, data_range=255)


class LPIPS:
    def __init__(self, device: str = "cuda"):
        self.loss_fn = lpips_lib.LPIPS(net="alex").to(device)
        self.device = device

    def __call__(self, img1: np.ndarray, img2: np.ndarray) -> float:
        t1 = torch.from_numpy(img1).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 127.5 - 1
        t2 = torch.from_numpy(img2).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 127.5 - 1
        with torch.no_grad():
            d = self.loss_fn(t1, t2)
        return d.item()
