import math
import torch
from typing import Union, Tuple

def calculate_psnr(img_pred: torch.Tensor, img_gt: torch.Tensor) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
        img_pred (torch.Tensor): Predicted image tensor
        img_gt (torch.Tensor): Ground truth image tensor
        
    Returns:
        float: PSNR value in dB
        
    Note:
        - Images should be in range [0, 1]
        - Higher PSNR indicates better quality
        - Typical values are between 20 and 40 dB
    """
    mse = torch.mean((img_pred - img_gt) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = torch.max(img_gt)
    psnr = 20 * math.log10(max_pixel) - 10 * math.log10(mse)
    return psnr

def calculate_batch_psnr(img_pred: torch.Tensor, img_gt: torch.Tensor) -> Tuple[float, float]:
    """Calculate PSNR for a batch of images.
    
    Args:
        img_pred (torch.Tensor): Predicted batch of images
        img_gt (torch.Tensor): Ground truth batch of images
        
    Returns:
        Tuple[float, float]: (mean PSNR, std PSNR) in dB
    """
    batch_size = img_pred.size(0)
    psnr_values = torch.zeros(batch_size)
    
    for i in range(batch_size):
        psnr_values[i] = calculate_psnr(img_pred[i], img_gt[i])
    
    return psnr_values.mean().item(), psnr_values.std().item() 