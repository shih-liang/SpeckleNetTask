import torch
import torch.nn as nn
import torch.nn.functional as F

class TVLoss(nn.Module):
    """Total Variation Loss for image smoothing.
    
    This loss encourages spatial smoothness in the image by penalizing
    differences between neighboring pixels.
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate Total Variation Loss.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: TV loss value
        """
        batch_size = x.size()[0]
        h_tv = torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2).sum()
        w_tv = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2).sum()
        
        if self.reduction == 'mean':
            return (h_tv + w_tv) / batch_size
        return h_tv + w_tv

class L1Loss(nn.Module):
    """L1 Loss (Mean Absolute Error) with optional reduction.
    
    This loss measures the absolute difference between the predicted
    and target values.
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.l1_loss = nn.L1Loss(reduction=reduction)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate L1 Loss.
        
        Args:
            pred (torch.Tensor): Predicted tensor
            target (torch.Tensor): Target tensor
            
        Returns:
            torch.Tensor: L1 loss value
        """
        return self.l1_loss(pred, target) 