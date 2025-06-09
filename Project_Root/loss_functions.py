import torch
import torch.nn as nn
import torch.nn.functional as F

class BreastClipLoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.0, i2i_weight: float = 1.0, t2t_weight: float = 0.5, loss_ratio: float = 1.0):
        super(BreastClipLoss, self).__init__()
        self.label_smoothing = label_smoothing
        self.i2i_weight = i2i_weight
        self.t2t_weight = t2t_weight
        self.loss_ratio = loss_ratio
        # CrossEntropyLoss applies LogSoftmax + NLLLoss under the hood
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
        B = logits_per_image.size(0)

        # Build targets [0,1,2,...,B-1]
        device = logits_per_image.device
        targets = torch.arange(B, dtype=torch.long, device=device)

        # Image->Text loss
        loss_i2t = self.criterion(logits_per_image, targets)

        # Text->Image loss
        loss_t2i = self.criterion(logits_per_text, targets)

        # Weighted sum and global scaling
        loss = (self.i2i_weight * loss_i2t + self.t2t_weight * loss_t2i) * self.loss_ratio
        return loss
