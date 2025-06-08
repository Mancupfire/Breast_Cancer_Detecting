import torch
import torch.nn as nn
import torch.nn.functional as F

class BreastClipLoss(nn.Module):
    def __init__(self,
                 label_smoothing: float = 0.0,
                 i2i_weight: float = 1.0,
                 t2t_weight: float = 0.5,
                 loss_ratio: float = 1.0):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.i2i_weight = i2i_weight
        self.t2t_weight = t2t_weight
        self.loss_ratio = loss_ratio

        # Use CrossEntropyLoss with optional label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self,
                logits_per_image: torch.Tensor,
                logits_per_text: torch.Tensor) -> torch.Tensor:

        device = logits_per_image.device
        B = logits_per_image.size(0)
        targets = torch.arange(B, dtype=torch.long, device=device)

        #    over B possible texts; correct index = i.
        loss_i2t = self.criterion(logits_per_image, targets)

        loss_t2i = self.criterion(logits_per_text, targets)

        # Weighted sum
        total_loss = (self.i2i_weight * loss_i2t + self.t2t_weight * loss_t2i) * self.loss_ratio
        return total_loss
