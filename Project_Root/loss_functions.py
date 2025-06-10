import torch
import torch.nn as nn
import torch.nn.functional as F

class BreastClipLoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.0, i2i_weight: float = 1.0, t2t_weight: float = 0.5,
                  loss_ratio: float = 1.0, multi_view: bool = False, view_weights: tuple = (0.5, 0.5)):
        super(BreastClipLoss, self).__init__()
        self.label_smoothing = label_smoothing
        self.i2i_weight = i2i_weight
        self.t2t_weight = t2t_weight
        self.loss_ratio = loss_ratio
        self.multi_view = multi_view
        self.view_weights = view_weights
        # shared CE criterion
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        if not self.multi_view:
            logits_per_image, logits_per_text = args
            return self._single_loss(logits_per_image, logits_per_text)

        # multi-view case
        if len(args) != 4:
            raise ValueError("Multi-view loss requires 4 inputs: i2t1, t2i1, i2t2, t2i2")
        w1, w2 = self.view_weights

        # view1 losses
        loss1 = self._single_loss(args[0], args[1])

        # view2 losses
        loss2 = self._single_loss(args[2], args[3])
        
        # weighted combination
        loss = (w1 * loss1 + w2 * loss2) * self.loss_ratio
        return loss

    def _single_loss(self, logits_per_image: torch.Tensor, logits_per_text : torch.Tensor) -> torch.Tensor:

        B = logits_per_image.size(0)
        device = logits_per_image.device
        targets = torch.arange(B, dtype=torch.long, device=device)
        loss_i2t = self.criterion(logits_per_image, targets)
        loss_t2i = self.criterion(logits_per_text,  targets)
        return (self.i2i_weight * loss_i2t + self.t2t_weight * loss_t2i) * self.loss_ratio
