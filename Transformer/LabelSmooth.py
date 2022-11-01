import torch
import torch.nn.functional as F
import torch.nn as nn


class LabelSmoothLoss(nn.Module):

    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, target, input):
        log_prob = F.log_softmax(input, dim=-1)
        target_one_hot = F.one_hot(torch.as_tensor(target, dtype=torch.int64), num_classes=11)
        target_one_hot_label_smooth = target_one_hot * (1 - self.smoothing) + (1 - target_one_hot) * self.smoothing / (
                target_one_hot.size(1) - 1)
        loss = (-target_one_hot_label_smooth * log_prob).sum(dim=-1).sum()
        return loss
