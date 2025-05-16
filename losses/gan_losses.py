import torch
import torch.nn as nn

class GeneratorLoss(nn.Module):
    def __init__(self, alpha=100):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCELoss()
        self.l1 = nn.L1Loss()

    def forward(self, fake, real, fake_pred):
        real_label = torch.ones_like(fake_pred)
        return self.bce(fake_pred, real_label) + self.alpha * self.l1(fake, real)

class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, fake_pred, real_pred):
        fake_label = torch.zeros_like(fake_pred)
        real_label = torch.ones_like(real_pred)
        return (self.bce(fake_pred, fake_label) + self.bce(real_pred, real_label)) / 2
