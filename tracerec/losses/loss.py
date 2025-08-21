import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        return NotImplemented
    
    def is_pairwise(self):
        """
        Check if the loss is pairwise.
        Returns:
            bool: True if the loss is pairwise, False otherwise.
        """
        return hasattr(self, "pairwise") and self.pairwise
