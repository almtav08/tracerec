from typing import Union
import torch
import torch.nn as nn

from tracerec.losses.loss import Loss


class Embedder(nn.Module):
    """
    Base class for all recommendation algorithms that generate embeddings.
    Derived models should implement contrastive learning for training.
    """
    def __init__(self):
        """
        Initializes the Embedder class.
        This class serves as a base for all embedding models in the recommendation system.
        """
        super(Embedder, self).__init__()

    def to_device(self, device='cpu'):
        """
        Moves the model to the specified device (CPU or GPU).
        Args:
            device: Device to move the model to ('cpu' or 'cuda').
        """
        self.device = device
        self.to(device)

    def compile(
        self,
        optimizer,
        criterion: Union[Loss, nn.Module],
    ):
        """
        Compile the model with the specified optimizer, criterion, and metrics.
        Args:
            optimizer: Optimizer to use for training
            criterion: Loss function to use for training
            metrics: List of metrics to monitor during training (default: ['loss'])
        """

        # Set optimizer and criterion
        self.optimizer = optimizer
        self.criterion = criterion

        return self
