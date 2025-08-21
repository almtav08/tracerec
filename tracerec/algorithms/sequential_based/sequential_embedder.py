import time
import torch
from torch.utils.data import DataLoader

from tracerec.algorithms.embedder import Embedder
from tracerec.data.datasets.path_dataset import PathDataset
from tracerec.utils.collates import path_collate


class SequentialEmbedder(Embedder):
    """
    A class to embed a user path into a vector space.
    This class is designed to handle user paths and their embeddings.
    It inherits from the Embedder class and implements methods for fitting and transforming data.
    """

    def __init__(
        self,
        embedding_dim,
        max_seq_length,
        dropout=0.2,
        pooling="last",
        device="cpu",
    ):
        """
        Initializes the CollaborativeEmbedder class.
        This class serves as a base for all collaborative filtering embedding models.
        """
        super(SequentialEmbedder, self).__init__()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.dropout = dropout
        self.device = device
        self.pooling = pooling
        self.last_loss = None
        self.history = {"train_loss": [], "epoch_time": [], "train_metric": {}}

    def to_device(self, device="cpu"):
        """Move the model to the specified device."""
        if hasattr(self, "criterion") and hasattr(self.criterion, "to"):
            self.criterion = self.criterion.to(self.device)
        super().to_device(device)

    def forward(self, input_embs):
        """
        Forward pass to compute embeddings for the given input embeddings.
        Args:
            input_embs: Input embeddings (as a PyTorch Tensor).
        Returns:
            Embeddings for the input embeddings.
        """
        raise NotImplementedError(
            "The forward method must be implemented by subclasses."
        )

    def fit(self,
        data,
        y,
        masks=None,
        num_epochs=100,
        batch_size=128,
        lr=0.001,
        shuffle=False,
        verbose=False,
        checkpoint_path=None,
    ):
        """
        Train the Sequential Embedder model using the provided data.

        Args:
            data: Training data containing paths (as a PyTorch Tensor)
            y: Ground truth labels for the training data (as a PyTorch Tensor)
            masks: Attention masks for the training data (as a PyTorch Tensor)
            num_epochs: Number of epochs to train the model (default: 100)
            batch_size: Batch size for training (default: 128)
            lr: Learning rate for the optimizer (default: 0.001)
            shuffle: Whether to shuffle the training data at each epoch (default: False)
            verbose: Whether to print training progress (default: False)
            checkpoint_path: Path to save model checkpoints (optional)

        Returns:
            Self
        """
        # Setup progress tracking
        best_train_metric = float("-inf")

        # Set model to training mode
        self.train()

        # Set optimizer
        self.optimizer = self.optimizer(self.parameters(), lr=lr)

        # Create DataLoader for batching
        dataset = PathDataset(data, y, masks)

        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=path_collate,
        )

        # Training loop
        for epoch in range(num_epochs):
            start_time = time.time()

            # Track total loss for this epoch
            total_loss = 0

            for paths, grades, masks in train_loader:
                # Clear gradients
                self.optimizer.zero_grad()

                embeddings = self(paths, mask=masks)
                loss = self.criterion(embeddings, grades)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

            # Average loss for the epoch
            avg_loss = total_loss / len(train_loader)
            self.last_loss = avg_loss

            # Print progress
            if verbose:
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Time: {time.time() - start_time:.2f}s"
                )

            # Track training metrics
            self.history["train_loss"].append(avg_loss)
            self.history["epoch_time"].append(time.time() - start_time)
            self.history["train_metric"]["loss"] = avg_loss

            # If checkpointing is enabled, save the model if it improves
            if checkpoint_path and -avg_loss > best_train_metric:
                best_train_metric = avg_loss
                torch.save(self, checkpoint_path)

        return self
