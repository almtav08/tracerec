import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tracerec.algorithms.embedder import Embedder
from tracerec.data.datasets.triples_dataset import TriplesDataset
from tracerec.utils.collates import pos_neg_triple_collate


class GraphEmbedder(Embedder):
    """
    A class to embed knowledge into a vector space.
    This class is designed to handle knowledge graphs and their embeddings.
    It inherits from the Embedder class and implements methods for fitting and transforming data.
    """

    def __init__(
        self,
        num_entities,
        num_relations,
        embedding_dim,
        device,
        norm,
    ):
        """
        Initializes the KnowledgeEmbedder class.
        This class serves as a base for all knowledge-based embedding models.
        """
        super(GraphEmbedder, self).__init__()

        # Initialize parameters
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.device = device
        self.norm = norm
        self.last_loss = None
        self.history = {"train_loss": [], "epoch_time": [], "train_metric": {}}

        # Initialize entity and relation embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        # Initialize embeddings
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)

        # Normalize the embeddings
        self.entity_embeddings.weight.data = F.normalize(
            self.entity_embeddings.weight.data, p=self.norm, dim=1
        )

    def to_device(self, device="cpu"):
        """Move the model to the specified device."""
        self.entity_embeddings = self.entity_embeddings.to(self.device)
        self.relation_embeddings = self.relation_embeddings.to(self.device)
        if hasattr(self, "criterion") and hasattr(self.criterion, "to"):
            self.criterion = self.criterion.to(self.device)
        super().to_device(device)

    def forward(self, triples):
        """
        Forward pass to compute embeddings for the given triples.
        Args:
            triples: Input triples (as a PyTorch Tensor).
        Returns:
            Embeddings for the input triples.
        """
        raise NotImplementedError(
            "The forward method must be implemented by subclasses."
        )

    def fit(
        self,
        data,
        data_neg,
        num_epochs=100,
        batch_size=128,
        lr=0.001,
        shuffle=False,
        verbose=False,
        checkpoint_path=None,
    ):
        """
        Train the TransE model using the provided triples.

        Args:
            data: Training data containing positive triples (as a PyTorch Tensor)
            data_neg: Negative triples for training (as a PyTorch Tensor)
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
        dataset = TriplesDataset(data, data_neg)

        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=pos_neg_triple_collate,
        )

        # Training loop
        for epoch in range(num_epochs):
            start_time = time.time()

            # Track total loss for this epoch
            total_loss = 0

            for batch in train_loader:
                # Clear gradients
                self.optimizer.zero_grad()

                pos_triples = batch[
                    :, :3
                ]  # Assuming batch contains positive triples in the first 3 columns
                neg_triples = batch[:, 3:]

                # Forward pass for positive triples
                pos_scores = self.forward(pos_triples)
                # Forward pass for negative triples
                neg_scores = self.forward(neg_triples)

                # Compute loss
                target = torch.tensor([-1], dtype=torch.float, device=self.device)
                loss = self.criterion(pos_scores, neg_scores, target)
                total_loss += loss.item()

                # Backward pass
                loss.backward()

                # Update parameters
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

    def transform(self, data):
        """
        Generate embeddings for the given entities.

        Args:
            data: Entity IDs for which to generate embeddings

        Returns:
            Entity embeddings for each input ID
        """
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.long, device=self.device)

        # Return the actual entity embeddings
        return self.entity_embeddings(data)
    
    def transform_relation(self, data):
        """
        Generate embeddings for the given relations.

        Args:
            data: Relation IDs for which to generate embeddings

        Returns:
            Relation embeddings for each input ID
        """
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.long, device=self.device)

        # Return the actual relation embeddings
        return self.relation_embeddings(data)
