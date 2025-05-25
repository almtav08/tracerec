import time
import torch

from tracerec.data.datasets.triples_dataset import TriplesDataset
from tracerec.utils.collates import pos_neg_triple_collate

from ...algorithms.embedder import Embedder
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np


class TransE(Embedder):
    """
    Implementation of TransE knowledge graph embedding model.
    TransE models entities and relations as vectors in the same space,
    with the goal that h + r ≈ t for true triples (h, r, t).
    """

    def __init__(
        self,
        num_entities,
        num_relations,
        embedding_dim=100,
        device="cpu",
        norm=1,
    ):
        """
        Initialize the TransE model with the given parameters.

        Args:
            num_entities: Number of unique entities in the knowledge graph
            num_relations: Number of unique relations in the knowledge graph
            embedding_dim: Dimension of the embedding vectors (default: 100)
            criterion: Loss function to use for training (default: None, will use margin ranking loss)
            device: Device to run the model on ('cpu' or 'cuda')
            norm: The p-norm to use for distance calculation (default: 1, Manhattan distance)
        """
        super(TransE, self).__init__()
        
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

        # Move model to the specified device
        self.to_device()

    def to_device(self):
        """Move the model to the specified device."""
        self.entity_embeddings = self.entity_embeddings.to(self.device)
        self.relation_embeddings = self.relation_embeddings.to(self.device)
        if hasattr(self, "criterion") and hasattr(self.criterion, "to"):
            self.criterion = self.criterion.to(self.device)

    def forward(self, triples):
        """
        Forward pass for the TransE model.

        Args:
            triples: Tensor of shape (batch_size, 3) containing (head, relation, tail) triples

        Returns:
            Tensor of scores for each triple
        """
        heads = triples[:, 0]
        relations = triples[:, 1]
        tails = triples[:, 2]
        head_embeddings = self.entity_embeddings(heads)
        relation_embeddings = self.relation_embeddings(relations)
        tail_embeddings = self.entity_embeddings(tails)

        # TransE score: || h + r - t ||
        scores = torch.norm(
            head_embeddings + relation_embeddings - tail_embeddings, p=self.norm, dim=1
        )
        return scores

    def compile(
        self,
        optimizer=None,
        criterion=None,
        metrics=["loss"],
    ):
        """
        Compile the model with the specified optimizer, criterion, and metrics.
        Args:
            optimizer: Optimizer to use for training (default: None, will use the one provided in __init__)
            criterion: Loss function to use for training (default: None, will use margin ranking loss)
            metrics: List of metrics to monitor during training (default: ['loss'])
        """
        if optimizer is not None:
            self.optimizer = optimizer

        if criterion is not None:
            self.criterion = criterion
        else:
            # Default to margin ranking loss
            self.criterion = nn.MarginRankingLoss(margin=1.0)

        # Move criterion to the specified device
        self.criterion = self.criterion.to(self.device)

        # Set metrics
        self.metrics = metrics

        return self

    def fit(
        self,
        data,
        data_neg,
        y,
        val_data=None,
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
            y: Ground truth labels for the training data (as a PyTorch Tensor)
            val_data: Validation data for monitoring performance (optional)
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
        best_model_state = None

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

                pos_triples = batch[:, :3]  # Assuming batch contains positive triples in the first 3 columns
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
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Time: {time.time() - start_time:.2f}s")

            # Track training metrics
            self.history["train_loss"].append(avg_loss)
            self.history["epoch_time"].append(time.time() - start_time)
            self.history["train_metric"]["loss"] = avg_loss

            # If checkpointing is enabled, save the model if it improves
            if checkpoint_path and -avg_loss > best_train_metric:
                best_train_metric = avg_loss
                best_model_state = self.state_dict()
                torch.save(self, checkpoint_path)

        return self

    def _get_ground_truth(self, data):
        """
        Get ground truth for the given data.

        Args:
            data: Training data containing positive triples

        Returns:
            Ground truth for the training data
        """
        # Assuming data is a PyTorch Dataset with triples
        return [triple for triple in data]

    def transform(self, X):
        """
        Generate embeddings for the given entities.

        Args:
            X: Entity IDs for which to generate embeddings

        Returns:
            Entity embeddings for each input ID
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.long, device=self.device)

        # Return the actual entity embeddings
        return self.entity_embeddings(X)
