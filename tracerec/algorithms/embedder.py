import torch
import torch.nn as nn


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

    def fit(self, data, data_neg, y):
        """
        Trains the model with the provided data using contrastive learning.
        Args:
            data: Positive input data.
            data_neg: Negative data for contrast.
            y: Labels or target values.
        """
        raise NotImplementedError("The fit method must be implemented by subclasses.")

    def transform(self, X):
        """
        Generates embeddings from the input data.
        Args:
            X: Input data.
        Returns:
            Generated embeddings.
        """
        raise NotImplementedError(
            "The transform method must be implemented by subclasses."
        )

    def fit_transform(self, X, y=None, X_neg=None):
        """
        Trains the model and generates embeddings for all entities in a single step.
        
        Args:
            X: Positive input data (e.g., user-item interactions).
            y: Labels or target values (optional).
            X_neg: Negative data for contrast (optional, but recommended for contrastive learning).
            
        Returns:
            Embeddings for all entities in the model.
        """
        self.fit(X, y, X_neg)

        # For knowledge graph embeddings, return embeddings for all entities
        if hasattr(self, 'num_entities'):
            # Create tensor with all entity IDs
            all_entities = torch.arange(self.num_entities, device=getattr(self, 'device', 'cpu'))
            return self.transform(all_entities)
        else:
            # Default behavior for other embedding models
            return self.transform(X)
