import torch
from ...algorithms.embedder import Embedder
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TransE(Embedder):
    """
    Implementation of TransE knowledge graph embedding model.
    TransE models entities and relations as vectors in the same space,
    with the goal that h + r ≈ t for true triples (h, r, t).
    """
    
    def __init__(self, num_entities, num_relations, embedding_dim=100, criterion=None, device='cpu', norm=1):
        """
        Initialize the TransE model with the given parameters.
        
        Args:
            num_entities: Total number of entities in the knowledge graph
            num_relations: Total number of relation types in the knowledge graph
            embedding_dim: Dimension of the embedding vectors (default: 100)
            criterion: Loss function to use for training (default: None, will use margin ranking loss)
            device: Device to run the model on ('cpu' or 'cuda')
            norm: The p-norm to use for distance calculation (default: 1, Manhattan distance)
        """
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.device = device
        self.norm = norm
        self.last_loss = None
        
        # Initialize entity and relation embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
          # Initialize embeddings
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)
        
        # Normalize the embeddings
        self.entity_embeddings.weight.data = F.normalize(self.entity_embeddings.weight.data, p=self.norm, dim=1)
        
        # Set default criterion if none is provided
        if criterion is None:
            self.criterion = nn.MarginRankingLoss(margin=1.0)
        else:
            self.criterion = criterion
        
        # Move model to the specified device
        self.to_device()
        
    def to_device(self):
        """Move the model to the specified device."""
        self.entity_embeddings = self.entity_embeddings.to(self.device)
        self.relation_embeddings = self.relation_embeddings.to(self.device)
        if hasattr(self, 'criterion') and hasattr(self.criterion, 'to'):
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
        scores = torch.norm(head_embeddings + relation_embeddings - tail_embeddings, p=self.norm, dim=1)
        return scores
    
    def fit(self, X, y=None, X_neg=None):
        """
        Train the TransE model using the provided triples.
        
        Args:
            X: Positive triples in the form of (head, relation, tail)
            y: Ignored, included for compatibility with the Embedder interface
            X_neg: Negative triples for training (corrupted triples)
        
        Returns:
            Self
        """
        if X_neg is None:
            raise ValueError("Negative samples (X_neg) are required for training TransE")
        
        # Convert to tensors if they are not already
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.long, device=self.device)
        if not isinstance(X_neg, torch.Tensor):
            X_neg = torch.tensor(X_neg, dtype=torch.long, device=self.device)
            
        # Calculate scores for positive and negative triples
        pos_scores = self.forward(X)
        neg_scores = self.forward(X_neg)
        
        # Target tensor: positive scores should be lower than negative scores
        target = torch.tensor([-1], dtype=torch.float, device=self.device)
        
        # Calculate loss (margin ranking loss)
        loss = self.criterion(pos_scores, neg_scores, target)
        self.last_loss = loss.item()
        
        # Update the embeddings
        loss.backward()
        
        # Normalize entity embeddings after update
        with torch.no_grad():
            self.entity_embeddings.weight.data = F.normalize(self.entity_embeddings.weight.data, p=self.norm, dim=1)
        
        return self
        
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
