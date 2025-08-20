import torch
import torch.nn as nn

from .graph_embedder import GraphEmbedder


class SimplE(GraphEmbedder):
    """
    Implementation of SimplE knowledge graph embedding model.
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
        Initialize the SimplE model with the given parameters.

        Args:
            num_entities: Number of unique entities in the knowledge graph
            num_relations: Number of unique relations in the knowledge graph
            embedding_dim: Dimension of the embedding vectors (default: 100)
            device: Device to run the model on ('cpu' or 'cuda')
            norm: The p-norm to use for distance calculation (default: 1, Manhattan distance)
        """
        super(SimplE, self).__init__(
            num_entities, num_relations, embedding_dim, device, norm
        )

        self.tail_embeddings = nn.Embedding(num_entities, embedding_dim)
        nn.init.xavier_uniform_(self.tail_embeddings.weight.data)

        self.relation_inv_embeddings = nn.Embedding(num_relations, embedding_dim)
        nn.init.xavier_uniform_(self.relation_inv_embeddings.weight.data)

    def to_device(self, device="cpu"):
        self.tail_embeddings = self.tail_embeddings.to(device)
        self.relation_inv_embeddings = self.relation_inv_embeddings.to(device)
        return super().to_device(device)

    def forward(self, triples):
        """
        Forward pass for the SimplE model.

        Args:
            triples: Tensor of shape (batch_size, 3) containing (head, relation, tail) triples

        Returns:
            Tensor of scores for each triple (distance)
        """
        heads = triples[:, 0]
        relations = triples[:, 1]
        tails = triples[:, 2]

        head_h = self.entity_embeddings(heads)
        head_t = self.tail_embeddings(heads)
        tail_h = self.entity_embeddings(tails)
        tail_t = self.tail_embeddings(tails)

        rel = self.relation_embeddings(relations)       
        rel_inv = self.relation_inv_embeddings(relations)

        # Triple products
        score_forward = torch.sum(head_h * rel * tail_t, dim=1)
        score_backward = torch.sum(tail_h * rel_inv * head_t, dim=1)

        scores = -0.5 * (score_forward + score_backward)

        return scores
