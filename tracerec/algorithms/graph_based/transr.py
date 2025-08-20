import torch
import torch.nn as nn

from .graph_embedder import GraphEmbedder


class TransR(GraphEmbedder):
    """
    Implementation of TransR knowledge graph embedding model.
    """

    def __init__(
        self,
        num_entities,
        num_relations,
        embedding_dim=100,
        projection_dim=100,
        device="cpu",
        norm=1,
    ):
        """
        Initialize the TransR model with the given parameters.

        Args:
            num_entities: Number of unique entities in the knowledge graph
            num_relations: Number of unique relations in the knowledge graph
            embedding_dim: Dimension of the embedding vectors (default: 100)
            device: Device to run the model on ('cpu' or 'cuda')
            norm: The p-norm to use for distance calculation (default: 1, Manhattan distance)
        """
        super(TransR, self).__init__(
            num_entities, num_relations, embedding_dim, device, norm
        )

        self.projection_dim = projection_dim
        self.proj_relations = nn.Embedding(
            num_relations, embedding_dim * projection_dim
        ).to(self.device)
        nn.init.xavier_uniform_(self.proj_relations.weight.data)

        self.relation_embeddings = nn.Embedding(num_relations, projection_dim)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)

        self.to_device(device)

    def to_device(self, device="cpu"):
        self.proj_relations = self.proj_relations.to(device)
        return super().to_device(device)

    def forward(self, triples):
        """
        Forward pass for the TransR model.

        Args:
            triples: Tensor of shape (batch_size, 3) containing (head, relation, tail) triples

        Returns:
            Tensor of scores for each triple (distance)
        """
        heads = triples[:, 0]
        relations = triples[:, 1]
        tails = triples[:, 2]
        head_embeddings = self.entity_embeddings(heads)
        relation_embeddings = self.relation_embeddings(relations)
        tail_embeddings = self.entity_embeddings(tails)
        projections = self.proj_relations(relations)

        rp = projections.view(-1, self.embedding_dim, self.projection_dim)

        hp = torch.matmul(head_embeddings.unsqueeze(1), rp).squeeze(1)
        tp = torch.matmul(tail_embeddings.unsqueeze(1), rp).squeeze(1)

        scores = torch.norm(hp + relation_embeddings - tp, p=self.norm, dim=1)

        return scores
