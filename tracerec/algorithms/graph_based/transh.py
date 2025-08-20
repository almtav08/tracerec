import torch
import torch.nn as nn

from .graph_embedder import GraphEmbedder


class TransH(GraphEmbedder):
    """
    Implementation of TransH knowledge graph embedding model.
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
        Initialize the TransH model with the given parameters.

        Args:
            num_entities: Number of unique entities in the knowledge graph
            num_relations: Number of unique relations in the knowledge graph
            embedding_dim: Dimension of the embedding vectors (default: 100)
            device: Device to run the model on ('cpu' or 'cuda')
            norm: The p-norm to use for distance calculation (default: 1, Manhattan distance)
        """
        super(TransH, self).__init__(
            num_entities, num_relations, embedding_dim, device, norm
        )

        self.normal_vectors = nn.Embedding(num_relations, embedding_dim)
        nn.init.xavier_uniform_(self.normal_vectors.weight.data)

        self.to_device(device)

    def to_device(self, device="cpu"):
        self.normal_vectors = self.normal_vectors.to(device)
        return super().to_device(device)

    def forward(self, triples):
        """
        Forward pass for the TransH model.

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
        normal_embeddings = self.normal_vectors(relations)

        h_proj = head_embeddings - torch.sum(head_embeddings * normal_embeddings, dim=1, keepdim=True) * normal_embeddings
        t_proj = tail_embeddings - torch.sum(tail_embeddings * normal_embeddings, dim=1, keepdim=True) * normal_embeddings

        scores = torch.norm(h_proj + relation_embeddings - t_proj, p=self.norm, dim=1)

        return scores
