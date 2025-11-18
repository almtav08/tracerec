import numpy as np
import torch
import torch.nn as nn

from .graph_embedder import GraphEmbedder


class RotatE(GraphEmbedder):
    """
    Implementation of RotatE knowledge graph embedding model.
    """

    def __init__(
        self,
        num_entities,
        num_relations,
        embedding_dim=100,
        gamma=12.0,
        device="cpu",
        norm=1,
    ):
        """
        Initialize the RotatE model with the given parameters.

        Args:
            num_entities: Number of unique entities in the knowledge graph
            num_relations: Number of unique relations in the knowledge graph
            embedding_dim: Dimension of the embedding vectors (default: 100)
            device: Device to run the model on ('cpu' or 'cuda')
            norm: The p-norm to use for distance calculation (default: 1, Manhattan distance)
        """
        super(RotatE, self).__init__(
            num_entities, num_relations, embedding_dim, device, norm
        )

        self.gamma = gamma
        self.epsilon = 2.0
        self.embedding_range = (self.gamma + self.epsilon) / embedding_dim

        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim // 2)
        nn.init.uniform_(
            self.relation_embeddings.weight.data,
            -self.embedding_range,
            self.embedding_range,
        )

    def forward(self, triples):
        """
        Forward pass for the RotatE model.

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

        reh, imh = torch.chunk(head_embeddings, 2, dim=-1)
        ret, imt = torch.chunk(tail_embeddings, 2, dim=-1)

        phase = relation_embeddings / (self.embedding_range / np.pi)
        rcos, rsin = torch.cos(phase), torch.sin(phase)

        reh_rot = reh * rcos - imh * rsin
        imh_rot = reh * rsin + imh * rcos

        scores = torch.norm(reh_rot - ret, p=self.norm, dim=1) + torch.norm(
            imh_rot - imt, p=self.norm, dim=1
        )

        return scores
