import torch
import torch.nn as nn

from .graph_embedder import GraphEmbedder


class ComplEx(GraphEmbedder):
    """
    Implementation of ComplEx knowledge graph embedding model.
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
        Initialize the ComplEx model with the given parameters.

        Args:
            num_entities: Number of unique entities in the knowledge graph
            num_relations: Number of unique relations in the knowledge graph
            embedding_dim: Dimension of the embedding vectors (default: 100)
            device: Device to run the model on ('cpu' or 'cuda')
            norm: The p-norm to use for distance calculation (default: 1, Manhattan distance)
        """
        super(ComplEx, self).__init__(
            num_entities, num_relations, embedding_dim, device, norm
        )

        self.entity_img_embeddings = nn.Embedding(num_entities, embedding_dim)
        nn.init.xavier_uniform_(self.entity_img_embeddings.weight.data)

        self.relation_img_embeddings = nn.Embedding(num_relations, embedding_dim)
        nn.init.xavier_uniform_(self.relation_img_embeddings.weight.data)

    def to_device(self, device="cpu"):
        self.entity_img_embeddings = self.entity_img_embeddings.to(device)
        self.relation_img_embeddings = self.relation_img_embeddings.to(device)
        return super().to_device(device)

    def forward(self, triples):
        """
        Forward pass for the ComplEx model.

        Args:
            triples: Tensor of shape (batch_size, 3) containing (head, relation, tail) triples

        Returns:
            Tensor of scores for each triple (distance)
        """
        heads = triples[:, 0]
        relations = triples[:, 1]
        tails = triples[:, 2]

        head_real = self.entity_embeddings(heads)
        head_imag = self.entity_img_embeddings(heads)
        rel_real = self.relation_embeddings(relations)
        rel_imag = self.relation_img_embeddings(relations)
        tail_real = self.entity_embeddings(tails)
        tail_imag = self.entity_img_embeddings(tails)

        scores = torch.sum(
            head_real * rel_real * tail_real
            + head_real * rel_imag * tail_imag
            + head_imag * rel_real * tail_imag
            - head_imag * rel_imag * tail_real,
            dim=1,
        )

        return -scores
