import torch
import torch.nn as nn

from .graph_embedder import GraphEmbedder


class Tucker(GraphEmbedder):
    """
    Implementation of Tucker knowledge graph embedding model.
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
        Initialize the Tucker model with the given parameters.

        Args:
            num_entities: Number of unique entities in the knowledge graph
            num_relations: Number of unique relations in the knowledge graph
            embedding_dim: Dimension of the embedding vectors (default: 100)
            device: Device to run the model on ('cpu' or 'cuda')
            norm: The p-norm to use for distance calculation (default: 1, Manhattan distance)
        """
        super(Tucker, self).__init__(
            num_entities, num_relations, embedding_dim, device, norm
        )

        self.core_tensor = nn.Parameter(torch.empty(self.embedding_dim, self.embedding_dim, self.embedding_dim))
        nn.init.xavier_uniform_(self.core_tensor.data)

    def forward(self, triples):
        """
        Forward pass for the Tucker model.

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

        W_mat = torch.einsum("ijk,bj->bik", self.core_tensor, relation_embeddings)

        hr = torch.einsum("bi,bik->bk", head_embeddings, W_mat)

        scores = -torch.sum(hr * tail_embeddings, dim=1)
        
        return scores
