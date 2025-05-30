import torch

from .knowledge_embedder import KnowledgeEmbedder


class TransE(KnowledgeEmbedder):
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
            device: Device to run the model on ('cpu' or 'cuda')
            norm: The p-norm to use for distance calculation (default: 1, Manhattan distance)
        """
        super(TransE, self).__init__(num_entities, num_relations, embedding_dim, device, norm)

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
