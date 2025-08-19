import random
import torch

from tracerec.data.triples.triples_manager import TriplesManager
from tracerec.samplers.negative_triple_sampler import NegativeTripleSampler


class PathBasedNegativeSampler(NegativeTripleSampler):
    """
    A sampler that generates negative samples based on a path-based approach.
    It uses the `NegativeTriplesSampler` as a base class.
    """

    def __init__(
        self,
        triples_manager: TriplesManager,
        corruption_ratio=0,
        device="cpu",
        min_distance=1.0,
    ):
        """
        Initializes the PathBasedNegativeSampler with the given parameters.

        Args:
            all_triples (list): All triples in the dataset
            all_entities (set): All entities in the dataset
            corruption_ration (float): Ratio of head/tail corruption for negative sampling (by default, 0, only tails are corrupted)
            device (str): Device to run the model on ('cpu' or 'cuda')
            entity_paths (dict): Precomputed paths for each entity
            min_distance (float): Minimum distance for path-based sampling (default: 1.0)
        """
        all_triples = triples_manager.get_triples()
        all_entities = triples_manager.get_entities()
        entity_paths = triples_manager.get_entity_paths()

        super().__init__(all_triples, all_entities, corruption_ratio, device)

        self.entity_paths = entity_paths
        self.min_distance = min_distance

    def sample(self, positive_triples, num_samples=1, random_state=None):
        """
        Sample negative triples based on the path-based approach.

        Args:
            positive_triples (list): List of positive triples to sample from
            num_samples (int): Number of negative samples to generate per positive triple
            random_state (int, optional): Random seed for reproducibility
        Returns:
            tensor: Tensor of shape (num_positive_triples, num_samples, 3) containing negative triples
        """
        super().sample(num_samples, random_state)

        num_triples = len(self.all_triples)

        # Create a tensor to store the negative samples
        neg_triples = torch.zeros((num_triples, num_samples, 3), dtype=torch.long)

        for i, triple in enumerate(positive_triples):
            head, relation, tail = triple

            for j in range(num_samples):
                # Decide whether to corrupt head or tail
                corrupt_tail = random.random() > self.corruption_ratio

                if corrupt_tail:
                    source_entity = head
                    entity_to_corrupt = tail
                else:
                    source_entity = tail
                    entity_to_corrupt = head

                # Get paths for this relation
                if (
                    relation in self.entity_paths
                    and source_entity in self.entity_paths[relation]
                ):
                    paths = self.entity_paths[relation][source_entity]
                    neg_candidates = []

                    # Collect entities that are further than min_distance
                    for entity in self.all_entities:
                        # Skip the entity in the triple
                        if entity == entity_to_corrupt:
                            continue

                        # If entity is not in paths or distance > min_distance, it's a good candidate
                        if entity not in paths or paths[entity] > self.min_distance:
                            neg_candidates.append(entity)

                    if neg_candidates:
                        corrupted_entity = random.choice(neg_candidates)
                        if corrupt_tail:
                            neg_triples[i, j] = torch.tensor(
                                [head, relation, corrupted_entity]
                            )
                        else:
                            neg_triples[i, j] = torch.tensor(
                                [corrupted_entity, relation, tail]
                            )
                        continue

                # If we can't find candidates with path-based approach, fallback to random
                if corrupt_tail:
                    corrupted_tail = random.choice(
                        [e for e in self.all_entities if e != tail]
                    )
                    neg_triples[i, j] = torch.tensor([head, relation, corrupted_tail])
                else:
                    corrupted_head = random.choice(
                        [e for e in self.all_entities if e != head]
                    )
                    neg_triples[i, j] = torch.tensor([corrupted_head, relation, tail])

        return neg_triples

    def set_min_distance(self, distance):
        """
        Update the minimum distance for path-based sampling.

        Args:
            distance (int): New minimum distance
        """
        self.min_distance = distance

    def set_entity_paths(self, entity_paths):
        """
        Update the precomputed entity paths.

        Args:
            entity_paths (dict): New precomputed paths for each entity
        """
        self.entity_paths = entity_paths
