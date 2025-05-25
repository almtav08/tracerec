"""
Negative triple sampler for knowledge graph embedding models.
This module provides sampling strategies for generating negative triples.
"""

import random
import torch

from tracerec.data.triples.triples_manager import TriplesManager


class NegativeTripleSampler:
    """
    Class for sampling negative triples from a knowledge graph.
    Negative triples are used in contrastive learning for training
    knowledge graph embedding models.
    """

    def __init__(self, triples, triples_manager: TriplesManager, strategy='random', min_distance=2, corruption_ratio=0, device='cpu'):
        """
        Initialize the negative triple sampler.
        
        Args:
            triples_manager (list): List of positive triples
            strategy (str): Strategy for negative sampling:
                           'random': Randomly corrupt head or tail entities
                           'path_based': Use path distances to select negative entities
            min_distance (int): Minimum distance to consider for path-based sampling (only used if strategy is 'path_based')
            corruption_ratio (float): Probability of corrupting tail (vs. head) (by default, 0, only tails are corrupted)
            device (str): Device to run the sampler on ('cpu' or 'cuda')
        """
        self.triples = triples
        self.triples_manager = triples_manager
        self.strategy = strategy
        self.min_distance = min_distance
        self.corruption_ratio = corruption_ratio
        self.device = device

    def sample(self, num_samples=1, random_state=None):
        """
        Sample negative triples for the given positive triples.
        If train_ratio and test_ratio are provided,
        the function will return train, test, and validation datasets.
        No validation dataset will be returned if train_ration + test_ratio = 1.

        Args:
            num_samples: Number of negative samples to generate per positive triple
            random_state: Random seed for reproducibility

        Returns:
            Tuple of TripleDataset objects: (train_dataset, test_dataset, val_dataset) or (train_dataset, test_dataset)
        """
        if random_state is not None:
            random.seed(random_state)

        # Use appropriate sampling strategy
        neg_triples = eval(f'self._sample_{self.strategy}')(num_samples)

        return neg_triples

    def _sample_random(self, num_samples=1):
        """
        Sample negative triples by randomly corrupting head or tail entities.
        
        Args:
            num_samples: Number of negative samples per positive triple
            
        Returns:
            torch.Tensor: Tensor of shape (num_triples, num_samples, 3) containing negative triples
        """
        entities_list = list(self.triples_manager.get_entities())
        triples = self.triples
        num_triples = len(triples)
        all_triples = set((h, r, t) for h, r, t in self.triples_manager.get_triples())

        # Create a tensor to store the negative samples
        neg_triples = torch.zeros((num_triples, num_samples, 3), dtype=torch.long)

        for i, triple in enumerate(triples):
            head, relation, tail = triple

            for j in range(num_samples):
                # Decide whether to corrupt head or tail
                corrupt_tail = random.random() > self.corruption_ratio
                valid_triple_found = False

                # Keep trying until we find a valid negative triple
                max_attempts = 100  # Prevent infinite loops
                attempts = 0

                while not valid_triple_found and attempts < max_attempts:
                    if corrupt_tail:
                        # Corrupt tail entity
                        corrupted_tail = random.choice([e for e in entities_list if e != tail])
                        candidate_triple = (head, relation, corrupted_tail)
                        # Check if the triple already exists
                        if candidate_triple not in all_triples:
                            neg_triples[i, j] = torch.tensor([head, relation, corrupted_tail])
                            valid_triple_found = True
                    else:
                        # Corrupt head entity
                        corrupted_head = random.choice([e for e in entities_list if e != head])
                        candidate_triple = (corrupted_head, relation, tail)
                        # Check if the triple already exists
                        if candidate_triple not in all_triples:
                            neg_triples[i, j] = torch.tensor([corrupted_head, relation, tail])
                            valid_triple_found = True
                    attempts += 1

                # If we couldn't find a valid triple after max attempts, just use the last attempt
                if not valid_triple_found:
                    if corrupt_tail:
                        corrupted_tail = random.choice([e for e in entities_list if e != tail])
                        neg_triples[i, j] = torch.tensor([head, relation, corrupted_tail])
                    else:
                        corrupted_head = random.choice([e for e in entities_list if e != head])
                        neg_triples[i, j] = torch.tensor([corrupted_head, relation, tail])

        return neg_triples

    def _sample_path_based(self, num_samples=1):
        """
        Sample negative triples based on path distances between entities.
        Entities that are further away are preferred as negative samples.
        
        Args:
            num_samples: Number of negative samples per positive triple
            
        Returns:
            torch.Tensor: Tensor of shape (num_triples, num_samples, 3) containing negative triples
        """
        entities_list = list(self.triples_manager.get_entities())
        triples = self.triples_manager.get_triples()
        num_triples = len(triples)
        entity_paths = self.triples_manager.get_entity_paths()

        # Create a tensor to store the negative samples
        neg_triples = torch.zeros((num_triples, num_samples, 3), dtype=torch.long)

        for i, triple in enumerate(triples):
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
                if relation in entity_paths and source_entity in entity_paths[relation]:
                    paths = entity_paths[relation][source_entity]
                    neg_candidates = []

                    # Collect entities that are further than min_distance
                    for entity in entities_list:
                        # Skip the entity in the triple
                        if entity == entity_to_corrupt:
                            continue

                        # If entity is not in paths or distance > min_distance, it's a good candidate
                        if entity not in paths or paths[entity] > self.min_distance:
                            neg_candidates.append(entity)

                    if neg_candidates:
                        corrupted_entity = random.choice(neg_candidates)
                        if corrupt_tail:
                            neg_triples[i, j] = torch.tensor([head, relation, corrupted_entity])
                        else:
                            neg_triples[i, j] = torch.tensor([corrupted_entity, relation, tail])
                        continue

                # If we can't find candidates with path-based approach, fallback to random
                if corrupt_tail:
                    corrupted_tail = random.choice([e for e in entities_list if e != tail])
                    neg_triples[i, j] = torch.tensor([head, relation, corrupted_tail])
                else:
                    corrupted_head = random.choice([e for e in entities_list if e != head])
                    neg_triples[i, j] = torch.tensor([corrupted_head, relation, tail])

        return neg_triples

    def set_strategy(self, strategy):
        """
        Update the sampling strategy.
        
        Args:
            strategy (str): New strategy for negative sampling
        """
        self.strategy = strategy

    def set_corruption_ratio(self, ratio):
        """
        Update the head/tail corruption ratio.
        
        Args:
            ratio (float): New probability of corrupting tail (vs. head)
        """
        self.sampling_ratio = ratio

    def set_min_distance(self, distance):
        """
        Update the minimum distance for path-based sampling.
        
        Args:
            distance (int): New minimum distance
        """
        self.min_distance = distance
