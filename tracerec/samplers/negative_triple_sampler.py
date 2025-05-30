"""
Negative triple sampler for knowledge graph embedding models.
This module provides sampling strategies for generating negative triples.
"""

import random


class NegativeTripleSampler:
    """
    Class for sampling negative triples from a knowledge graph.
    Negative triples are used in contrastive learning for training
    knowledge graph embedding models.
    """
    def __init__(self, all_triples, all_entities, corruption_ration, device):
        """
        Initialize the negative triple sampler.
        
        Args:
            all_triples (list): List of all triples in the dataset
            all_entities (set): Set of all entities in the dataset
            corruption_ration (float): Ratio of head/tail corruption for negative sampling (by default, 0, only tails are corrupted)
            device (str): Device to run the sampler on ('cpu' or 'cuda')
        """
        self.all_triples = all_triples
        self.all_entities = all_entities
        self.corruption_ratio = corruption_ration
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

    def set_corruption_ratio(self, ratio):
        """
        Update the head/tail corruption ratio.
        
        Args:
            ratio (float): New probability of corrupting tail (vs. head)
        """
        self.sampling_ratio = ratio
