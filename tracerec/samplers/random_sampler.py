import torch
import random

from tracerec.data.triples.triples_manager import TriplesManager
from tracerec.samplers.negative_triple_sampler import NegativeTripleSampler


class RandomNegativeSampler(NegativeTripleSampler):
    """
    A sampler that generates negative samples based on a path-based approach.
    It uses the `NegativeTriplesSampler` as a base class.
    """

    def __init__(
        self,
        triples_manager: TriplesManager,
        corruption_ratio=0,
        device="cpu",
    ):
        """
        Initializes the PathBasedNegativeSampler with the given parameters.

        Args:
            all_triples (list): All triples in the dataset
            all_entities (set): All entities in the dataset
            corruption_ration (float): Ratio of head/tail corruption for negative sampling (by default, 0, only tails are corrupted)
            device (str): Device to run the model on ('cpu' or 'cuda')
        """
        all_triples = triples_manager.get_triples()
        all_entities = triples_manager.get_entities()
        super().__init__(all_triples, all_entities, corruption_ratio, device)

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
                valid_triple_found = False

                # Keep trying until we find a valid negative triple
                max_attempts = 100  # Prevent infinite loops
                attempts = 0

                while not valid_triple_found and attempts < max_attempts:
                    if corrupt_tail:
                        # Corrupt tail entity
                        corrupted_tail = random.choice(
                            [e for e in self.all_entities if e != tail]
                        )
                        candidate_triple = (head, relation, corrupted_tail)
                        # Check if the triple already exists
                        if candidate_triple not in self.all_triples:
                            neg_triples[i, j] = torch.tensor(
                                [head, relation, corrupted_tail]
                            )
                            valid_triple_found = True
                    else:
                        # Corrupt head entity
                        corrupted_head = random.choice(
                            [e for e in self.all_entities if e != head]
                        )
                        candidate_triple = (corrupted_head, relation, tail)
                        # Check if the triple already exists
                        if candidate_triple not in self.all_triples:
                            neg_triples[i, j] = torch.tensor(
                                [corrupted_head, relation, tail]
                            )
                            valid_triple_found = True
                    attempts += 1

                # If we couldn't find a valid triple after max attempts, just use the last attempt
                if not valid_triple_found:
                    if corrupt_tail:
                        corrupted_tail = random.choice(
                            [e for e in self.all_entities if e != tail]
                        )
                        neg_triples[i, j] = torch.tensor(
                            [head, relation, corrupted_tail]
                        )
                    else:
                        corrupted_head = random.choice(
                            [e for e in self.all_entities if e != head]
                        )
                        neg_triples[i, j] = torch.tensor(
                            [corrupted_head, relation, tail]
                        )

        return neg_triples
