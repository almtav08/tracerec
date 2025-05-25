"""
Specific dataset for working with triples in PyTorch.
"""
import torch

from tracerec.data.datasets.base_dataset import BaseRecDataset


class TriplesDataset(BaseRecDataset):
    """
    PyTorch dataset for data triples (subject, relation, object).
    """

    def __init__(self, positive_triples, negative_triples=None):
        """
        Initializes the triples dataset.

        Args:
            positive_triples (list): List of positive triples in the form (subject, relation, object)
            negative_triples (list, optional): List of negative triples in the same form.
        """
        self.positive_triples = positive_triples
        self.negative_triples = negative_triples
        super().__init__(data=self.positive_triples)

    def __getitem__(self, idx):
        """
        Gets an element from the dataset.

        Args:
            idx (int): Index of the element to get

        Returns:
            tuple: (positive_triple, negative_triple(s)) if negative triples are available,
                   otherwise returns only the positive triple
        """
        if self.negative_triples is not None:
            return self.positive_triples[idx], self.negative_triples[idx]
        return self.positive_triples[idx]
