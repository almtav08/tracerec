"""
Specific dataset for working with triples in PyTorch.
"""

from tracerec.data.datasets.base_dataset import BaseRecDataset
from tracerec.data.triples.triples_manager import TriplesManager


class TriplesDataset(BaseRecDataset):
    """
    PyTorch dataset for data triples (subject, relation, object).
    """

    def __init__(self, triples_manager=None):
        """
        Initializes the triples dataset.

        Args:
            triples_manager (TriplesManager): Triple manager
        """
        self.triples_manager = (
            triples_manager if triples_manager is not None else TriplesManager()
        )
        super().__init__(data=self.triples_manager.triples)

    def add_triple(self, subject, relation, object_):
        """
        Adds a triple to the dataset.

        Args:
            subject: Subject of the triple
            relation: Relation of the triple
            object_: Object of the triple
        """
        self.triples_manager.add_triple(subject, relation, object_)
        self.data = self.triples_manager.triples

    def get_entity_count(self):
        """
        Gets the number of unique entities.

        Returns:
            int: Number of entities
        """
        return self.triples_manager.get_entity_count()

    def get_relation_count(self):
        """
        Gets the number of unique relations.

        Returns:
            int: Number of relations
        """
        return self.triples_manager.get_relation_count()
