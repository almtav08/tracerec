"""
Triple manager for recommendation systems.
Triples are representations of (subject, relation, object).
"""

import random
import torch
from networkx import DiGraph, all_pairs_shortest_path_length


class TriplesManager:
    """
    Class to manage data triples in recommendation systems.
    Triples can be (subject, relation, object).
    """

    def __init__(self, triples=None):
        """
        Initializes the triple manager.

        Args:
            triples (list): Initial list of triples, if any
        """
        self.triples = triples if triples is not None else []
        self.entities = set()
        self.relations = set()

        if self.triples:
            self._extract_entities_and_relations()

        self.relation_graphs = self._build_relation_graphs()
        self.entity_paths = self._build_entity_paths()

    def _extract_entities_and_relations(self):
        """
        Extracts entities and relations from triples.
        """
        for s, r, o in self.triples:
            self.entities.add(s)
            self.entities.add(o)
            self.relations.add(r)

    def get_triples(self):
        """
        Returns the list of triples.

        Returns:
            list: List of triples
        """
        return self.triples

    def get_entities(self):
        """
        Returns the set of unique entities.

        Returns:
            set: Set of unique entities
        """
        return self.entities

    def add_triple(self, subject, relation, object_):
        """
        Adds a triple to the collection.

        Args:
            subject: Subject of the triple (can be a user)
            relation: Relation of the triple (can be an action or rating)
            object_: Object of the triple (can be an item)
        """
        self.triples.append((subject, relation, object_))
        self.entities.add(subject)
        self.entities.add(object_)
        self.relations.add(relation)

        # Recalculate relation graphs and entity paths
        self.relation_graphs = self._build_relation_graphs()
        self.entity_paths = self._build_entity_paths()

    def get_entity_paths(self):
        """
        Returns the entity paths for each relation.

        Returns:
            dict: Dictionary where keys are relations and values are dictionaries
                  of entity paths with their lengths
        """
        return self.entity_paths

    def filter_by_relation(self, relation):
        """
        Filters triples by a specific relation.

        Args:
            relation: The relation to filter by

        Returns:
            list: List of triples containing that relation
        """
        return [t for t in self.triples if t[1] == relation]

    def filter_by_subject(self, subject):
        """
        Filters triples by a specific subject.

        Args:
            subject: The subject to filter by

        Returns:
            list: List of triples containing that subject
        """
        return [t for t in self.triples if t[0] == subject]

    def filter_by_object(self, object_):
        """
        Filters triples by a specific object.

        Args:
            object_: The object to filter by

        Returns:
            list: List of triples containing that object
        """
        return [t for t in self.triples if t[2] == object_]

    def get_entity_count(self):
        """
        Gets the number of unique entities.

        Returns:
            int: Number of entities
        """
        return len(self.entities)

    def get_relation_count(self):
        """
        Gets the number of unique relations.

        Returns:
            int: Number of relations
        """
        return len(self.relations)

    def __len__(self):
        """
        Returns the number of triples.

        Returns:
            int: Number of triples
        """
        return len(self.triples)

    def _build_relation_graphs(self):
        """
        Build a relation graph from the given triples.

        Args:
            triples (list): List of triples in the form (subject, relation, object).
            relations (set): Set of relations to consider.

        Returns:
            dict: A dictionary where keys are relations and values are lists of tuples
                representing the edges in the relation graph.
        """
        relation_graphs = {relation: DiGraph() for relation in self.relations}

        for subject, relation, object_ in self.triples:
            if relation in relation_graphs:
                relation_graphs[relation].add_edge(subject, object_)

        return relation_graphs

    def _build_entity_paths(self):
        """
        Build entity paths from the given triples.

        Args:
            triples (list): List of triples in the form (subject, relation, object).
            relations (set): Set of relations to consider.

        Returns:
            dict: A dictionary where keys are relations and values are lists of tuples
                representing the paths for each relation.
        """
        entity_paths = {relation: {} for relation in self.relations}

        for relation, graph in self.relation_graphs.items():
            entity_paths[relation] = dict(all_pairs_shortest_path_length(graph))

        return entity_paths

    def _calc_ground_truth(self, subject, relation):
        """
        Calculate the ground truth for a given subject and relation.

        Args:
            subject: The subject to calculate ground truth for
            relation: The relation to calculate ground truth for

        Returns:
            set: Set of objects that are related to the subject by the given relation
        """
        return [o for s, r, o in self.triples if s == subject and r == relation]

    def split(
        self, train_ratio=0.8, relation_ratio=False, random_state=None, device="cpu"
    ):
        """
        Splits the triples into training and testing sets.

        Args:
            train_ratio (float): Proportion of triples to include in the training set
            relation_ratio (bool): If True, ensures that the train/test split maintains the same ratio of relations
            random_state (int, optional): Random seed for reproducibility
            device (str): Device to run the split on ('cpu' or 'cuda')

        Returns:
            tuple (Tensor): (train_triples, train_truth, test_triples, test_truth)
        """
        if random_state is not None:
            random.seed(random_state)
            random.shuffle(self.triples)

        if relation_ratio:
            # Group triples by relation
            relation_groups = {}
            for triple in self.triples:
                relation = triple[1]
                if relation not in relation_groups:
                    relation_groups[relation] = []
                relation_groups[relation].append(triple)

            # Split each group into train and test
            train_triples = []
            test_triples = []
            for group in relation_groups.values():
                train_size = int(len(group) * train_ratio)
                train_triples.extend(group[:train_size])
                test_triples.extend(group[train_size:])
        else:
            # Simple split without relation ratio
            train_size = int(len(self.triples) * train_ratio)
            train_triples = self.triples[:train_size]
            test_triples = self.triples[train_size:]

        train_truth = [self._calc_ground_truth(s, r) for s, r, _ in train_triples]
        test_truth = [self._calc_ground_truth(s, r) for s, r, _ in test_triples]

        # Convert to tensors
        train_triples = torch.tensor(train_triples, dtype=torch.long, device=device)
        test_triples = torch.tensor(test_triples, dtype=torch.long, device=device)
        train_truth = torch.tensor(train_truth, dtype=torch.long, device=device)
        test_truth = torch.tensor(test_truth, dtype=torch.long, device=device)

        return train_triples, train_truth, test_triples, test_truth
