"""
Path manager for interaction paths in recommendation systems.
Paths represent sequences of interactions between users and items.
"""

import torch
import random
from tracerec.algorithms.graph_based.graph_embedder import GraphEmbedder


class PathManager:
    """
    Class to manage interaction paths in recommendation systems.
    A path is an ordered sequence of elements a user has interacted with.
    """

    def __init__(
        self,
        paths=None,
        grades=None,
        max_seq_length=128,
        item_embedder: GraphEmbedder = None,
        att_mask=True,
    ):
        """
        Initializes the path manager.

        Args:
            paths (dict): Initial paths dictionary, if any.
                          The key is the user ID and the value is a list of elements the user has interacted with.
            grades (list): List of grades corresponding to the paths.
            item_embedder (GraphEmbedder): Item embedder for transforming item IDs into embeddings.
        """
        self.paths = paths.copy() if paths is not None else {}
        self.grades = grades.copy() if grades is not None else []
        self.max_seq_length = max_seq_length
        self.item_embedder = item_embedder
        self.att_mask = att_mask
        self.train_indexes = []
        self.test_indexes = []
        if self.att_mask:
            self.attention_mask = {}

        self.entities = set()

        if self.paths:
            if self.max_seq_length > 0:
                self._pad_paths()
            self._extract_entities()

        if self.item_embedder is not None:
            self._embed_paths()

    def _pad_paths(self):
        """
        Pads the paths to the maximum sequence length.
        """
        for user, items in self.paths.items():

            length = len(items)
            mask = torch.zeros(self.max_seq_length, dtype=torch.bool).to(
                self.item_embedder.device
            )

            if length < self.max_seq_length:
                padding = [0] * (self.max_seq_length - length)
                self.paths[user] = items + padding
                mask[length:] = 1
            elif length > self.max_seq_length:
                self.paths[user] = items[-self.max_seq_length :]

            self.attention_mask[user] = mask

    def _embed_paths(self):
        """
        Embeds the interaction paths using the item embedder.
        """
        if self.item_embedder is not None:
            for user, items in self.paths.items():
                item_embeddings = self.item_embedder.transform(items).detach()
                self.paths[user] = item_embeddings

    def _extract_entities(self):
        """
        Extracts entities from the paths.
        """
        for user, items in self.paths.items():
            self.entities.add(user)
            for item in items:
                self.entities.add(item)

    def add_interaction(self, user_id, item):
        """
        Adds an interaction to a user's path.

        Args:
            user_id: User identifier
            item: The element the user has interacted with
        """
        if user_id not in self.paths:
            self.paths[user_id] = []
            self.entities.add(user_id)

        self.paths[user_id].append(item)
        self.entities.add(item)

    def get_user_path(self, user_id):
        """
        Gets the complete path of a user.

        Args:
            user_id: User identifier

        Returns:
            list: List of elements the user has interacted with, or None if the user doesn't exist
        """
        return self.paths.get(user_id)

    def get_user_path_length(self, user_id):
        """
        Gets the length of a user's path.

        Args:
            user_id: User identifier

        Returns:
            int: Number of elements the user has interacted with, or 0 if the user doesn't exist
        """
        if user_id in self.paths:
            return len(self.paths[user_id])
        return 0

    def get_users(self):
        """
        Gets the list of users with paths.

        Returns:
            list: List of user identifiers
        """
        return list(self.paths.keys())

    def get_entity_count(self):
        """
        Gets the number of unique entities (users + elements).

        Returns:
            int: Number of entities
        """
        return len(self.entities)

    def get_items(self):
        """
        Gets the list of unique elements users have interacted with.

        Returns:
            set: Set of element identifiers
        """
        items = set()
        for user_id in self.paths:
            items.update(self.paths[user_id])
        return items

    def __len__(self):
        """
        Returns the number of users with paths.

        Returns:
            int: Number of users
        """
        return len(self.paths)

    def split(
        self, train_ratio=0.8, relation_ratio=False, random_state=None, device="cpu"
    ):
        """
        Splits the paths into training and testing sets.

        Args:
            train_ratio (float): Proportion of paths to include in the training set
            relation_ratio (bool): If True, ensures that the train/test split maintains the same ratio of grades
            random_state (int, optional): Random seed for reproducibility
            device (str): Device to run the split on ('cpu' or 'cuda')

        Returns:
            tuple (Tensor): (train_paths, train_grades, test_paths, test_grades)
        """
        indexes = list(self.paths.keys())
        if random_state is not None:
            random.seed(random_state)
        random.shuffle(indexes)

        if relation_ratio:
            # Group paths by grade
            grade_groups = {}
            for idx in indexes:
                grade = self.grades[idx]
                if grade not in grade_groups:
                    grade_groups[grade] = []
                grade_groups[grade].append(idx)

            # Split each group into train and test
            train_paths = []
            train_grades = []
            test_paths = []
            test_grades = []

            if self.att_mask:
                train_masks = []
                test_masks = []

            for grade, group in grade_groups.items():
                train_size = int(len(group) * train_ratio)
                train_indexes = group[:train_size]
                test_indexes = group[train_size:]
                self.train_indexes.extend(train_indexes)
                self.test_indexes.extend(test_indexes)

                for idx in train_indexes:
                    train_paths.append(self.paths[idx])
                    train_grades.append(self.grades[idx])
                    if self.att_mask:
                        train_masks.append(self.attention_mask[idx])
                for idx in test_indexes:
                    test_paths.append(self.paths[idx])
                    test_grades.append(self.grades[idx])
                    if self.att_mask:
                        test_masks.append(self.attention_mask[idx])

            if self.att_mask:
                return (
                    train_paths,
                    train_grades,
                    train_masks,
                    test_paths,
                    test_grades,
                    test_masks,
                )
            return train_paths, train_grades, test_paths, test_grades

        train_size = int(len(indexes) * train_ratio)
        self.train_indexes = indexes[:train_size]
        self.test_indexes = indexes[train_size:]

        train_paths = [self.paths[k] for k in self.train_indexes]
        train_grades = [self.grades[k] for k in self.train_indexes]
        test_paths = [self.paths[k] for k in self.test_indexes]
        test_grades = [self.grades[k] for k in self.test_indexes]

        if self.att_mask:
            train_masks = [self.attention_mask[k] for k in self.train_indexes]
            test_masks = [self.attention_mask[k] for k in self.test_indexes]
            return (
                train_paths,
                train_grades,
                train_masks,
                test_paths,
                test_grades,
                test_masks,
            )

        return train_paths, train_grades, test_paths, test_grades
