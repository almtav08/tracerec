"""
Specific dataset for working with interaction paths in PyTorch.
"""

import random

from tracerec.data.datasets.base_dataset import BaseRecDataset
from tracerec.data.paths.path_manager import PathManager


class PathDataset(BaseRecDataset):
    """
    PyTorch dataset for interaction paths between users and items.
    """

    def __init__(self, paths, grades, masks=None, pairwise=False):
        """
        Initializes the paths dataset.

        Args:
            paths (list): List of user paths
            grades (list): List of grades corresponding to the paths
        """
        self.paths = paths
        self.grades = grades
        self.masks = masks
        self.pairwise = pairwise

        if pairwise:
            self.grade_to_indices = {}
            for i, g in enumerate(grades):
                if g not in self.grade_to_indices:
                    self.grade_to_indices[g] = []
                self.grade_to_indices[g].append(i)

        super().__init__(data=paths)

    def __getitem__(self, idx):
        """
        Gets a single item from the dataset.

        Args:
            idx (int): Index of the item to retrieve

        Returns:
            tuple: (path, grade) for the specified index
        """
        path = self.paths[idx]
        grade = self.grades[idx]
        mask = self.masks[idx] if self.masks is not None else None

        if not self.pairwise:
            return path, grade, mask

        anchor = (path, mask)

        # Positivo: mismo grade, distinto índice
        pos_idx = idx
        while pos_idx == idx:
            pos_idx = random.choice(self.grade_to_indices[grade])
        pos_path = self.paths[pos_idx]
        pos_mask = self.masks[pos_idx] if self.masks is not None else None
        positive = (pos_path, pos_mask)

        # Negativo: distinto grade
        neg_grade = random.choice(
            [g for g in self.grade_to_indices.keys() if g != grade]
        )
        neg_idx = random.choice(self.grade_to_indices[neg_grade])
        neg_path = self.paths[neg_idx]
        neg_mask = self.masks[neg_idx] if self.masks is not None else None
        negative = (neg_path, neg_mask)

        return anchor, positive, negative
