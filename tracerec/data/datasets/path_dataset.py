"""
Specific dataset for working with interaction paths in PyTorch.
"""

from tracerec.data.datasets.base_dataset import BaseRecDataset
from tracerec.data.paths.path_manager import PathManager


class PathDataset(BaseRecDataset):
    """
    PyTorch dataset for interaction paths between users and items.
    """

    def __init__(self, paths, grades, masks = None):
        """
        Initializes the paths dataset.

        Args:
            paths (list): List of user paths
            grades (list): List of grades corresponding to the paths
        """
        self.paths = paths
        self.grades = grades
        self.masks = masks
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
        if mask is not None:
            return path, grade, mask
        return path, grade
