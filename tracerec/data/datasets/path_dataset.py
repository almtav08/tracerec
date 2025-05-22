"""
Specific dataset for working with interaction paths in PyTorch.
"""

from tracerec.data.datasets.base_dataset import BaseRecDataset
from tracerec.data.paths.path_manager import PathManager


class PathDataset(BaseRecDataset):
    """
    PyTorch dataset for interaction paths between users and items.
    """

    def __init__(self, path_manager=None,):
        """
        Initializes the paths dataset.

        Args:
            path_manager (PathManager): Path manager
        """
        self.path_manager = path_manager if path_manager is not None else PathManager()
        # Convert paths to a list to be compatible with Dataset
        paths_list = [
            (user, interactions)
            for user, interactions in self.path_manager.paths.items()
        ]
        super().__init__(data=paths_list)

    def add_interaction(self, user_id, item):
        """
        Adds an interaction to a user's path.

        Args:
            user_id: User identifier
            item: The item the user has interacted with
        """
        self.path_manager.add_interaction(user_id, item)
        # Update the data list
        self.data = [
            (user, interactions)
            for user, interactions in self.path_manager.paths.items()
        ]

    def get_user_path(self, user_id):
        """
        Gets the complete path of a user.

        Args:
            user_id: User identifier

        Returns:
            list: List of items the user has interacted with, or None if the user doesn't exist
        """
        return self.path_manager.get_user_path(user_id)

    def get_users(self):
        """
        Gets the list of users with paths.

        Returns:
            list: List of user identifiers
        """
        return self.path_manager.get_users()

    def get_entity_count(self):
        """
        Gets the number of unique entities.

        Returns:
            int: Number of entities
        """
        return self.path_manager.get_entity_count()
