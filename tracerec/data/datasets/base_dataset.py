"""
Base Dataset for implementing PyTorch datasets for recommendation systems.
"""

from torch.utils.data import Dataset


class BaseRecDataset(Dataset):
    """
    Base class for recommendation datasets based on PyTorch Dataset.
    """

    def __init__(self, data=None):
        """
        Initializes the dataset.

        Args:
            data: Data to use in the dataset
        """
        self.data = data

    def __len__(self):
        """
        Returns the size of the dataset.

        Returns:
            int: Number of elements in the dataset
        """
        if self.data is None:
            return 0
        return len(self.data)

    def __getitem__(self, idx):
        """
        Gets an element from the dataset.

        Args:
            idx (int): Index of the element to get

        Returns:
            The transformed element if there is a defined transformation
        """
        if self.data is None:
            raise IndexError("Empty dataset")

        item = self.data[idx]

        return item
