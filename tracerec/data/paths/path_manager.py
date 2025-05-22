"""
Path manager for interaction paths in recommendation systems.
Paths represent sequences of interactions between users and items.
"""


class PathManager:
    """
    Class to manage interaction paths in recommendation systems.
    A path is an ordered sequence of elements a user has interacted with.
    """

    def __init__(self, paths=None):
        """
        Initializes the path manager.

        Args:
            paths (dict): Initial paths dictionary, if any.
                          The key is the user ID and the value is a list of elements the user has interacted with.
        """
        self.paths = paths if paths is not None else {}
        self.entities = set()

        if self.paths:
            self._extract_entities()

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
