"""
Triple manager for recommendation systems.
Triples are representations of (subject, relation, object).
"""


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

    def _extract_entities_and_relations(self):
        """
        Extracts entities and relations from triples.
        """
        for s, r, o in self.triples:
            self.entities.add(s)
            self.entities.add(o)
            self.relations.add(r)

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
