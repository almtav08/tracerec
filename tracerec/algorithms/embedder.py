class Embedder:
    """
    Base class for all recommendation algorithms that generate embeddings.
    Derived models should implement contrastive learning for training.
    """

    def fit(self, X, y=None, X_neg=None):
        """
        Trains the model with the provided data using contrastive learning.
        Args:
            X: Positive input data (e.g., user-item interactions).
            y: Labels or target values (optional).
            X_neg: Negative data for contrast (optional, but recommended for contrastive learning).
        """
        raise NotImplementedError("The fit method must be implemented by subclasses.")

    def transform(self, X):
        """
        Generates embeddings from the input data.
        Args:
            X: Input data.
        Returns:
            Generated embeddings.
        """
        raise NotImplementedError(
            "The transform method must be implemented by subclasses."
        )

    def fit_transform(self, X, y=None, X_neg=None):
        """
        Trains the model and generates embeddings in a single step using contrastive learning.
        """
        self.fit(X, y, X_neg)
        return self.transform(X)
