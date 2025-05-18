class Embedder:
    """
    Clase base para todos los algoritmos de recomendación que generan embeddings.
    Los modelos derivados deben implementar aprendizaje por contraste para el entrenamiento.
    """

    def fit(self, X, y=None, X_neg=None):
        """
        Entrena el modelo con los datos proporcionados usando aprendizaje por contraste.
        Args:
            X: Datos positivos de entrada (por ejemplo, interacciones usuario-item).
            y: Etiquetas o valores objetivo (opcional).
            X_neg: Datos negativos para contraste (opcional, pero recomendado para aprendizaje por contraste).
        """
        raise NotImplementedError(
            "El método fit debe ser implementado por las subclases."
        )

    def transform(self, X):
        """
        Genera embeddings a partir de los datos de entrada.
        Args:
            X: Datos de entrada.
        Returns:
            Embeddings generados.
        """
        raise NotImplementedError(
            "El método transform debe ser implementado por las subclases."
        )

    def fit_transform(self, X, y=None, X_neg=None):
        """
        Entrena el modelo y genera embeddings en un solo paso usando aprendizaje por contraste.
        """
        self.fit(X, y, X_neg)
        return self.transform(X)
