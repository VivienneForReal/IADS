# -*- coding: utf-8 -*-
# @Author: H. T. Duong Vu

# Import necessary libraries
import numpy as np

from src.classif.base import Base

class KNN(Base):
    """
    K-Nearest Neighbors (KNN) classification algorithm.
    This class implements the KNN algorithm for classification tasks.
    """

    def __init__(self, input_dimension: int, k: int = 3):
        """
        Initialize the KNN classifier with the input dimension and number of neighbors.

        :param input_dimension: The number of features in the input data.
        :param k: The number of nearest neighbors to consider.
        """
        super().__init__(input_dimension)
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the KNN model to the training data.

        :param X: Training data features.
        :param y: Training data labels.
        """
        self.X_train = X
        self.y_train = y

    def score(self, x):
        """ Rend la proportion des labels parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        nearest_indices = np.argsort(distances)[:self.k]
        nearest_labels = self.y_train[nearest_indices]
        label_counts = np.bincount(nearest_labels)
        return np.argmax(label_counts)

    def predict(self, x):
        """ Rend la prédiction sur x (label de 0 à 9)
            x: une description : un ndarray
        """
        return self.score(x)