# -*- coding: utf-8 -*-
# @Author: H. T. Duong Vu

# Import necessary libraries
import numpy as np
import pandas as pd
from typing import Any

class Base:
    """
    Base class for classification tasks.
    Note: This class is intended to be extended by other classes that implement specific classification algorithms.
    """

    def __init__(self, input_dimension: int):
        """
        Initialize the base class with the input dimension.

        :param input_dimension: The number of features in the input data.
        """
        self.input_dimension = input_dimension

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model to the training data.

        :param X: Training data features.
        :param y: Training data labels.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def score(self, X: np.ndarray) -> Any[float, int]:
        """
        Evaluate the model on the given data.

        :param X: Data features to evaluate.
        :return: The accuracy of the model on the provided data.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the labels for the given data.

        :param X: Data features to predict.
        :return: Predicted labels.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the accuracy of the model on the provided data.

        :param X: Data features.
        :param y: True labels.
        :return: Accuracy of the model.
        """
        pred = [self.predict(X[i]) for i in range(len(X))]

        good_rate = 0
        for i in range(len(X)):
            if pred[i] == y[i]:
                good_rate += 1
        return good_rate / len(X) if len(X) > 0 else 0.0
    
    def __str__(self):
        """
        String representation of the base class.

        :return: A string indicating the class name and input dimension.
        """
        return f"{self.__class__.__name__}(input_dimension={self.input_dimension})"