# -*- coding: utf-8 -*-
# @Author: H. T. Duong Vu

# Import necessary libraries
import numpy as np

from src.classif.base import Base

class Perceptron(Base):
    """
    Perceptron of Rosenblatt classification algorithm.
    """
    def __init__(self, input_dimension: int, num_labels: int, learning_rate: float = 1e-2, init: bool=True):
        """
        Initialize the Perceptron classifier with the input dimension and number of labels.

        :param input_dimension: The number of features in the input data.
        :param num_labels: The number of unique labels in the classification task.
        :param learning_rate: The learning rate for weight updates.
        :param init: Whether to initialize weights randomly or not.
        """
        super().__init__(input_dimension)
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        
        self.w = np.zeros((num_labels, input_dimension))  # Each label gets its own weight vector
        if not init:
            for i in range(num_labels):
                for j in range(input_dimension):
                    self.w[i, j] = ((2*np.random.random()-1) * 0.001)

        self.allw = [self.w.copy()]  # stockage des premiers poids

    def train_step(self, X: np.ndarray, y: np.ndarray):
        """
        Perform a single training step on the provided data.

        :param X: Training data features.
        :param y: Training data labels.
        """
        data = list(zip(X, y))
        np.random.shuffle(data)

        for x_i, y_i in data:
            predict = self.predict(x_i)
            if y_i != predict:
                self.w[y_i] += self.learning_rate * x_i
                self.w[predict] -= self.learning_rate * x_i

            self.allw.append(self.w.copy())
        return self.w
    
    def score(self, x):
        """
        Compute the score for a given input x.

        :param x: Input feature vector.
        :return: The predicted label for the input x.
        """
        scores = np.dot(self.w, x)
        return scores
    
    def predict(self, x):
        """
        Predict the label for a given input x.

        :param x: Input feature vector.
        :return: The predicted label for the input x.
        """
        scores = self.score(x)
        return np.argmax(scores)
    
    def get_allw(self):
        """
        Get all the weights recorded during training.

        :return: A list of weight vectors recorded at each training step.
        """
        return self.allw
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, seuil: float=1e-3):
        """
        Train the Perceptron model on the provided data for a specified number of epochs.

        :param X: Training data features.
        :param y: Training data labels.
        :param epochs: Number of epochs to train the model.
        """
        norm_diff_values = []      
        i = 0

        while i < epochs:
            w_old = self.w.copy()
            self.w = self.train_step(X, y)
            norm_diff = np.linalg.norm(w_old - self.w)
            norm_diff_values.append(norm_diff)
            if norm_diff < seuil:
                break
            i += 1
        return norm_diff_values