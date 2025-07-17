# -*- coding: utf-8 -*-
# @Author: H. T. Duong Vu

# Import necessary libraries
import numpy as np 
import random 
random.seed(42)  # For reproducibility

def generate_train_test_uniform(X: np.ndarray, y: np.ndarray, n_per_class: int) -> tuple:
    """
    Generate training and testing datasets with a uniform distribution of classes.

    :param X: Input features.
    :param y: Input labels.
    :param n_per_class: Number of samples per class for training.
    :return: Tuple containing training features, training labels, testing features, and testing labels.
    """

    # Check conditions
    if len(X) != len(y):
        raise ValueError("The number of samples in X and y must be the same.")
    if n_per_class <= 0:
        raise ValueError("n_per_class must be a positive integer.")
    if n_per_class > len(y) // len(np.unique(y)):
        raise ValueError("n_per_class is too large for the number of classes in y.")

    unique_classes = np.unique(y)
    train_X, train_y = [], []
    test_X, test_y = [], []

    for cls in unique_classes:
        indices = np.where(y == cls)[0]
        random.shuffle(indices)  # Shuffle indices to ensure randomness
        selected_indices = indices[:n_per_class]
        
        train_X.append(X[selected_indices])
        train_y.append(y[selected_indices])
        
        remaining_indices = indices[n_per_class:]
        test_X.append(X[remaining_indices])
        test_y.append(y[remaining_indices])

    return (np.vstack(train_X), np.concatenate(train_y), 
            np.vstack(test_X), np.concatenate(test_y))

def generate_dataset_uniform(p: int, n: int, binf: int=-1, bsup: int=1) -> tuple:
    """
    Generate a uniform dataset with specified parameters.

    :param p: Number of features.
    :param n: Number of samples.
    :param binf: Lower bound for feature values.
    :param bsup: Upper bound for feature values.
    :return: Tuple containing features and labels.
    """
    
    if p <= 0 or n <= 0:
        raise ValueError("Both p and n must be positive integers.")
    
    X = np.random.uniform(binf, bsup, (n, p))
    y = np.random.randint(0, 2, n)  # Binary labels (0 or 1)
    
    return X, y


def normalization(df):
    """
    Normalize the features in the DataFrame.

    :param df: Input DataFrame.
    :return: Normalized DataFrame.
    """
    return (df - df.mean()) / df.std()

def dist_euclidean(x1, x2):
    """
    Calculate the Euclidean distance between two points.

    :param x1: First point.
    :param x2: Second point.
    :return: Euclidean distance between x1 and x2.
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))

def centroide(data):
    """
    Calculate the centroid of a dataset.

    :param data: Input dataset.
    :return: Centroid of the dataset.
    """
    return np.mean(data, axis=0)

def dist_centroid(group1, group2):
    """
    Calculate the distance from each point in the dataset to the centroid.

    :param data: Input dataset.
    :param centroid: Centroid of the dataset.
    :return: Array of distances from each point to the centroid.
    """
    return dist_euclidean(centroide(group1), centroide(group2))