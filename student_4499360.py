import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets, ensemble, metrics, svm, model_selection, linear_model

def training_test_split(X, y, test_size=0.3, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    X_train = X[train_indices]
    X_test = X[test_indices]

    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test

def true_positives(true_labels, predicted_labels, positive_class):
    pos_true = true_labels == positive_class
    pos_predicted = predicted_labels == positive_class
    match = pos_true & pos_predicted
    return np.sum(match)

def false_positives(true_labels, predicted_labels, positive_class):
    pos_predicted = predicted_labels == positive_class
    neg_true = true_labels != positive_class
    match = pos_predicted & neg_true
    return np.sum(match)

def true_negatives(true_labels, predicted_labels, positive_class):
    neg_true = true_labels != positive_class
    neg_pred = predicted_labels != positive_class
    match = neg_true & neg_pred
    return np.sum(match)

def false_negatives(true_labels, predicted_labels, positive_class):
    pos_true = true_labels == positive_class
    neg_pred = predicted_labels != positive_class
    match = pos_true & neg_pred
    return np.sum(match)

def precision(true_labels, predicted_labels, positive_class):
    TP = true_positives(true_labels, predicted_labels, positive_class)
    FP = false_positives(true_labels, predicted_labels, positive_class)
    return TP / (TP + FP)

def recall(true_labels, predicted_labels, positive_class):
    TP = true_positives(true_labels, predicted_labels, positive_class)
    FN = false_negatives(true_labels, predicted_labels, positive_class)
    return TP / (TP + FN)

def accuracy(true_labels, predicted_labels, positive_class):
    TP = true_positives(true_labels, predicted_labels, positive_class)
    TN = true_negatives(true_labels, predicted_labels, positive_class)
    FP = false_positives(true_labels, predicted_labels, positive_class)
    FN = false_negatives(true_labels, predicted_labels, positive_class)
    return (TP + TN) / (TP + TN + FP + FN)

def specificity(true_labels, predicted_labels, positive_class):
    TN = true_negatives(true_labels, predicted_labels, positive_class)
    FP = false_positives(true_labels, predicted_labels, positive_class)
    return TN / (TN + FP)

def balanced_accuracy(true_labels, predicted_labels, positive_class):
    rec = recall(true_labels, predicted_labels, positive_class)
    spec = specificity(true_labels, predicted_labels, positive_class)
    return (rec + spec) / 2

def F1(true_labels, predicted_labels, positive_class):
    prec = precision(true_labels, predicted_labels, positive_class)
    rec = recall(true_labels, predicted_labels, positive_class)
    return 2 * (prec * rec) / (prec + rec)