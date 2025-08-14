import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from cvxopt import matrix, solvers
from itertools import combinations
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt 
from collections import defaultdict
import json


class SVM_QP:
    def __init__(self, kernel='linear', C=1.0, gamma=1.0, degree=3):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.alpha = None
        self.support_vectors_ = None
        self.support_y = None
        self.b = 0

    def kernel_func(self, x1, x2):
        if self.kernel == 'linear':
            return np.dot(x1, x2.T)
        elif self.kernel == 'rbf':
            sq_dists = np.sum(x1**2, axis=1).reshape(-1, 1) + \
                       np.sum(x2**2, axis=1) - 2 * np.dot(x1, x2.T)
            return np.exp(-self.gamma * np.clip(sq_dists, 0, None))
        elif self.kernel == 'poly':
            return (np.dot(x1, x2.T) + 1) ** self.degree
        else:
            raise ValueError("Unsupported kernel")

    def fit(self, X, y):
        n_samples, n_features = X.shape
        K = self.kernel_func(X, X)

        P = matrix(np.outer(y, y) * K)
        q = matrix(-np.ones(n_samples))
        G_std = matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
        h_std = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))
        A = matrix(y.reshape(1, -1).astype(np.double))
        b = matrix(0.0)

        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G_std, h_std, A, b)
        alpha = np.ravel(solution['x'])

        sv = alpha > 1e-3
        self.alpha = alpha[sv]
        self.support_vectors_ = X[sv]
        self.support_y = y[sv]

        self.b = np.mean([
            y_k - np.sum(self.alpha * self.support_y * self.kernel_func(x_k[None, :], self.support_vectors_))
            for x_k, y_k in zip(self.support_vectors_, self.support_y)
        ])

    def project(self, X):
        K = self.kernel_func(X, self.support_vectors_)
        return np.dot(K, self.alpha * self.support_y) + self.b

    def predict(self, X):
        return np.sign(self.project(X))
    
    # ==== Multi-class One-vs-Rest ====

class OneVsRestSVM_QP:
    def __init__(self, kernel='linear', C=1.0, gamma=1.0, degree=3):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.classifiers = {}
        self.classes = []

    def fit(self, X, y):
        self.classes = np.unique(y)
        for cls in self.classes:
            y_binary = np.where(y == cls, 1, -1)
            clf = SVM_QP(kernel=self.kernel, C=self.C, gamma=self.gamma, degree=self.degree)
            clf.fit(X, y_binary)
            self.classifiers[cls] = clf

    def predict(self, X):
        decision_scores = np.zeros((X.shape[0], len(self.classes)))
        for idx, cls in enumerate(self.classes):
            clf = self.classifiers[cls]
            decision_scores[:, idx] = clf.project(X)
        return self.classes[np.argmax(decision_scores, axis=1)]
