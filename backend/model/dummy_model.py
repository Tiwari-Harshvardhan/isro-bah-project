import numpy as np


class DummyModel:
    def predict(self, X):
        X = np.asarray(X)
        return np.full(len(X), 34.0)
