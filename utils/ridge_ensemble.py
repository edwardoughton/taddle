'''
Ensembles ridge regression models to do grid-level predictions with equal voting from each model
'''
import numpy as np
class RidgeEnsemble:
    def __init__(self, ridges, scalers):
        assert type(ridges) == list and type(scalers) == list and len(ridges) == len(scalers)
        self.ridges = ridges
        self.scalers = scalers
    
    def predict(self, x):
        predictions = np.zeros(len(x))
        for ridge, scalar in zip(self.ridges, self.scalers):
            feats = scalar.transform(x)
            predictions += ridge.predict(feats)
        predictions /= len(self.ridges)
        return predictions
