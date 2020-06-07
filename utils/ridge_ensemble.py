'''
Ensembles ridge regression models to do grid-level predictions with equal voting from each model
'''
import numpy as np
class RidgeEnsemble:
    def __init__(self, ridges):
        assert type(ridges) == list
        self.ridges = ridges
    
    def predict(self, x):
        predictions = np.zeros(len(x))
        for ridge in self.ridges:
            predictions += ridge.predict(x)
        predictions /= len(self.ridges)
        return predictions
