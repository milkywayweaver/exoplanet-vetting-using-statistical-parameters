import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from scipy.stats import skew

class AutoLogTransform(BaseEstimator,TransformerMixin):
    '''
    Automatic log transform based on Feng et al (2016)
    '''
    def __init__(self,beta=None,beta_multiplier=10):
        self.beta = beta
        self.beta_multiplier = beta_multiplier
    
    def fit(self,X:np.ndarray,y=None):
        X = np.asarray(X, dtype=float)
        if self.beta is None:
            self.beta = skew(X,axis=0)*self.beta_multiplier
        self.beta = np.where(self.beta == 0, 1e-5, self.beta)
        
        self.xmin_ = np.min(X,axis=0)
        self.xmax_ = np.max(X,axis=0)
        self.R_ = self.xmax_ - self.xmin_
        return self
    
    def transform(self,X:np.ndarray):
        return self.__autolog(X)

    def __autolog(self,X:np.ndarray):
        X = np.asarray(X, dtype=float)
        for i in range(X.shape[1]):
            if self.beta[i] > 0:
                X[:,i] = np.log(X[:,i] - self.xmin_[i] + self.R_[i]/self.beta[i])
            elif self.beta[i] < 0:
                X[:,i] = -np.log(self.xmax_[i] - X[:,i] - self.R_[i]/self.beta[i])
        return X