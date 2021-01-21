### IMPORTS ###

import numpy as np
import copy

### CLASSES ###

class EstimationMixin:
    '''Base class for iterative parameter estimation.'''
    
    def __init__(self):
        self.converged = False
        self.estimation = {'scores': []}
        for param, value in self.params.items():
            self.estimation[param] = [value]
        
    @property
    def iteration(self):
        iteration = len(self.estimation['scores']) - 1
        return iteration

    def _initialise_estimation(self, y, X, weights):
        '''Prepares instance for estimation'''
        EstimationMixin.__init__(self)
        self.estimation['scores'] += [self.score(y=y, X=X, weights=weights)]
    
    @staticmethod
    def _prepare_y(y):
        '''Prepares y for estimation.'''
        y = np.array(y)
        return y
    
    @staticmethod
    def _prepare_X(y, X):
        '''Prepares X for estimation.'''
        if X is None:
            pass
        else:
            X = np.array(X)
            assert X.shape[0] == y.shape[0], \
                'X needs to match shape of y'
        return X
    
    @staticmethod
    def _prepare_weights(y, weights):
        '''Prepares the weights for estimation.'''
        if weights is None:
            weights = np.ones(y.shape)
        else:
            weights = np.array(weights)
            assert weights.shape[0] == y.shape[0], \
                'weights need to match number of observations'
        return weights
    
    @classmethod
    def _prepare_data(cls, y, X, weights=None):
        '''Prepares data formats for estimation.'''
        y = cls._prepare_y(y=y)
        X = cls._prepare_X(y=y, X=X)
        weights = cls._prepare_weights(y=y, weights=weights)
        return (y, X, weights)
    
    def _check_convergence(self, threshold):
        '''Checks convergence of the estimation given a threshold.'''
        assert self.iteration >= 1, \
            '''Not enough iterations to check convergence.'''
        if abs(self.estimation['scores'][-1]-self.estimation['scores'][-2]) < threshold:
            self.converged = True
            
    def _update_estimation(self, y, X, weights):
        '''Updates estimation attributes.'''
        self.estimation['scores'] += [self.score(y=y, X=X, weights=weights)]
        for param, value in self.params.items():
            self.estimation[param] += [value]
        
    def step(self, y, X=None, weights=None):
        '''Performs one estimation step.'''
        # initialise estimation step
        if not hasattr(self, 'iteration'):
            self._initialise_estimation(y=y, X=X, weights=weights)
        y, X, weights = self._prepare_data(y=y, X=X, weights=weights)
        
        # call defined step
        self._step(y, X, weights)
        
        # update estimation attributes
        self._update_estimation(y, X, weights)   
        
        
    def _estimate(self, y, X=None, weights=None, max_iter=100, threshold=1e-6):
        '''Performs parameter estimation through an iterative procedure.
        Requires the class to define a step() method that defines one fitting iteration.
        '''
        # initialise
        self._initialise_estimation(y, X, weights)
        y, X, weights = self._prepare_data(y, X, weights=weights)
        
        # run
        while self.iteration < max_iter and self.converged == False:
            self.step(y, X=X, weights=weights)
            self._check_convergence(threshold=threshold)
            if self.iteration == max_iter:
                warnings.warn('maximum number of iterations reached')
            
        return self
      
    def fit(self, y, X=None, weights=None):
        '''Fits the model parameters to an observation sequence.
        weights are optional.
        '''
        self =  self._estimate(y, X=X, weights=weights, max_iter=100, threshold=1e-6)
        self.is_fitted = True 
    
    def __repr__(self):
        return str(self)