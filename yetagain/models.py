import numpy as np
import scipy as sp
import warnings
import copy
# from scipy.stats import norm

from yetagain.dists import NormalDistribution, MixtureDistribution, StudentTDistribution
from yetagain.estimation import EstimationMixin


class ModelMixin:
    '''Mixin class for models.'''

    def __init__(self):
        self.is_fitted = False

    def errors(self, y, method='mean'):
        '''Returns errors made by the model when predicting input data.'''
        errors = y - self.predict(y, X=None, method=method)
        return errors

    def squared_errors(self, y, method='mean'):
        '''Returns errors made by the model when predicting input data.'''
        squared_errors = self.errors(y, method='mean')**2
        return squared_errors

    def iterate(self, steps=1):
        '''Iterates the model forward the input number of steps.'''
        return self

    def copy(self):
        '''Returns a deep copy with new memory address.'''
        _model = copy.deepcopy(self)
        return _model

    def __repr__(self):
        return str(self)

    @property
    def params_(self):
        assert self.is_fitted, \
            'Model has no fitted parameters.'
        return self.params

    def predict(self, y, X=None, method='mean'):
        '''Returns an array with predicted values for an input sample.'''
        y = np.array(y)
        if method == 'mean':
            predictions = np.full(shape=y.shape, fill_value=self.mean())
        else:
            raise NotImplementedError('Prediction method not implemented')
        return predictions


class NormalModel(ModelMixin, EstimationMixin, NormalDistribution):
    '''i.i.d. normal distribution model.'''
    
    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma
        ModelMixin.__init__(self)
    
    @property
    def loc(self):
        return self.mu
    
    @property
    def scale(self):
        return self.sigma
    
    @property
    def params(self):
        params = {'mu': self.mu,
                  'sigma': self.sigma}
        return params
    
    @params.setter
    def params(self, params):
        for k, v in params.items():
            setattr(self, k, v)
            
    def _step(self, y, X, weights):
        '''Performs one estimation step.
        Recalculates the distribution mean and variance.
        '''
        # estimate mean
        mean = np.average(y, weights=weights)
        
        # estimate variance
        errors = (y-mean)**2
        variance = np.average(errors, weights=weights)
            
        # set attributes
        self.mu = float(mean)
        self.sigma = float(np.sqrt(variance))
        self.converged = True


    @property
    def distribution(self):
        '''Extracts and returns a NormalDistribution object
        with the the same parameters as the model.
        '''
        norm = NormalDistribution(mu=self.mu, sigma=self.sigma)
        return norm
    

    def __str__(self):
        '''Returns a summarizing string.'''
        string = 'NormalModel(mu={}, sigma={})'.format(round(self.mu, 4), round(self.sigma, 4))
        return string


class StudentTModel(ModelMixin, EstimationMixin, StudentTDistribution):
    '''i.i.d. normal distribution model.'''
    
    def __init__(self, mu=0, sigma=1, df=np.inf):
        self.mu = mu
        self.sigma = sigma
        self.df = df
        ModelMixin.__init__(self)
    
    @property
    def loc(self):
        return self.mu
    
    @property
    def scale(self):
        return self.sigma
    
    @property
    def params(self):
        params = {'df': self.df,
                  'mu': self.mu,
                  'sigma': self.sigma}
        return params
    
    @params.setter
    def params(self, params):
        for k, v in params.items():
            setattr(self, k, v)

            
    def _e_step(self, y):
        '''Performs the expectation step to update estimation weights.'''
        # intialise the EM algorithm with the equally weighted scipy implementation
        if self.iteration == 0:
            (self.df, self.mu, self.sigma) = sp.stats.t.fit(y)
        
        # update weights
        w_ = ((self.df+1)*self.sigma**2) / (self.df*self.sigma**2 + self.squared_errors(y))
        self.w_ = w_

    def _m_step(self, y, weights):
        '''Performs the maximisation step to update location and scale of the distribution.'''
        # update mu
        self.mu = np.average(y, weights=weights*self.w_)
        
        # update sigma
        squared_errors = self.squared_errors(y)
        self.sigma = np.sqrt(np.average(squared_errors*self.w_, weights=weights))
        
        # update df
        const = 1 - np.log((self.df+1)/2) + np.average(np.log(self.w_)-self.w_, weights=weights) + sp.special.digamma((self.df+1)/2)
        fun = lambda df: np.log(df/2) - sp.special.digamma(df/2) + const
        self.df = sp.optimize.fsolve(fun, self.df)[0]

    def _step(self, y, X, weights):
        '''Performs one estimation step.
        Recalculates the distribution mean and variance.
        '''
        self._e_step(y)
        self._m_step(y, weights)
        
    
    @property
    def distribution(self):
        '''Extracts and returns a NormalDistribution object
        with the the same parameters as the model.
        '''
        distribution = StudentTDistribution(mu=self.mu, sigma=self.sigma, df=self.df)
        return distribution


    def __str__(self):
        '''Returns a summarizing string.'''
        string = 'StudentTModel(mu={}, sigma={}, df={})'.format(round(self.mu, 4), round(self.sigma, 4), round(self.df, 4))
        return string
        
        
class MixtureModel(ModelMixin, MixtureDistribution):
    '''mixture model of arbitrary distributions.'''
    
    def __init__(self, components=[]):
        self.components = components
        
    def fit(self, Y, weights=None, method='em'):
        ### use EM algorithm
        raise NotImplementedError('fit method not implemented')

    
    @property
    def distribution(self):
        raise NotImplementedError('distribution not implemented')
        

    def __str__(self):
        '''Returns a summarizing string'''
        string = 'MixtureModel(\n'
        for (component, weight) in self.components:
            string += '\t {}, weight={},\n'.format(component.__str__(), weight)
        string += ')'
        return string