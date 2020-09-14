import numpy as np
import warnings
from scipy.stats import norm

from yetagain.dists import NormalDistribution, MixtureDistribution


class BaseModel:
    
    '''
    Base class for models
    '''
    
    def __init__(self):
        pass
    

    def score(self, Y):
        
        '''
        Returns the log-likelihood of an observation sequence
        '''
        
        score = np.log(self.pdf(Y)).sum()
        return score


    def iterate(self, steps=1):

        '''
        Iterates the model forward the input number of steps.
        '''

        return self

    
    def __repr__(self):
        return str(self)


class NormalModel(BaseModel, NormalDistribution):
    
    '''
    i.i.d. normal distribution model
    '''
    
    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma
    
    @property
    def loc(self):
        return self.mu
    
    @property
    def scale(self):
        return self.sigma
    
        
    def fit(self, Y, weights=None):
        
        '''
        Fits the model parameters to an observation sequence.
        weights are optional.
        '''
        
        # prepare
        Y = np.array(Y)
        if weights is None:
            weights = np.ones(Y.shape)
        else:
            weights = np.array(weights)
        
        # estimate mean
        mean = np.average(Y, weights=weights)
        
        # estimate variance
        errors = (Y-mean)**2
        variance = np.average(errors, weights=weights)
        
        # update
        self.mu = float(mean)
        self.sigma = float(np.sqrt(variance))


    @property
    def distribution(self):

        '''
        Extracts and returns a NormalDistribution object
        with the the same parameters as the model.
        '''

        norm = NormalDistribution(mu=self.mu, sigma=self.sigma)
        return norm


    def __str__(self):

        '''
        Returns a summarizing string
        '''

        string = 'NormalModel(mu={}, sigma={})'.format(round(self.mu, 4), round(self.sigma, 4))
        return string
        
        
class MixtureModel(BaseModel, MixtureDistribution):
    
    '''
    mixture model of arbitrary distributions
    '''
    
    def __init__(self, components=[]):
        self.components = components
        
        
    def fit(self, y):
        ### use EM algorithm
        raise NotImplementedError('fit method not implemented')

    
    @property
    def distribution(self):
        raise NotImplementedError('distribution not implemented')
        

    def __str__(self):

        '''
        Returns a summarizing string
        '''

        string = 'MixtureModel(\n'
        for (component, weight) in self.components:
            string += '\t {}, weight={},\n'.format(component.__str__(), weight)
        string += ')'
        return string
    