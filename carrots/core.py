
'''
This module contains:
- Classes
    - MixtureDistribution
    - MarkovChain
    - ProductDistribution
- Functions
    - 
'''

# IMPORTS

import numpy as np
import scipy as sp
import pandas as pd

import scipy.stats as ss

import carrots.utils as utils

# CLASSES

class MixtureDistribution:
    
    '''
    A Mixture is a list of triples that parametrise the components of a Gaussian mixture distribution.
    Each triple is a tuple of mean, standard deviation and probability weight of the component.
    '''
    
    def __init__(self, components=[]):
        self.components = components
        self.n_components = len(self.components)
        
    def add_component(self,component):
        self.components += [component]
        self.n_components += 1
        
    def central_moment(self, moment):

        '''
        Compute the central moments of a mixture of normal components.
        Moment is the order of the central moment to compute.
        '''

        mean = sum([w*m for (m, s, w) in self.components])
    
        if moment is 1:
            return mean
        else:
            mixture_moment = 0
            for (m, s, w) in self.components:
                for k in range(moment+1):
                    product = sp.special.comb(moment, k) * (m-mean)**(moment-k) * utils.normal_central_moment(s, k)
                    mixture_moment += w * product
            return mixture_moment
        
    def standardised_moment(self, moment):
    
        '''
        Normalised moment of a mixture distribution.
        '''
    
        if (moment<=2):
            mixture_moment = self.central_moment(moment)
        else:
            mixture_variance = self.central_moment(2)
            mixture_central_moment = self.central_moment(moment)
            mixture_moment = mixture_central_moment / mixture_variance**(moment/2)
            if (moment%2==0):
                bias = utils.normal_central_moment(1,moment)
                mixture_moment -= bias
        return mixture_moment
    
    def mean(self):
        return self.standardised_moment(1)
    
    def var(self):
        return self.standardised_moment(2)
    
    def std(self):
        return self.standardised_moment(2)**0.5
    
    def skew(self):
        return self.standardised_moment(3)
    
    def kurt(self):
        
        '''
        Note that the output value is the excess kurtosis.
        '''
        
        return self.standardised_moment(4)
    
    def mvsk(self):
    
        '''
        The first four standardised moments about the mean of a mixture distribution.
        '''
    
        m = self.mean()
        v = self.var()
        s = self.skew()
        k = self.kurt()
        return (m,v,s,k)
            
    def rvs(self, sample_size=1):
    
        '''
        Draw a random sample from a mixture distribution
        '''
    
        weights = [p for (m,s,p) in self.components]
        norm_params = [(m,s) for (m,s,p) in self.components]
        draw_from = np.random.choice(self.n_components, size=sample_size, replace=True, p=weights)
        sample = np.fromiter((ss.norm.rvs(*(norm_params[i])) for i in draw_from),dtype=np.float64)
        if sample_size is 1:
            sample = sample[0]
        return sample
    
    def pdf(self, x):
        y = np.zeros(np.array(x).shape)
        for (m, s, w) in self.components:
            y += w*sp.stats.norm.pdf(x, m, s)
        return y
    
    def cdf(self, x):
        y = np.zeros(np.array(x).shape)
        for (m, s, w) in self.components:
            y += w*sp.stats.norm.cdf(x, m, s)
        return y
    
    def entropy(self):
        
        '''
        Calculate Shannon's entropy based on logarithms with base n of the n component probabilities.
        '''
        
        entropy = 0
        for (m, s, w) in self.components:
            if w == 0:
                pass
            else:
                entropy += w*np.log(w)/np.log(self.n_components)
        return abs(entropy)
    
    def get_component_means(self):
        means = [m for (m,s,w) in self.components]
        return means
    
    def get_component_stds(self):
        stds = [s for (m,s,w) in self.components]
        return stds
    
    def get_component_weights(self):
        weights = [w for (m,s,w) in self.components]
        return weights
    
    

class ProductDistribution:
    
    '''
    A ProducDistribution is a list of tuples that contains the first central moments of the factor distributions.
    Note that the input moments have to be non-standardised and factor draws have to be independent.
    '''
    
    def __init__(self, factors_moments=[]):
        self.factors_moments = factors_moments
        self.n_factors = len(self.factors_moments)
        
    def add_factor(self,factors_moments):
        self.factors_moments += [factors_moments]
        self.n_factors += 1
        
    def mean(self):
        prod = 1
        for factor in self.factors_moments:
            m = factor[0]
            prod *= m
        mean = prod
        return mean

    def var(self):
        prod1,prod2 = 1,1
        for factor in self.factors_moments:
            (m,s) = (factor[0],factor[1])
            prod1 *= m**2+s
            prod2 *= m**2
        var = prod1 - prod2
        return var
    
    def std(self):
        return self.var()**0.5

    def skew(self):
        prod1,prod2,prod3 = 1,1,1
        for factor in self.factors_moments:
            (m,s,g) = (factor[0],factor[1],factor[2])
            prod1 *= g+3*m*s+m**3
            prod2 *= m*s+m**3
            prod3 *= m**3
        third_central_moment = prod1 - 3*prod2 + 2*prod3
        skew = third_central_moment/(self.var()**1.5)
        return skew

    def kurt(self):
        
        '''
        Note that the output value is the excess kurtosis.
        '''
        
        prod1,prod2,prod3,prod4 = 1,1,1,1
        for factor in self.factors_moments:
            (m,s,g,k) = (factor[0],factor[1],factor[2],factor[3])
            prod1 *= k+4*m*g+6*m**2*s+m**4
            prod2 *= m*g+3*m**2*s+m**4
            prod3 *= m**2*s+m**4
            prod4 *= m**4
        fourth_central_moment = prod1 - 4*prod2 + 6*prod3 - 3*prod4
        kurt = fourth_central_moment/(self.var()**2)-3
        return kurt

    def mvsk(self):
        m = self.mean()
        v = self.var()
        s = self.skew()
        k = self.kurt()
        return (m,v,s,k)
    
    

class MarkovChain:
    
    '''
    A MarkovChain
    '''
    
    def __init__(self, transition_matrix=[], state_vector=[]):
        self.transition_matrix = transition_matrix
        self.n_states = np.array(self.transition_matrix).shape[0]
        self.state_vector = state_vector
        if self.state_vector == []:
            self.state_vector = [1/self.n_states] * self.n_states
        
    def set_transition_matrix(self, transition_matrix):
        self.transition_matrix = transition_matrix
        
    def set_state_vector(state_vector):
        self.state_vector = state_vector
    
    def calculate_steady_state(self, set_state=False):
        dim = np.array(self.transition_matrix).shape[0]
        q = np.c_[(self.transition_matrix-np.eye(dim)),np.ones(dim)]
        QTQ = np.dot(q, q.T)
        steady_state_probabilities = np.linalg.solve(QTQ,np.ones(dim))
        if set_state:
            self.state_vector = steady_state_probabilities
        return steady_state_probabilities

    def iterate(self, steps=1, return_state_vector=False):
        self.state_vector = np.dot(self.state_vector, np.linalg.matrix_power(self.transition_matrix, steps))
        if return_state_vector:
            return self.state_vector
    
    def rvs(self, T=1):
        draw = np.random.choice(self.n_states, size=1, p=self.state_vector)[0]
        sample = [draw]
        for t in range(1,T):
            draw = np.random.choice(self.n_states, size=1, p=self.transition_matrix[draw])[0]
            sample += [draw]
        if T is 1:
            sample = sample[0]
        return sample
    
    def calculate_expected_durations(self):
        expected_durations = (np.ones(self.n_states)-np.diag(self.transition_matrix))**-1
        return expected_durations