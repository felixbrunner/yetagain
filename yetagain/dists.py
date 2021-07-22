### IMPORTS ###

import numpy as np
import scipy as sp
import pandas as pd
import warnings
import copy

### CLASSES ###

class DistributionMixin:
    '''Mixin class for distributions.'''

    @property
    def std(self):
        '''Returns the distribution standard deviation.'''
        return self.var**0.5

    @property
    def exkurt(self):
        '''Returns the excess kurtosis.'''
        return self.kurt - 3

    def mvsk(self):
        '''Returns the first four standardised moments about the mean.'''
        m = self.mean
        v = self.var
        s = self.skew
        k = self.kurt
        return (m, v, s, k)

    def standardised_moment(self, moment):
        '''Returns the normalised moment of input order.'''
        variance = self.central_moment(2)
        central_moment = self.central_moment(moment)
        standardised_moment = central_moment / variance**(moment/2)
        return standardised_moment

    def excess_moment(self, moment):
        '''Returns the normalised moment of input order
        in excess of normal distribution.
        '''
        assert moment > 2, \
            'no excess at low moment order'
        standardised_moment = self.standardised_moment(moment)
        if (moment%2==0):
            bias = sp.stats.norm(loc=0, scale=1).moment(moment)
            excess_moment = standardised_moment - bias
        return excess_moment

    def likelihood(self, y, X=None):
        '''Returns the likelihoods of the observations in a sample.'''
        likelihood = self.pdf(y)
        return likelihood

    def score(self, y, X=None, weights=None):
        '''Returns the (weighted) log-likelihood of a sample.'''
        # weights
        if weights is None:
            weights = np.ones(np.array(y).shape)
        else:
            weights = np.array(weights)

        # score likelihoods
        score = (weights * np.log(self.likelihood(y=y, X=X))).sum()
        return score

    def copy(self):
        '''Returns a deep copy with new memory address.'''
        _dist = copy.deepcopy(self)
        return _dist

    def __repr__(self):
        return str(self)



class NormalDistribution(DistributionMixin):
    '''A normal distribution.
    If no parameters are specified, defaults to a standard normal distribution.
    '''
    
    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma

    @property
    def mu(self):
        '''The distribution mean.'''
        return self._mu
    
    @mu.setter
    def mu(self, mu):
        assert type(mu) in [int, float, np.float64], \
            'mu needs to be numeric'
        self._mu = mu        
        
    @property
    def sigma(self):
        '''The distribution standard deviation.'''
        return self._sigma
    
    @sigma.setter
    def sigma(self, sigma):
        assert type(sigma) in [int, float, np.float64], \
            'sigma needs to be numeric'
        self._sigma = sigma
    
    def central_moment(self, moment):
        '''Returns the central moments of input order.'''
        
        assert moment>0 and type(moment)==int, \
            'moment needs to be a positive integer'

        if moment % 2 == 1:
            #odd moments of a normal are zero
            central_moment = 0 
        else:
            #even moments are given by sigma^n times the double factorial
            central_moment = self.sigma**moment * sp.special.factorialk(moment-1, 2)
        return central_moment
    
    @property
    def mean(self):
        '''Returns the distribution mean.'''
        return self.mu

    @mean.setter
    def mean(self, new_mean):
        self.mu = new_mean

    @property
    def var(self):
        '''Returns the distribution variance.'''
        var = self.sigma**2
        return var

    @var.setter
    def var(self, new_var):
        assert new_var > 0, \
            'new_var needs to be a positive number'
        self.sigma = new_var**0.5

    @property
    def skew(self):
        '''Returns the distribution skewness.'''
        skew = self.standardised_moment(3)
        return skew
    
    @property
    def kurt(self):
        '''Returns the distribution kurtosis.'''
        kurt = self.standardised_moment(4)
        return kurt

    @property
    def mode(self):
        '''Returns the mode of the distribution.'''
        return self.mu

    @property
    def median(self):
        '''Returns the median of the distribution.'''
        return self.mu
    
    def pdf(self, x):
        '''Returns the probability density function value for input numbers.'''
        fx = sp.stats.norm.pdf(x, loc=self.mu, scale=self.sigma)
        return fx
    
    def cdf(self, x):
        '''Returns the cumulative density function value for input numbers.'''
        Fx = sp.stats.norm.cdf(x, loc=self.mu, scale=self.sigma)
        return Fx
    
    def draw(self, size=1):
        '''Draws random numbers from the distribution.'''
        sample = sp.stats.norm.rvs(size=size, loc=self.mu, scale=self.sigma)
        if size == 1:
            sample = sample[0]
        return sample

    def __str__(self):
        '''Returns a summarizing string.'''
        string = 'NormalDistribution(mu={}, sigma={})'.format(round(self.mu, 4), round(self.sigma, 4))
        return string



class StudentTDistribution(DistributionMixin):
    '''A student t distribution.
    If no parameters are specified, a standard normal distribution.
    '''
    
    def __init__(self, mu=0, sigma=1, df=np.inf):
        self.mu = mu
        self.sigma = sigma
        self.df = df

    @property
    def mu(self):
        '''The distribution mean.'''
        return self._mu

    @mu.setter
    def mu(self, mu):
        assert type(mu) in [int, float, np.float64], \
            'mu needs to be numeric'
        self._mu = mu

    @property
    def sigma(self):
        '''The distribution standard deviation.'''
        return self._sigma

    @sigma.setter
    def sigma(self, sigma):
        assert type(sigma) in [int, float, np.float64], \
            'sigma needs to be numeric'
        self._sigma = sigma

    @property
    def df(self):
        '''The distribution degrees of freedom.'''
        return self._df

    @df.setter
    def df(self, df):
        assert type(df) in [int, float, np.float64], \
            'df needs to be numeric'
        assert df > 0, \
            'df needs to be a postive number'
        self._df = df

    def central_moment(self, moment):
        '''Returns the central moments of input order.'''
        
        assert moment>0 and type(moment)==int, \
            'moment needs to be a positive integer'

        if moment % 2 == 1:
            #odd moments of a student t are zero
            central_moment = 0
        else:
            #even moments are given by...
            assert self.df > moment, \
                'moment {} does not exist for distribution with {} degrees of freedom'.format(round(moment, 4), round(self.df, 4))
            central_moment = sp.special.gamma((moment+1)/2) /np.sqrt(np.pi) * self.df**(moment/2) / (self.df/2-np.arange(1, (moment/2)+1)).prod()
        return central_moment
    
    @property
    def mean(self):
        '''Returns the distribution mean.'''
        return self.mu

    @mean.setter
    def mean(self, new_mean):
        self.mu = new_mean
    
    @property
    def var(self):
        '''Returns the distribution variance.'''
        assert self.df > 2, \
            'Variance does not exist for distribution with {} degrees of freedom.'.format(round(self.df, 4))
        var = self.sigma**2 * (self.df/(self.df-2))
        return var

    @var.setter
    def var(self, new_var):
        assert new_var > 0, \
            'variance needs to be a positive number'
        self.sigma = (new_var * (self.df-2) / self.df)**0.5

    @property
    def skew(self):
        '''Returns the distribution skewness.'''
        skew = self.standardised_moment(3)
        return skew

    @property
    def kurt(self):
        '''Returns the distribution kurtosis.'''
        kurt = self.standardised_moment(4)
        return kurt

    @property
    def mode(self):
        '''Returns the mode of the distribution.'''
        return self.mu

    @property
    def median(self):
        '''Returns the median of the distribution.'''
        return self.mu

    def pdf(self, x):
        '''Returns the probability density function value for input numbers.'''
        y = sp.stats.t.pdf(x, loc=self.mu, scale=self.sigma, df=self.df)
        return y

    def cdf(self, x):
        '''Returns the cumulative density function value for input numbers.'''
        y = sp.stats.t.cdf(x, loc=self.mu, scale=self.sigma, df=self.df)
        return y

    def draw(self, size=1):
        '''Draws random numbers from the distribution.'''
        sample = sp.stats.t.rvs(size=size, loc=self.mu,
                                scale=self.sigma, df=self.df)
        if size == 1:
            sample = sample[0]
        return sample

    def __str__(self):
        '''Returns a summarizing string.'''
        string = 'StudentTDistribution(mu={}, sigma={}, df={})'.format(round(self.mu, 4), round(self.sigma, 4), round(self.df, 4))
        return string



class MixtureDistribution(DistributionMixin):
    '''A mixture distribution is a list of tuples that parametrise the components of a Mixture distribution.
    Each component tuple is a pair of distribution and probability weight of the component.
    '''
    
    def __init__(self, components=None):
        self.components = components
        
    def _check_component(self, component):
        '''Checks component inputs.'''
        dist, weight = component
        assert isinstance(dist, DistributionMixin), \
            'unknown component distribution type'
        assert type(weight) == float or type(weight) == int or type(weight) == np.float64, \
            'weight needs to be numeric'
    
    @property
    def components(self):
        '''List of (distribution, weight) tuples.
        Distributions need to be instances of Base Distribution.
        Weights need to be numbers.
        '''
        return self._components
    
    @components.setter
    def components(self, components):
        assert type(components) == list, \
            'components needs to be a list of tuples'
        for component in components:
            self._check_component(component)
        self._components = components
        
    @property
    def distributions(self):
        '''Returns a list of component distributions.'''
        distributions = [component[0] for component in self.components]
        return distributions
        
    @property
    def weights(self):
        '''Returns a list of component weights.'''
        weights = [component[1] for component in self.components]
        return weights
    
    @property
    def n_components(self):
        '''Returns the number of components.'''
        return len(self.components)
    
    def add_component(self, distribution, weight):
        '''Adds a component to the mixture distribution.
        Inputs needs to be a distribution and a weight.
        '''        
        component = (distribution, weight)
        self._check_component(component)
        self.components = self.components + [component]
    
    @property
    def mean(self):        
        '''Returns the mean.'''
        mean = sum([component.mean*weight for (component, weight) in self.components])
        return mean

    @mean.setter
    def mean(self, new_mean):
        '''Moves the mixture distribution such that it has a new mean, while
        its scale and shape remain unchanged. Adjusts the means of all
        component distributions accordingly inplace.
        '''
        # setup
        addition = new_mean - self.mean
        new_components = []

        # adjust all component means
        for component, weight in self.components:
            component.mean += addition
    
    def central_moment(self, moment):
        '''Returns the central moment of input order.'''

        assert moment > 0, \
            'moment needs to be positive'
    
        if moment is 1:
            return 0
        else:
            mean = self.mean
            inputs = [(component.mean, component.std, weight) \
                            for (component, weight) in self.components]
            central_moment = 0
            for (m, s, w) in inputs:
                for k in range(moment+1):
                    product = sp.special.comb(moment, k) \
                                * (m-mean)**(moment-k) \
                                * sp.stats.norm(loc=0, scale=s).moment(k)
                    central_moment += w * product
            return central_moment
    
    @property
    def var(self):
        '''Returns the distribution variance.'''
        return self.central_moment(2)

    @var.setter
    def var(self, new_var):
        '''Scales the mixture distribution such that it has a new variance,
        while its location and shape remain unchanged. Adjusts the means and
        variances of all component distributions accordingly inplace.
        '''
        # setup
        scaling = new_var / self.var
        mean = self.mean
        new_components = []

        # scale all component means & variances
        for component, weight in self.components:
            component.mean = mean + (component.mean - mean) * np.sqrt(scaling)
            component.var *= scaling

    @property
    def skew(self):
        '''Returns the distribution skewness.'''
        return self.standardised_moment(3)

    @property
    def kurt(self):
        '''Returns the distribution kurtosis.'''
        return self.standardised_moment(4)

    @property
    def mode(self):
        '''Returns the mode of the distribution.'''
        pdfs = [self.pdf(comp_mode) for comp_mode in self.component_modes]
        mode = self.component_modes[np.argmax(pdfs)]
        return mode

    @property
    def median(self):
        '''Returns the median of the distribution.'''
        raise NotImplementedError('MixtureDistribution median not implemented')
    
    def entropy(self, level='state'):
        '''Returns Shannon's entropy based on logarithms with base n of the n component probabilities.'''
        if level == 'state':
            entropy = sp.stats.entropy(self.weights, base=self.n_components)
        else:
            raise NotImplementedError('random variable entropy not implemented')
        return entropy
    
    @property
    def component_means(self):
        '''Returns a list of component means.'''
        means = [distribution.mean for (distribution, weight) in self.components]
        return means
    
    @property
    def component_stds(self):
        '''Returns a list of component standard deviations.'''
        stds = [distribution.std for (distribution, weight) in self.components]
        return stds

    @property
    def component_modes(self):
        '''Returns a list of component standard deviations.'''
        modes = [distribution.mode for (distribution, weight) in self.components]
        return modes 
    
    def pdf(self, x):
        '''Evaluates the probability density function at x.'''
        fx = np.zeros(np.array(x).shape)
        for (component, weight) in self.components:
            fx += weight*component.pdf(x)
        return fx
    
    def cdf(self, x):
        '''Evaluates the cumulative density function at x.'''
        Fx = np.zeros(np.array(x).shape)
        for (component, weight) in self.components:
            Fx += weight*component.cdf(x)
        return Fx

    def draw(self, size=1, return_states=False):
        '''Draw a random sample from a mixture distribution.'''
        states = np.random.choice(self.n_components, size=size,
                                  replace=True, p=self.weights)
        sample = np.fromiter((self.components[i][0].draw() for i in states),
                             dtype=np.float64)
        
        if size is 1:
            sample = sample[0]
            
        if return_states:
            return (sample, states)
        else:
            return sample

    def __str__(self):
        '''Returns a summarizing string.'''
        string = 'MixtureDistribution(\n'
        for (component, weight) in self.components:
            string += '\t {}, weight={},\n'.format(component.__str__(), weight)
        string += ')'
        return string
    


class ProductDistribution(DistributionMixin):
    '''A ProducDistribution is a list of tuples that contains the first central moments of the factor distributions.
    Note that the input moments have to be non-standardised and factor draws have to be independent.
    '''
    
    def __init__(self, factors=None):
        self.factors = factors
        
    def _check_factor(self, factor):
        '''Checks factor inputs.'''
        assert isinstance(factor, DistributionMixin), \
            'unknown factor distribution type'
        
    @property
    def factors(self):
        '''List of factor distributions.
        Each element needs to be instances of BaseDistribution.
        '''
        return self._factors
    
    @factors.setter
    def factors(self, factors):
        assert type(factors) == list, \
            'factors needs to be a list of tuples'
        for factor in factors:
            self._check_factor(factor)
        self._factors = factors
    
    @property
    def n_factors(self):
        '''Returns the number of factors.'''
        return len(self.factors)
    
    def add_factor(self, factor):
        '''Adds a factor to the mixture distribution.
        Input needs to instance of BaseDistribution.
        '''
        self._check_factor(factor)
        self.factors = self.factors + [factor]

    @property
    def mean(self):
        ''' Returns the distribution mean.'''
        prod = 1
        for factor in self.factors:
            m = factor.mean
            prod *= m
        mean = prod
        return mean
    
    @property
    def var(self):
        '''Returns the distribution variance.'''
        prod1, prod2 = 1, 1
        for factor in self.factors:
            (m, s) = (factor.mean, factor.var)
            prod1 *= m**2+s
            prod2 *= m**2
        var = prod1 - prod2
        return var

    @property
    def skew(self):
        '''Returns the distribution skewness.'''
        prod1, prod2, prod3 = 1, 1, 1
        for factor in self.factors:
            (m, s, g) = (factor.mean, factor.var, factor.central_moment(3))
            prod1 *= g+3*m*s+m**3
            prod2 *= m*s+m**3
            prod3 *= m**3
        third_central_moment = prod1 - 3*prod2 + 2*prod3
        skew = third_central_moment/(self.var()**1.5)
        return skew

    @property
    def kurt(self):
        '''Returns the distribution kurtosis.'''
        prod1, prod2, prod3, prod4 = 1, 1, 1, 1
        for factor in self.factors:
            (m, s, g, k) = (factor.mean, factor.var, factor.central_moment(3), factor.central_moment(4))
            prod1 *= k+4*m*g+6*m**2*s+m**4
            prod2 *= m*g+3*m**2*s+m**4
            prod3 *= m**2*s+m**4
            prod4 *= m**4
        fourth_central_moment = prod1 - 4*prod2 + 6*prod3 - 3*prod4
        kurt = fourth_central_moment/(self.var()**2)#-3
        return kurt

    @property
    def mode(self):
        '''Returns the mode of the distribution.'''
        raise NotImplementedError('ProductDistribution mode not implemented')

    @property
    def median(self):
        '''Returns the median of the distribution.'''
        raise NotImplementedError('ProductDistribution mode not implemented')

    def draw(self, size=1, add_one=False):
        '''Returns a random sample drawn from the product distribution.'''
        sample = np.ones(size)
        if add_one:
            for factor in self.factors:
                sample *= factor.draw(size=size)+1
            sample -= 1
        else:
            for factor in self.factors:
                sample *= factor.draw(size=size)

        if size is 1:
            sample = sample[0]
        
        return sample
    
    def pdf(self):
        raise NotImplementedError('exact pdf unknown')
    
    def cdf(self):
        raise NotImplementedError('exact cdf unknown')

    def __str__(self):
        '''Returns a summarizing string.'''
        string = 'ProductDistribution(\n'
        for factor in self.factors:
            string += '\t {},\n'.format(factor.__str__())
        string += ')'
        return string