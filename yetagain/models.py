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

    def errors(self, y, X=None, method='mean'):
        '''Returns errors made by the model when predicting input data.'''
        assert method != 'distribution', \
            'distribution not an allowed prediction method to calculate errors'
        errors = y - self.predict(y=y, X=X, method=method)
        return errors

    def squared_errors(self, y, X=None, method='mean'):
        '''Returns errors made by the model when predicting input data.'''
        squared_errors = self.errors(y=y, X=X, method=method)**2
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

    def predict(self, y, X=None, method='distribution', **kwargs):
        '''Returns an array with predictions for an input sample.'''
        y = np.atleast_1d(y)
        if method == 'distribution' or method == None:
            return [self.distribution.copy() for y_t in y]
        elif method == 'mean':
            return np.full(shape=y.shape, fill_value=self.mean)
        elif method == 'mode':
            return np.full(shape=y.shape, fill_value=self.mode)
        elif method == 'median':
            return np.full(shape=y.shape, fill_value=self.median)
        elif method == 'var':
            return np.full(shape=y.shape, fill_value=self.var)
        elif method == 'std':
            return np.full(shape=y.shape, fill_value=self.std)
        else:
            raise NotImplementedError('Prediction method not implemented')

    def draw(self, size=1, return_distributions=False):
        '''Draw a random sequence of specified length.'''
        # draw sample from distribution
        sample = self.distribution.draw(size=size)

        # return sequence of distributions if required
        if return_distributions:
            if size is 1:
                distributions = self.distribution.copy()
            else:
                distributions = [self.distribution.copy() for i in range(size)]
            return (sample, distributions)
        else:
            return sample

    def forecast(self, horizons=[1], method=None, **kwargs):
        '''returns a forecast of h steps ahead.'''
        # make sure horizons is iterable
        horizons = np.atleast_1d(horizons)

        # calculate forecasts
        forecasts = []
        for horizon in horizons:
            forecast_model = self.iterate(horizon)
            distribution = forecast_model.distribution

            # extract forecast statistic
            if method == None or method == 'distribution':
                forecasts += [distribution]
            elif method == 'mean':
                forecasts += [distribution.mean]
            elif method == 'mode':
                forecasts += [distribution.mode]
            elif method == 'median':
                forecasts += [distribution.median]
            elif method == 'var':
                forecasts += [distribution.var]
            elif method == 'std':
                forecasts += [distribution.std]
            else:
                raise NotImplementedError('Forecast method not implemented')
        return forecasts

    def likelihood(self, y, X=None, **kwargs):
        '''Returns the likelihoods of the observations in a sample.'''
        distributions = self.predict(y=y, X=X, **kwargs)
        likelihood = [dist_t.pdf(y_t) for y_t, dist_t in zip(y, distributions)]
        return likelihood

    def score(self, y, X=None, weights=None, **kwargs):
        '''Returns the (weighted) log-likelihood of a sample.'''

        # weights
        if weights is None:
            weights = np.ones(np.array(y).shape)
        else:
            weights = np.array(weights)

        # score log-likelihood
        score = (weights * np.log(self.likelihood(y=y, X=X, weights=weights, **kwargs))).sum()
        return score


class NormalModel(ModelMixin, EstimationMixin, NormalDistribution):
    '''i.i.d. normal distribution model.'''
    
    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma
        ModelMixin.__init__(self)
    
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
        self.mu = float(mean)
        
        # estimate variance
        errors = self.squared_errors(y)
        variance = np.average(errors, weights=weights)
        self.sigma = float(np.sqrt(variance))
            
        # set status
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
        w_ = ((self.df+1)*self.sigma**2) \
                / (self.df*self.sigma**2 + self.squared_errors(y))
        self.w_ = w_

    def _m_step(self, y, weights):
        '''Performs the maximisation step to update location and
        scale of the distribution.
        '''
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
        distribution = StudentTDistribution(mu=self.mu,
                                            sigma=self.sigma,
                                            df=self.df)
        return distribution

    def __str__(self):
        '''Returns a summarizing string.'''
        string = 'StudentTModel(mu={}, sigma={}, df={})'\
            .format(round(self.mu, 4), round(self.sigma, 4), round(self.df, 4))
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