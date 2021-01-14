import numpy as np
import scipy as sp
import warnings
# from scipy.stats import norm

from yetagain.dists import NormalDistribution, MixtureDistribution, StudentTDistribution


class BaseModel:
    '''Base class for models.'''
    
    def __init__(self):
        pass

    def score(self, Y):
        '''Returns the log-likelihood of an observation sequence.'''
        score = np.log(self.pdf(Y)).sum()
        return score

    def iterate(self, steps=1):
        '''Iterates the model forward the input number of steps.'''
        return self

    def __repr__(self):
        return str(self)


class NormalModel(BaseModel, NormalDistribution):
    '''i.i.d. normal distribution model.'''
    
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
        '''Fits the model parameters to an observation sequence.
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
        '''Extracts and returns a NormalDistribution object
        with the the same parameters as the model.
        '''
        norm = NormalDistribution(mu=self.mu, sigma=self.sigma)
        return norm


    def __str__(self):
        '''Returns a summarizing string.'''
        string = 'NormalModel(mu={}, sigma={})'.format(round(self.mu, 4), round(self.sigma, 4))
        return string


class StudentTModel(BaseModel, StudentTDistribution):
    '''i.i.d. normal distribution model.'''
    
    def __init__(self, mu=0, sigma=1, df=np.inf):
        self.mu = mu
        self.sigma = sigma
        self.df = df
        self.is_fitted = False
    
    @property
    def loc(self):
        return self.mu
    
    @property
    def scale(self):
        return self.sigma
    
        
    def fit(self, Y, weights=None, method='em'):
        '''Fits the model parameters to an observation sequence.
        Weights are optional.
        '''        
        # prepare weights
        Y = np.array(Y)
        if weights is None:
            weights = np.ones(Y.shape)
        else:
            weights = np.array(weights)
        
        if method == 'em':
            self = self._estimate_em_algorithm(Y, weights)
            self.is_fitted = True
        else:
            raise NotImplementedError('fitting algorithm not implemented')
        
        
    def _estimate_em_algorithm(self, Y, weights, max_iter=100, threshold=1e-6, return_fit=False):
        '''Performs parameter estimation with the Expectation Maximisation algorithm.
        Returns a fitted model.
        Returns the the fitted model and parameters of the estimation if return_fit=True.
        '''
        
        # initialise
        Y = np.array(Y)
        params_ = self._initialise_em(Y)
        score_ = self._score(Y, weights, params_)
        
        # store
        iteration = 0
        scores = {iteration: score_}
        
        while iteration < max_iter:
            iteration += 1
            w_ = self._do_e_step(Y, weights, params_)
            params_ = self._do_m_step(Y, w_, params_[0])
            score_= self._score(Y, weights, params_)
            scores[iteration] = score_
            
            if abs(scores[iteration]-scores[iteration-1]) < threshold:
                converged = True
                
                # approximate df
                params_ = (params_[0]*weights.mean(), params_[1], params_[2])
                warnings.warn('df parameter is only approximated')
                
                break
        else:
            converged = False
            warnings.warn('maximum number of iterations reached')
                
        self._update_params(params_)
        
        if return_fit:
            fit = {'converged': converged,
                   'iterations': iteration,
                   'scores': scores,
                  }
            
            return self, fit
        else:
            return self
        
    def _initialise_em(self, Y):
        '''Intialises the EM algorithm using the equally weighted scipy implementation.'''
        params_ = sp.stats.t.fit(Y)
        return params_
    
    def _score(self, Y, weights, params_):
        '''Scores a (weighted) sample as log-likelihood given a set of parameters.'''
        likelihoods_ = sp.stats.t(*params_).pdf(Y) 
        score_ = sum(np.log(likelihoods_)* weights)
        return score_
        
    def _do_e_step(self, Y, weights, params_):
        '''Performs the expectation step to update estimation weights.'''
        (df_, mu_, sigma_) = params_
        w_ = ((df_+1)*sigma_**2) / (df_*sigma_**2 + (Y-mu_)**2)
        w_ *= weights
        return w_
        
    def _do_m_step(self, Y, w_, df_):
        '''Performs the maximisation step to update location and scale of the distribution.'''
        # estimate mu
        mu_ = np.average(Y, weights=w_)
        
        # estimate sigma
        errors = (Y-mu_)**2
        sigma_ = np.sqrt(np.average(errors, weights=w_))
        
        # approximate df
        df_ = df_#*sum(w_)/len(w_)

        return (df_, mu_, sigma_)
    
    def _update_params(self, params_):
        '''Updates the stored model parameters.'''
        (df_, mu_, sigma_) = params_
        self.df = df_
        self.mu = mu_
        self.sigma = sigma_
        
    
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
        
        
class MixtureModel(BaseModel, MixtureDistribution):
    '''mixture model of arbitrary distributions.'''
    
    def __init__(self, components=[]):
        self.components = components
        
    def fit(self, y):
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