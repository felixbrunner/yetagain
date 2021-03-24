### IMPORTS ###

import numpy as np
import scipy as sp
import warnings
import copy

from yetagain.models import ModelMixin, NormalModel
from yetagain.estimation import EstimationMixin

### CLASSES ###

class GARCHModel(ModelMixin, EstimationMixin):
    '''Generalised Auto-Regressive Conditional Heteroskedasticity model class.'''
    
    def __init__(self, means_model=NormalModel(), omega=0, alphas=[0], betas=[0], state=None):
        self.means_model = means_model
        self.omega = omega
        self.alphas = alphas
        self.betas = betas
        self.state = state
        ModelMixin.__init__(self)
        
    @property
    def p(self):
        '''The order of GARCH (AR) terms.'''
        p = len(self.betas)
        return p
    
    @property
    def q(self):
        '''The order of ARCH (MA) terms.'''
        q = len(self.alphas)
        return q
    
    @property
    def memory(self):
        '''Maximum length of memory in the model.'''
        memory = max(self.p, self.q)
        return memory
    
    @property
    def means_model(self):
        return self._means_model
    
    @means_model.setter
    def means_model(self, means_model):
        assert isinstance(means_model, ModelMixin), \
            'means_model needs to be a model'
        self._means_model = means_model
    
    @property
    def omega(self):
        '''The state equation intercept.'''
        return self._omega
    
    @omega.setter
    def omega(self, omega):
        assert type(omega) in [int, float, np.float64], \
            'omega needs to be numeric'
        self._omega = omega
        
    @property
    def alphas(self):
        '''The state equation MA parameters.'''
        return self._alphas
    
    @alphas.setter
    def alphas(self, alphas):
        if type(alphas) == np.ndarray:
            alphas = alphas.tolist()
        if type(alphas) in [int, float, np.float64]:
            alphas = [alphas]
        if len(alphas)==0:
            raise NotImplementedError('GARCH(p,0) is currently not implemented')
        self._alphas = alphas
        
    @property
    def betas(self):
        '''The state equation AR parameters.'''
        return self._betas
    
    @betas.setter
    def betas(self, betas):
        if type(betas) == np.ndarray:
            betas = betas.tolist()
        if type(betas) in [int, float, np.float64]:
            betas = [betas]
        self._betas = betas
        
    @property
    def mu(self):
        return self.means_model.mean()
        
    @property
    def sigma(self):
        return self.means_model.std()
        
    @property
    def state(self):
        return self._state
    
    @property
    def params(self):
        # collect garch equation parameters
        params = {'omega': self.omega,
                  'alphas': self.alphas,
                  'betas': self.betas,
                 }
        
        # add means_model params to dict
        params.update(self.means_model.params)
        return params
    
    @params.setter
    def params(self, params):
        for k, v in params.items():
            setattr(self, k, v)
            
    @property
    def param_vector(self):
        theta = np.array([self.omega] + self.alphas + self.betas).reshape(-1, 1)
        return theta
    
    @state.setter
    def state(self, state):
        if state is None:
            pass
        else:
            assert type(state) in [int, float, np.float64], \
                'state needs to be numeric'
            assert state > 0, \
                'state needs to be positive'
        self._state = state
        
    def __str__(self):
        '''Returns a summarizing string.'''
        string = 'GARCHModel(({}, {}),\n\t means_model={},\n\t omega={}, alphas={}, betas={}, \n\t state={})' \
            .format(self.p, self.q, self.means_model.__str__(), self.omega, self.alphas, self.betas, self.state)
        return string
    
    @property
    def distribution(self):
        distribution = self.means_model.distribution.copy()
        if self.state:
            distribution.set_variance(self.state)
        return distribution
    
    @property
    def steady_state(self):
        '''The unconditional error variance expectation.'''
        steady_state = self.omega/(1-sum(self.alphas)-sum(self.betas))
        return steady_state
    
    def set_to_steady_state(self):
        '''Set the steady state as the model state.'''
        self.state = self.steady_state
        
    @property
    def steady_state_kurtosis(self):
        ''''''
        raise NotImplementedError('unconditional kurtosis not implemented')
        
    def draw(self, size=1, return_variances=False):
        '''Draw a random sequence of specified length.'''
        # set up
        if self.is_fitted:
            states, epsilons = self._start_recursion(warm_start=True)
        else:
            states, epsilons = self._start_recursion(warm_start=False)
        means_model = self.means_model.copy()
        variances = []
        sample = []
        
        for step in range(size):
            # draw
            variances += [self.omega + np.flip(self.betas)@states + np.flip(self.alphas)@epsilons**2]
            means_model.set_variance(variances[-1])
            sample += [means_model.draw()]
            
            # update states & epsilons
            states = np.append(states[1:], variances[-1])
            epsilons = np.append(epsilons[1:], means_model.errors(sample[-1]))
        
        if size is 1:
            sample = sample[0]
            variances = variances[0]
            
        if return_variances:
            return (sample, variances)
        else:
            return sample
        
    def iterate(self, steps=1, set_state=False):
        '''Iterates the model the specified number of steps.
        steps should be a positive integer.
        If set_state=True, GARCHModel object is modified in place.
        '''
        # define past states
        states, epsilons = self._start_recursion()
        
        # iterate state
        new_state = self.state
        for step in range(steps):
            # draw
            new_state = self.omega + np.flip(self.betas)@states + np.flip(self.alphas)@epsilons**2
            
            # update states & epsilons
            states = np.append(states[1:], new_state)
            epsilons = np.append(epsilons[1:], self.steady_state**0.5)
        
        # outputs
        if set_state:
            self.state = new_state
        else:
            new_garch = self.copy()
            new_garch.state = new_state
            return new_garch
    
    def _start_recursion(self, warm_start=False):
        '''Returns arrays of lagged states and errors.'''
        # warm start case
        if warm_start:
            assert self.state is not None, \
                'warm start requires the model to have state'
            if self.is_fitted: #if model has state history
                states = self.states_[-self.p:]
                epsilons = self.epsilons_[-self.q:]
            else: # if model has state but not state history
                states = np.append(np.full(max(0, self.p-1), self.steady_state), self.state)
                epsilons = np.full(max(0, self.q), self.steady_state**0.5)
                
        # start from steady state case
        else:
            states = np.full(max(0, self.p), self.steady_state)
            epsilons = np.full(max(0, self.q), self.steady_state**0.5)
        
        # return empty arrays if order zero
        if self.p==0:
            states = np.array([])
        if self.q==0:
            epsilons = np.array([])
        return (states, epsilons)
        
    def filter_states(self, y, X=None, weights=None, warm_start=False, return_errors=False):
        '''Filters the variance states from a given series.'''
        # prepare importance weights
        if weights is None:
            weights = np.ones(np.array(y).shape)
        else:
            weights = np.array(weights)
        weights = weights/weights.mean()
        
        # get previous states
        lagged_states, lagged_epsilons = self._start_recursion(warm_start=warm_start)
        
        # set up storage arrays
        if warm_start:
            states_ = self.states_.copy()
            epsilons_ = self.epsilons_.copy()
        else:
            states_ = []
            epsilons_ = []
            
        # recursive filter
        for y_t, w_t in zip(y, weights):
            states_ += [self.omega + np.flip(self.betas)@lagged_states + np.flip(self.alphas)@lagged_epsilons**2]
            epsilons_ += [self.means_model.errors(y_t)] #variance parameter is not required to calculate errors

            # update states_ & epsilons_ if included
            if self.p > 0:
                lagged_states = np.append(lagged_states[1:], states_[-1])
            if self.q > 0:
                lagged_epsilons = np.append(lagged_epsilons[1:], w_t*abs(epsilons_[-1])+(1-w_t)*states_[-1]**0.5)
        
        if return_errors:
            return (states_, epsilons_)
        else:
            return states_
        
    def _variance_gradients(self, y, X, weights):
        '''Returns a numpy array with recursively calculated state
        variable gradients.
        '''
        # build data matrix Z
        Z = np.ones([len(y), 1])
        weighted_epsilons = weights*self.epsilons_**2 + (1-weights)*self.states_ #weighted average of squared error and its expectation
        for arch in range(1, 1+self.q):
            Z = np.concatenate(
                    [Z,
                     np.array([self.steady_state]*arch+weighted_epsilons.tolist()[:-arch]).reshape(-1, 1),
                    ], axis=1
                )
        for garch in range(1, 1+self.p):
            Z = np.concatenate(
                    [Z,
                     np.array([self.steady_state]*garch+self.states_[:-garch]).reshape(-1, 1),
                    ], axis=1
                )

        # steady state gradient
        denominator = 1 - sum(self.alphas + self.betas)
        steady_state_gradient = np.concatenate(
                                    [
                                    np.array([[1/denominator]]),
                                    np.full([1, self.p+self.q], self.omega/denominator**2)
                                    ], axis=1
                                )
        
        # recursively calculate gradients
        alphas = np.array(self.alphas).reshape(-1, 1)
        betas = np.array(self.betas).reshape(-1, 1)
        lagged_gradients = np.repeat(steady_state_gradient, repeats=self.p, axis=0)
        variance_gradients = np.empty(Z.shape)
        for row, (z_t, lagged_weights) in enumerate(zip(Z, weights)):
            gradient_t = z_t + betas.T@lagged_gradients + (alphas*lagged_weights).T@lagged_gradients
            variance_gradients[row] = gradient_t
            if self.p > 0:
                lagged_gradients = np.concatenate(
                                        [
                                        gradient_t,
                                        lagged_gradients[:-1],
                                        ], axis=0
                                    ) 
        
        return variance_gradients
    
    def _initialise_parameters(self, y, X, weights):
        '''Fixes initial parameter values to start estimation.'''
        # update means model
        self.means_model.step(y=y, X=None, weights=weights)
        self.epsilons_ = self.means_model.errors(y)
        
        # initialise state equation parameters with ARCH regression (OLS)
        if sum(self.alphas + self.betas)==0:
            X = np.concatenate(
                    [np.ones([len(y-self.q),1]),
                     np.array([np.roll(self.epsilons_, arch)**2 for arch in range(1, self.q+1)]).reshape(-1, 1)
                    ], axis=1)
            y = np.array(self.epsilons_).reshape(-1, 1)**2
            omega_inv = sp.sparse.diags(weights)
            init_ = np.linalg.inv(X.T@omega_inv@X) @ X.T@omega_inv@y
            self.omega = init_[0][0]
            self.alphas = init_[1:].flatten().tolist()
            self.betas = [0]*self.p
            
        # fix intercept
        if self.omega <= 0:
            self.omega = self.means_model.var() * (1-sum(self.alphas+self.betas))
            
    def _e_step(self, y, X, weights):
        '''Updates the state variable given the model parameters.'''
        self.states_ = self.filter_states(y, X=X, weights=weights, warm_start=False, return_errors=False)
        self.state = self.states_[-1]
        
    def _m_step(self, y, X, weights, learning_rate=None):
        '''Updates the model parameters given the state variable.'''
        # update state equation parameters using the scoring method
        W = (1/2**0.5) * self._variance_gradients(y=y, X=X, weights=weights) / np.array(self.states_).reshape(-1, 1)
        weighted_epsilons = weights*self.epsilons_**2 + (1-weights)*self.states_
        v = (1/2**0.5) * (weighted_epsilons/np.array(self.states_)-1).reshape(-1, 1)
        omega_inv = sp.sparse.diags(weights)
        delta_ = (np.linalg.inv(W.T@omega_inv@W) @ W.T@omega_inv@v)
        
        # update state equation parameters
        if learning_rate is None:
            learning_rate = 0.2-0.15/(self.iteration+1)**1 # learning rate converges to 0.2
        state_params_ = model.param_vector + learning_rate*delta_
        if abs(state_params_[1:].sum()) > 1:
            raise ValueError('encountered nonstationary parameters during iteration')
        self.omega = state_params_.squeeze()[0]
        self.alphas = state_params_.squeeze()[1:1+self.q]
        self.betas = state_params_.squeeze()[1+self.q:1+self.q+self.p]
        
        # update means model
        w_ = weights / np.array(self.states_)
        self.means_model.step(y=y, X=None, weights=w_)
        self.epsilons_ = self.means_model.errors(y)
    
    def _step(self, y, X, weights, learning_rate=None):
        '''Performs one estimation step.
        Recalculates the latent variables and the model parameters.
        '''
        # initialise
        if self.iteration == 0:
            self._initialise_parameters(y=y, X=X, weights=weights)
        
        # state filter
        self._e_step(y=y, X=X, weights=weights)
        
        # parameter update
        self._m_step(y=y, X=X, weights=weights, learning_rate=learning_rate)
        
    def score(self, y, X=None, weights=None, warm_start=False):
        '''Returns the log-likelihood of a sample.
        The state variable is filtered given the model paramters to perform the calculations.
        '''        
        # weights
        if weights is None:
            weights = np.ones(y.shape)
        else:
            weights = np.array(weights)
            
        # filter state sequence
        states_ = self.filter_states(y=y, X=X, weights=weights, warm_start=warm_start)

        # score
        score = (weights * np.log(self.means_model.pdf(y, scale=np.array(states_)**0.5))).sum()
        return score