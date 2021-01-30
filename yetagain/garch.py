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
            alphas = alphas.to_list()
        if type(alphas) in [int, float, np.float64]:
            alphas = [alphas]
        self._alphas = np.array(alphas)
        
    @property
    def betas(self):
        '''The state equation AR parameters.'''
        return self._betas
    
    @betas.setter
    def betas(self, betas):
        if type(betas) == np.ndarray:
            betas = betas.to_list()
        if type(betas) in [int, float, np.float64]:
            betas = [betas]
        self._betas = np.array(betas)
        
    @property
    def mu(self):
        return self.means_model.mean()
        
    @property
    def sigma(self):
        return self.means_model.std()
        
    @property
    def state(self):
        return self._state
    
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
    
    def set_steady_state(self):
        '''Set the steady state as the model state.'''
        self.state = self.steady_state
        
    def _construct_lags(self):
        '''Returns arrays of lagged states and errors.'''
        if self.is_fitted:
            states = self.states_[-self.p:]
            epsilons = self.epsilons_[-self.q:]
        else:
            if self.state:
                states = np.append(np.full(self.p-1, self.steady_state), self.state)
            else:
                states = np.full(self.p, self.steady_state)
            epsilons = np.full(self.q, self.steady_state**0.5)
            
        return (states, epsilons)
        
    def draw(self, size=1, return_variances=False):
        '''Draw a random sequence of specified length.'''
        # set up
        states, epsilons = self._construct_lags()
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
        states, epsilons = self._construct_lags()
        
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