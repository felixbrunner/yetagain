import pandas as pd
import numpy as np
import warnings
import copy

# from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
# from hmmlearn.hmm import GaussianHMM

from yetagain.dists import MixtureDistribution, ProductDistribution
from yetagain.markov import MarkovChain
from yetagain.models import ModelMixin, NormalModel, MixtureModel
from yetagain.estimation import EstimationMixin


class HiddenMarkovModel(ModelMixin, EstimationMixin, MarkovChain):#, MixtureModel, ):
    '''Hidden Markov Model class.'''
    
    def __init__(self, emission_models=[NormalModel(), NormalModel()],
                       transition_matrix=None,
                       state=None):
        self.emission_models = emission_models
        self.transition_matrix = transition_matrix
        self.state = state
        ModelMixin.__init__(self)
        
    @property
    def emission_models(self):
        '''A tuple of emission models associated with the Markov states.'''
        return self._emission_models
    
    @emission_models.setter
    def emission_models(self, emission_models):
        for emission_model in emission_models:
            assert isinstance(emission_model, ModelMixin), \
                'unknown emission_model type'
        if hasattr(self, 'transition_matrix'):
            assert len(emission_models) == len(self.transition_matrix), \
                'number of emission models inconsitent'
        self._emission_models = emission_models

    @property
    def k_models(self):
        '''Return the number of component models.'''
        return len(self.emission_models)
    
    @property
    def transition_matrix(self):
        '''The Markov state transition probability matrix.
        Needs to be square.
        '''
        return self._transition_matrix

    @transition_matrix.setter
    def transition_matrix(self, transition_matrix):
        if transition_matrix is None:
            k = self.k_models
            transition_matrix = np.full([k, k], 1/k)
            # transition_matrix = 0.9 * np.eye(k) + np.full([k, k], 0.1/k)
        else:
            assert len(transition_matrix) == len(self.emission_models), \
                'transition matrix and number of emission models inconsistent'
        MarkovChain.transition_matrix.fset(self, transition_matrix)

    @property
    def components(self):
        '''The mixture distribution components.'''
        if self.state is not None:
            weights = self.state.squeeze()
        else:
            weights = self.steady_state().squeeze()
        components = [(component.distribution, float(weight)) 
                            for (component, weight) 
                            in zip(self.emission_models, weights)]
        return components

    @property
    def distribution(self):
        '''Extracts and returns a MixtureDistribution object
        with the current state vector as weights.
        '''
        mix = MixtureDistribution(components=self.components)
        return mix
    
    @property
    def markov_chain(self):
        '''Extracts and returns a MarkovChain object
        with the transition matrix and state vector as parameters.
        '''     
        mc = MarkovChain(transition_matrix=self.transition_matrix,
                         state=self.state)
        return mc

    def iterate(self, steps=1, set_state=False):
        '''Iterates the model the specified number of steps.
        steps should be a positive integer.
        (negative steps work, but tend to break when going before the initial state)
        If set_state=True, HiddenMarkovModel object is modified in place.
        '''
        new_state = self.markov_chain.iterate(steps=steps).state
        new_models = [model.iterate(steps=steps) for model in self.emission_models]
        
        if set_state:
            self.state = new_state
            self.emission_models = new_models
        else:
            new_hmm = HiddenMarkovModel(emission_models=new_models,
                                        transition_matrix=self.transition_matrix,
                                        state=new_state)
            return new_hmm

    @property
    def steady_state(self):
        '''Returns the steady state probabilities of the HMMs Markov chain.'''
        return MarkovChain(transition_matrix=self.transition_matrix).steady_state

    def set_to_steady_state(self):
        '''Set the steady state as the HiddenMarkovModel object state.
        Additionally, all emission_models are also set to their steady_states.
        '''
        for model in self.emission_models:
            if hasattr(model, 'state'):
                model.set_to_steady_state()
        self.state = self.steady_state
    
    @property
    def params(self):
        # collect hmm state equation parameters
        params = {'transition_matrix': self.transition_matrix,
                 }
        
        # add emission_models params to dict
        for i, emission_model in enumerate(self.emission_models):
            model_params = emission_model.params
            model_params = {'state{}_'.format(i+1)+k:v
                            for k, v in model_params.items()}
            params.update(model_params)
        return params
    
    @params.setter
    def params(self, params):
        for k, v in params.items():
            setattr(self, k, v)
        warnings.warn('parameter setting not tested')
    
    def __str__(self):
        '''Returns a summarizing string.'''
        string = 'HiddenMarkovModel(\n'
        string += 'transition_matrix=\n{},\n'.format(self.transition_matrix.__str__())
        string += 'state=\n{},\nemission_models=\n('.format(self.state.__str__())
        for model in self.emission_models:
            string += '{},\n'.format(model.__str__())
        string += '))'
        return string

    def draw(self, size=1, return_distributions=False, return_states=False):
        '''Draw a random sequence of specified length.'''
        states = self.markov_chain.draw(size=size)
        emission_model_draws = [model.draw(size=size, return_distributions=True)
                        for model in self.emission_models]

        sample = [emission_model_draws[state][0][i] 
                    for (i, state) in enumerate(states)]
        distributions = [emission_model_draws[state][1][i] 
                    for (i, state) in enumerate(states)]
        
        if size is 1:
            states = states[0]
            sample = sample[0]
            distributions = distributions[0]

        if return_distributions:
            if return_states:
                return (sample, distributions, states)
            else:
                return (sample, distributions)
        else:
            if return_states:
                return (sample, states)
            else:
                return sample

    def _emission_likelihoods(self, y, X=None, weights=None, filter=False):
        '''Return a list of lists with emission likelihoods for each
        observation for each model.
        '''

        # overall observation importance weights
        importance_weights = self._prepare_importance_weights(y=y, weights=weights)

        # state weights
        if hasattr(self, 'states_'):
            state_odds = np.array(self.states_)
        else:
            state_odds = np.kron(np.ones([len(y), 1]), self.state)
        state_odds = state_odds.T

        # evaluate model likelihoods
        B = []
        for model, model_odds  in zip(self.emission_models, state_odds):
            weights = model_odds * importance_weights
            model_likelihoods = model.likelihood(y=y, X=X, weights=weights, filter=filter)
            B += [model_likelihoods]

        return np.array(B).T

    def _start_recursion(self, warm_start=True):
        '''Returns an array of lagged states.'''
        # warm start case
        if warm_start:
            # if model is fitted or has no state memory
            if self.is_fitted or not hasattr(self, 'states_'): 
                initial_state = self.state
            # if model has state history
            else:
                initial_state = self.states_[0]
                
        # start from steady state case
        else:
            initial_state = self.steady_state

        return initial_state.reshape(1, -1)

    def _forward_pass(self, A, B, pi, weights):
        '''Returns filtered probabilities together with each scaling factor.'''
        # initialise forward pass with first observation
        alpha_0 = pi * B[0]
        c_0 = 1/alpha_0.sum()
        
        # save values & scaling factor
        Alpha = alpha_0*c_0
        C = [c_0]
        
        # iterate
        for b_t, w_t in zip(B[1:], weights):
            # calculate
            alpha_t = (w_t * b_t * Alpha[-1] @ A
                        + (1-w_t) * Alpha[-1] @ A) \
                        .reshape(1, -1)
            c_t = 1/alpha_t.sum()

            # save
            Alpha = np.concatenate((Alpha, alpha_t*c_t), axis=0)
            C += [c_t]
        
        # outputs
        C = np.array(C).reshape(-1, 1)
        return Alpha, C

    def filter(self, y, X=None, weights=None, warm_start=False, return_scaling=False):
        '''Filters the hidden markov states from a given series.
        Performs a forward pass.
        '''
        # prepare inputs
        weights = self._prepare_importance_weights(y=y, weights=weights)
        B = self._emission_likelihoods(y=y, X=X, weights=weights, filter=False)
        A = self.transition_matrix
        pi = self._start_recursion(warm_start=warm_start)

        # perform forward pass
        filtered_states, scaling_factors = self._forward_pass(A, B, pi, weights)
        
        return filtered_states

    def _backward_pass(self, A, B, pi, C, weights):
        '''Returns backwards filtered probabilities.'''
        # initialise backward pass as one
        beta_T = np.ones(pi.shape)
        
        # save values & scaling factor
        Beta = beta_T*C[-1]

        # iterate
        for b_t, c_t, w_t in zip(B[:0:-1], C[-2::-1], weights[:0:-1]):
            # calculate
            beta_t = (w_t * b_t * Beta[0] @ A.T
                        + (1-w_t) * Beta[0] @ A.T) \
                            .reshape(1, -1)

            # save
            Beta = np.concatenate((beta_t*c_t, Beta), axis=0)
            
        return Beta

    def _emission_odds(self, Alpha, Beta):
        '''Returns odds for each observation to be emitted by each component model.'''
        total = Alpha * Beta
        Gamma = total / total.sum(axis=1).reshape(-1, 1)
        return Gamma
    
    def smoothen(self, y, X=None, weights=None, warm_start=False, return_filtered=False):
        '''Smoothens the hidden markov states from a given series.
        Performs a forward pass, then a backward pass.
        Return of filtered states is optional.
        '''
        # prepare inputs
        weights = self._prepare_importance_weights(y=y, weights=weights)
        B = self._emission_likelihoods(y=y, X=X, weights=weights, filter=False)
        P = self.transition_matrix
        pi = self._start_recursion(warm_start=warm_start)

        # perform smoothing pass
        filtered_states, scaling_factors = self._forward_pass(P, B, pi, weights)
        backfiltered_states = self._backward_pass(P, B, pi, scaling_factors, weights)
        smoothened_states = self._emission_odds(filtered_states, backfiltered_states)
        
        if return_filtered:
            return smoothened_states, filtered_states
        else:
            return smoothened_states

    def predict(self, y, X=None, weights=None, filter=True, warm_start=False, method='distribution'):
        '''Returns an array with predictions for an input sample.'''
        # predict emission_models
        emission_model_densities = []
        for model in self.emission_models:
            if hasattr(model, 'filter'):
                emission_model_densities += [model.predict(y=y, X=X, weights=weights,
                                                           filter=filter,
                                                           warm_start=warm_start)]
            else:
                emission_model_densities += [model.predict(y=y, X=X)]
        emission_model_densities = [list(i) for i in zip(*emission_model_densities)]

        # states
        if filter:
            _states = self.smoothen(y=y, X=X, weights=weights)
        else:
            _states = self.states_

        # compile mixture distributions
        distributions = []
        for d_t, p_t in zip(emission_model_densities, _states):
            distributions += [MixtureDistribution(components=[(d, w) for d, w in zip(d_t, p_t)])]

        # output
        if method == 'distribution' or method == None:
            return distributions
        elif method == 'mean':
            return [dist.mean for dist in distributions]
        elif method == 'mode':
            return [dist.mode for dist in distributions]
        elif method == 'median':
            return [dist.median for dist in distributions]
        elif method == 'var':
            return [dist.var for dist in distributions]
        elif method == 'std':
            return [dist.std for dist in distributions]
        else:
            raise NotImplementedError('Prediction method not implemented')

    def _e_step(self, y, X, weights):
        '''Updates the state variable given the model parameters.'''
        # prepare inputs
        B = self._emission_likelihoods(y=y, X=X, weights=weights, filter=False)
        P = self.transition_matrix
        pi = self._start_recursion(warm_start=False)

        # perform smoothing pass
        Alpha, C = self._forward_pass(P, B, pi, weights)
        Beta = self._backward_pass(P, B, pi, C, weights)
        Gamma = self._emission_odds(Alpha, Beta)
        Xi = self._transition_odds(B, Alpha, Beta)
        
        # update model states
        self.states_ = Gamma
        self.state = self.states_[-1]

        # return emission and transition odds
        return (Gamma, Xi)

    def _transition_odds(self, B, Alpha, Beta):
        '''Returns the odds of each observation to have transitioned from each
        state to each other state including itself.
        '''
        Alpha_block = np.kron(Alpha[:-1], np.ones(self.k_models))
        B_Beta_block = np.kron(np.ones(self.k_models), B[1:]*Beta[1:])
        total = Alpha_block * \
                    B_Beta_block * \
                    self.transition_matrix.reshape(1, -1)
        Xi = total / total.sum(axis=1).reshape(-1, 1)
        return Xi

    def _update_transition_matrix(self, Gamma, Xi):
        '''Sets the HMM transition matrix to an updated version.'''
        numerator = Xi.sum(axis=0)
        denominator = np.kron(Gamma[:-1], np.ones(self.k_models)).sum(axis=0)
        self.transition_matrix = (numerator/denominator) \
                                    .reshape(self.transition_matrix.shape)

    def _update_emission_models(self, y, X, weights, Gamma,
                                threshold=1e-2, max_iter=10):
        '''Updates the emission model parameters.'''
        for model, Gamma_i in zip(self.emission_models, Gamma.T):
            model.fit(y=y, X=X, weights=weights.flatten()*Gamma_i,
                      threshold=threshold, max_iter=max_iter)

    def _m_step(self, y, X, weights, Gamma, Xi):
        '''Performs all steps of the M-step and returns temporary variables.
        All component models are reestimated and parameters updated.
        Part of Baum-Welch algorithm.
        '''
        self._update_transition_matrix(Gamma, Xi)
        self._update_emission_models(y, X, weights, Gamma)

    def _initialise_parameters(self, y, X, weights):
        '''Fixes initial parameter values to start estimation.'''
        
        # initialise transition matrix
        # self.transition_matrix *= np.random.lognormal(mean=0, sigma=0.1,
        #                                 size=self.transition_matrix.shape)

        k = self.k_models
        if self.steady_state.std() < 0.1/k:
            self.transition_matrix = 0.7 * np.eye(k) \
                                        + 0.1 * np.full([k, k], 1/k) \
                                        + 0.2 * self.transition_matrix
            
        # initialise emission models to prevent identical models
        for model in self.emission_models:
            if not model.is_fitted:
                model.fit(y=y, X=X, weights=weights, max_iter=10)
            disturbed_params = model.params.copy()
            for key, param in model.params.items():
                if type(param) != list:
                    disturbed_params[key] = param \
                                    * np.random.lognormal(mean=0, sigma=0.1)
            model.params = disturbed_params

    def _step(self, y, X, weights):
        '''Performs one estimation step.
        Recalculates the latent variable probabilities and the model parameters.
        '''
        # initialise
        if self.iteration == 0:
            self._initialise_parameters(y=y, X=X, weights=weights)

        # state filter
        Gamma, Xi = self._e_step(y=y, X=X, weights=weights)
        
        # parameter update
        self._m_step(y=y, X=X, weights=weights, Gamma=Gamma, Xi=Xi)