import pandas as pd
import numpy as np
import warnings
import copy

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from hmmlearn.hmm import GaussianHMM

from yetagain.dists import MixtureDistribution, ProductDistribution
from yetagain.markov import MarkovChain
from yetagain.models import MixtureModel


class HiddenMarkovModel(MixtureModel, MarkovChain):
    
    '''
    Hidden Markov Model class
    '''
    
    def __init__(self, emission_models=None, transition_matrix=None, state_vector=None):
        self.transition_matrix = transition_matrix
        self.state_vector = state_vector
        self.emission_models = emission_models
        self.is_fitted = False
        
        
    @property
    def emission_models(self):
        
        '''
        A tuple of emission models associated with the Markov states.
        '''
        
        return self._emission_models
    
    @emission_models.setter
    def emission_models(self, emission_models):
        if emission_models is not None:
            emission_models = tuple(emission_models)
            if self.transition_matrix is not None:
                assert len(emission_models) == self.transition_matrix.shape[0], \
                    'number of emission models inconsitent'
            elif self.state_vector is not None:
                assert len(emission_models) == self.state_vector.shape[1], \
                    'number of emission models inconsitent'
            self._emission_models = emission_models
            
        else:
            self._emission_models = None

    
    @property
    def n_components(self):

        '''
        Return the number of component models.
        '''

        if self.emission_models is not None:
            return len(self.emission_models)
        else:
            return self.markov_chain.n_states
            
            
    @property
    def components(self):
        
        '''
        The mixture distribution components.
        '''
        
        assert self.state_vector is not None, \
            'state vector not set'
        weights = self.state_vector.squeeze()
        components = [(component.distribution, float(weight)) for (component, weight) in zip(self.emission_models, weights)]
        return components
        
    
    def fit(self, Y, method='baumwelch', **kwargs):
        
        '''
        Fits the model to a sample of data.
        '''
        
        if method == 'baumwelch':
            self = self._estimate_baum_welch(Y, **kwargs)
            self.is_fitted = True
        else:
            raise NotImplementedError('fitting algorithm not implemented')
    
    
    def _estimate_baum_welch(self, Y, max_iter=100, threshold=1e-6, return_fit=False):
        
        '''
        Performs parameter estimation with the Baum-Welch algorithm.
        Returns a fitted model.
        Returns the the fitted model and parameters of the estimation if return_fit=True.
        '''
        
        # initialise
        Y = np.array(Y)
        A_, pi_, models_ = self._initialise_baum_welch()
        self._check_baum_welch_inputs(A_, pi_, models_)
        score_, B_ = self._score(Y, models_, pi_)
        
        # store
        iteration = 0
        scores = {iteration: score_}
        
        while iteration < max_iter:
            iteration += 1
            Alpha, Gamma, Xi = self._do_e_step(Y, A_, B_, pi_)          
            A_, models_, pi_ = self._do_m_step(Y, models_, Gamma, Xi)
            score_, B_ = self._score(Y, models_, Gamma)
            scores[iteration] = score_
            
            if abs(scores[iteration]-scores[iteration-1]) < threshold:
                converged = True
                break
        else:
            converged = False
            warnings.warn('maximum number of iterations reached')
                
        self._update_attributes(A_, models_, Alpha)
        
        if return_fit:
            fit = {'converged': converged,
                   'iterations': iteration,
                   'scores': scores,
                   'pdfs': B_,
                   'smoothened_probabilities': Gamma,
                   'filtered_probabilities': Alpha}
            
            return self, fit
        else:
            return self
    
    
    def _initialise_baum_welch(self):
        
        '''
        Returns initial values for the Baum-Welch algorithm.
        Part of Baum-Welch algorithm.
        '''
        
        assert self.emission_models is not None, \
            'emission models not specified'
            
        if self.state_vector is None:
            if self.transition_matrix is None:
                self.state_vector = np.full([1, self.n_components], 1/self.n_components)
            else:
                self.steady_state(set_state=True)
        if self.transition_matrix is None:
            self.transition_matrix = np.full([self.n_components, self.n_components], 1/self.n_components)
        
        A = self.transition_matrix
        models = self.emission_models
        pi = self.state_vector
        return A, pi, models
    
    
    def _check_baum_welch_inputs(self, A, pi, models):
        
        '''
        Checks the dimension match of algorithm inputs.
        Part of Baum-Welch algorithm.
        '''

        assert len(models) == A.shape[0] == A.shape[1] == pi.shape[1], \
            'dimension mismatch'
    
    
    def _score(self, Y, emission_models, Gamma):
        
        '''
        Returns the overall model score and component model pdf values for each observation.
        Part of Baum-Welch algorithm.
        '''
        
        B = self._evaluate_emission_models(Y, emission_models)
        score = np.log((B * Gamma).sum(axis=1)).sum(axis=0)
        return score, B
    
    
    def _evaluate_emission_models(self, Y, emission_models):
        
        '''
        Returns component model pdf values for each observation.
        Part of Baum-Welch algorithm.
        '''
        
        B = np.concatenate([model.pdf(Y).reshape(-1, 1) for model in emission_models], axis=1)
        return B
    
    
    def _do_e_step(self, Y, A, B, pi):
        
        '''
        Performs all steps of the E-step and returns temporary variables.
        All data state probabilities are updated based on the existing component models.
        Part of Baum-Welch algorithm.
        '''
        
        Alpha, C = self._forward_pass(A, B, pi)
        Beta = self._backward_pass(A, B, pi, C)
        Gamma = self._emission_odds(Alpha, Beta)
        Xi = self._transition_odds(A, B, Alpha, Beta)
        return Alpha, Gamma, Xi
    
    
    def _forward_pass(self, A, B, pi):
        
        '''
        Returns filtered probabilities of the data together with each scaling factor.
        Part of Baum-Welch algorithm.
        '''
        
        # initialise forward pass with first observation
        alpha_0 = pi * B[0]
        c_0 = 1/alpha_0.sum()
        
        # save values & scaling factor
        Alpha = alpha_0*c_0
        C = [c_0]
        
        # iterate
        for b_t in B[1:]:
            # calculate
            alpha_t = (b_t * Alpha[-1] @ A).reshape(1, -1)
            c_t = 1/alpha_t.sum()
            
            # save
            Alpha = np.concatenate((Alpha, alpha_t*c_t), axis=0)
            C += [c_t]
            
        C = np.array(C).reshape(-1, 1)
        return Alpha, C

    
    def _backward_pass(self, A, B, pi, C):
        
        '''
        Returns smoothened probabilities of the data.
        Part of Baum-Welch algorithm.
        '''
        
        # initialise backward pass as one
        beta_T = np.ones(pi.shape)
        
        # save values & scaling factor
        Beta = beta_T*C[-1]
        
        # iterate
        for b_t, c_t in zip(B[:0:-1],C[len(C)-2::-1]):
            # calculate
            beta_t = (b_t * Beta[0] @ A.T).reshape(1, -1)
            
            # save
            Beta = np.concatenate((beta_t*c_t, Beta), axis=0)
            
        return Beta
    
    
    def _emission_odds(self, Alpha, Beta):
        
        '''
        Returns odds for each observation to be emitted by each component model.
        Part of Baum-Welch algorithm.
        '''
        
        total = Alpha * Beta
        Gamma = total/total.sum(axis=1).reshape(-1, 1)
        return Gamma
    
    
    def _transition_odds(self, A, B, Alpha, Beta):
        
        '''
        Returns the odds of each state to transition from each state to each state.
        Part of Baum-Welch algorithm.
        '''
        
        Alpha_block = np.kron(Alpha[:-1], np.ones(A.shape[0]))
        B_Beta_block = np.kron(np.ones(A.shape[0]), B[1:]*Beta[1:])
        total = Alpha_block * B_Beta_block * A.reshape(1, -1)
        Xi = total/total.sum(axis=1).reshape(-1, 1)
        return Xi
    
    
    def _do_m_step(self, Y, models, Gamma, Xi):
        
        '''
        Performs all steps of the M-step and returns temporary variables.
        All component models are reestimated and parameters updated.
        Part of Baum-Welch algorithm.
        '''
        
        A_ = self._update_transition_matrix(Gamma, Xi)
        models_ = self._update_model_parameters(Y, models, Gamma)
        pi_ = self._update_initial_state(Gamma)
        return A_, models_, pi_
    

    def _update_transition_matrix(self, Gamma, Xi):
        
        '''
        Returns an updated Markov transition matrix.
        Part of Baum-Welch algorithm.
        '''
        
        numerator = Xi.sum(axis=0)
        denominator = np.kron(Gamma[:-1], np.ones(Gamma.shape[1])).sum(axis=0)
        A_ = (numerator/denominator).reshape(Gamma.shape[1], Gamma.shape[1])
        return A_
    
    
    def _update_model_parameters(self, Y, emission_models, Gamma):
        
        '''
        Returns updated emission models.
        Part of Baum-Welch algorithm.
        '''
        
        models_ = []
        for model, weights in zip(emission_models, Gamma.T):
            model.fit(Y, weights)
            models_ += [model]
        return tuple(models_)
    
    
    def _update_initial_state(self, Gamma):
        
        '''
        Returns updated initial state probabilities.
        Part of Baum-Welch algorithm.
        '''
        
        return Gamma[0].reshape(1, -1)
    
    
    def _update_attributes(self, A_, models_, Alpha):
        
        '''
        Updates the HMM attributes in place.
        Part of Baum-Welch algorithm.
        '''
        
        # ensure total transition probabilities are 1
        # if (A_.sum(axis=1) != 1).any():
        #     A_ = A_.round(6)/A_.round(6).sum(axis=1)
        #     warnings.warn('Transition matrix rounded to 6 decimal places')
        self.transition_matrix = A_
        
        self.emission_models = models_
        
        state_vector = Alpha[-1]
        # ensure total state probability is 1
        # if state_vector.sum() != 1:
        #     state_vector = state_vector.round(8)/state_vector.round(8).sum()
        #     warnings.warn('State vector rounded to 8 decimal places')
        self.state_vector = state_vector
    
    
    @property
    def distribution(self):
        
        '''
        Extracts and returns a MixtureDistribution object
        with the current state vector as weights.
        '''
        
        mix = MixtureDistribution(components=self.components)
        return mix
    
    
    @property
    def mixture_distribution(self):
        
        '''
        Extracts and returns a MixtureDistribution object
        with the current state vector as weights.
        '''
        
        return self.distribution
    
    @property
    def markov_chain(self):
        
        '''
        Extracts and returns a MarkovChain object
        with the transition matrix and state vector as parameters.
        '''
                
        mc = MarkovChain(transition_matrix=self.transition_matrix, state_vector=self.state_vector)
        return mc


    def iterate(self, steps=1, set_state=False):
        
        '''
        Iterates the model the specified number of steps.
        steps should be a positive integer.
        (negative steps work, but tend to break when going before the initial state)
        If set_state=True, HiddenMarkovModel object is modified in place.
        '''
        
        new_state = self.markov_chain.iterate(steps=steps).state_vector
        new_models = [model.iterate(steps=steps) for model in self.emission_models]
        
        if set_state:
            self.state_vector = new_state
            self.emission_models = new_models
        else:
            new_hmm = HiddenMarkovModel(emission_models=new_models,
                                        transition_matrix=self.transition_matrix,
                                        state_vector=new_state)
            return new_hmm


    def product_distribution(self, horizon=1, include_current=False):

        '''
        
        '''

        assert horizon > 0 and type(horizon) == int, \
            'horizon needs to be a positive integer'

        factors = []
        if include_current:
            factors += [self.distribution]

        for h in range(1, horizon+1):
            factors += [self.iterate(h).distribution]

        product_distribution = ProductDistribution(factors=factors)
        return product_distribution


    def copy(self):

        '''
        Returns a deep copy with new memory address.
        '''

        new_hmm = copy.deepcopy(self)
        return new_hmm

    
    def __str__(self):

        '''
        Returns a summarizing string
        '''

        string = 'HiddenMarkovModel(\n'
        string += 'P=\n{},\n'.format(self.transition_matrix.__str__())
        string += 'pi=\n{},\nM=\n('.format(self.state_vector.__str__())
        for model in self.emission_models:
            string += '{},\n'.format(model.distribution.__str__())
        string += '))'
        return string


    def rvs(self, size=1, return_states=False):
    
        '''
        Draw a random sequence of specified length.
        '''
    
        states = self.markov_chain.rvs(size=size)
        sample = np.fromiter((self.components[i][0].rvs() for i in states), dtype=np.float64)
        
        if size is 1:
            sample = sample[0]
            
        if return_states:
            return (sample, states)
        else:
            return sample


# #### LEGACY CODE ####

# class HMM(MarkovChain):
#     def __init__(self, emission_models=(), transition_matrix=None, start_probas=None, switch_var=True, switch_const=True, k=None):
        
#         '''
        
#         '''

#         self.emission_models = emission_models
#         self.transition_matrix = transition_matrix
#         self.start_probas = start_probas
        
#         self.switch_var = switch_var
#         self.switch_const = switch_const
#         self.k = k
        
#         self.params_ = None
#         self.se_ = None
#         self.tstats_ = None

#         self.metrics_ = None
#         self.smooth_prob_ = None
#         self.filt_prob_ = None


#     def fit(self, y, package='baumwelch', start_params=None, iter=100, **kwargs):
        
#         '''
#         Fits the Gaussian HMM to the series y.
#         '''
        
#         assert package in ['statsmodels', 'hmmlearn', 'baumwelch'], 'package unknown'
        
#         if package == 'statsmodels':            
#             from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
            
#             #if start_params is None:
#                 #start_params = np.random.randn(self.k+self.k**2)*0.01
#                 #m = y.mean()
#                 #s = y.std()
#                 #v = y.var()
#                 #start_params = np.full(self.**2-self.k, 1/self.k).tolist()\
#                 #                        +(np.random.randn(self.k)*s/2+m).tolist()\
#                 #                        +(np.random.randn(self.k)*v+s).tolist()
#             model = MarkovRegression(endog=y, switching_variance=self.switch_var, switching_trend=self.switch_const, k_regimes=self.k)\
#                                 .fit(start_params=start_params, maxiter=iter, **kwargs)
#             self.params_ = model.params
#             self.se_ = model.bse
#             self.tstats_ = model.tvalues
#             self.metrics_ = pd.Series({'llf': model.llf, 'aic': model.aic, 'bic': model.bic,})
#             self.smooth_prob_ = model.smoothed_marginal_probabilities
#             self.filt_prob_ = model.filtered_marginal_probabilities
        
#         if package == 'hmmlearn':
#             from hmmlearn.hmm import GaussianHMM
            
#             assert self.switch_var is True and self.switch_const is True, 'only implemented for fully parametrised components'
#             t_index = y.index
#             y = np.expand_dims(y.values, axis=1)
#             model = GaussianHMM(n_components=self.k, n_iter=iter, **kwargs).fit(y)
#             trans_probas = model.transmat_.T.reshape(self.k**2,1)[:self.k**2-self.k]
#             states = np.arange(self.k)
#             p_index=[f'p[{j}->{i}]' for i in states[:-1] for j in states]\
#                         +[f'const[{i}]' for i in states]\
#                         +[f'sigma2[{i}]' for i in states]
#             self.params_ = pd.Series(np.concatenate((trans_probas, model.means_, model.covars_.squeeze(axis=1))).squeeze(), index=p_index)
#             llf = model.score(y)
#             self.metrics_ = pd.Series({'llf': llf,
#                                        'aic': 2*len(self.params_)-2*llf,
#                                        'bic': len(self.params_)*np.log(len(y))-2*llf})
#             self.smooth_prob_ = pd.DataFrame(model.predict_proba(y), index=t_index)

#         if package == 'baumwelch':
#             self = self._estimate_baum_welch(np.array(y), max_iter=iter, **kwargs)

#         return self
    
    
#     @property
#     def estimates_(self):
#         estimates = pd.DataFrame({'estimate': self.params_,
#                                   's.e.': self.se_,
#                                   't-stat': self.tstats_})
#         return estimates
    

#     @property
#     def transition_matrix_(self):
#         k = self.k
#         trans_matrix = np.matrix(self.params_[:k**2-k].values.reshape(k-1, k).T)
#         trans_matrix = np.append(trans_matrix, 1-trans_matrix.sum(axis=1), axis=1)
#         return trans_matrix



#     def get_mixture_distribution(self, state='steady_state'):
#         if state == 'steady_state':
#             probas = self.steady_state_
#         elif state == 'latest':
#             probas = self.filt_prob_.iloc[-1]
#         else:
#             assert len(state) == self.k, 'wrong number of state probabilities'
#             probas = state

#         components = [(self.params_[f'const[{i}]'], self.params_[f'sigma2[{i}]']**0.5, probas[i]) for i in range(self.k)]
#         mix = GaussianMixtureDistribution(components=components)
#         return mix


#     def filtered_moments(self):
#         filt_mom = pd.DataFrame(index=self.filt_prob_.index, columns=['mean','var','skew','kurt','entropy'])
#         for date, probas in self.filt_prob_.iterrows():
#             mix = self.get_mixture_distribution(state=probas.values)
#             filt_mom.loc[date] = [*mix.mvsk(), mix.entropy()]

#         return filt_mom


#     def smoothened_moments(self):
#         smooth_mom = pd.DataFrame(index=self.smooth_prob_.index, columns=['mean','var','skew','kurt','entropy'])
#         for date, probas in self.smooth_prob_.iterrows():
#             mix = self.get_mixture_distribution(state=probas.values)
#             smooth_mom.loc[date] = [*mix.mvsk(), mix.entropy()]
#         return smooth_mom