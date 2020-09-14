import numpy as np
from scipy.stats import entropy
import warnings

class MarkovChain:
    
    '''
    A MarkovChain
    '''
    
    def __init__(self, transition_matrix=None, state_vector=None):
        self.transition_matrix = transition_matrix
        self.state_vector = state_vector

    @property
    def transition_matrix(self):
        
        '''
        The Markov state transition probability matrix.
        Needs to be square.
        '''
        
        return self._transition_matrix
        
    @transition_matrix.setter
    def transition_matrix(self, transition_matrix):
        if transition_matrix is not None:
            transition_matrix = np.array(transition_matrix)
            assert transition_matrix.shape[0] == transition_matrix.shape[1], \
                'transition matrix needs to be square'
            assert all(transition_matrix.sum(axis=1).round(8) == 1), \
                'transition matrix rows need to sum to one'
            if hasattr(self, 'state_vector') and self.state_vector is not None:
                assert transition_matrix.shape[0] == self.state_vector.shape[1], \
                    'state vector dimension mismatch'
            self._transition_matrix = transition_matrix
        else:
            self._transition_matrix = None
        
    
    @property
    def state_vector(self):
        
        '''
        The current state vector.
        '''
        
        return self._state_vector
    
    @state_vector.setter
    def state_vector(self, state_vector):
        if state_vector is not None:
            state_vector = np.array(state_vector).reshape(1,-1)
            assert state_vector.sum(axis=1).round(8) == 1, \
                'state vector needs to sum to one'
            assert (state_vector>=0).all() and (state_vector<=1).all(), \
                'probabilites need to be bounded between zero and one'
            if hasattr(self, 'transition_matrix') and self.transition_matrix is not None:
                assert state_vector.shape[1] == self.transition_matrix.shape[0], \
                    'transition matrix dimension mismatch'
            self._state_vector = state_vector
        else:
            self._state_vector = None
    

    def steady_state(self, set_state=False):
        
        '''
        Returns the steady state probabilities of the Markov chain.
        If set_state=True, MarkovChain object is modified in place.
        '''
        
        dim = np.array(self.transition_matrix).shape[0]
        q = np.c_[(self.transition_matrix-np.eye(dim)),np.ones(dim)]
        QTQ = np.dot(q, q.T)
        steady_state = np.linalg.solve(QTQ,np.ones(dim))
        if set_state:
            self.state_vector = steady_state
        else:
            return steady_state
        
        
    def expected_durations(self):
        
        '''
        Returns the expected state durations of the MarkovChain object.
        '''
        
        expected_durations = (np.ones(self.n_states)-np.diag(self.transition_matrix))**-1
        return expected_durations
    
    
    @property
    def n_states(self):
        
        '''
        Returns the number of states of the MarkovChain object.
        '''
        
        if self.state_vector is not None:
            return self.state_vector.shape[1]
        elif self.transition_matrix is not None:
            return self.transition_matrix.shape[0]
        else:
            raise TypeError('MarkovChain object empty')


    def iterate(self, steps=1, set_state=False):
        
        '''
        Iterates the MarkovChain object the specified number of steps.
        steps should be a positive integer.
        (negative steps work, but tend to break when going before the initial state)
        If set_state=True, MarkovChain object is modified in place.
        '''
        
        new_state = np.dot(self.state_vector, np.linalg.matrix_power(self.transition_matrix, steps))
        
        # ensure total probability is 1
        # if new_state.sum() != 1:
        #     new_state = new_state.round(8)/new_state.round(8).sum()
        #     warnings.warn('state vector probabilities rounded to 8 digits')
        
        if set_state:
            self.state_vector = new_state
        else:
            new_mc = MarkovChain(transition_matrix=self.transition_matrix,
                                 state_vector=new_state)
            return new_mc
        
        
    def forecast(self, horizons=[1]):
        
        '''
        Returns forecasted state probabilities for a set of horizons.
        horizons needs to be an iterable.
        '''
        
        horizons_states = np.array([]).reshape(0, self.n_states)
        for horizon in horizons:
            pi_ = np.dot(self.state_vector, np.linalg.matrix_power(self.transition_matrix, horizon))
            horizons_states = np.concatenate([horizons_states, pi_.reshape(1, self.n_states)], axis=0)
        return horizons_states
    

    def rvs(self, size=0, random_state=1):
        
        '''
        Draws a random sample sequence from the MarkovChain object.
        t_steps is the number of time steps forward to be drawn.
        If t_steps is zero, only the current state is drawn.
        '''
        
        sample = np.random.choice(self.n_states, size=1, p=self.state_vector.squeeze())
        for t in range(1, size+1):
            sample = np.concatenate([sample, \
                        np.random.choice(self.n_states, size=1, p=self.transition_matrix[sample[-1]])])
        return sample
    
    
    def entropy(self, horizons=None):
        
        '''
        Calculate Shannon's entropy of the n state probabilities based on logarithms with base n.
        '''
        
        if horizons is None:
            state_entropy = entropy(self.state_vector.squeeze(), base=self.n_states)
        
        else:
            horizon_states = self.forecast(horizons)
            state_entropy = []
            for horizon in horizon_states:
                state_entropy += [entropy(horizon.squeeze(), base=self.n_states)]
            
        return np.array(state_entropy)


    def __repr__(self):
        return str(self)


    def __str__(self):
        '''
        Returns a summarizing string
        '''

        string = 'MarkovChain(\n'
        string += 'P=\n{},\n'.format(self.transition_matrix.__str__())
        string += 'pi=\n{}\n)'.format(self.state_vector.__str__())
        return string
    