import numpy as np
from scipy.stats import entropy
import warnings

class MarkovChain:
    '''A MarkovChain'''
    
    def __init__(self, transition_matrix=np.full([2, 2], 1/2), state=None):
        self.transition_matrix = transition_matrix
        self.state = state

    @property
    def transition_matrix(self):
        '''The Markov state transition probability matrix.
        Needs to be square.
        '''
        return self._transition_matrix
        
    @transition_matrix.setter
    def transition_matrix(self, transition_matrix):
        transition_matrix = np.array(transition_matrix)
        
        # make sure matrix is square
        assert transition_matrix.shape[0] == transition_matrix.shape[1], \
            'transition matrix needs to be square'

        # check if matrix is transition matrix (rows sum to one)
        if any(transition_matrix.sum(axis=1) != 1):
            transition_matrix /= transition_matrix.sum(axis=1).reshape(-1, 1)
            warnings.warn('transition probabilities transformed into probabilities')
        
        # check if dimensions match
        if hasattr(self, 'state') and self.state is not None:
            assert len(transition_matrix) == self.state.shape[1], \
                    'state vector dimension mismatch'
        
        self._transition_matrix = transition_matrix
    
    @property
    def state(self):
        '''The current state vector.'''
        return self._state
    
    @state.setter
    def state(self, state):
        # set to steady state if no state is provided
        if state is None:
            self._state = self.steady_state

        # process input
        else:
            state = np.array(state).reshape(1,-1)

            # make sure all values are probabilities
            assert (state>=0).all() and (state<=1).all(), \
                'probabilites need to be bounded between zero and one'

            # make sure input is a state vector (probabilities sum to one)
            if state.sum() != 1:
                state /= state.sum()
                warnings.warn('state probability vector transformed into probabilities')

            # check if dimensions match
            if hasattr(self, 'transition_matrix') \
                and self.transition_matrix is not None:
                assert state.shape[1] == self.n_states, \
                    'transition matrix dimension mismatch'

            self._state = state

    @property
    def steady_state(self):
        '''Returns the steady state probabilities of the Markov chain.
        If set_state=True, MarkovChain object is modified in place.
        '''
        dim = self.n_states
        q = np.c_[(self.transition_matrix-np.eye(dim)), np.ones(dim)]
        QTQ = np.dot(q, q.T)
        steady_state = np.linalg.solve(QTQ, np.ones(dim))
        return steady_state.reshape(1, -1)

    def set_to_steady_state(self):
        '''Set the steady state as the MarkovChain object state.'''
        self.state = self.steady_state
        
    def expected_durations(self):
        '''Returns the expected state durations of the MarkovChain object.'''
        expected_durations = (np.ones(self.n_states) \
                                - np.diag(self.transition_matrix))**-1
        return expected_durations
    
    @property
    def n_states(self):
        '''Returns the number of states of the MarkovChain object.'''
        return self.transition_matrix.shape[0]

    def iterate(self, steps=1, set_state=False):
        '''Iterates the MarkovChain object the specified number of steps.
        steps should be a positive integer.
        (negative steps work, but tend to break when going before the initial state)
        If set_state=True, MarkovChain object is modified in place.
        '''
        # calculate new state vector
        new_state = np.dot(self.state, np.linalg.matrix_power(self.transition_matrix, steps))
        
        # outputs
        if set_state:
            self.state = new_state
        else:
            new_mc = MarkovChain(transition_matrix=self.transition_matrix,
                                 state=new_state)
            return new_mc
        
    def forecast(self, horizons=[1]):
        '''Returns forecasted state probabilities for a set of horizons.
        horizons input needs to be iterable.
        '''
        # make sure horizons is iterable
        horizons = np.atleast_1d(horizons)

        # calculate forecasts
        forecasts = []
        for horizon in horizons:
            forecasts += [self.iterate(horizon).state_vector\
                                .flatten()\
                                .tolist()]
        return forecasts

        ## LEGACY
        # horizons_states = np.array([]).reshape(0, self.n_states)
        # for horizon in horizons:
        #     pi_ = np.dot(self.state, np.linalg.matrix_power(self.transition_matrix, horizon))
        #     horizons_states = np.concatenate([horizons_states, pi_.reshape(1, self.n_states)], axis=0)
        # return horizons_states
    
    def draw(self, size=1):
        '''Draws a random sample sequence from the MarkovChain object.
        size is the number of time steps forward to be drawn.
        If size is zero, only the current state is drawn.
        '''
        # setup
        sample = []
        draw_probabilities = (self.state @ self.transition_matrix).squeeze()

        for step in range(size):
            # draw new state
            sample += [np.random.choice(self.n_states,
                                        size=1,
                                        p=draw_probabilities)[0]]

            # update drawing probabilities for next draw
            draw_probabilities = self.transition_matrix[sample[-1]]

        return sample
    
    @property
    def entropy(self):
        '''Calculate Shannon's entropy of the n state probabilities based
        on logarithms with base n.
        '''
        state_entropy = entropy(self.state.squeeze(),
                                base=self.n_states)

        return state_entropy

    def __repr__(self):
        return str(self)

    def __str__(self):
        '''Returns a summarizing string.'''
        string = 'MarkovChain(\n'
        string += 'transition_matrix=\n{},\n'.format(self.transition_matrix.__str__())
        string += 'state=\n{}\n)'.format(self.state.__str__())
        return string

    @property
    def is_ergodic(self):
        raise NotImplementedError('ergodicity not implemented')
    
    @property
    def is_aperiodic(self):
        raise NotImplementedError('aperiodicity not implemented')

    @property
    def is_irreducible(self):
        raise NotImplementedError('irreducibility not implemented')