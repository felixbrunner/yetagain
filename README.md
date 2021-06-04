# yetagain (in construction)
This is a package for estimation, simulation and forecasting with Hidden Markov Models.

It serves to estimate discrete time HMMs with continuous emission models.

Currently it implements:
- Markov Chains
- Distributions
    - Distribution Mixin
    - Normal distribution
    - Student-T distribution
    - Mixture distribution
    - Product distribution
- Models
    - Model Mixin
    - Normal model
    - Student-T model
    - Hidden Markov model
    - GARCH model
    - Mixture model
- Estimation
    -Estimation Mixin

## Mixins
### DistributionMixin

### ModelMixin

### EstimationMixin
Requires the following methods to work on a model:
- _step
- score

## Markov Chains

## Distributions
Distributions cannot be fitted.
Distributions implement:
- pdf
- cdf
- score
- draw
- set_variance: sets the distribution paramters such that a variance is matched
- central_moment

## Models
Models can be fitted using the methods of EstimationMixin.
To be compatible to the estimation of higher hierarchy models,
they should implement a _step method that can be called in a higher estimation.
Convergence towards a ML esitmate needs to be ensured in every step.
Additionally, the fitting algorithm of any model should implement importance weightings.


All models implement:
- iterate: iterates the model through time
- distribution (property): gives distribution (can be conditional on state)
- draw: draw a random sample from the model as DGP,
        needs to have 'return_distributions' argument.
- params (property): returns a dictionary with the model parameters
- _step: defines one iterative estimation step for numerical optimization
- score

If the model instance is a state space model, also implements:
- filter_states: method
- states_
- steady_state (property): asymptotic state
- state (property)
- set_to_steady_state

- ADD: distributions_ for all models ??
