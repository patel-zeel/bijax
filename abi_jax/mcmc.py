from .core import (
    sample_dist,
    log_prob_dist,
)

import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions


class MCMC:
    """
    A helper class to facilitate the use of the MCMC algorithm from external libraries such as `blackjax`.
    """

    def __init__(self, prior, log_likelihood_fn):

        self.prior = prior
        self.log_likelihood_fn = log_likelihood_fn

    def init(self, seed):
        return sample_dist(self.prior, seed)

    def log_joint(self, params, outputs, inputs):
        p_log_prob = log_prob_dist(self.prior, params)
        log_likelihood = self.log_likelihood_fn(latent_sample=params, outputs=outputs, inputs=inputs)
        return p_log_prob + log_likelihood
