from .base import (
    sample_dist,
    log_prob_dist,
)

import tensorflow_probability.substrates.jax as tfp
from ajax.base import check_distribution_zero_batch

tfd = tfp.distributions


class MCMC:
    """
    A helper class to facilitate the use of the MCMC algorithm from external libraries such as `blackjax`.
    """

    def __init__(self, prior, log_likelihood_fn):
        check_distribution_zero_batch(prior)  # Assert that the prior distribution has no batch dimension.

        self.prior = prior
        self.log_likelihood_fn = log_likelihood_fn

    def init(self, seed):
        return sample_dist(self.prior, seed)

    def log_joint(self, params, batch, aux):
        p_log_prob = log_prob_dist(self.prior, params)
        log_likelihood = self.log_likelihood_fn(latent_sample=params, data=batch, aux=aux)
        return p_log_prob + log_likelihood
