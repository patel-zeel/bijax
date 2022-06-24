import jax
from sklearn.covariance import log_likelihood

from .base import (
    inverse_transform_dist,
    sample_dist,
    transform_tree,
    transform_dist_params,
    log_prob_dist,
)

from .utils import initialize_params

import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions


class MCMC:
    def __init__(self, prior, get_log_likelihood):
        self.prior = prior
        self.check_prior_zero_batch()  # Assert that the prior distribution has no batch dimension.
        self.get_log_likelihood = get_log_likelihood

    def check_prior_zero_batch(self):
        is_leaf = lambda x: isinstance(x, tfd.Distribution)

        def check(dist):
            if isinstance(dist, tfd.Distribution):
                return len(dist.batch_shape)
            else:
                return 0

        batch_lens = jax.tree_map(lambda dist: check(dist), self.prior, is_leaf=is_leaf)
        assert sum(jax.tree_leaves(batch_lens)) == 0, "The prior distributions must have no batch dimension."

    def init(self, seed):
        return sample_dist(self.prior, seed)

    def log_joint(self, params, batch, aux):
        p_log_prob = log_prob_dist(self.prior, params)
        log_likelihood = self.get_log_likelihood(params, aux, batch)
        return p_log_prob + log_likelihood
