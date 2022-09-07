import jax
from jax.flatten_util import ravel_pytree
import jax.tree_util as jtu

from bijax.variational_distributions import VariationalDistribution
from bijax.core import DistributionPyTree


class ADVI:
    def __init__(self, prior, bijector, likelihood_fn, vi_type="mean_field", rank=None):
        self.prior = prior
        self.bijector = bijector
        self.likelihood_fn = likelihood_fn

        self.variational_distribution = VariationalDistribution(
            prior=self.prior, bijector=self.bijector, vi_type=vi_type, rank=rank
        )
        self.prior_distribution = DistributionPyTree(self.prior, {})

    def init(self, key):
        return {"variational_params": self.variational_distribution._initialise_params(key)}

    def loss_fn(self, params, outputs, inputs, total_size=None, seed=None, num_mc_samples=1):
        if total_size is None:
            total_size = outputs.shape[0]

        variational_params = params["variational_params"]
        sample, log_q = self.variational_distribution.sample_and_log_prob(
            seed=seed, sample_shape=(num_mc_samples,), params=variational_params
        )

        def log_p_plus_log_likelihood(sample_value):
            log_p = ravel_pytree(jtu.tree_leaves(self.prior_distribution.log_prob(sample_value)))[0].sum()
            likelihood = self.likelihood_fn(sample_value, inputs, **params)
            log_likelihood = likelihood.log_prob(outputs).sum()
            reweighted_log_likelihood = total_size * log_likelihood / len(outputs)
            return log_p, reweighted_log_likelihood

        log_p, reweighted_log_likelihood = jax.vmap(log_p_plus_log_likelihood)(sample)
        return (log_q - log_p - reweighted_log_likelihood).mean()
