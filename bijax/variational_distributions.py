import jax
from jax.flatten_util import ravel_pytree
import jax.tree_util as jtu
import jax.numpy as jnp

from bijax.core import DistributionPyTree
from bijax.utils import constrain, unconstrain, fill_in_bijectors, get_inverse_log_det_jacobian

import tensorflow_probability.substrates.jax as tfp
from chex import dataclass
from dataclasses import field

tfb = tfp.bijectors
tfd = tfp.distributions


@dataclass
class VariationalDistribution:
    prior: dict = field(default_factory=dict)
    bijector: dict = field(default_factory=dict)
    vi_type: str = "mean_field"
    rank: int = None

    def __post_init__(self):
        self.bijector = fill_in_bijectors(self.bijector, self.prior)
        prior = DistributionPyTree(self.prior, {})

        constrained_sample = prior.sample(seed=jax.random.PRNGKey(0))
        sample = unconstrain(constrained_sample, self.bijector)
        flat_sample, self.unravel_fn = ravel_pytree(sample)
        self.length = len(flat_sample)

        assert self.vi_type in [
            "mean_field",
            "full_rank",
            "low_rank",
        ], "vi_type must be one of mean_field, full_rank, low_rank"

        if self.vi_type == "low_rank":
            assert self.rank is not None, "rank must be specified for low_rank variational distribution"
        else:
            assert self.rank is None, "rank must only be specified for low_rank variational distribution"

    def get_variational_distribution(self, params):
        loc = params["loc"]
        if self.vi_type == "mean_field":
            scale_diag = jnp.exp(params["log_scale_diag"])
            return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)
        elif self.vi_type == "low_rank":
            cov_perturb_factor = params["cov_perturb_factor"]
            cov_diag_factor = jnp.exp(params["log_cov_diag_factor"])
            return tfd.MultivariateNormalDiagPlusLowRankCovariance(
                loc=loc,
                cov_diag_factor=cov_diag_factor,
                cov_perturb_factor=cov_perturb_factor,
            )
        elif self.vi_type == "full_rank":
            scale_tril = tfb.CorrelationCholesky()(params["scale_tril_inv_corr_chol"])
            return tfd.MultivariateNormalTriL(loc=loc, scale_tril=scale_tril)

    def _initialise_params(self, key):
        if self.vi_type == "mean_field":
            return {"loc": jnp.zeros(self.length), "log_scale_diag": jnp.zeros(self.length)}
        elif self.vi_type == "low_rank":
            return {
                "loc": jnp.zeros(self.length),
                "log_cov_diag_factor": jnp.ones(self.length),
                "cov_perturb_factor": jnp.ones((self.length, self.rank)),
            }
        elif self.vi_type == "full_rank":
            n = self.length - 1
            num_elements = int(n * (n + 1) / 2)
            return {"loc": jnp.zeros(self.length), "scale_tril_inv_corr_chol": jnp.ones(num_elements)}

    def sample(self, sample_shape, seed, params):
        distribution = self.get_variational_distribution(params)
        flat_sample = distribution.sample(sample_shape, seed)

        def sample_fn(flat_sample_value):
            sample = self.unravel_fn(flat_sample_value)
            constrained_sample = constrain(sample, self.bijector)
            return constrained_sample

        for _ in range(len(sample_shape)):
            sample_fn = jax.vmap(sample_fn)
        return sample_fn(flat_sample)

    def log_prob(self, constrained_sample, sample_shape, params):
        distribution = self.get_variational_distribution(params)

        def log_prob_fn(constrained_sample_value):
            sample = unconstrain(constrained_sample_value, self.bijector)
            flat_sample, _ = ravel_pytree(sample)
            log_prob = distribution.log_prob(flat_sample)
            inv_log_det_jac = get_inverse_log_det_jacobian(constrained_sample_value, self.bijector)
            # jax.debug.print(
            #     "intlog_prob: {log_prob}, inv_log_det_jac: {inv_log_det_jac}, sample: {sample}",
            #     log_prob=log_prob,
            #     inv_log_det_jac=inv_log_det_jac,
            #     sample=(constrained_sample_value, sample),
            # )
            return log_prob + inv_log_det_jac

        for _ in range(len(sample_shape)):
            log_prob_fn = jax.vmap(log_prob_fn)
        return log_prob_fn(constrained_sample)

    def prob(self, constrained_sample, sample_shape, params):
        return jtu.tree_map(jnp.exp, self.log_prob(constrained_sample, sample_shape, params))

    def sample_and_log_prob(self, sample_shape, seed, params):
        distribution = self.get_variational_distribution(params)
        flat_sample = distribution.sample(sample_shape, seed)

        def sample_and_log_prob_fn(flat_sample_value):
            sample = self.unravel_fn(flat_sample_value)
            log_prob = distribution.log_prob(flat_sample_value)
            constrained_sample = constrain(sample, self.bijector)
            inverse_log_det_jacobian = get_inverse_log_det_jacobian(constrained_sample, self.bijector)
            return constrained_sample, log_prob + inverse_log_det_jacobian

        for _ in range(len(sample_shape)):
            sample_and_log_prob_fn = jax.vmap(sample_and_log_prob_fn)
        return sample_and_log_prob_fn(flat_sample)
