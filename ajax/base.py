######### Avoid TypeCheck Warning #########
import logging

logger = logging.getLogger()


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()


logger.addFilter(CheckTypesFilter())
############################################

import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from .utils import seeds_like


class Posterior:
    def __init__(self, posterior, dill_unravel, bijector):
        assert isinstance(posterior, tfd.Distribution)
        self.posterior = posterior
        self.unravel_fn = dill_unravel.function
        self.bijector = bijector

    def sample(self, seed, sample_shape=()):
        sample = self.posterior.sample(seed=seed, sample_shape=sample_shape)

        def f(x):
            return transform_tree(self.unravel_fn(x), self.bijector)

        for _ in range(len(sample_shape)):
            f = jax.vmap(f)
        return f(sample)

    def log_prob(self, sample, sample_shape=()):
        def log_prob_per_sample(sample_tree):
            approx_normal_sample_tree = inverse_transform_tree(sample_tree, self.bijector)
            approx_normal_sample, _ = ravel_pytree(approx_normal_sample_tree)
            normal_log_prob = self.posterior.log_prob(approx_normal_sample)
            jacobian_tree = jax.tree_map(
                lambda sample, bijector: bijector.inverse_log_det_jacobian(sample), sample_tree, self.bijector
            )
            jacobian = ravel_pytree(jacobian_tree)[0].sum()
            return normal_log_prob + jacobian

        f = log_prob_per_sample
        for _ in range(len(sample_shape)):
            f = jax.vmap(f)
        return f(sample)


def inverse_transform_dist(dist_pytree, bijector_pytree):
    is_leaf = lambda x: isinstance(x, tfd.Distribution)
    return jax.tree_map(
        lambda dist, bijector: tfd.TransformedDistribution(dist, tfb.Invert(bijector)),
        dist_pytree,
        bijector_pytree,
        is_leaf=is_leaf,
    )


def sample_dist(dist_pytree, seed, sample_shape=()):
    is_leaf = lambda dist: isinstance(dist, tfd.Distribution)
    seeds = seeds_like(dist_pytree, seed, is_leaf=is_leaf)
    samples = jax.tree_map(
        lambda dist, seed: dist.sample(seed=seed, sample_shape=sample_shape), dist_pytree, seeds, is_leaf=is_leaf
    )
    return samples


def log_prob_dist(dist_pytree, sample_pytree):
    def is_leaf(dist):
        return isinstance(dist, tfd.Distribution)

    log_probs = jax.tree_map(lambda dist, sample: dist.log_prob(sample), dist_pytree, sample_pytree, is_leaf=is_leaf)
    return ravel_pytree(log_probs)[0].sum()


def transform_tree(pytree, bijector_pytree):
    return jax.tree_map(lambda param, bijector: bijector(param), pytree, bijector_pytree)


def inverse_transform_tree(pytree, bijector_pytree):
    return jax.tree_map(lambda param, bijector: bijector.inverse(param), pytree, bijector_pytree)


def transform_dist_params(dist, ordered_posterior_bijectors):
    params, treedef = jax.tree_flatten(dist)
    params = jax.tree_map(lambda param, bijector: bijector(param), params, ordered_posterior_bijectors)
    return jax.tree_unflatten(treedef, params)


def get_normal_posterior_size_and_unravel_fn(approx_normal_prior):
    seed = jax.random.PRNGKey(0)
    samples = sample_dist(approx_normal_prior, seed=seed, sample_shape=())
    # for name, sample in samples.items():
    #     assert (
    #         sample.ndim < 2
    #     ), f"'{name}' returned {sample.ndim} dimensional sample after transformation but should be 1d or scalar"
    array, unravel_fn = ravel_pytree(samples)
    return len(array), unravel_fn


def get_mean_field(approx_normal_prior, ordered_posterior_bijectors=None):
    if ordered_posterior_bijectors is None:
        ordered_posterior_bijectors = [tfb.Identity(), tfb.Exp()]
    size, unravel_fn = get_normal_posterior_size_and_unravel_fn(approx_normal_prior)
    return (
        tfd.MultivariateNormalDiag(loc=jnp.zeros(size), scale_diag=jnp.ones(size)),
        unravel_fn,
        ordered_posterior_bijectors,
    )


def get_low_rank(approx_normal_prior, rank=1, ordered_posterior_bijectors=None):
    if ordered_posterior_bijectors is None:
        ordered_posterior_bijectors = [tfb.Identity(), tfb.Exp(), tfb.Identity()]
    size, unravel_fn = get_normal_posterior_size_and_unravel_fn(approx_normal_prior)
    scale_perturb_factor = jnp.ones((size, rank))
    return (
        tfd.MultivariateNormalDiagPlusLowRank(
            loc=jnp.zeros(size), scale_diag=jnp.ones(size), scale_perturb_factor=scale_perturb_factor
        ),
        unravel_fn,
        ordered_posterior_bijectors,
    )


def get_full_rank(approx_normal_prior, ordered_posterior_bijectors=None):
    if ordered_posterior_bijectors is None:
        ordered_posterior_bijectors = [tfb.Identity(), tfb.Identity()]
    size, unravel_fn = get_normal_posterior_size_and_unravel_fn(approx_normal_prior)
    return (
        tfd.MultivariateNormalTriL(loc=jnp.zeros(size), scale_tril=jnp.eye(size)),
        unravel_fn,
        ordered_posterior_bijectors,
    )
