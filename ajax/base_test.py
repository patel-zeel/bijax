import pytest

import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp

import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from ajax.base import (
    seeds_like,
    inverse_transform_dist,
    sample_dist,
)


@pytest.mark.parametrize(
    "params,is_leaf,size",
    [
        ({"normal": tfd.Normal(0, 1), "beta": tfd.Beta(1, 1)}, None, 4),
        ({"normal": tfd.Normal(0, 1), "beta": tfd.Beta(1, 1)}, lambda x: isinstance(x, tfd.Distribution), 2),
    ],
)
def seeds_like_test(params, is_leaf, size):
    seed = jax.random.PRNGKey(0)
    seeds = seeds_like(params, seed, is_leaf)
    assert len(jax.tree_flatten(seeds)[0]) == size


def inverse_transform_dist_test():
    dist_pytree = {"normal": tfd.Normal(0, 1), "beta": tfd.Beta(1, 1)}
    bijector_pytree = {"normal": tfb.Exp(), "beta": tfb.Exp()}
    transformed_dists = inverse_transform_dist(dist_pytree, bijector_pytree)
    assert isinstance(transformed_dists["normal"], tfd.TransformedDistribution)
    assert isinstance(transformed_dists["beta"], tfd.TransformedDistribution)


def sample_dist_test():
    dist_pytree = {"normal": tfd.MultivariateNormalDiag(jnp.zeros(2), jnp.ones(2)), "beta": tfd.Beta(1, 1)}
    seed = jax.random.PRNGKey(0)
    samples = sample_dist(dist_pytree, seed)
    assert samples["normal"].shape == (2,)
    assert samples["beta"].shape == ()
