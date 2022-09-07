import pytest
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

import jax

from bijax.core import DistributionPyTree

distribution_pytree1 = {"x": tfd.Normal(0.0, 1.0), "y": tfd.Beta(2.0, 1.0)}
bijector_pytree1 = {"x": tfb.Identity(), "y": tfb.Sigmoid()}
final_bijector_pytree1 = {"x": tfb.Identity(), "y": tfb.Sigmoid()}

distribution_pytree2 = {"x": tfd.Normal(0.0, 1.0), "y": tfd.Beta(2.0, 1.0), "z": tfd.Normal(0.0, 1.0)}
bijector_pytree2 = {"y": tfb.Sigmoid()}
final_bijector_pytree2 = {"x": tfb.Identity(), "y": tfb.Sigmoid(), "z": tfb.Identity()}


distribution_pytree3 = {"x": tfd.Normal(0.1, 1.0), "y": lambda x: tfd.Normal(x, 1.0)}
bijector_pytree3 = {}


@pytest.mark.parametrize(
    "distribution_pytree,bijector_pytree,final_bijector_pytree",
    [
        (distribution_pytree1, bijector_pytree1, final_bijector_pytree1),
        (distribution_pytree2, bijector_pytree2, final_bijector_pytree2),
    ],
)
def auto_fill_bijector_test(distribution_pytree, bijector_pytree, final_bijector_pytree):
    distribution = DistributionPyTree(distribution_pytree, bijector_pytree)
    assert distribution.bijector_pytree == final_bijector_pytree


@pytest.mark.parametrize(
    "distribution_pytree,bijector_pytree",
    [
        (distribution_pytree1, bijector_pytree1),
        (distribution_pytree2, bijector_pytree2),
        (distribution_pytree3, bijector_pytree3),
    ],
)
def sample_and_log_prob_test(distribution_pytree, bijector_pytree):
    distribution = DistributionPyTree(distribution_pytree, bijector_pytree)
    seed = jax.random.PRNGKey(0)
    sample, log_prob = distribution.sample_and_log_prob(seed=seed)
