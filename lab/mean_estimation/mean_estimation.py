# import os
# os.environ["JAX_CHECK_TRACER_LEAKS"] = "1"

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
import distrax

tfd = tfp.distributions
tfb = tfp.bijectors

import optax

from ajax import Prior, Likelihood, Variational, ADVI
from ajax.utils import fill_params

# generate data
N = 100
gt_dist = distrax.MultivariateNormalFullCovariance(loc=[0.5, 2.0], covariance_matrix=[[1.0, 0.5], [0.5, 1.0]])
data = gt_dist.sample(seed=jax.random.PRNGKey(0), sample_shape=(N,))

# define prior
prior = Prior(
    distributions={"mean": tfd.MultivariateNormalDiag(loc=jnp.array([0.0, 5.0]), scale_diag=jnp.array([1.0, 1.0]))},
    bijectors={"mean": tfb.Identity()},
)

# define link function & likelihood
def link_function(sample):
    loc = sample["mean"]
    covariance_matrix = jnp.eye(len(sample["mean"]))
    return {"loc": loc, "covariance_matrix": covariance_matrix}


likelihood = Likelihood(tfd.MultivariateNormalFullCovariance, link_function)

# define variational

variational = Variational(prior=prior, vi_type="full_rank")

# Define ADVI
advi = ADVI(prior, likelihood, variational, data)


seed = jax.random.PRNGKey(0)
n_samples = 5

params = variational.get_params()
params = fill_params(seed, params, jax.random.normal)
# print(params)
# print(advi.loss(params, prior, likelihood, variational, data, n_samples, seed))
for _ in range(10):
    val, grads = advi.value_and_grad(params, n_samples=10, seed=seed)
    print(grads)
    params = jax.tree_map(lambda x, g: x - 0.01 * g, params, grads)
