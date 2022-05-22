import jax
import jax.numpy as jnp
import distrax
from advi_jax import ADVI

## Generate data

key = jax.random.PRNGKey(0)
N = 1000

data = jax.random.normal(key, shape=(N, 2))
data.shape

## Model specification

# prior
loc = jnp.zeros((2,))
cov = jnp.eye(2)
prior = {"mean": distrax.MultivariateNormalFullCovariance(loc=loc, covariance_matrix=cov)}
prior_transforms = {"mean": distrax.Lambda(lambda x: x)}

# likelihood
def log_likelihood(params, data):
    fixed_cov = jnp.eye(2)
    dist = distrax.MultivariateNormalFullCovariance(loc=params["mean"], covariance_matrix=fixed_cov)
    return jax.vmap(dist.log_prob)(data).sum()


# Model
model = ADVI(prior, prior_transforms, log_likelihood, vi_type="mean_field")

keys = jax.random.split(key, 10)
model.objective_fun(model._params, keys, data)
