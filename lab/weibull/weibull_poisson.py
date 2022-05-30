import logging

logger = logging.getLogger()


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "user_dtype" not in record.getMessage()


logger.addFilter(CheckTypesFilter())

import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

try:
    import optax
except ModuleNotFoundError:
    # %pip install -qq optax
    import optax

try:
    from ajax import Prior, Likelihood, Variational, ADVI
except:
    # %pip install -qq git+https://github.com/patel-zeel/ajax
    from ajax import Prior, Likelihood, Variational, ADVI

from ajax.utils import fill_params
from tqdm import trange

### Generate dataset

# Define prior
prior = Prior({"rate": tfd.Weibull(concentration=1.5, scale=1)})

# Define likelihood
def link_function(sample):
    return sample


likelihood = Likelihood(tfd.Poisson, link_function)

# Generate samples
N = 100
seed = jax.random.PRNGKey(1)
mean_sample = prior.sample(seed)
data = likelihood.sample(seed, mean_sample, sample_shape=(N,))

# Plot the samples
plt.hist(data)

## ADVI


def fit_advi(vi_type, epochs):
    bijectors = {"rate": tfb.Softplus()}
    variational = Variational(prior, bijectors, vi_type=vi_type)

    advi = ADVI(prior, likelihood, variational, data)

    # n_samples = 10
    optimizer = optax.adam(learning_rate=0.1)
    params = variational.get_params()
    state = optimizer.init(params)

    value_and_grad_fun = jax.jit(jax.value_and_grad(advi.loss), static_argnames=["n_samples"])
    seed = jax.random.PRNGKey(0)
    for _ in trange(epochs):
        seed = jax.random.split(seed, 1)[0]
        loss_value, grads = value_and_grad_fun(params, seed)
        updates, state = optimizer.update(grads, state)
        params = optax.apply_updates(params, updates)
    return jax.tree_leaves(variational.transform_dist(params["rate"]))


## Fit Mean-field  VI

results = {}

fit_advi(vi_type="mean_field", epochs=200)

## Fit Full-rank VI

loc, L_chol = fit_advi(vi_type="full_rank", epochs=200)
print("loc", loc)
print("covariance_matrix", L_chol @ L_chol.T)
results["full_rank"] = {"loc": loc, "covar": L_chol @ L_chol.T}

## Closed form

prior_cov = jnp.diag(jax.tree_leaves(prior.distributions["mean"])[1])
sigma_y_inv = jnp.linalg.inv(jnp.array([[0.28, 0.2], [0.2, 0.31]]))
optimal_cov_inv = jnp.linalg.inv(prior_cov) + N * sigma_y_inv
optimal_cov = jnp.linalg.inv(optimal_cov_inv)

optimal_mean = optimal_cov @ (sigma_y_inv @ (N * data.mean(axis=0)))
results["analytic"] = {"loc": optimal_mean, "covar": optimal_cov}

## Results

results["mean_field"]

results["full_rank"]

results["analytic"]
