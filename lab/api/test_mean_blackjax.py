import os

os.environ["JAX_CHECK_TRACER_LEAKS"] = "1"
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
import distrax

tfd = tfp.distributions
import optax
import numpy as np
import blackjax

from ajax.base import Prior, Variational

# generate data
N = 100
gt_dist = distrax.MultivariateNormalFullCovariance(loc=[1.0, 2.0], covariance_matrix=[[1.0, 0.5], [0.5, 1.0]])
data = gt_dist.sample(seed=jax.random.PRNGKey(0), sample_shape=(N,))

# define prior
prior = Prior(
    distributions={"mean": distrax.MultivariateNormalDiag(loc=jnp.array([0.0, 0.0]), scale_diag=jnp.array([1.0, 1.0]))},
    transforms={"mean": distrax.Lambda(lambda x: x)},
)

# define likelihood
def log_likelihood_fun(sample, data):
    return jax.vmap(_log_likelihood_fun, in_axes=(0, None))(sample, data)


def _log_likelihood_fun(sample, data):
    cov = jnp.eye(len(sample["mean"]))
    mean = sample["mean"]
    dist = distrax.MultivariateNormalFullCovariance(loc=mean, covariance_matrix=cov)
    return jax.vmap(dist.log_prob)(data).sum()


def log_prob(mean1, mean2, data=data):
    sample = {"mean": jnp.array([mean1, mean2])}
    log_prior = prior._log_prob(sample)
    log_likelihood = _log_likelihood_fun(sample, data)
    return log_prior + log_likelihood


logprob = lambda x: log_prob(**x)

inv_mass_matrix = np.array([0.5, 0.5])
num_integration_steps = 60
step_size = 1e-3

hmc = blackjax.hmc(logprob, step_size, inv_mass_matrix, num_integration_steps)
initial_position = {"mean1": 0.1, "mean2": 0.2}
initial_state = hmc.init(initial_position)

hmc_kernel = jax.jit(hmc.step)


def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


rng_key = jax.random.PRNGKey(0)
states = inference_loop(rng_key, hmc_kernel, initial_state, 10000)

print("Done")
