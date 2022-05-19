import blackjax
import jax
import jax.numpy as jnp
import distrax
from advi import ADVI_MeanField as ADVI
import optax
import matplotlib.pyplot as plt

# Define prior distributions
prior_dist = distrax.MultivariateNormalFullCovariance(jnp.array([1.0, 10.0]), jnp.array([[1.0, 0.2], [0.2, 1.0]]))
lik_dist = lambda theta: distrax.MultivariateNormalFullCovariance(theta, jnp.array([[1.0, 0.5], [0.5, 1.0]]))


def likelihood_sample(key, theta):
    key = jax.random.split(key, 1)[0]
    return lik_dist(theta).sample(seed=key)


key = jax.random.PRNGKey(0)
keys = jax.random.split(key, 100)
data = jax.vmap(lambda key: likelihood_sample(key, prior_dist.sample(seed=key)))(keys)

# Define log likelihood function
def log_likelihood_fun(theta):
    return jax.vmap(lik_dist(theta).log_prob)(data).sum()


def log_prior_fun(theta):
    return prior_dist.log_prob(theta)


def log_prob_fun(theta):
    return log_prior_fun(theta) + log_likelihood_fun(theta)


rng_key = jax.random.PRNGKey(314)
M = 2
w0 = jax.random.multivariate_normal(rng_key, 0.1 + jnp.zeros(M), jnp.eye(M))

rmh = blackjax.rmh(log_prob_fun, sigma=jnp.ones(M) * 0.7)
state = rmh.init(w0)


def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


print("initial_state", state)
num_samples = 1000
rng_key = jax.random.split(rng_key, 1)[0]
states = inference_loop(rng_key, rmh.step, state, num_samples)

print("mean\n", states.position[100:, :].mean(axis=0))
print("cov\n", jnp.cov(states.position[100:, :].T))

p_inv = jnp.linalg.inv(prior_dist.covariance())
l_inv = jnp.linalg.inv(lik_dist(data[0]).covariance())

real_post_covar = jnp.linalg.inv(p_inv + data.shape[0] * l_inv)

real_post_mean = real_post_covar @ (l_inv @ (data.shape[0] * data.mean(axis=0)) + p_inv @ prior_dist.mean())

print("real_post_mean\n", real_post_mean)
print("real_post_covar\n", real_post_covar)
print("Okay")
