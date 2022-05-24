import os

os.environ["JAX_CHECK_TRACER_LEAKS"] = "1"
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
import distrax

tfd = tfp.distributions

from ajax.base import Prior, Variational

# generate data
N = 1000
gt_dist = distrax.MultivariateNormalFullCovariance(loc=[1.0, 2.0], covariance_matrix=[[1.0, 0.5], [0.5, 1.0]])
data = gt_dist.sample(seed=jax.random.PRNGKey(0), sample_shape=(N,))

# define prior
prior = Prior(
    distributions={
        "mean": distrax.MultivariateNormalDiag(loc=jnp.array([0.0, 0.0]), scale_diag=jnp.array([1.0, 1.0])),
        "L_corr": tfd.CholeskyLKJ(dimension=2, concentration=2.0),
        "sigma": distrax.MultivariateNormalDiag(loc=jnp.array([0.0, 0.0]), scale_diag=jnp.array([1.0, 1.0])),
    },
    transforms={
        "mean": distrax.Lambda(lambda x: x),
        "L_corr": distrax.Lambda(lambda x: jnp.exp(x)),
        "sigma": distrax.Lambda(lambda x: jnp.exp(x)),
    },
)

# define likelihood
def log_likelihood_fun(sample, data):
    return jax.vmap(_log_likelihood_fun, in_axes=(0, None))(sample, data)


def _log_likelihood_fun(sample, data):
    sigma = jnp.diag(sample["sigma"])
    L_corr = sample["L_corr"]
    corr = L_corr @ L_corr.T
    cov = sigma @ corr @ sigma
    mean = sample["mean"]
    dist = distrax.MultivariateNormalFullCovariance(loc=mean, covariance_matrix=cov)
    return jax.vmap(dist.log_prob)(data).sum()


# define variational
N = 100
variational = Variational(prior=prior, vi_type="mean_field", rank=None)

# loss function
def loss(params, seed, n_samples=5):
    variational.raw_params = params
    sample, log_q_prob = variational.sample_and_log_prob(seed=seed, sample_shape=(n_samples,))
    log_likelihood = log_likelihood_fun(sample, data)
    print(sample)
    log_prior = prior.log_prob(sample)
    # print(log_prior)
    log_p_prob = log_prior + log_likelihood
    # print(log_p_prob, log_q_prob)
    return jnp.mean(log_q_prob - log_p_prob)


# run variational inference
grad_fun = jax.grad(loss)
epochs = 10
lr = 0.1
seed = jax.random.PRNGKey(123)
for _ in range(epochs):
    seed = jax.random.split(seed, 1)[0]
    value = loss(variational.raw_params, seed)
    print(value)
    grads = grad_fun(variational.raw_params, seed)
    variational.raw_params = jax.tree_map(lambda p, g: p - lr * g, variational.raw_params, grads)
