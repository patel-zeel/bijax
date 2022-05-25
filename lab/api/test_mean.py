from lib2to3.pytree import LeafPattern
import os

os.environ["JAX_CHECK_TRACER_LEAKS"] = "1"
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
import distrax

tfd = tfp.distributions
import optax

from ajax.base2 import Prior, Variational

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


# define variational
N = 100
variational = Variational(prior=prior, vi_type="full_rank", rank=None)

# loss function
def loss(params, prior, seed, n_samples=10):
    variational = Variational(prior=prior, vi_type="full_rank", rank=None)
    variational.raw_params = params
    sample, log_q_prob = variational.sample_and_log_prob(seed=seed, sample_shape=(n_samples,))
    log_likelihood = log_likelihood_fun(sample, data)
    log_prior = prior.log_prob(sample)
    # print(log_prior)
    log_p_prob = log_prior + log_likelihood
    # print(log_p_prob, log_q_prob)
    return jnp.mean(log_q_prob - log_p_prob)


# run variational inference
grad_fun = jax.jit(jax.grad(loss), static_argnums=1)
epochs = 50
tx = optax.adam(learning_rate=0.1)
state = tx.init(variational.raw_params)

seed = jax.random.PRNGKey(123)
for _ in range(epochs):
    seed = jax.random.split(seed, 1)[0]
    value = loss(variational.raw_params, prior, seed)
    print(value)
    grads = grad_fun(variational.raw_params, prior, seed)
    updates, state = tx.update(grads, state)
    variational.raw_params = optax.apply_updates(variational.raw_params, updates)

print(variational.raw_params["loc"])
print(variational.raw_params["scale_tri"] @ variational.raw_params["scale_tri"].T)
print("Done")
