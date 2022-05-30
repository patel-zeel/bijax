# import os
# os.environ["JAX_CHECK_TRACER_LEAKS"] = "1"

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
import distrax
import matplotlib.pyplot as plt

tfd = tfp.distributions
tfb = tfp.bijectors

import optax

from ajax import Prior, Likelihood, Variational, ADVI
from ajax.utils import fill_params

# generate data
N = 10000
gt_corr = jnp.array([[1.0, -0.4], [-0.4, 1.0]])
gt_sigma = jnp.diag(jnp.array([0.7, 1.5]))
gt_cov = gt_sigma @ gt_corr @ gt_sigma
print("gt_cov:", gt_cov)
gt_dist = distrax.MultivariateNormalFullCovariance(loc=[1.0, -2.0], covariance_matrix=gt_cov)
data = gt_dist.sample(seed=jax.random.PRNGKey(0), sample_shape=(N,))
plt.scatter(data[:, 0], data[:, 1])
plt.savefig("abc.png")

# define prior
prior = Prior(
    distributions={
        "mean": tfd.MultivariateNormalDiag(loc=jnp.array([0.0, 0.0]), scale_diag=jnp.array([1.5, 1.5])),
        "corr": tfd.CholeskyLKJ(dimension=2, concentration=2.0),
        "sigma": tfd.Independent(tfd.Exponential(rate=[0.1, 0.1]), reinterpreted_batch_ndims=1),
    }
)

# define link function & likelihood
def link_function(sample):
    loc = sample["mean"]
    sigma_diag = jnp.diag(sample["sigma"])
    L_corr = sample["corr"]
    scale_tril = sigma_diag @ L_corr
    return {"loc": loc, "scale_tril": scale_tril}


likelihood = Likelihood(tfd.MultivariateNormalTriL, link_function)

# define variational

variational = Variational(
    prior=prior,
    bijectors={"mean": tfb.Identity(), "corr": tfb.CorrelationCholesky(), "sigma": tfb.Exp()},
    vi_type="full_rank",
)

# Define ADVI
advi = ADVI(prior, likelihood, variational, data)


seed = jax.random.PRNGKey(1)
n_samples = 100

params = variational.get_params()
params = fill_params(seed, params, jax.random.normal)
# print(params)
# print(advi.loss(params, prior, likelihood, variational, data, n_samples, seed))
tx = optax.adam(learning_rate=0.1)
state = tx.init(params)

value_and_grad_fun = jax.jit(jax.value_and_grad(advi.loss), static_argnames=["n_samples"])

vals = []

for _ in range(200):
    seed = jax.random.split(seed, 1)[0]
    val, grads = value_and_grad_fun(params, n_samples=10, seed=seed)
    updates, state = tx.update(grads, state)
    params = optax.apply_updates(params, updates)
    print("loss", val)
    vals.append(val)
    print("mean", params["mean"].distribution.loc)
    print("sigma", jnp.exp(params["sigma"].distribution.mean()))
    l = tfb.CorrelationCholesky().forward(params["corr"].distribution.mean())
    print("corr", l @ l.T)
    # l = jnp.eye(2)
    # sigma = jnp.diag(jnp.exp(params["sigma"].distribution.loc))
    # print(sigma @ l @ l.T @ sigma)
    # print(advi.loss(params, prior, likelihood, variational, data, n_samples, seed))

plt.figure()
plt.plot(vals[50:])
plt.savefig("losses.png")

plt.figure()
plt.scatter(data[:, 0], data[:, 1], label="gt", alpha=0.5)
variational.set_params(params)
sample = variational.sample(seed)
seed = jax.random.split(seed, 1)[0]
samples = likelihood.sample(seed, sample, sample_shape=(10000,))
plt.scatter(samples[:, 0], samples[:, 1], label="posterior", alpha=0.5)
plt.legend()
plt.savefig("posterior.png")
