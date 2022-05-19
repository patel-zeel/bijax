import jax
import jax.numpy as jnp
import distrax
from advi import ADVI_MeanField as ADVI
import optax
import matplotlib.pyplot as plt

# Define prior distributions
prior_dists = {
    "theta": lambda: distrax.MultivariateNormalFullCovariance(
        jnp.array([1.0, 10.0]), jnp.array([[1.0, 0.2], [0.2, 1.0]])
    ),
}

transforms = {"theta": lambda: distrax.Lambda(lambda x: x)}

# Define log likelihood function
def log_likelihood_fun(data, theta):
    lik_dist = distrax.MultivariateNormalFullCovariance(theta["theta"], jnp.array([[1.0, 0.5], [0.5, 1.0]]))
    return jax.vmap(lik_dist.log_prob)(data).sum()


def likelihood_sample(key, theta):
    key = jax.random.split(key, 1)[0]
    lik_dist = distrax.MultivariateNormalFullCovariance(theta, jnp.array([[1.0, 0.0], [0.0, 1.0]]))
    return lik_dist.sample(seed=key)


# Define model
model = ADVI(prior_dists, transforms, log_likelihood_fun)

# Initialize
key = jax.random.PRNGKey(0)
params = model.init(key, distrax.Normal(loc=0.0, scale=1.0))
keys = jax.random.split(key, 1000)
data = jax.vmap(lambda key: likelihood_sample(key, prior_dists["theta"]().sample(seed=key)))(keys)

# Setup ADVI
key = jax.random.split(key, 1)[0]
n_iterations = 200
n_mc_samples = 1000
learning_rate = 0.1

# Optimize
sample_epsilon = jax.jit(model.sample_epsilon, static_argnums=1)
value_and_grad_fun = jax.jit(jax.value_and_grad(model.objective_fun))
tx = optax.adam(learning_rate=learning_rate)
state = tx.init(params)
losses = []
for i in range(n_iterations):
    key = jax.random.PRNGKey(i)
    epsilons = sample_epsilon(key=key, sample_shape=(n_mc_samples,))
    value, grads = value_and_grad_fun(params, epsilons, data)
    updates, state = tx.update(grads, state)
    params = optax.apply_updates(params, updates)

    print(i, value)
    losses.append(value)
    print("means", params["mean"]["theta"])
    print("scales", jnp.exp(params["log_scale"]["theta"]))

plt.plot(losses)
plt.savefig("advi_bi.png")
