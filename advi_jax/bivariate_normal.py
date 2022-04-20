import jax
import jax.numpy as jnp
import distrax
from advi import ADVI_MeanField as ADVI
import optax
import matplotlib.pyplot as plt

# Define prior distributions
prior_dists = {
    "theta": distrax.MultivariateNormalDiag(
        jnp.array([0.2, 10.0]),
        jnp.array([1.5, 2.5]),
    ),
}

# Transform prior distributions
transforms = {"theta": lambda x: x}

# Define log likelihood function
def log_likelihood_fun(data, theta):
    return 0.0


# Define model
model = ADVI(prior_dists, transforms, log_likelihood_fun)

# Initialize
key = jax.random.PRNGKey(0)
params = model.init(key, distrax.Normal(loc=0.0, scale=1.0))
data = (None, None)

# Define objective
def objective_step(params, samples, data):
    loss_val = jax.vmap(model.objective_per_mc_sample, in_axes=(None, 0, None))(params, samples, data)

    return loss_val.mean()


# Setup ADVI
key = jax.random.split(key, 1)[0]
n_iterations = 100
n_mc_samples = 100
learning_rate = 0.1
samples = model.epsilon_dist.sample(seed=key, sample_shape=(n_iterations, n_mc_samples))

print(model.objective_per_mc_sample(params, samples[0, 0], data))
print(objective_step(params, samples[0], data))

# Optimize
value_and_grad_fun = jax.value_and_grad(objective_step)
tx = optax.adam(learning_rate=learning_rate)
state = tx.init(params)
for i in range(n_iterations):
    value, grads = value_and_grad_fun(params, samples[i], data)
    updates, state = tx.update(grads, state)
    params = optax.apply_updates(params, updates)

    print(value)

print(params["mean"], jnp.exp(params["log_scale"]))
print("Done")
