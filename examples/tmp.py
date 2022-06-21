import jax
import jax.numpy as jnp
import optax
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from ajax.advi import ADVI
from ajax.utils import train_model

import matplotlib.pyplot as plt
import seaborn as sns

a0, b0 = 5, 1

prior = {"p": tfd.Beta(a0, b0)}
bijector = {"p": tfb.Sigmoid()}
data = jnp.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
aux = None


def get_likelihood(likelihood_params, aux=None):
    return tfd.Bernoulli(probs=likelihood_params["p"])


model = ADVI(prior, bijector, get_likelihood, vi_type="mean_field")

optimizer = optax.adam(learning_rate=0.1)
seed = jax.random.PRNGKey(0)

loss_fn_kwargs = {"data": data, "aux": aux, "data_size": len(data), "seed": seed, "n_samples": 100}

params, losses = train_model(model, loss_fn_kwargs, optimizer, n_epochs=100, seed=seed)

plt.plot(losses)

true_posterior = tfd.Beta(prior["p"].concentration1 + a0, prior["p"].concentration0 + b0)
posterior = model.apply(params)

x = jnp.linspace(0.01, 0.99, 1000)
seed = jax.random.PRNGKey(1)

posterior_probs = posterior.log_prob({"p": x}, sample_shape=(len(x),))
true_posterior_probs = true_posterior.log_prob(x)

# plt.plot(x, jnp.exp(posterior_probs), label='ADVI')
# sns.kdeplot(posterior.sample(seed, sample_shape=(10,)))
# plt.plot(x, jnp.exp(true_posterior_probs), label='True')
# plt.legend();
posterior.sample(seed, sample_shape=(10,))
