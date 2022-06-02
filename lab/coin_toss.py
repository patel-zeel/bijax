import optax
import tensorflow_probability.substrates.jax as tfp
import jax.numpy as jnp
import seaborn as sns

from ajax import Prior, Likelihood, Variational, ADVI
from ajax.utils import fill_params
import jax
import matplotlib.pyplot as plt
import pickle

tfd = tfp.distributions
tfb = tfp.bijectors

import pandas as pd

data = pd.read_pickle("https://raw.githubusercontent.com/AnandShegde/pml_baselines/main/data/coin_toss")

dataset = data["samples"]
alpha_param = data["prior"]["alpha"]
beta_param = data["prior"]["beta"]

prior = Prior(distributions={"theta": tfd.Beta(alpha_param, beta_param)})


def link_function(samples):
    probs = samples["theta"]
    return {"probs": probs}


likelihood = Likelihood(tfd.Bernoulli, link_function)

variational = Variational(prior=prior, bijectors={"theta": tfb.Sigmoid()}, vi_type="full_rank")

advi = ADVI(prior, likelihood, variational, dataset)

params = variational.get_params()

tx = optax.adam(learning_rate=0.1)
state = tx.init(params)
value_and_grad_fun = jax.value_and_grad(advi.loss)


def update_func(carry, x):
    params = carry["params"]
    state = carry["state"]
    seed = carry["seed"]
    seed = jax.random.split(seed, 1)[0]
    loss, grads = value_and_grad_fun(params, seed=seed)
    updates, state = tx.update(grads, state)
    params = optax.apply_updates(params, updates)
    carry = {"params": params, "state": state, "seed": seed}
    return carry, loss


carry = {"params": params, "state": state, "seed": jax.random.PRNGKey(10)}  # key value matters keep it 10 :(
carry, loss = jax.lax.scan(update_func, carry, xs=None, length=50)
loss

keys = jax.random.PRNGKey(1)
variational.set_params(carry["params"])
sample = variational.sample(seed=keys, sample_shape=(19000,))

sample["theta"]
# print(variational)

beta_dist_one = tfd.Beta(alpha_param, beta_param)
x = jnp.linspace(0, 1, 100)
plt.plot(x, beta_dist_one.prob(x), label="prior")
one = jnp.sum(dataset == 1).astype("float32")
zero = jnp.sum(dataset == 0).astype("float32")
beta_dist = tfd.Beta(alpha_param + one, beta_param + zero)
post_pdf = beta_dist.prob(x)
plt.plot(x, post_pdf, label="True Posterior")
sns.kdeplot(sample["theta"], label="vi estimate")
plt.legend()
