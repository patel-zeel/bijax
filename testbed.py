import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import optax
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

import matplotlib.pyplot as plt
import seaborn as sns

from bijax import ADVI
from bijax.utils import train_fn

import pymc as pm
import numpy as np
import arviz as az

from bijax.variational_distributions import VariationalDistribution

support = jnp.linspace(0.01, 0.99, 100)
print(support)
b = tfb.Sigmoid()
v_dist = tfd.Normal(0, 1)


def prob(sample):
    log_prob = v_dist.log_prob(b.inverse(sample))
    inv_log_det_jac = b.inverse_log_det_jacobian(sample)
    # jax.debug.print(
    #     "log_prob: {log_prob}, inv_log_det_jac: {inv_log_det_jac} sample {sample}",
    #     log_prob=log_prob,
    #     inv_log_det_jac=inv_log_det_jac,
    #     sample=b.inverse(sample),
    # )
    return jnp.exp(log_prob + inv_log_det_jac)


prob = jax.vmap(prob)

prior = {"p": tfd.Normal(0, 1)}
bijector = {"p": tfb.Sigmoid()}

var_dist = VariationalDistribution(prior=prior, bijector=bijector)
params = var_dist._initialise_params(0)
params["log_scale_diag"] = jnp.log(1.0).reshape((1,))
probs = var_dist.prob({"p": support}, sample_shape=(len(support),), params=params)
plt.plot(support, probs, color="g")

plt.plot(support, tfd.Beta(2, 2).prob(support))
plt.plot(support, b(v_dist).prob(support))
plt.plot(support, prob(support), "--")
plt.savefig("test.png")
