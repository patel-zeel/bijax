import jax
import jax.numpy as jnp
import optax


class ADVI:
    def __init__(self, prior, likelihood, variational, data):
        self.prior = prior
        self.likelihood = likelihood
        self.variational = variational
        self.data = data

    def loss(self, params, seed, n_samples=1):
        self.variational.set_params(params)
        sample = self.variational.sample(seed, sample_shape=(n_samples,))
        q_log_prob = jax.vmap(self.variational.log_prob)(sample)
        p_log_prob = jax.vmap(self.prior.log_prob)(sample)
        log_likelihood = jax.vmap(self.likelihood.log_prob, in_axes=(0, None))(sample, self.data)
        return jnp.mean(q_log_prob - p_log_prob - log_likelihood)
