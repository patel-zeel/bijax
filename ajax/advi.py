import jax
import jax.numpy as jnp
import optax


class ADVI:
    def __init__(self, prior, likelihood, variational, data):
        self.prior = prior
        self.likelihood = likelihood
        self.variational = variational
        self.data = data

    def loss(self, params, prior, likelihood, variational, data, n_samples, seed):
        variational.set_params(params)
        # params = variational.get_params()
        # return jnp.concatenate(jax.tree_leaves(params)).sum()**2
        sample = variational.sample(seed, sample_shape=(n_samples,))
        # return sample["mean"].sum() ** 2
        q_log_prob = jax.vmap(variational.log_prob)(sample)
        p_log_prob = jax.vmap(prior.log_prob)(sample)
        log_likelihood = jax.vmap(likelihood.log_prob, in_axes=(0, None))(sample, data)
        print(q_log_prob, p_log_prob)
        return jnp.mean(q_log_prob - p_log_prob - log_likelihood)

    def value_and_grad(self, params, n_samples=1, seed=None):
        value_and_grad_fun = jax.value_and_grad(self.loss)
        value, grads = value_and_grad_fun(
            params, self.prior, self.likelihood, self.variational, self.data, n_samples, seed
        )
        return value, grads
