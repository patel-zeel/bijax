import jax
import jax.numpy as jnp
import distrax
from distrax import MultivariateNormalDiag
import tensorflow_probability.substrates.jax as tfp
from torch import inverse
from functools import reduce
from init import initialize
from jax.experimental.host_callback import id_print  # Useful for print debugging


class ADVI_MeanField:
    def __init__(self, prior_dists, prior_transforms, log_likelihood_fun):
        self.prior_dists = prior_dists
        self.prior_transforms = prior_transforms
        self.log_likelihood_fun = log_likelihood_fun

        self.event_dims = jax.tree_map(self._get_event_dim, self.prior_dists)
        self.n_event_dims = sum(jax.tree_leaves(self.event_dims))

        self.tree_def = jax.tree_structure(self.event_dims)

    @staticmethod
    def _get_event_dim(dist):
        event_shape = dist().event_shape
        return 1 if event_shape == () else event_shape[0]

    def init(self, key=None, initializer=None):
        mean = jax.tree_map(lambda dim: jnp.zeros((dim,)), self.event_dims)
        log_scale = jax.tree_map(lambda dim: jnp.zeros((dim,)), self.event_dims)

        params = {"mean": mean, "log_scale": log_scale}
        if initializer:
            assert key is not None, "key must be provided if initializer is provided"
            params = initialize(key, params, initializer)

        return params

    @staticmethod
    def _get_log_jac(f, x):
        return jnp.log(jnp.abs(jnp.linalg.det(jax.jacfwd(f().forward)(x))))

    def sample_epsilon(self, key, sample_shape):
        keys = jax.random.split(key, len(self.event_dims))
        keys = jax.tree_unflatten(self.tree_def, keys)
        return jax.tree_map(
            lambda key, dim: MultivariateNormalDiag(jnp.zeros((dim,)), jnp.ones((dim,))).sample(
                seed=key, sample_shape=sample_shape
            ),
            keys,
            self.event_dims,
        )

    @staticmethod
    def _get_q_prob(params, sample):
        log_probs = jax.tree_map(
            lambda mean, log_scale, sample: MultivariateNormalDiag(mean, jnp.exp(log_scale)).log_prob(sample),
            params["mean"],
            params["log_scale"],
            sample,
        )
        return jnp.sum(jnp.array(jax.tree_leaves(log_probs)))

    def objective_per_mc_sample(self, params, epsilons, data):
        rsample = jax.tree_map(
            lambda mean, log_scale, epsilon: mean + jnp.exp(log_scale) * epsilon,
            params["mean"],
            params["log_scale"],
            epsilons,
        )
        q_prob = self._get_q_prob(params, rsample)

        transformed_rsample = jax.tree_map(lambda f, x: f().forward(x), self.prior_transforms, rsample)
        log_jac = jax.tree_map(lambda f, x: self._get_log_jac(f, x), self.prior_transforms, rsample)
        log_prior = jax.tree_map(lambda dist, x: dist().log_prob(x), self.prior_dists, transformed_rsample)

        p_probs = jax.tree_map(lambda x, y: x + y, log_jac, log_prior)
        p_prob = sum(jax.tree_leaves(p_probs))

        log_likelihood = self.log_likelihood_fun(data, transformed_rsample)

        return q_prob - (p_prob + log_likelihood)

    def objective_fun(self, params, epsilons, data):
        idx = jnp.arange(jax.tree_leaves(epsilons)[0].shape[0])
        losses = jax.vmap(lambda i: self.objective_per_mc_sample(params, jax.tree_map(lambda x: x[i], epsilons), data))(
            idx
        )
        return jnp.mean(losses)

    def sample_posterior(self, key, params, sample_shape):
        keys = jax.random.split(key, len(self.event_dims))
        sample = jax.tree_map(
            lambda key, mean, log_scale: MultivariateNormalDiag(mean, jnp.exp(log_scale)).sample(
                seed=key, sample_shape=sample_shape
            ),
            keys,
            params["mean"],
            params["log_scale"],
        )
        return sample

    def log_prob(self, params, sample):
        inverse_transforms = jax.tree_map(distrax.Inverse, self.prior_transforms)
        normal_sample = jax.tree_map(inverse_transforms, sample)
        log_probs = jax.tree_map(
            lambda mean, log_scale: MultivariateNormalDiag(mean, jnp.exp(log_scale)).log_prob(sample),
            params["mean"],
            params["log_scale"],
            normal_sample,
        )

        log_jac = jax.tree_map(lambda f, x: self._get_log_jac(f, x), inverse_transforms, sample)

        p_probs = jax.tree_map(lambda x, y: x + y, log_jac, log_probs)
        p_prob = sum(jax.tree_leaves(p_probs))
        return p_prob
