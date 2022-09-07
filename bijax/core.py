import jax
from jax.flatten_util import ravel_pytree
import jax.tree_util as jtu
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from bijax.utils import fill_in_bijectors


class DistributionPyTree:
    def __init__(self, base_distribution, bijector):
        self.bijector = fill_in_bijectors(bijector, base_distribution)
        self.base_distribution = base_distribution

        # check if graph is valid
        _ = self.sample(seed=jax.random.PRNGKey(0))

    @staticmethod
    def get_distribution(distribution, bijector):
        if isinstance(distribution, tfd.Distribution):
            return bijector(distribution)
        elif callable(distribution):
            var_names = distribution.__code__.co_varnames
            distribution_fn = lambda **kwargs: bijector(distribution(**kwargs))
            return (distribution_fn, var_names)
        else:
            raise ValueError("distribution must be a tfd.Distribution or a callable")

    def sample(self, sample_shape=(), seed=None):
        sample = {}

        def sample_fn(name, key):
            if name in sample:
                return
            distribution = self.base_distribution[name]
            bijector = self.bijector[name]
            if isinstance(distribution, tfd.Distribution):
                sample[name] = bijector(distribution).sample(sample_shape, seed=key)
            elif callable(distribution):
                var_names = distribution.__code__.co_varnames
                keys = jax.random.split(key, len(var_names))
                keys = [key for key in keys]
                jtu.tree_map(sample_fn, var_names, keys)
                kwargs = {name: sample[name] for name in var_names}
                sample[name] = bijector(distribution(**kwargs)).sample(sample_shape, seed=key)

        names = list(self.base_distribution.keys())
        keys = jax.random.split(seed, len(self.base_distribution))
        keys = [key for key in keys]
        jtu.tree_map(sample_fn, names, keys)

        return sample

    def log_prob(self, sample):
        def log_prob_fn(sample_value, distribution, bijector):
            if isinstance(distribution, tfd.Distribution):
                return bijector(distribution).log_prob(sample_value)
            elif callable(distribution):
                var_names = distribution.__code__.co_varnames
                kwargs = {name: sample[name] for name in var_names}
                return bijector(distribution(**kwargs)).log_prob(sample_value)
            else:
                raise ValueError("distribution must be a Distribution or a callable")

        return jtu.tree_map(log_prob_fn, sample, self.base_distribution, self.bijector)
