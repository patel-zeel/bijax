import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from functools import partial

dist = tfp.distributions


def get_random_sample(key, shape, initializer):
    random_sample = initializer.sample(seed=key, sample_shape=shape)
    return random_sample


def initialize(key, params, initializer=dist.Normal(0.0, 1.0)):
    values, treedef = jax.tree_flatten(params)
    shapes = jax.tree_map(lambda x: jnp.asarray(x).shape, values)
    keys = jax.random.split(key, len(shapes))
    random_values = [get_random_sample(keys[i], shapes[i], initializer) for i in range(len(shapes))]
    return jax.tree_unflatten(treedef, random_values)
