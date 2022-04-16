import jax
import tensorflow_probability.substrates.jax as tfp

dist = tfp.distributions


def initialize(key, params, initializer=dist.Normal(0.0, 1.0)):
    values, tree_def = jax.tree_util.tree_flatten(params)
    random_values = initializer.sample(seed=key, sample_shape=(len(values),))
    return jax.tree_util.tree_unflatten(tree_def, random_values)
