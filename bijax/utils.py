import jax
from jax.flatten_util import ravel_pytree
import jax.tree_util as jtu
import jax.numpy as jnp
import optax

import tensorflow_probability.substrates.jax as tfp
from typing import Union
from jaxtyping import PyTree, Array

tfd = tfp.distributions
tfb = tfp.bijectors


def get_inverse_log_det_jacobian(objects, bijectors):
    values = jtu.tree_map(lambda sample, bijector: bijector.inverse_log_det_jacobian(sample), objects, bijectors)
    return ravel_pytree(values)[0].sum()


def constrain(
    objects: PyTree[Union[Array, tfd.Distribution]], bijectors: PyTree[tfb.Bijector]
) -> PyTree[Union[Array, tfd.Distribution]]:
    is_leaf = lambda x: isinstance(x, tfd.Distribution)
    return jax.tree_map(lambda object, bijector: bijector(object), objects, bijectors, is_leaf=is_leaf)


def unconstrain(
    objects: PyTree[Union[Array, tfd.Distribution]], bijectors: PyTree[tfb.Bijector]
) -> PyTree[Union[Array, tfd.Distribution]]:
    inverse_bijectors = jax.tree_map(tfb.Invert, bijectors, is_leaf=lambda x: isinstance(x, tfb.Bijector))
    return constrain(objects, inverse_bijectors)


def train_fn(loss_fn, params, optimizer, n_epochs, seed=None, return_args=set()):
    value_and_grad_fn = jax.value_and_grad(loss_fn)
    state = optimizer.init(params)

    if seed is None:

        @jax.jit
        def one_step(params_and_state, _):
            params, state = params_and_state
            loss, grads = value_and_grad_fn(params)
            updates, state = optimizer.update(grads, state)
            params = optax.apply_updates(params, updates)
            return (params, state), (loss, params)

        (params, states), (losses, params_list) = jax.lax.scan(one_step, (params, state), xs=None, length=n_epochs)
    else:

        @jax.jit
        def one_step(params_and_state, seed):
            params, state = params_and_state
            loss, grads = value_and_grad_fn(params, seed=seed)
            updates, state = optimizer.update(grads, state)
            params = optax.apply_updates(params, updates)
            return (params, state), (loss, params)

        seeds = jax.random.split(seed, n_epochs)
        (params, states), (losses, params_list) = jax.lax.scan(one_step, (params, state), xs=seeds)

    return_dict = {"params": params, "losses": losses}
    for key in return_args:
        return_dict[key] = locals()[key]
    return return_dict


def initialize_params(params, seed, initializer):
    """Initialize parameters with a given initializer.

    Args:
        seed: JAX PRNGKey
        params: parameters to initialize
        initializer: one of the jax.nn.initializers or a callable that takes a PRNGKey and shape and returns DeviceArray.
                     example: jax.nn.initializers.normal(stddev=0.1)
                     example: lambda seed, shape: jax.random.uniform(seed, shape, minval=0, maxval=1)
                     example: lambda seed, shape: jnp.zeros(shape)

    Returns:
        DeviceArray: initialized parameters
    """
    values, unravel_fn = ravel_pytree(params)
    random_values = initializer(seed, (len(values),))
    return unravel_fn(random_values)


def seeds_like(params, seed, is_leaf=None):
    """Generate seeds for a tree of parameters.

    Args:
        params: parameters.
        seed: JAX PRNGKey.
        is_leaf: same as `jax.tree_map`.

    Returns:
        seeds: JAX PRNGKeys for the tree of parameters.
    """
    values, treedef = jax.tree_flatten(params, is_leaf=is_leaf)
    return jax.tree_unflatten(treedef, jax.random.split(seed, len(values)))


def fill_in_bijectors(bijector, distribution):
    additional_keys = distribution.keys() - bijector.keys()
    identity_bijectors = {key: tfb.Identity() for key in additional_keys}
    return {**bijector, **identity_bijectors}

def get_shapes(params):
    return jtu.tree_map(lambda x:x.shape, params)

def svd_inverse(matrix, jitter = 1e-6):
    U, S, V = jnp.linalg.svd(matrix+jnp.eye(matrix.shape[0])*jitter)
    
    return V.T/S@U.T