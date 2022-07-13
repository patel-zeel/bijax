import jax
from jax.flatten_util import ravel_pytree
import optax


def train_fn(loss_fn, params, optimizer, n_epochs, seed, return_args={}):
    value_and_grad_fn = jax.value_and_grad(loss_fn)
    state = optimizer.init(params)

    @jax.jit
    def one_step(params_and_state, seed):
        params, state = params_and_state
        loss, grads = value_and_grad_fn(params, seed=seed)
        updates, state = optimizer.update(grads, state)
        params = optax.apply_updates(params, updates)
        return (params, state), (loss, params)

    seeds = jax.random.split(seed, n_epochs)
    (params, states), (losses, params_list) = jax.lax.scan(one_step, (params, state), xs=seeds)
    return_dict = {"params": params}
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
