import jax


def seeds_like(seed, params):
    values, treedef = jax.tree_flatten(params)
    return jax.tree_unflatten(treedef, jax.random.split(seed, len(values)))


def fill_params(seed, params, initializer):
    assert seed is not None
    values, treedef = jax.tree_flatten(params)
    seeds = seeds_like(seed, values)
    values = jax.tree_map(lambda seed, value: initializer(seed, value.shape), seeds, values)
    return jax.tree_unflatten(treedef, values)
