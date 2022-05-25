import jax


def seeds_like(seed, params):
    values, treedef = jax.tree_flatten(params)
    return jax.tree_unflatten(treedef, jax.random.split(seed, len(values)))


def fill_params(seed, params, initializer):
    assert seed is not None
    seeds = seeds_like(seed, params)
    params = jax.tree_map(lambda seed, param: initializer(seed, (len(param),)), seeds, params)
    return params
