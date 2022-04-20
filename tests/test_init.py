import jax
import jax.numpy as jnp
import distrax
from advi_jax.init import initialize


def test_init():
    key = jax.random.PRNGKey(0)
    params = {"a": jnp.array([1, 2, 3]), "b": jnp.array([4, 5, 6])}
    params = initialize(key, params)
