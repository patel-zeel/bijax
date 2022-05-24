## AJAX

This repo contains the ADVI implementation in JAX. The original paper is [here](https://www.jmlr.org/papers/volume18/16-107/16-107.pdf).

### Installation

```
pip install advi_jax
```

### Brain storming

* We may be able to use normal functions instead of distrax.bijector by inverting the jacobian.
* distrax bijectors have forward method and inverse method.
* vmap works on pytrees with vector values and functions with pytree outputs.