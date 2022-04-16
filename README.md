## ADVI in JAX

### Design considerations

* ADVI class is an object but `ADVI.objective_fun` is a pure function that can be optimized with `optax` or `jaxopt` or any other jax supported optimizers.
* variational distribution parameters can be initialized with `ADVI.init` using a `distrax` or `tfp` distribution as an initializer (or any jax distribution that implements `.sample()` method in a similar way).
* Users can pass the suitable bijectors of class `distrax.Bijector` to the variational distribution.
* Transformation is directly applied to posterior and thus prior and likelihood stay untouched during the entire process. This way, after the training, the variational distribution is ready for sampling without any additional transformations. Also, this gives freedom to variational distribution to be constructed in more complex way as it is separated from the other parts of the model (see the example below).
* If we do not change the `key` during the training, the method is called the deterministic ADVI.
* Users can implement their own `likelihood_log_prob_fun` because likelihood does not necessarily have to be a 
distribution.

### A Coin Toss Example

```py
import jax
import jax.numpy as jnp

from advi_jax import ADVI
from advi_jax.variational_distributions import MeanField
from advi_jax.init import initialize
import tensorflow_probability.substrates.jax as tfp
dist = tfp.distributions

# Data
tosses = jnp.array([0, 1, 0, 0, 1, 0])

# Prior and likelihood
prior_dist = dist.Beta(2.0, 3.0)
likelihood_log_prob_fun = lambda theta: dist.Bernoulli(probs=theta).log_prob(tosses).sum()

# ADVI model
model = ADVI(prior_dist, likelihood_log_prob_fun, tosses)

# Variational distribution and bijector
bijector = distrax.Sigmoid()
variational_dist = MeanField(u_mean = jnp.array(0.0), u_scale = jnp.array(0.0), bijector = bijector)

# Initialize the parameters of variational distribution
key = jax.random.PRNGKey(0)
variational_dist = initialize(key, variational_dist, initializer=dist.Normal(0.0, 1.0))

# Define the value and grad function
value_and_grad_fun = jax.jit(jax.value_and_grad(model.objective_fun, argnums=1), static_argnums=2)

# Do gradient descent!
learning_rate = 0.01
for i in range(100):
    key = jax.random.PRNGKey(i)  # If this is constant, this becomes deterministic ADVI
    loss_value, grads = value_and_grad_fun(key, variational_dist, n_samples=10)
    variational_dist = variational_dist - learning_rate * grad

# Get the posterior samples
key = jax.random.PRNGKey(2)
posterior_samples = variational_dist.sample(seed=key, sample_shape=(100,))
```
