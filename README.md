## AJAX

A JAX library for approximate Bayesian inference.

## Installation

```
pip install git+https://github.com/patel-zeel/ajax.git
```

## Methods implemented in AJAX

* `from ajax.advi import ADVI` - [Automatic Differentiation Variational Inference](https://arxiv.org/abs/1603.00788)
* `from ajax.laplace import ADLaplace` - Automatic Differentiation Laplace approximation.
* `from ajax.mcmc import MCMC` - A helper class for external Markov Chain Monte Carlo (MCMC) sampling.

## How to use AJAX?

AJAX is built without too many layers of abstractions or some new conventions. Thus, it is also useful for educational purposes. If you like to directly dive into the examples, please refer to the [examples](examples) directory.


There are a few core components of AJAX:

### Prior
`tensoflow_probability.substrates.jax` should be used to define the distributions for prior.

```python
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
```

Prior distribution for the coin toss problem can be defined as follows:

```python
prior = {"p_of_heads": tfd.Beta(0.5, 0.5)}
```

Prior distribution for the Linear Regression problem can be defined as follows:

```python
shape_of_weights = 5
prior = {"weights": tfd.MultivariateNormalDiag(
                                               loc=tf.zeros(shape_of_weights), 
                                               scale_diag=tf.ones(shape_of_weights)
                                            )}
```

### Bijectors
Bijectors available in `tensorflow_probability.substrates.jax` are used to facilitate the change of variable trick or change of support trick. Here, a bijector should transform a Gaussian random variable with infinite support to a transformed random variable with finite support.

```python
import tensorflow_probability.substrates.jax as tfp
tfb = tfp.bijectors
```

To perform Automatic Differentiation Variational Inference for the coin toss problem, a bijector can be defined as follows:

```python
prior = {"p_of_heads": tfd.Beta(0.5, 0.5)}
bijector = {"p_of_heads": tfb.Sigmoid()}
```

For the Linear Regression problem, a bijector can be defined as follows:

```python
shape_of_weights = 5
prior = {"weights": tfd.MultivariateNormalDiag(
                                               loc=tf.zeros(shape_of_weights), 
                                               scale_diag=tf.ones(shape_of_weights)
                                            )}
bijector = {"weights": tfb.Identity()}
```

### Likelihood
Users have total freedom on how to define the log likelihood function adhering to several conditions. The log likelihood function should take the following arguments:

* latent_sample: a dictionary of values that represents a sample taken from the latent (prior) parameter distributions. It will have same keys as the prior.
* data: Data generated from the likelihood. We will find log probability of the `data` given the prior sample.
* aux: Auxiliary data required to evaluate the likelihood. For example, in the Linear Regression problem, `X` is the auxiliary data. For the coin toss problem, `aux` is None.
* kwargs: We internally pass the trainable `params` as `kwargs` to the likelihood function. So, the user can mention additional learnable parameters in `kwargs` and they will be trained.

For coin toss problem, we can define the log likelihood function as follows:

```python
def log_likelihood_fn(latent_sample, data, aux, **kwargs):
    p_of_heads = latent_sample["p_of_heads"]
    log_likelihood = tfd.Bernoulli(probs=p_of_heads).log_prob(data).sum()
    return log_likelihood
```

For the Linear Regression problem with learnable noise variance, we can define the log likelihood function as follows:

```python
def log_likelihood_fn(latent_sample, data, aux, **kwargs):
    weights = latent_sample["weights"]
    loc = jnp.dot(weights, aux["X"])
    noise_variance = jnp.exp(kwargs["log_noise_scale"])
    log_likelihood = tfd.MultivariateNormalDiag(loc=loc, scale_diag=noise_variance).log_prob(data).sum()
    return log_likelihood
```

### Initialization
We can automatically initialize the parameters of the model.

Here is an example with ADVI model.
```python
model = ADVI(prior, bijector, log_likelihood_fn, vi_type="mean_field")
seed = jax.random.PRNGKey(0)
params = model.init(seed)
```

### Optimization
Models in AJAX have `loss_fn` method which can be used to compute the loss. The loss can be optimized with any method that work with `JAX`. We also have a utility function `from ajax.utils import train` to train the model using `optax` optimizers.

### Get the posterior distribution
Some of the models (`ADVI` and `ADLaplace`) support `.apply()` method to get the posterior distribution.

```python
posterior = model.apply(params, ...)
posterior.sample(...)
posterior.log_prob(...)
```

## Gotchas

* Each component in prior should return a single scalar `log_prob`. Use `tfd.Independent` wherever required to ensure that each component of prior has no batch dimensions.
