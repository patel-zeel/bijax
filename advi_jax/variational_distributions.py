from importlib.metadata import distribution
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from tinygp.helpers import JAXArray, dataclass, field
import distrax

dist = tfp.distributions


@dataclass
class MeanField:
    """
    u_mean: un-bounded mean
    u_scale: un-bounded scale

    # TODO: Think how to pass in the transform function
    transform_fn: transform function to apply to unbounded scale
    """

    u_mean: JAXArray = field(default_factory=lambda: jnp.zeros(()))
    u_scale: JAXArray = field(default_factory=lambda: jnp.zeros(()))
    bijector: distrax.Bijector = field(default_factory=lambda: distrax.Identity())

    def sample(self, key):
        scale = jnp.exp(self.u_scale)  # positivity constraint
        normal_posterior = dist.Normal(loc=self.u_mean, scale=scale)
        transformed_posterior = distrax.Transformed(normal_posterior, self.bijector)

        sample = transformed_posterior.sample(seed=key, sample_shape=self.u_mean.shape)

        return sample

    def log_prob(self, sample):
        scale = jnp.exp(self.u_scale)  # positivity constraint
        normal_posterior = dist.Normal(loc=self.u_mean, scale=scale)
        transformed_posterior = distrax.Transformed(normal_posterior, self.bijector)

        return transformed_posterior.log_prob(sample)

    def prob(self, sample):
        return jnp.exp(self.log_prob(sample))

    def sample_and_log_prob(self, key):
        scale = jnp.exp(self.u_scale)  # positivity constraint
        normal_posterior = dist.Normal(loc=self.u_mean, scale=scale)
        transformed_posterior = distrax.Transformed(normal_posterior, self.bijector)

        sample, log_prob = transformed_posterior.sample_and_log_prob(seed=key, sample_shape=self.u_mean.shape)

        return sample, log_prob


# Another version


@dataclass
class MeanFieldV1:
    """
    u_mean: un-bounded mean
    u_scale: un-bounded scale

    # TODO: Think how to pass in the transform function
    transform_fn: transform function to apply to unbounded scale
    """

    u_mean: JAXArray = field(default_factory=lambda: jnp.zeros(()))
    u_scale: JAXArray = field(default_factory=lambda: jnp.zeros(()))

    def sample(self, key):
        standard_normal_dist = dist.Normal(loc=0.0, scale=1.0)
        sample = standard_normal_dist.sample(seed=key, sample_shape=self.u_mean.shape)

        scale = jnp.exp(self.u_scale)  # positivity constraint

        reparametrized_sample = self.u_mean + scale * sample
        return reparametrized_sample

    def log_prob(self, sample):
        scale = jnp.exp(self.u_scale)  # positivity constraint
        distribution = dist.Normal(loc=self.u_mean, scale=scale)
        return distribution.log_prob(sample)

    def sample_and_log_prob(self, key):
        sample = self.sample(key)
        return sample, self.log_prob(sample)
