from importlib.metadata import distribution
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from tinygp.helpers import JAXArray, dataclass, field
import distrax

dist = tfp.distributions


@dataclass  # dataclass_pytree
class MeanField:
    """
    u_mean: un-bounded mean
    u_scale: un-bounded scale
    """

    u_mean: JAXArray = field(default_factory=lambda: jnp.zeros(()))
    u_scale: JAXArray = field(default_factory=lambda: jnp.zeros(()))
    bijector: distrax.Bijector = field(default_factory=lambda: distrax.Lambda(lambda x: x))

    def sample(self, key):
        scale = jnp.exp(self.u_scale)  # positivity constraint
        normal_posterior = dist.Independent(dist.Normal, self.u_mean.shape[0])(loc=self.u_mean, scale_diag=scale)
        bijector = distrax.Block(distrax.Lambda(lambda x: x), (self.u_mean.shape[0],))
        transformed_posterior = distrax.Transformed(normal_posterior, bijector)

        sample = transformed_posterior.sample(seed=key)

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

        sample, log_prob = transformed_posterior.sample_and_log_prob(seed=key)

        return sample, log_prob

    # def __repr__(self):
    #     return "MeanField(mean={}, scale={})".format(self.u_mean, jnp.exp(self.u_scale))


@dataclass  # dataclass_pytree
class FullRank:
    """
    u_mean: un-bounded mean
    u_tril: un-bounded tril
    """

    u_mean: JAXArray = field(default_factory=lambda: jnp.zeros(()))
    u_scale_tril: JAXArray = field(default_factory=lambda: jnp.zeros((1, 1)))
    bijector: distrax.Bijector = 1.0  # = field(default_factory=lambda: distrax.Lambda(lambda x: x))

    def _get_transformed_dist(self):
        normal_posterior = dist.MultivariateNormalTriL(loc=self.u_mean, scale_tril=self.u_scale_tril)
        # return distrax.Transformed(normal_posterior, self.bijector)
        return normal_posterior

    def sample(self, key):
        return self._get_transformed_dist().sample(seed=key)

    def log_prob(self, sample):
        return self._get_transformed_dist().log_prob(sample)

    def prob(self, sample):
        return jnp.exp(self.log_prob(sample))

    def sample_and_log_prob(self, key):
        sample, log_prob = self._get_transformed_dist()._sample_and_log_prob(seed=key, sample_shape=(1,))
        return sample, log_prob


# Another version


@dataclass
class MeanFieldV1:
    """
    u_mean: un-bounded mean
    u_scale: un-bounded scale
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
