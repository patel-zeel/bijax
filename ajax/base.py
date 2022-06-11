import logging

logger = logging.getLogger()


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()


logger.addFilter(CheckTypesFilter())

from functools import reduce
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from .utils import seeds_like


class Prior:
    def __init__(self, distributions):
        self.distributions = distributions
        self.guide = {key: 0 for key in self.distributions}

    def log_prob(self, sample):
        log_probs = jax.tree_map(lambda value, dist: dist.log_prob(value), sample, self.distributions)
        return sum(jax.tree_leaves(log_probs))

    def sample(self, seed, sample_shape=()):
        seeds = seeds_like(seed, self.guide)
        return jax.tree_map(
            lambda seed, dist: dist.sample(seed=seed, sample_shape=sample_shape), seeds, self.distributions
        )


class Likelihood:
    def __init__(self, likelihood, get_likelihood_params=None):
        self.likelihood = likelihood
        if get_likelihood_params is None:
            self.get_likelihood_params = lambda params: params
        else:
            self.get_likelihood_params = get_likelihood_params

    def get_likelihood(self, params):
        likelihood_params = self.get_likelihood_params(params)
        return self.likelihood(**likelihood_params)

    def sample(self, seed, params, sample_shape=()):
        likelihood = self.get_likelihood(params)
        return likelihood.sample(seed=seed, sample_shape=sample_shape)

    def log_prob(self, params, data):
        likelihood = self.get_likelihood(params)
        return jnp.sum(likelihood.log_prob(data))


class Variational:
    def __init__(self, prior, bijectors, vi_type="full_rank"):
        self.prior = prior
        self.bijectors = bijectors
        self.vi_type = vi_type
        # self.distribution = tfd.Normal(loc=0, scale=1)

        # dummy pytree to help vectorization with tree_map
        self.guide = {key: 0 for key in self.prior.distributions.keys()}

        # bijectors for mean and variance of the variational distribution

        if vi_type == "mean_field":
            # Be cautious with the order of the bijectors
            self.params_transforms = [tfb.Identity(), tfb.Exp()]
        elif vi_type == "full_rank":
            self.params_transforms = [tfb.Identity(), tfb.Identity()]
        else:
            raise ValueError(f"Unknown vi_type {vi_type}")

        # initialize shapes
        def get_shape(dist):
            return tuple(dist.batch_shape.as_list() + dist.event_shape.as_list())

        def get_flat_shape(dist):
            if isinstance(dist, tfd.CholeskyLKJ):
                assert dist.batch_shape.as_list() == [], "CholeskyLKJ multi-batch is not supported"
                return (dist.event_shape[0] * (dist.event_shape[0] - 1)) // 2
            shape = get_shape(dist)
            if shape == ():
                return 1
            return reduce(lambda a, b: a * b, shape)

        self.shapes = jax.tree_map(lambda _, dist: get_shape(dist), self.guide, self.prior.distributions)
        self.flat_shapes = jax.tree_map(lambda _, dist: get_flat_shape(dist), self.guide, self.prior.distributions)
        self.params = self.init_params()

    def init_params(self):
        def get_variational(bijector, flat_shape):
            if self.vi_type == "mean_field":
                normal_dist = tfd.MultivariateNormalDiag(loc=jnp.zeros(flat_shape), scale_diag=jnp.ones(flat_shape))
            elif self.vi_type == "full_rank":
                normal_dist = tfd.MultivariateNormalTriL(loc=jnp.zeros(flat_shape), scale_tril=jnp.eye(flat_shape))
            return tfd.TransformedDistribution(normal_dist, bijector)

        return jax.tree_map(
            lambda _, flat_shape, bijector: get_variational(bijector, flat_shape),
            self.guide,
            self.flat_shapes,
            self.bijectors,
        )

    def transform_dist(self, dist):
        values, treedef = jax.tree_flatten(dist)
        values = jax.tree_map(lambda value, transform: transform(value), values, self.params_transforms)
        return jax.tree_unflatten(treedef, values)

    def sample(self, seed, sample_shape=()):
        def sample_dist(seed, dist, sample_shape, shape):
            dist = self.transform_dist(dist)
            sample = dist.sample(seed=seed, sample_shape=sample_shape)
            final_shape = sample_shape + shape
            return sample.reshape(final_shape)

        seeds = seeds_like(seed, self.guide)
        return jax.tree_map(
            lambda seed, dist, shape: sample_dist(seed, dist, sample_shape, shape), seeds, self.params, self.shapes
        )

    def log_prob(self, sample):
        def log_prob_dist(dist, sample, flat_shape):
            if not isinstance(dist.bijector, tfb.CorrelationCholesky):
                sample = sample.reshape(-1, flat_shape)
            dist = self.transform_dist(dist)
            return dist.log_prob(sample)

        log_probs = jax.tree_map(
            lambda tmp_sample, dist, flat_shape: log_prob_dist(dist, tmp_sample, flat_shape),
            sample,
            self.params,
            self.flat_shapes,
        )
        return sum(jax.tree_leaves(log_probs))

    def get_params(self):
        return jax.tree_map(lambda x: x, self.params)

    def set_params(self, params):
        self.params = params

