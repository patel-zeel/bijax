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
    def __init__(self, distributions, bijectors):
        self.distributions = distributions
        self.bijectors = bijectors
        self.guide = {key: 0 for key in self.distributions}

    def log_prob(self, sample):
        log_probs = jax.tree_map(lambda value, dist: dist.log_prob(value), sample, self.distributions)
        return sum(jax.tree_leaves(log_probs))

    def sample(self, seed, sample_shape=None):
        seeds = seeds_like(seed, self.guide)
        return jax.tree_map(
            lambda seed, dist: dist.sample(seed=seed, sample_shape=sample_shape), seeds, self.distributions
        )


class Likelihood:
    def __init__(self, likelihood, link_function):
        self.likelihood = likelihood
        self.link_function = link_function

    def get_likelihood(self, params):
        likelihood_params = self.link_function(params)
        return self.likelihood(**likelihood_params)

    def sample(self, seed, params, sample_shape):
        likelihood = self.get_dist(params)
        return likelihood.sample(seed=seed, sample_shape=sample_shape)

    def log_prob(self, params, data):
        likelihood = self.get_likelihood(params)
        return jnp.sum(jax.vmap(likelihood.log_prob)(data))


class Variational:
    def __init__(self, prior, vi_type="full_rank"):
        self.prior = prior
        self.vi_type = vi_type

        self.guide = {key: 0 for key in self.prior.distributions.keys()}
        self.distribution = jax.tree_map(
            lambda _, dist, bijector: self.get_variational(dist, bijector),
            self.guide,
            self.prior.distributions,
            self.prior.bijectors,
        )

    def get_variational(self, dist, bijector):  # Call only once
        seed = jax.random.PRNGKey(0)
        sample = dist.sample(seed=seed)
        shape = sample.shape
        if isinstance(dist, tfd.CholeskyLKJ):
            dim = shape[0] * (shape[0] - 1) // 2
        else:
            dim = reduce(lambda x, y: x * y, shape)
        flat_shape = (dim,)

        if self.vi_type == "mean_field":
            params = {"loc": jnp.zeros(flat_shape), "scale_diag": jnp.ones(flat_shape)}
            params_transforms = [tfb.Identity().foward, tfb.Exp().forward]
            normal_dist = tfd.MultivariateNormalDiag
        elif self.vi_type == "full_rank":
            params = {"loc": jnp.zeros(flat_shape), "scale_tril": jnp.ones(dim * (dim + 1) // 2)}
            params_transforms = [tfb.Identity().forward, tfp.math.fill_triangular]
            normal_dist = tfd.MultivariateNormalTriL

        transformed_params = jax.tree_map(lambda param, transform: transform(param), params, params_transforms)
        dist = normal_dist(**transformed_params)
        transformed_dist = tfd.TransformedDistribution(dist, bijector)
        return {
            "shape": shape,
            "params": params,
            "revert_shape": flat_shape,
            "params_transforms": params_transforms,
            "transformed_dist": transformed_dist,
        }

    def _sample_dist(self, dist_dict, seed, sample_shape=()):
        sample = dist_dict["transformed_dist"].sample(seed=seed, sample_shape=sample_shape)
        final_shape = sample_shape + dist_dict["shape"]
        return sample.reshape(final_shape)

    def sample(self, seed, sample_shape=()):
        seeds = seeds_like(seed, self.guide)
        sample = jax.tree_map(
            lambda seed, dist_dict: self._sample_dist(dist_dict, seed, sample_shape), seeds, self.distribution
        )
        return sample

    def _log_prob_dist(self, dist_dict, sample):
        sample = sample.reshape(dist_dict["revert_shape"])
        return dist_dict["transformed_dist"].log_prob(sample)

    def log_prob(self, sample):
        log_probs = jax.tree_map(
            lambda _, dist_dict, tmp_sample: self._log_prob_dist(dist_dict, tmp_sample),
            self.guide,
            self.distribution,
            sample,
        )
        return sum(jax.tree_leaves(log_probs))

    def sample_and_log_prob(self, seed, sample_shape):
        sample = self.sample(seed=seed, sample_shape=sample_shape)
        log_prob = jax.vmap(self.log_prob)(sample)
        return sample, log_prob

    def update_dist(self, dist_dict, params):
        transformed_params = jax.tree_map(
            lambda param, transform: transform(param), params, dist_dict["params_transforms"]
        )
        tree_def = jax.tree_structure(dist_dict["transformed_dist"])
        transformed_dist = jax.tree_unflatten(tree_def, jax.tree_leaves(transformed_params))
        dist_dict["transformed_dist"] = transformed_dist

    def set_params(self, params):
        # ToDo: check if params matches required shape
        jax.tree_map(lambda _, dist_dict, param: self.update_dist, self.guide, self.distribution, params)
        jax.tree_map(lambda _, x, param: x.update({"params": param}), self.guide, self.distribution, params)

    def get_params(self):
        params = jax.tree_map(lambda _, x: x["params"], self.guide, self.distribution)
        return jax.tree_map(lambda x: x, params)  # Returns a copy
