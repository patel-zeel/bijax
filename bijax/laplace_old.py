import logging

logger = logging.getLogger()


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()


logger.addFilter(CheckTypesFilter())

import numpy as np
import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from .core import seeds_like, inverse_transform_dist, Posterior, transform_dist_params


class ADLaplace:
    def __init__(self, prior, bijector, get_likelihood=None):
        self.prior = prior
        self.bijectors = bijector
        self.get_likelihood = get_likelihood

        self.approx_normal_distributions = inverse_transform_dist(prior, bijector)

    def init(self, seed):
        seeds = seeds_like(self.prior, seed)
        normal_prior = inverse_transform_dist(self.prior, self.bijector)
        params = jax.tree_map(
            lambda seed, dist: dist.sample(seed=seed),
            seeds,
            normal_prior,
        )
        for param in params.values():
            assert param.ndim <= 1, "params must be scalar or vector"
        return params

    def loss_fn(self, params, outputs, inputs=None):
        transformed_params = jax.tree_map(lambda param, bijector: bijector.forward(param), params, self.bijectors)
        prior_log_probs = jax.tree_map(
            lambda param, dist: dist.log_prob(param),
            params,
            self.approx_normal_distributions,
        )

        def likelihood_log_prob(params, outputs, inputs):
            likelihood = self.get_likelihood(params, inputs)
            return likelihood.log_prob(outputs)

        likelihood_log_probs = jax.vmap(likelihood_log_prob, in_axes=(None, 0, 0))(transformed_params, outputs, inputs)
        loss = -(likelihood_log_probs.sum() + sum(jax.tree_leaves(prior_log_probs)))
        return loss

    def apply(self, params, outputs, inputs=None):
        precision = jax.hessian(self.loss_fn)(params, outputs, inputs)
        event_shapes = jax.tree_map(lambda _, dist: dist.event_shape, self.guide, self.prior)
        return Posterior(params, precision, self.bijectors, event_shapes)


# class Posterior:
#     def __init__(self, params, precision, bijectors, event_shapes):
#         self.params = jax.tree_map(lambda x: x, params)
#         self.guide = jax.tree_structure(self.params)
#         self.precision = jax.tree_map(lambda x: x, precision)
#         self.bijectors = jax.tree_map(lambda x: x, bijectors)
#         self.event_shapes = jax.tree_map(lambda x: x, event_shapes)

#         self.size = jax.tree_map(lambda param: param.size, self.params)
#         self.cumsum_size = np.cumsum(jax.tree_leaves(self.size))[:-1]

#     def untree_precision(self):
#         precision_matrix = []
#         for start in self.precision:
#             precision_matrix.append([])
#             for end in self.precision[start]:
#                 element = self.precision[start][end].reshape((self.size[start], self.size[end]))
#                 precision_matrix[-1].append(element)
#         return jnp.block(precision_matrix)

#     def get_normal_dist(self, keys):
#         precision_matrix = self.untree_precision()
#         covariance_matrix = jnp.linalg.inv(precision_matrix)
#         params = jax.tree_map(lambda x: x, self.params)
#         rm_keys = set(params) - set(keys)
#         for key in rm_keys:
#             params.pop(key)
#         loc, untree_fn = ravel_pytree(params)
#         flat_size, unravel = ravel_pytree(self.size)
#         cumsum_size = unravel(np.cumsum([0] + flat_size.tolist())[:-1].astype(np.int32))
#         idx_list = []
#         for key in keys:
#             idx_list.extend(range(cumsum_size[key], cumsum_size[key] + self.params[key].size))
#         idx_list = np.array(idx_list)
#         dist = tfd.MultivariateNormalFullCovariance(loc, covariance_matrix[jnp.ix_(idx_list, idx_list)])
#         return dist, untree_fn

#     def sample(self, seed, sample_shape=()):
#         dist, untree_fn = self.get_normal_dist(self.params.keys())
#         normal_sample = dist.sample(sample_shape=sample_shape, seed=seed)
#         f = untree_fn
#         for _ in range(len(sample_shape)):  # vectorized transform
#             f = jax.vmap(f)
#         split_sample = f(normal_sample)
#         return jax.tree_map(lambda sample, bijector: bijector.forward(sample), split_sample, self.bijectors)

#     # To Do: This function is problamatic so, correct it with tricks like vectorized transform in "sample" method
#     # inverse_log_det_jacobian event_shape is merely doing summation, need to check if it is correct.
#     def log_prob(self, sample, sample_shape=()):
#         def log_prob_per_sample(sample):
#             bijectors = jax.tree_map(lambda x: x, self.bijectors)
#             rm_keys = set(bijectors) - set(sample)
#             for key in rm_keys:
#                 bijectors.pop(key)
#             normal_sample = jax.tree_map(lambda sample, bijector: bijector.inverse(sample), sample, bijectors)
#             log_jacobian = jax.tree_map(
#                 lambda samp, bijector: bijector.inverse_log_det_jacobian(samp), sample, bijectors
#             )
#             log_jacobian = ravel_pytree(log_jacobian)[0].sum()
#             flat_sample, _ = ravel_pytree(normal_sample)
#             dist, _ = self.get_normal_dist(sample.keys())
#             normal_log_prob = dist.log_prob(flat_sample)
#             return normal_log_prob + log_jacobian

#         f = log_prob_per_sample
#         for _ in range(len(sample_shape)):
#             f = jax.vmap(f)
#         return f(sample)

#     def sample_and_log_prob(self, seed, sample_shape=()):
#         samples = self.sample(seed, sample_shape)
#         return samples, self.log_prob(samples, sample_shape)
