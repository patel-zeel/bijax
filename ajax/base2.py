import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
import distrax

tfd = tfp.distributions


class Prior:
    def __init__(self, distributions, transforms):
        self.distributions = jax.tree_map(lambda x: x, distributions)
        self.transforms = jax.tree_map(lambda x: x, transforms)
        self.inverse_transforms = {key: distrax.Inverse(transform) for key, transform in transforms.items()}

    def _log_prob(self, sample):
        log_probs = jax.tree_map(lambda value, dist: dist.log_prob(value), sample, self.distributions)
        return sum(jax.tree_leaves(log_probs))

    def log_prob(self, sample):
        return jax.vmap(self._log_prob)(sample)

    def sample(self, seed, sample_shape=None):
        sample = {}
        for key, dist in self.distributions.items():
            seed = jax.random.split(seed, 1)[0]
            dist_sample = dist.sample(seed=seed, sample_shape=sample_shape)
            sample.update({key: dist_sample})
        return sample


class Variational:
    def __init__(self, prior, vi_type="mean_field", rank=None):
        self.prior = prior
        self.vi_type = vi_type
        self.rank = rank

        # initialize variational distribution
        self.lengths, self.shapes = self.get_params_stats()
        self.init_dist()

    def get_params_stats(self):
        lengths = {}
        shapes = {}
        for key, dist in self.prior.distributions.items():
            shape = dist.batch_shape + dist.event_shape
            if shape == ():
                lengths.update({key: 1})
            elif isinstance(dist, tfd.CholeskyLKJ):
                n = dist.event_shape[0]
                lengths.update({key: (n * (n + 1)) // 2})
            else:
                lengths.update({key: sum(shape)})
            shapes.update({key: shape})
        return lengths, shapes

    def init_dist(self):
        length = sum(jax.tree_leaves(self.lengths))
        loc = jnp.zeros((length,))
        if self.vi_type == "mean_field":
            scale_diag = jnp.ones((length,))
            self.raw_params = {"loc": loc, "scale_diag": scale_diag}
            self.params_transforms = {"loc": lambda x: x, "scale_diag": lambda x: jnp.exp(x)}
            self.variational_dist = distrax.MultivariateNormalDiag
        elif self.vi_type == "full_rank":
            scale_tri = jnp.eye(length)
            self.raw_params = {"loc": loc, "scale_tri": scale_tri}
            self.params_transforms = {"loc": lambda x: x, "scale_tri": lambda x: x}
            self.variational_dist = distrax.MultivariateNormalTri

    def _normal_sample(self, seed, sample_shape):
        params = jax.tree_map(lambda transform, params: transform(params), self.params_transforms, self.raw_params)
        dist = self.variational_dist(**params)
        sample = dist.sample(seed=seed, sample_shape=sample_shape)
        return sample

    def get_jac(self, transform, sample):
        return jnp.abs(jnp.linalg.det(jax.jacobian(transform.forward)(sample)))

    def _log_prob_from_normal_sample(self, normal_sample):
        params = jax.tree_map(lambda transform, params: transform(params), self.params_transforms, self.raw_params)
        dist = self.variational_dist(**params)
        log_prob = dist.log_prob(normal_sample)
        sample_dict = self._unflatten_sample(normal_sample)
        jac_dict = jax.tree_map(
            lambda sample, transform: self.get_jac(transform, sample.ravel()), sample_dict, self.prior.transforms
        )
        log_jac = jnp.log(jnp.prod(jnp.array(jax.tree_leaves(jac_dict))))
        return log_prob - log_jac

    def log_prob_from_normal_sample(self, normal_sample):
        return jax.vmap(self._log_prob_from_normal_sample)(normal_sample)

    def _transform_sample(self, normal_sample):
        sample = self._unflatten_sample(normal_sample)
        transformed_sample = jax.tree_map(
            lambda sample, transform: transform.forward(sample), sample, self.prior.transforms
        )
        return transformed_sample

    def transform_sample(self, sample):
        return jax.vmap(self._transform_sample)(sample)

    def sample(self, seed, sample_shape=None, return_normal_sample=False):
        normal_sample = self._normal_sample(seed, sample_shape)
        transformed_sample = jax.vmap(self._transform_sample)(normal_sample)

    def log_prob(self, sample):
        normal_sample = jax.tree_map(lambda transform, sample: transform(sample), self.prior.inverse_transforms, sample)
        normal_sample = self._flatten_sample(normal_sample)
        return self._log_prob_from_normal_sample(normal_sample)

    def _unflatten_sample(self, sample):
        offset = 0
        sample_dict = {}
        for key in self.shapes:
            shape = self.shapes[key]
            length = self.lengths[key]
            if isinstance(self.prior.distributions[key], tfd.CholeskyLKJ):
                dist_sample = tfp.math.fill_triangular(sample[offset : offset + length], upper=False)
            else:
                dist_sample = sample[offset : offset + length].reshape(shape)
            sample_dict.update({key: dist_sample})
            offset += length
        return sample_dict

    def _flatten_sample(self, sample):
        return jnp.concatenate(leaf.ravel() for leaf in jax.tree_leaves(sample))

    def _sample_and_log_prob(self, normal_sample):
        log_prob = self._log_prob_from_normal_sample(normal_sample)
        sample = self._unflatten_sample(normal_sample)
        transformed_sample = jax.tree_map(lambda transform, sample: transform(sample), self.prior.transforms, sample)
        return transformed_sample, log_prob

    def sample_and_log_prob(self, seed, sample_shape=None):
        normal_sample = self._normal_sample(seed, sample_shape)
        log_prob = self.log_prob_from_normal_sample(normal_sample)
        sample = self.transform_sample(normal_sample)
        return sample, log_prob
