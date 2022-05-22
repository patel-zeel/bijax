import jax
import numpy as np
import jax.numpy as jnp
import distrax
from distrax import MultivariateNormalDiag, MultivariateNormalDiagPlusLowRank, MultivariateNormalTri
from jax.experimental.host_callback import id_print  # Useful for print debugging


class ADVI:
    def __init__(self, prior_dists, prior_transforms, log_likelihood_fun, vi_type="mean_field", rank=None):
        self.prior_dists = jax.tree_map(lambda x: x, prior_dists)  # sorting
        self.prior_transforms = prior_transforms
        self.log_likelihood_fun = log_likelihood_fun
        self.vi_type = vi_type

        self.event_dims = {key: self._get_event_dim(dist) for key, dist in self.prior_dists.items()}
        self.n_event_dims = sum(jax.tree_leaves(self.event_dims))

        self.tree_def = jax.tree_structure(self.event_dims)

        if self.vi_type == "low_rank":
            self.rank = rank

        # initialize variational distribution and params
        self.init_variational_dist_and_params()

    @staticmethod
    def _get_event_dim(dist):
        event_shape = dist.event_shape
        return 1 if event_shape == () else event_shape[0]

    def init_variational_dist_and_params(self):
        # Initialize variational distribution
        loc = jnp.zeros((self.n_event_dims,))
        if self.vi_type == "mean_field":
            scale_diag = jnp.ones((self.n_event_dims,))
            self._params = {"loc": loc, "scale_diag": scale_diag}
            self._params_transforms = {"loc": lambda x: x, "scale_diag": lambda x: jnp.exp(x)}
            self.variational_dist = MultivariateNormalDiag
        elif self.vi_type == "low_rank":
            scale_diag = jnp.ones((self.n_event_dims,))
            scale_u_matrix = jnp.ones((self.n_event_dims, self.rank))
            self._params = {"loc": loc, "scale_diag": scale_diag, "scale_u_matrix": scale_u_matrix}
            self._params_transforms = {
                "loc": lambda x: x,
                "scale_diag": lambda x: jnp.exp(x),
                "scale_u_matrix": lambda x: x,
            }
            self.variational_dist = MultivariateNormalDiagPlusLowRank
        elif self.vi_type == "full_covariance":
            scale_tri = jnp.ones((self.n_event_dims, self.n_event_dims))
            self._params = {"loc": loc, "scale_tri": scale_tri}
            self._params_transforms = {"loc": lambda x: x, "scale_tri": lambda x: jnp.tril}
            self.variational_dist = MultivariateNormalTri
        else:
            raise ValueError("Unknown variational distribution type")

    # @property
    # def params(self):
    #     return jax.tree_map(lambda f, param: f(param), self._params_transforms, self._params)

    def objective_per_mc_sample(self, params, key, data):
        # Sample epsilons
        sample = jax.random.normal(key, shape=(self.n_event_dims,))

        # Get log_prob of sample in variational distribution
        params = jax.tree_map(lambda f, param: f(param), self._params_transforms, self._params)
        q_prob = self.variational_dist(**params).log_prob(sample)

        split_sample = jnp.split(sample, np.cumsum(jax.tree_leaves(self.event_dims))[:-1])
        split_sample = jax.tree_unflatten(self.tree_def, split_sample)
        transformed_sample = jax.tree_map(
            lambda x, transform: transform.forward(x), split_sample, self.prior_transforms
        )
        transformed_prior = [
            distrax.Transformed(dist, transform)
            for dist, transform in zip(self.prior_dists.values(), self.prior_transforms.values())
        ]
        transformed_prior = jax.tree_unflatten(self.tree_def, transformed_prior)

        log_prior = sum(jax.tree_map(lambda sample, dist: dist.log_prob(sample), transformed_sample, transformed_prior))
        log_likelihood = self.log_likelihood_fun(data, split_sample)

        return q_prob - (log_prior + log_likelihood)

    def objective_fun(self, params, keys, data):
        losses = jax.vmap(self.objective_per_mc_sample, in_axes=(None, 0, None))(params, keys, data)
        return losses.mean()

    def sample_posterior(self, key, sample_shape):
        key = jax.random.split(key, len(self.event_dims))

    def log_prob(self, params, sample):
        inverse_transforms = jax.tree_map(distrax.Inverse, self.prior_transforms)
        normal_sample = jax.tree_map(inverse_transforms, sample)
        log_probs = jax.tree_map(
            lambda mean, log_scale: MultivariateNormalDiag(mean, jnp.exp(log_scale)).log_prob(sample),
            params["mean"],
            params["log_scale"],
            normal_sample,
        )

        log_jac = jax.tree_map(lambda f, x: self._get_log_jac(f, x), inverse_transforms, sample)

        p_probs = jax.tree_map(lambda x, y: x + y, log_jac, log_probs)
        p_prob = sum(jax.tree_leaves(p_probs))
        return p_prob
