from multiprocessing.dummy import current_process
import jax
import jax.numpy as jnp
import distrax
import tensorflow_probability.substrates.jax as tfp
from init import initialize
from copy import deepcopy
from jax.experimental.host_callback import id_print


class ADVI_MeanField:
    def __init__(self, prior_dists, prior_transforms, log_likelihood_fun):
        self.prior_dists = {key: prior_dists[key] for key in sorted(prior_dists)}
        self.prior_transforms = {key: prior_transforms[key] for key in sorted(prior_transforms)}
        self.log_likelihood_fun = log_likelihood_fun

        # Get the number of event dimensions
        # TODO: jax.tree_map event dims
        self.event_dims = {key: self._get_event_dim(value) for key, value in self.prior_dists.items()}
        self.n_event_dims = sum(self.event_dims.values())

        self.epsilon_dist = distrax.MultivariateNormalDiag(
            jnp.zeros((self.n_event_dims,)), jnp.ones((self.n_event_dims,))
        )

    @staticmethod
    def _get_event_dim(dist):
        if dist.event_shape == ():
            return 1
        else:
            return dist.event_shape[0]

    def _get_q_prob(self, params, sample):
        current_variational_dist = distrax.MultivariateNormalDiag(params["mean"], jnp.exp(params["log_scale"]))
        return current_variational_dist.log_prob(sample)

    def objective_per_mc_sample(self, params, theta, data):
        rsample = params["mean"] + jnp.exp(params["log_scale"]) * theta

        q_prob = self._get_q_prob(params, rsample)

        offset = 0
        p_prob = 0.0
        transformed_sample = {}
        for key in self.prior_dists:
            event_dim = self.event_dims[key]
            transform = self.prior_transforms[key]
            prior_slice = self.prior_dists[key]
            rsample_slice = rsample[offset : offset + event_dim]
            transformed_rsample_slice = transform(rsample_slice)
            transformed_sample[key] = transformed_rsample_slice

            jac = jax.jacfwd(transform)(rsample_slice)
            log_jac = jnp.log(jnp.abs(jnp.linalg.det(jac)))
            log_prior = prior_slice.log_prob(transformed_rsample_slice)

            p_prob = p_prob + log_prior + log_jac

            offset += event_dim

        log_likelihood = self.log_likelihood_fun(data, transformed_sample)

        return q_prob - (p_prob + log_likelihood)

    def init(self, key=None, initializer=None):
        mean = jnp.zeros((self.n_event_dims,))
        log_scale = jnp.zeros((self.n_event_dims,))
        dists = {"mean": mean, "log_scale": log_scale}
        if initializer:
            assert key is not None, "key must be provided if initializer is provided"
            dists = initialize(key, dists, initializer)

        return dists

    def sample_posterior(self, key, params, n_samples):
        post_dist = distrax.MultivariateNormalDiag(params["mean"], jnp.exp(params["log_scale"]))

        sample = post_dist.sample(seed=key, sample_shape=(n_samples,))

        offset = 0
        transformed_sample = {}
        for key in self.prior_dists:
            event_dim = self.event_dims[key]
            transform = self.prior_transforms[key]
            sample_slice = sample[:, offset : offset + event_dim]
            transformed_sample_slice = transform(sample_slice)
            transformed_sample[key] = transformed_sample_slice

        return transformed_sample


# class ADVI:
#     def __init__(self, prior_dist, likelihood_log_prob_fun, data):
#         self.prior_dist = prior_dist
#         self.likelihood_log_prob_fun = likelihood_log_prob_fun
#         self.data = data

#         # parallelize the objective function
#         self._objective_fun_vmap = jax.vmap(self._objective_fun_per_sample, in_axes=(0, None, None))

#         # Define value and grad function
#         self.value_and_grad_fun = jax.jit(jax.value_and_grad(self.objective_fun, argnums=1), static_argnums=3)

#     def log_prior_likelihood(self, sample, data, **dists):
#         log_prior = self.prior_dist.log_prob(sample)
#         log_likelihood = self.likelihood_log_prob_fun(sample, data, **dists)
#         return log_prior + log_likelihood

#     def _objective_fun_per_sample(self, key, dists, data):
#         variational_dist = dists["variational_dist"]
#         sample, q_prob = variational_dist.sample_and_log_prob(key)
#         p_prob = self.log_prior_likelihood(sample, data, **dists)

#         return q_prob - p_prob

#     def objective_fun(self, key, dists, data, n_samples=1):
#         keys = jax.random.split(key, n_samples)
#         return self._objective_fun_vmap(keys, dists, data).mean()


# Another version


# class ADVI_V1:
#     def __init__(self, prior_dist, likelihood_log_prob_fun, bijector, data):
#         self.prior_dist = prior_dist
#         self.likelihood_log_prob_fun = likelihood_log_prob_fun
#         self.bijector = bijector
#         self.data = data

#         # parallelize the objective function
#         self._objective_fun_vmap = jax.vmap(self._objective_fun_per_sample, in_axes=(0, None))

#     def log_prior_likelihood(self, normal_sample):
#         transformed_sample = self.bijector.forward(normal_sample)
#         transformed_prior_dist = distrax.Transformed(self.prior_dist, distrax.Inverse(self.bijector))

#         log_prior = transformed_prior_dist.log_prob(normal_sample)
#         log_likelihood = self.likelihood_log_prob_fun(transformed_sample, self.data)
#         return log_prior + log_likelihood

#     def _objective_fun_per_sample(self, key, variational_dist):
#         normal_sample, q_prob = variational_dist.sample_and_log_prob(key)
#         p_prob = self.log_prior_likelihood(normal_sample)

#         return q_prob - p_prob

#     def objective_fun(self, key, variational_dist, n_samples=1):
#         keys = jax.random.split(key, n_samples)
#         return self._objective_fun_vmap(keys, variational_dist).mean()
