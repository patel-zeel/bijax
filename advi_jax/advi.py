import jax
import distrax
import tensorflow_probability.substrates.jax as tfp

dist = tfp.distributions


class ADVI:
    def __init__(self, prior_dist, likelihood_log_prob_fun, data):
        self.prior_dist = prior_dist
        self.likelihood_log_prob_fun = likelihood_log_prob_fun
        self.data = data

        # parallelize the objective function
        self._objective_fun_vmap = jax.vmap(self._objective_fun_per_sample, in_axes=(0, None))

    def log_prior_likelihood(self, sample):
        log_prior = self.prior_dist.log_prob(sample)
        log_likelihood = self.likelihood_log_prob_fun(sample, self.data)
        return log_prior + log_likelihood

    def _objective_fun_per_sample(self, key, variational_dist):
        sample, q_prob = variational_dist.sample_and_log_prob(key)
        p_prob = self.log_prior_likelihood(sample)

        return q_prob - p_prob

    def objective_fun(self, key, variational_dist, n_samples=1):
        keys = jax.random.split(key, n_samples)
        return self._objective_fun_vmap(keys, variational_dist).mean()


# Another version


class ADVI_V1:
    def __init__(self, prior_dist, likelihood_log_prob_fun, bijector, data):
        self.prior_dist = prior_dist
        self.likelihood_log_prob_fun = likelihood_log_prob_fun
        self.bijector = bijector
        self.data = data

        # parallelize the objective function
        self._objective_fun_vmap = jax.vmap(self._objective_fun_per_sample, in_axes=(0, None))

    def log_prior_likelihood(self, normal_sample):
        transformed_sample = self.bijector.forward(normal_sample)
        transformed_prior_dist = distrax.Transformed(self.prior_dist, distrax.Inverse(self.bijector))

        log_prior = transformed_prior_dist.log_prob(normal_sample)
        log_likelihood = self.likelihood_log_prob_fun(transformed_sample, self.data)
        return log_prior + log_likelihood

    def _objective_fun_per_sample(self, key, variational_dist):
        normal_sample, q_prob = variational_dist.sample_and_log_prob(key)
        p_prob = self.log_prior_likelihood(normal_sample)

        return q_prob - p_prob

    def objective_fun(self, key, variational_dist, n_samples=1):
        keys = jax.random.split(key, n_samples)
        return self._objective_fun_vmap(keys, variational_dist).mean()
