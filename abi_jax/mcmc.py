import jax
from jax.flatten_util import ravel_pytree
from .core import (
    sample_dist,
    log_prob_dist,
    fill_in_bijector,
    inverse_transform_dist,
    transform_tree,
)

import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions


class MCMC:
    def __init__(self, prior, bijector={}, log_likelihood_fn=None):
        self.prior = prior
        self.bijector = fill_in_bijector(bijector, self.prior)
        self.log_likelihood_fn = log_likelihood_fn

        self.approx_normal_prior = inverse_transform_dist(self.prior, self.bijector)

        dummy_params = self.init(jax.random.PRNGKey(0))
        _, self.unravel_fn = ravel_pytree(dummy_params)

    def init(self, seed):
        return sample_dist(self.approx_normal_prior, seed)

    def log_joint(self, params, outputs, inputs):
        p_log_prob = log_prob_dist(self.approx_normal_prior, params)
        constrained_params = transform_tree(params, self.bijector)
        log_likelihood = self.log_likelihood_fn(latent_sample=constrained_params, outputs=outputs, inputs=inputs)
        return p_log_prob + log_likelihood

    def get_nuts_kernel(self, outputs, inputs, step_size=1e-3, **kwargs):
        def log_joint_fn(params):
            log_joint = jax.tree_util.Partial(self.log_joint, outputs=outputs, inputs=inputs)
            return log_joint(self.unravel_fn(params))

        return tfp.mcmc.NoUTurnSampler(log_joint_fn, step_size=step_size, **kwargs)

    def sample(self, seed, init_params_pytree, kernel, n_samples=100, n_burnin=100, trace_fn=None):
        if trace_fn is None:
            trace_fn = lambda _, results: results.target_log_prob
        init_params, _ = ravel_pytree(init_params_pytree)

        states, *other_results = tfp.mcmc.sample_chain(
            num_results=n_samples,
            current_state=init_params,
            trace_fn=trace_fn,
            kernel=kernel,
            num_burnin_steps=n_burnin,
            seed=seed,
        )

        states_pytree = jax.vmap(self.unravel_fn)(states)
        constrained_states = transform_tree(states_pytree, self.bijector)
        return constrained_states, other_results
