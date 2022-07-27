import jax

from .core import (
    inverse_transform_dist,
    transform_tree,
    transform_dist_params,
    log_prob_dist,
    get_full_rank,
    get_low_rank,
    get_mean_field,
    fill_in_bijector,
)

from .core import Prior, Posterior, GenerativeDistribution
from .utils import initialize_params

import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from typing import Dict, Callable, Literal, Optional


class ADVI:
    def __init__(
        self,
        prior: Dict[str, tfd.Distribution],
        likelihood_fn: Callable[..., tfd.Distribution],
        prior_constraints: Optional[Dict[str, tfb.Bijector]] = {},
        vi_type: Optional[Literal["mean_field", "low_rank", "full_rank"]] = "mean_field",
        rank: Optional[int] = None,
        ordered_posterior_bijectors=None,
    ):
        """Automatic Differentiation Variational Inference
        A model class that implements the ADVI algorithm.

        Args:
            prior: A dictionary of prior distributions.
            likelihood_fn: A callable that takes params as argument and returns a likelihood distribution.
            prior_constraints: A dictionary of constraints over the latent variables. The keys should be identical with `prior`.

            vi_type: type of variational inference ("mean_field", "full_rank", "low_rank"). Defaults to "mean_field".
            rank: Rank of posterior covariance matrix in case where `vi_type` is "low_rank". Defaults to None.
            ordered_posterior_bijectors: A dictionary of bijectors for posterior parameters. `ordered` here means ordered as per the alphabetical order. Defaults to None.
        """

        self.prior = prior
        self.prior_constraints = fill_in_bijector(prior_constraints, self.prior)

        assert (
            self.prior_constraints.keys() == self.prior.keys()
        ), "The keys in `prior` and `prior_constraints` must be the same."

        assert (vi_type == "low_rank") == (
            rank is not None
        ), "`rank` must be specified only if `vi_type` is `low_rank`."

        self.likelihood_fn = likelihood_fn
        self.vi_type = vi_type

        self.approx_normal_prior = inverse_transform_dist(self.prior, self.prior_constraints)

        if vi_type == "mean_field":
            self.approx_posterior, self.unravel_fn, self.posterior_params_bijector = get_mean_field(
                self.approx_normal_prior, ordered_posterior_bijectors
            )
        elif vi_type == "low_rank":
            self.approx_posterior, self.unravel_fn, self.posterior_params_bijector = get_low_rank(
                self.approx_normal_prior, rank, ordered_posterior_bijectors
            )
        elif vi_type == "full_rank":
            self.approx_posterior, self.unravel_fn, self.posterior_params_bijector = get_full_rank(
                self.approx_normal_prior, ordered_posterior_bijectors
            )

    def init(self, seed, initializer=jax.nn.initializers.normal(stddev=1.0)):
        return {"approx_posterior": initialize_params(self.approx_posterior, seed, initializer)}

    def loss_fn(self, params, outputs, inputs, full_data_size, seed, n_samples=1):
        approx_posterior_raw = params["approx_posterior"]
        approx_posterior = transform_dist_params(approx_posterior_raw, self.posterior_params_bijector)
        samples = approx_posterior.sample(seed=seed, sample_shape=(n_samples,))
        
        def loss_fn_no_batch(sample):
            q_log_prob = approx_posterior.log_prob(sample)
            sample_tree = self.unravel_fn(sample)
            p_log_prob = log_prob_dist(self.approx_normal_prior, sample_tree)
            transformed_sample_tree = transform_tree(sample_tree, self.prior_constraints)
            likelihood = self.likelihood_fn(transformed_sample_tree, inputs=inputs)
            log_likelihood = likelihood.log_prob(outputs).sum()
            log_likelihood = (log_likelihood / len(outputs)) * full_data_size  # normalize by outputs size
            return q_log_prob - p_log_prob - log_likelihood

        return jax.vmap(loss_fn_no_batch)(samples).mean()

    def get_params_posterior(self, params):
        approx_posterior = params["approx_posterior"]
        approx_posterior = transform_dist_params(approx_posterior, self.posterior_params_bijector)
        return Posterior(approx_posterior, self.approx_normal_prior, self.prior_constraints)

    def get_generative_distribution(self, latent_distribution: Posterior or Dict[str, tfd.Distribution]):
        if isinstance(latent_distribution, Posterior):
            return GenerativeDistribution(latent_distribution, self.likelihood_fn)
        elif isinstance(latent_distribution, dict):
            return GenerativeDistribution(Prior(latent_distribution), self.likelihood_fn)
            