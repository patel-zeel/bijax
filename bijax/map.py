from sklearn.covariance import log_likelihood
from .core import (
    fill_in_bijector,
    inverse_transform_dist,
    inverse_transform_tree,
    log_prob_dist,
    sample_dist,
    transform_tree,
)


class MAP:
    def __init__(self, prior, likelihood_fn, prior_constraints):
        self.prior = prior
        self.likelihood_fn = likelihood_fn
        self.prior_constraints = fill_in_bijector(prior_constraints, self.prior)

        assert (
            self.prior_constraints.keys() == self.prior.keys()
        ), "The keys in `prior` and `prior_constraints` must be the same."

    def init(self, seed):
        params = sample_dist(self.prior, seed)
        unconstrained_params = inverse_transform_tree(params, self.prior_constraints)
        return {"params": unconstrained_params}

    def neg_log_joint(self, params, outputs, inputs=None):
        params = params["params"]
        params = transform_tree(params, self.prior_constraints)
        log_prior = log_prob_dist(self.prior, params)
        likelihood = self.likelihood_fn(params, inputs=inputs)
        log_likelihood = likelihood.log_prob(outputs).sum()
        return -(log_prior + log_likelihood)

    def apply(self, params):
        params = params["params"]
        return transform_tree(params, self.prior_constraints)
