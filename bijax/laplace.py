import logging
import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow_probability.substrates.jax as tfp


import optax
from sklearn.datasets import make_classification

tfd = tfp.distributions

class LaplaceApproximation:
    def __init__(self, prior, likelihood, bijector=tfp.bijectors.Identity()) -> None:
        self.prior = prior
        self.bijector = bijector
        self.likelihood = likelihood

    def __neg_log_joint(self, theta, data):
        dist_prior = tfd.TransformedDistribution(
            distribution=self.prior, bijector=tfp.bijectors.Invert(self.bijector)
        )

        return -(
            dist_prior.log_prob(theta) + self.likelihood(self.bijector(theta), data)
        ).squeeze()

    def map(self, data, lr=0.01, max_iter=1000, seed=jax.random.PRNGKey(0)):
        gradient = jax.jit(jax.value_and_grad(self.__neg_log_joint))

        d = data.shape[1] if len(data.shape) > 1 else 1

        theta_map = jax.random.uniform(shape=(d,), key=seed)

        optimizer = optax.adam(learning_rate=lr)
        state = optimizer.init(theta_map)
        losses = []
        for i in range(max_iter):
            val, grad = gradient(theta_map, data)
            losses.append(val)
            update, state = optimizer.update(grad, state)
            theta_map = optax.apply_updates(theta_map, update)
        self.losses = losses
        return theta_map

    def approx_posterior(self, data, lr=0.01, max_iter=1000, seed=jax.random.PRNGKey(0)):
        theta_map = self.map(data, lr=lr, max_iter=max_iter, seed=seed)
        hessian = jax.hessian(self.__neg_log_joint)(theta_map, data)
        # hessian = jnp.reshape(hessian, (-1, 1))
        cov = jnp.linalg.inv(hessian)
        self.laplace_approx = tfd.TransformedDistribution(
            distribution=tfd.MultivariateNormalFullCovariance(
                loc=theta_map, covariance_matrix=cov
            ),
            bijector=self.bijector,
        )
        self.theta_map = theta_map
        self.cov = cov
        return self.laplace_approx

    def plot_approx_posterior(self, x=jnp.linspace(-5, 5, 1000), true_posterior=None):
        if self.laplace_approx is None:
            raise ValueError("Run approx_posterior first")
        x = x.reshape(-1, 1)
        fig = plt.figure()
        if true_posterior is not None:
            plt.plot(x, true_posterior.prob(x), label="true_posterior", color="orange")
        plt.plot(
            x,
            self.laplace_approx.prob(x),
            label="Laplace Approximation",
            linestyle="dashed",
            color="k",
        )
        plt.title("Approximate Posterior")  # with n_samples = '+str(n_samples))
        plt.xlabel("x")
        plt.ylabel("pdf(x)")
        plt.legend()
        sns.despine()
        return fig

    def plot_log_approx_posterior(
        self, x=jnp.linspace(-5, 5, 1000), true_posterior=None
    ):
        if self.laplace_approx is None:
            raise ValueError("Run approx_posterior first")
        x = x.reshape(-1, 1)
        fig = plt.figure()
        if true_posterior is not None:
            plt.plot(
                x, true_posterior.log_prob(x), label="true_posterior", color="orange"
            )
        plt.plot(
            x,
            self.laplace_approx.log_prob(x),
            label="Laplace Approximation",
            linestyle="dashed",
            color="k",
        )
        plt.title("Approximate Posterior")  # with n_samples = '+str(n_samples))
        plt.xlabel("x")
        plt.ylabel("logpdf(x)")
        plt.legend()
        sns.despine()
        return fig
