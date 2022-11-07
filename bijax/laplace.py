import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from functools import partial
from jax.flatten_util import ravel_pytree
import optax
import matplotlib.pyplot as plt
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

from bijax.baselaplace import BaseLaplace
from bijax.core import DistributionPyTree

class Laplace:
    def __init__(self, prior, bijector, likelihood_fn=None, model=None, is_flax_model=False, subset_of_weights='all', hessian_structure='kfac'):
        self.prior = prior
        self.bijector = bijector
        self.likelihood_fn = likelihood_fn

        if is_flax_model:
            assert model is not None, "Flax model cannot be None"
            self.model = model
            self.__get_class_partial = partial(self.__get_class, model=model, loss_fn=self.loss_fn_flax, subset_of_weights=subset_of_weights, hessian_structure=hessian_structure)
        else:
            raise NotImplementedError
        # self.variational_distribution = self.VariationalDistribution(
        #     prior=self.prior, bijector=self.bijector, vi_type=vi_type, rank=rank
        # )
        # self.prior_distribution = DistributionPyTree(self.prior, self.bijector)

    # def init(self, key):
    #     return {"variational_params": self.variational_distribution._initialise_params(key)}

    def fit(self, X, y, key=jax.random.PRNGKey(0), lr=0.03, epochs=1000, verbose=True, only_map=False):
        map_params, _ = self.__get_map_params(X, y, key, lr, epochs, verbose)
        # print("map_params",map_params, type(map_params))
        self.laplace_distribution = self.__get_class_partial(map_params=map_params)
        if not only_map:
            self.laplace_distribution.fit(X, y)
        return map_params
            
    def loss_fn_flax(self, params, X, y, noise_var = 0.1):
        y_pred = self.model.apply(params, X)
        flat_params = ravel_pytree(params)[0]
        log_prior = jax.scipy.stats.norm.logpdf(flat_params).sum()
        # log_prior = tfd.Normal(0., 1.).log_prob(flat_params).sum()#/len(flat_params)
        log_likelihood = jax.scipy.stats.norm.logpdf(y, loc=y_pred, scale=noise_var).sum()
        
        return -(log_prior + log_likelihood)

    def loss_fn_no_flax(self, params, outputs, inputs, total_size=None, seed=None, num_mc_samples=1):
        """Not updated/implemented"""
        if total_size is None:
            total_size = outputs.shape[0]

        variational_params = params["variational_params"]
        sample, log_q = self.variational_distribution.sample_and_log_prob(
            seed=seed, sample_shape=(num_mc_samples,), params=variational_params
        )

        def log_p_plus_log_likelihood(sample_value):
            log_p = ravel_pytree(jtu.tree_leaves(self.prior_distribution.log_prob(sample_value)))[0].sum()
            likelihood = self.likelihood_fn(sample_value, inputs, **params)
            log_likelihood = likelihood.log_prob(outputs).sum()
            reweighted_log_likelihood = total_size * log_likelihood / len(outputs)
            return log_p, reweighted_log_likelihood

        log_p, reweighted_log_likelihood = jax.vmap(log_p_plus_log_likelihood)(sample)
        return (log_q - log_p - reweighted_log_likelihood).mean()

    def __get_map_params(self, X, y, key, lr=0.03, epochs=1000, verbose=True):
        # key = jax.random.PRNGKey(0)
        params = self.model.init(key, X)#.unfreeze()
        value_and_grad_fn = jax.jit(jax.value_and_grad(self.loss_fn_flax))
        opt = optax.adam(lr)
        state = opt.init(params)

        def one_step(params_and_state, xs):
            params, state = params_and_state
            loss, grads = value_and_grad_fn(params, X, y)
            updates, state = opt.update(grads, state)
            params = optax.apply_updates(params, updates)
            return (params, state), loss
            
        (params, state), losses = jax.lax.scan(one_step, init=(params, state), xs=None, length=epochs)

        if verbose:
            plt.plot(losses);
            print("Final Loss:", losses[-1])
        
        return params, losses

    def __get_class(self, map_params, model, loss_fn, subset_of_weights, hessian_structure):
        laplace_map = {subclass._key: subclass for subclass in self.__all_subclasses(BaseLaplace)
                    if hasattr(subclass, '_key')}
        if laplace_map is None:
            raise NotImplementedError
        laplace_class = laplace_map[(subset_of_weights, hessian_structure)]
        return laplace_class(map_params, model, loss_fn)

    def __all_subclasses(self, cls):
        return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in self.__all_subclasses(c)])