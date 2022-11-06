import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from functools import partial
from jax.flatten_util import ravel_pytree
from bijax.utils import svd_inverse, get_shapes


class BaseLaplace:
    def __init__(self, map_params, model, loss_fn=None):
        self.model = model
        self.loss_fn = model.loss_fn if loss_fn is None else loss_fn
        assert self.loss_fn is not None, "Loss Function must be defined"

    def _vectorize(self, f, seed, shape, f_kwargs={}):
        # length = math.prod(shape)
        length = shape[0]
        seeds = jax.random.split(seed, num=length).reshape(shape+(2,))
        
        sample_fn = partial(f, **f_kwargs)
        for _ in shape:
            sample_fn = jax.vmap(sample_fn)
        
        return sample_fn(seed=seeds)
    
    def _sample(self, seed):
        sample = jax.random.multivariate_normal(seed, mean=self.mean, cov=self.cov)
        return self.unravel_fn(sample)

    def sample(self, seed, n_samples):
        return self._vectorize(self._sample, seed, n_samples)
    
    def _predict(self, X, seed):
        sample = self._sample(seed)
        return self.model.apply(sample, X)
    
    def predict(self, X, seed, shape):
        return self._vectorize(self._predict, seed, shape, {'X': X})
    
class FullHessianLaplace(BaseLaplace):
    _key = ("all", "full")
    def __init__(self, map_params, model, loss_fn=None):
        super().__init__(map_params, model, loss_fn)
        flat_params, self.unravel_fn = ravel_pytree(map_params)
        self.mean = flat_params

    def fit(self, X, y):
        def neg_log_joint_flat(flat_params, X, y):
            params = self.unravel_fn(flat_params)
            return self.loss_fn(params, X, y)

        self.H = jax.hessian(neg_log_joint_flat)(self.mean, X, y)
        self.cov = svd_inverse(self.H)

class KFACHessianLaplace(BaseLaplace):
    _key = ("all", "kfac")
    def __init__(self, map_params, model, loss_fn=None):
        super().__init__(map_params, model, loss_fn)
        layers, self.tree_def = jtu.tree_flatten(map_params, is_leaf=lambda x: 'bias' in x)
        flat_layers = [ravel_pytree(layer) for layer in layers]
        self.means = list(map(lambda x: x[0], flat_layers))
        self.unravel_fn_list = list(map(lambda x: x[1], flat_layers))
        self.flat_layers = flat_layers
        self.loss_fn = model.loss_fn if loss_fn is None else loss_fn

    def fit(self, X, y):
        def neg_log_joint_flat(flat_params, X, y):
            flat_layers = [self.unravel_fn_list[i](flat_params[i]) for i in range(len(flat_params))]
            params = self.tree_def.unflatten(flat_layers)
            return self.loss_fn(params, X, y)

        self.H = jax.hessian(neg_log_joint_flat)(self.means, X, y)
        self.useful_H = [self.H[i][i] for i in range(len(self.H))]
        
        self.covs = [svd_inverse(matrix) for matrix in self.useful_H]

    def _sample_partial(self, seed, unravel_fn, mean, cov):
        sample = jax.random.multivariate_normal(seed, mean=mean, cov=cov)
        return unravel_fn(sample)
    
    def _sample(self, seed):
        seeds = [seed for seed in jax.random.split(seed, num=len(self.means))]
        flat_sample = jtu.tree_map(self._sample_partial, seeds, self.unravel_fn_list, self.means, self.covs)
        sample = self.tree_def.unflatten(flat_sample)
        return sample

class FullLLLaplace(BaseLaplace):
    _key = ("last_layer", "full")
    def __init__(self, map_params, model, loss_fn=None):
        super().__init__(map_params, model, loss_fn)
        # print("Inside Full LL")
        layers, self.tree_def = jtu.tree_flatten(map_params, is_leaf=lambda x: 'bias' in x)
        flat_layers = [ravel_pytree(layer) for layer in layers]
        self.means = list(map(lambda x: x[0], flat_layers))
        self.unravel_fn_list = list(map(lambda x: x[1], flat_layers))
        self.flat_layers = flat_layers
        self.loss_fn = model.loss_fn if loss_fn is None else loss_fn

        flat_last = self.means[-1]
        self.len_last=len(flat_last)
        self.flat_params, self.unravel_fn = ravel_pytree(map_params)


    def fit(self, X, y):
        def neg_log_joint_flat(flat_params, X, y):
            flat_layers = [self.unravel_fn_list[i](flat_params[i]) for i in range(len(flat_params))]
            params = self.tree_def.unflatten(flat_layers)
            return self.loss_fn(params, X, y)

        self.H = jax.hessian(neg_log_joint_flat)(self.means, X, y)
        # self.useful_H = [self.H[i][i] for i in range(len(self.H))]
        self.useful_H = self.H[-1][-1]
        
        self.mean = self.flat_params[(len(self.flat_params)-(self.len_last)):]
        self.cov = svd_inverse(self.useful_H)

    # def _sample_partial(self, seed, unravel_fn, mean, cov):
    #     sample = jax.random.multivariate_normal(seed, mean=mean, cov=cov)
    #     return unravel_fn(sample)
    
    # def _sample(self, seed):
    #     seeds = [seed for seed in jax.random.split(seed, num=len(self.means))]
    #     flat_sample = jtu.tree_map(self._sample_partial, seeds, self.unravel_fn_list, self.means, self.covs)
    #     sample = self.tree_def.unflatten(flat_sample)
    #     return sample    
    def _sample(self, seed):
        flat_sample = jax.random.multivariate_normal(seed, mean=self.mean, cov=self.cov)
        return flat_sample
    
    def predict(self, X, seed, shape):
        flat_sample = self.sample(seed, shape)
        flat_thetas_exceptlast=jnp.broadcast_to(self.flat_params[:-self.len_last].reshape(-1,1).T,(shape, len(self.flat_params)-self.len_last))
        # print("self.flat_params[:-self.len_last].reshape(-1,1).T.shape,flat_thetas_exceptlast.shape",
        #       self.flat_params[:-self.len_last].reshape(-1,1).T.shape,flat_thetas_exceptlast.shape)
        # print("flat_sample.shape", flat_sample.shape)
        flat_thetas_sampled=jnp.c_[flat_thetas_exceptlast,flat_sample]

        apply_model = lambda params, x: self.model.apply(self.unravel_fn(params), x)
        return jax.vmap(apply_model, in_axes=(0, None))(flat_thetas_sampled, X)
        