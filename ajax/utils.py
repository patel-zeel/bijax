import jax
import optax


def train_model(model, loss_fn_kwargs, optimizer, n_epochs, seed, return_args={}):
    params = model.init(seed)
    value_and_grad_fn = jax.value_and_grad(model.loss_fn)
    state = optimizer.init(params)

    @jax.jit
    def one_step(params_and_state, xs):
        params, state = params_and_state
        loss, grads = value_and_grad_fn(params, **loss_fn_kwargs)
        updates, state = optimizer.update(grads, state)
        params = optax.apply_updates(params, updates)
        return (params, state), (loss, params)

    (params, states), (losses, params_list) = jax.lax.scan(one_step, (params, state), xs=None, length=n_epochs)
    return_dict = {"best_params": params}
    for key in return_args:
        return_dict[key] = locals()[key]
    return return_dict
