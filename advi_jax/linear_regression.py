import jax
import jax.numpy as jnp
import distrax
from advi import ADVI_MeanField as ADVI
import optax
import matplotlib.pyplot as plt

# Data generation
key = jax.random.PRNGKey(0)
X = jax.random.normal(key, shape=(100, 3))
X_ = jnp.linspace(-3, 3, 100).reshape(-1, 1)
true_theta = jnp.array([0.5, 2.0, 10.0])
true_noise_scale = jnp.array(1.0)
y = jax.vmap(lambda x: jnp.dot(x, true_theta))(X)
key = jax.random.split(key)[0]
y_noisy = y + jax.random.normal(key, shape=y.shape) * true_noise_scale
print(X.shape, y.shape, y_noisy.shape)

# plt.scatter(X, y_noisy)
# plt.savefig("data.png")

# Define prior distributions
prior_dists = {
    "theta": lambda: distrax.MultivariateNormalDiag(jnp.zeros((X.shape[1],)), jnp.ones((X.shape[1],))),
    "noise_scale": lambda: distrax.Gamma(jnp.array(1.0), jnp.array(1.0)),
}

# Transform prior distributions
transforms = {
    "theta": lambda: distrax.Lambda(lambda x: x),
    "noise_scale": lambda: distrax.Lambda(lambda x: jnp.log(1 + jnp.exp(x))),
}

# Define log likelihood function
def log_likelihood_fun(data, theta):
    X, y = data
    mean = jnp.dot(X, theta["theta"])
    scale_diag = jnp.ones_like(y) * theta["noise_scale"]
    likelihood = distrax.MultivariateNormalDiag(mean, scale_diag)
    return likelihood.log_prob(y)


# Define model
model = ADVI(prior_dists, transforms, log_likelihood_fun)

# Initialize
key = jax.random.split(key, 6)[-1]
params = model.init(key, distrax.Normal(loc=0.0, scale=1.0))
data = (X, y_noisy)


# Setup ADVI
key = jax.random.split(key, 1)[0]
n_iterations = 100
n_mc_samples = 1000
learning_rate = 0.1

# Optimize
value_and_grad_fun = jax.jit(jax.value_and_grad(model.objective_fun))
tx = optax.adam(learning_rate=learning_rate)
state = tx.init(params)
for i in range(n_iterations):
    key = jax.random.PRNGKey(i)
    epsilons = model.sample_epsilon(key=key, sample_shape=(n_mc_samples,))
    value, grads = value_and_grad_fun(params, epsilons, data)
    updates, state = tx.update(grads, state)
    params = optax.apply_updates(params, updates)

    # y_mean = X.ravel() * params["mean"][1]
    # y_std = ((X.ravel() ** 2 * jnp.exp(params["log_scale"][1])) + jnp.exp(params["mean"][0])) ** 0.5

    print(i, value)

    plt.figure()
    # noise_var = (jnp.exp(params["mean"][0]) ** 2) / 0.01
    # y_map = X @ (jnp.linalg.inv(X.T @ X + noise_var) @ X.T @ y_noisy)
    plt.scatter(X, y_noisy, label="noisy data", color="k")
    # plt.plot(X, y_map, label="MAP", color="k")
    key = jax.random.PRNGKey(i)
    post_samples = model.sample_posterior(key, params, n_samples=100)
    for p_sample, n_sample in zip(post_samples["theta"][:5], post_samples["noise_scale"][:5]):
        post_dist = distrax.MultivariateNormalDiag((X_ * p_sample).ravel(), jnp.ones_like(y) * n_sample)
        key = jax.random.split(key, 1)[0]
        plt.scatter(X_, post_dist.sample(seed=key), alpha=0.6, marker="*")
    plt.scatter(X_, post_dist.sample(seed=key), label="posterior samples", alpha=0.6, marker="*")
    plt.plot(X_, X_ * post_samples["theta"].mean(), label="estimated posterior mean", color="r")
    # plt.scatter(X, X * p_sample, c="r", label="posterior samples", alpha=0.5)
    # plt.fill_between(X.ravel(), y_mean - 2 * y_std, y_mean - 2 * y_std, alpha=0.5)
    plt.ylim(-5, 5)
    plt.xlim(-2, 2.5)
    plt.title("Iteration %d" % i)
    plt.legend(loc="upper left")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.savefig(f"advi_jax_{str(i).zfill(3)}.png")
    plt.close()


print("Done")
