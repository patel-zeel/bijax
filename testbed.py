import jax
import jax.numpy as jnp
import distrax

mean = jnp.array([0.1, 0.2])
scale = jnp.array([0.4, 0.5])
key = jax.random.PRNGKey(1)

dist_uni = distrax.Normal(mean, scale)
trans_uni = distrax.Transformed(dist_uni, distrax.Sigmoid())
trans_uni_sample, log_prob = trans_uni.sample_and_log_prob(seed=key)
print(f"{trans_uni_sample=}, {log_prob=}, {log_prob.sum()=}")

# dist_bi = distrax.MultivariateNormalDiag(mean, scale)
# trans_bi = distrax.Transformed(dist_bi, distrax.Sigmoid(event_ndims_in=2))
# trans_bi_sample, log_prob = trans_bi.sample_and_log_prob(seed=key)
# print(f"{trans_bi_sample=}, {log_prob}")

dist_bi = distrax.MultivariateNormalDiag(mean, scale)
transform = jax.nn.sigmoid
trans_bi_sample, log_prob = dist_bi.sample_and_log_prob(seed=key)
log_jac = jnp.log(jnp.linalg.det(jax.jacfwd(transform)(trans_bi_sample)))
print(f"{transform(trans_bi_sample)=}, {log_prob - log_jac}")
