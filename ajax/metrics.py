# TODO: work in progress

import jax


def kl_divergence_per_sample(key, p, q):
    sample = p.sample(key)
    return p.log_prob(sample) - q.log_prob(sample)


kl_divergence_per_sample_vmap = jax.vmap(kl_divergence_per_sample, in_axes=(0, None, None))


def kl_divergence(key, p, q, n_samples=1):
    keys = jax.random.split(key, n_samples)
    return kl_divergence_per_sample_vmap(keys, p, q).mean()
