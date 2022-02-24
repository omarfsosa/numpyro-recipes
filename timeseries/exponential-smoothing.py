import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro import sample
from numpyro.contrib.control_flow import scan


def exponential_smoothing(n_timesteps, y=None):
    alpha = sample("alpha", dist.Beta(5, 5))
    sigma = sample("sigma", dist.LogNormal())
    l0 = sample("l0", dist.Normal())

    def transition(carry, _):
        l_prev = carry
        y_t = numpyro.sample("y", dist.Normal(l_prev, sigma))
        e_t = y_t - l_prev
        l_t = l_prev + alpha * e_t
        return l_t, y_t

    timesteps = jnp.arange(n_timesteps)
    init = l0
    with numpyro.handlers.condition(data={"y": y}):
        scan(transition, init, timesteps)
