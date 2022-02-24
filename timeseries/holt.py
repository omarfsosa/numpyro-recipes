import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro import sample
from numpyro.contrib.control_flow import scan


rng = jax.random.PRNGKey(0)


def holt(n_timesteps, y=None):
    alpha = sample("alpha", dist.Beta(5, 5))
    beta = sample("beta", dist.Beta(5, 5))
    sigma = sample("sigma", dist.LogNormal())
    l0 = sample("l0", dist.Normal())
    b0 = sample("b0", dist.Normal())

    def transition(carry, _):
        l_prev, b_prev = carry
        y_t = numpyro.sample("y", dist.Normal(l_prev + b_prev, sigma))
        e_t = y_t - l_prev - b_prev
        l_t = l_prev + b_prev + alpha * e_t
        b_t = b_prev + alpha * beta * e_t
        carry = (l_t, b_t)
        return carry, y_t

    timesteps = jnp.arange(n_timesteps)
    init = (l0, b0)
    with numpyro.handlers.condition(data={"y": y}):
        scan(transition, init, timesteps)
