import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro import sample, plate
from numpyro.contrib.control_flow import scan


def holt_winters(n_timesteps, seasonality, y=None):
    alpha = sample("alpha", dist.Beta(5, 5))
    beta = sample("beta", dist.Beta(5, 5))
    gamma = sample("gamma", dist.Beta(5, 5))

    l0 = sample("l0", dist.Normal(0, 1))
    b0 = sample("b0", dist.Normal(0, 1))
    with plate("plate_period", seasonality):
        s0 = sample("s0", dist.Normal(0, 1))

    sigma = sample("sigma", dist.LogNormal())

    def transition(carry, _):
        l_prev, b_prev, *s_prev = carry
        m_t = l_prev + b_prev + s_prev[0]
        y_t = sample("y", dist.Normal(m_t, sigma))
        e_t = y_t - m_t
        l_t = l_prev + b_prev + alpha * e_t
        b_t = b_prev + alpha * beta * e_t
        s_t = s_prev[0] + gamma * (1 - alpha) * e_t
        s_new = jnp.concatenate((jnp.array(s_prev[1:]), jnp.array([s_t])))
        carry = (l_t, b_t, s_new)
        return carry, y_t

    timesteps = jnp.arange(n_timesteps)
    init = (l0, b0, s0)
    with numpyro.handlers.condition(data={"y": y}):
        scan(transition, init, timesteps)
