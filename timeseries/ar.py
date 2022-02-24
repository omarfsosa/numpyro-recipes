import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro import sample
from numpyro.contrib.control_flow import scan
from numpyro.infer import Predictive, NUTS, MCMC

rng = jax.random.PRNGKey(0)


def ar2(num_timesteps, y0, y1, y=None):
    """
    An auto regressive (K=2) model.

    Parameters
    ----------
    num_timesteps: int, positive
        The total number of timesteps to model

    y: ndarray, shape (num_timesteps,)
        The observed values beyond y0 and y1

    y0, y1: floats
        The initial values of the process
    """
    a1 = sample("a1", dist.Normal())
    a2 = sample("a2", dist.Normal())
    const = sample("const", dist.Normal())
    sigma = sample("sigma", dist.Exponential())

    def transition(carry, _):
        y_prev, y_prev_prev = carry
        m_t = const + a1 * y_prev + a2 * y_prev_prev
        y_t = sample("y", dist.Normal(m_t, sigma))
        carry = (y_t, y_prev)
        return carry, None

    timesteps = jnp.arange(num_timesteps)
    init = (y0, y1)
    with numpyro.handlers.condition(data={"y": y}):
        scan(transition, init, timesteps)


#  Prior simulation
num_timesteps = 40
y0, y1 = 0.3, -0.1

prior = Predictive(ar2, num_samples=10)
prior_samples = prior(rng, num_timesteps, y0, y1)

#  Fitting
y_true = None  # Replace None with the true data
mcmc_settings = dict(num_warmup=1000, num_samples=1000)
y0, y1, *y = y_true
y = jnp.array(y)
num_timesteps = len(y)
mcmc = MCMC(NUTS(ar2), **mcmc_settings)
mcmc.run(rng, num_timesteps, y0, y1, y)

# Forecast
num_forecast = 10
y0, y1 = y_true[-2:]
forecaster = Predictive(ar2, posterior_samples=mcmc.get_samples())
forecast_samples = forecaster(rng, num_forecast, y0, y1)
