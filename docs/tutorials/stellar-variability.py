# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + nbsphinx="hidden"
import lightkurve as lk

# %matplotlib inline

# + nbsphinx="hidden"
# %run notebook_setup
# -

# # Gaussian process models for stellar variability

# When fitting exoplanets, we also need to fit for the stellar variability and Gaussian Processes (GPs) are often a good descriptive model for this variation.
# [PyMC3 has support for all sorts of general GP models](https://docs.pymc.io/gp.html), but *exoplanet* interfaces with the [celerite2](https://celerite2.readthedocs.io/) library to provide support for scalable 1D GPs (take a look at the [Getting started](https://celerite2.readthedocs.io/en/latest/tutorials/first/) tutorial on the *celerite2* docs for a crash course) that can work with large datasets.
# In this tutorial, we go through the process of modeling the light curve of a rotating star observed by Kepler using *exoplanet* and *celerite2*.
#
# First, let's download and plot the data:

# +
import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt

lcf = lk.search_lightcurvefile("TIC 10863087").download_all(
    quality_bitmask="hardest"
)
lc = lcf.PDCSAP_FLUX.stitch().remove_nans().remove_outliers()
lc = lc[:5000]
_, mask = lc.flatten().remove_outliers(sigma=3.0, return_mask=True)
lc = lc[~mask]

x = np.ascontiguousarray(lc.time, dtype=np.float64)
y = np.ascontiguousarray(lc.flux, dtype=np.float64)
yerr = np.ascontiguousarray(lc.flux_err, dtype=np.float64)
mu = np.mean(y)
y = (y / mu - 1) * 1e3
yerr = yerr * 1e3 / mu

plt.plot(x, y, "k")
plt.xlim(x.min(), x.max())
plt.xlabel("time [days]")
plt.ylabel("relative flux [ppt]")
_ = plt.title("TIC 10863087")
# -

# ## A Gaussian process model for stellar variability
#
# This looks like the light curve of a rotating star, and [it has been shown](https://arxiv.org/abs/1706.05459) that it is possible to model this variability by using a quasiperiodic Gaussian process.
# To start with, let's get an estimate of the rotation period using the Lomb-Scargle periodogram:

# +
import exoplanet as xo

results = xo.estimators.lomb_scargle_estimator(
    x, y, max_peaks=1, min_period=0.1, max_period=2.0, samples_per_peak=50
)

peak = results["peaks"][0]
freq, power = results["periodogram"]
plt.plot(1 / freq, power, "k")
plt.axvline(peak["period"], color="k", lw=4, alpha=0.3)
plt.xlim((1 / freq).min(), (1 / freq).max())
plt.yticks([])
plt.xlabel("period [days]")
_ = plt.ylabel("power")
# -

# Now, using this initialization, we can set up the GP model in *exoplanet* and *celerite2*.
# We'll use the [RotationTerm](https://celerite2.readthedocs.io/en/latest/api/python/#celerite2.terms.RotationTerm) kernel that is a mixture of two simple harmonic oscillators with periods separated by a factor of two.
# As you can see from the periodogram above, this might be a good model for this light curve and I've found that it works well in many cases.

# +
import pymc3 as pm
import pymc3_ext as pmx
import theano.tensor as tt
from celerite2.theano import terms, GaussianProcess

with pm.Model() as model:

    # The mean flux of the time series
    mean = pm.Normal("mean", mu=0.0, sd=10.0)

    # A jitter term describing excess white noise
    jitter = pm.Lognormal("jitter", mu=np.log(np.mean(yerr)), sd=2.0)

    # A term to describe the non-periodic variability
    sigma = pm.InverseGamma(
        "sigma", **pmx.estimate_inverse_gamma_parameters(1.0, 5.0)
    )
    rho = pm.InverseGamma(
        "rho", **pmx.estimate_inverse_gamma_parameters(0.5, 2.0)
    )

    # The parameters of the RotationTerm kernel
    sigma_rot = pm.InverseGamma(
        "sigma_rot", **pmx.estimate_inverse_gamma_parameters(1.0, 5.0)
    )
    period = pm.Lognormal("period", mu=np.log(peak["period"]), sd=2.0)
    Q0 = pm.Lognormal("Q0", mu=0.0, sd=2.0)
    dQ = pm.Lognormal("dQ", mu=0.0, sd=2.0)
    f = pm.Uniform("f", lower=0.1, upper=1.0)

    # Set up the Gaussian Process model
    kernel = terms.SHOTerm(sigma=sigma, rho=rho, Q=1 / 3.0)
    kernel += terms.RotationTerm(
        sigma=sigma_rot, period=period, Q0=Q0, dQ=dQ, f=f
    )
    gp = GaussianProcess(
        kernel, t=x, diag=yerr ** 2 + jitter ** 2, mean=mean, quiet=True
    )

    # Compute the Gaussian Process likelihood and add it into the
    # the PyMC3 model as a "potential"
    gp.marginal("gp", observed=y)

    # Compute the mean model prediction for plotting purposes
    pm.Deterministic("pred", gp.predict(y))

    # Optimize to find the maximum a posteriori parameters
    map_soln = pmx.optimize()
# -

# Now that we have the model set up, let's plot the maximum a posteriori model prediction.

plt.plot(x, y, "k", label="data")
plt.plot(x, map_soln["pred"], color="C1", label="model")
plt.xlim(x.min(), x.max())
plt.legend(fontsize=10)
plt.xlabel("time [days]")
plt.ylabel("relative flux [ppt]")
_ = plt.title("TIC 10863087; map model")

# That looks pretty good!
# Now let's sample from the posterior using [the PyMC3 Extras (`pymc3-ext`) library](https://github.com/exoplanet-dev/pymc3-ext):

np.random.seed(10863087)
with model:
    trace = pmx.sample(
        tune=2500,
        draws=2000,
        start=map_soln,
        cores=2,
        chains=2,
        target_accept=0.95,
    )

# Now we can do the usual convergence checks:

with model:
    summary = pm.summary(
        trace,
        var_names=[
            "f",
            "dQ",
            "Q0",
            "period",
            "sigma_rot",
            "rho",
            "sigma",
            "jitter",
            "mean",
        ],
    )
summary

# And plot the posterior distribution over rotation period:

period_samples = trace["period"]
plt.hist(period_samples, 25, histtype="step", color="k", density=True)
plt.yticks([])
plt.xlabel("rotation period [days]")
_ = plt.ylabel("posterior density")

# ## Variational inference
#
# One benefit of building our model within PyMC3 is that we can take advantage of the other inference methods provided by PyMC3, like [Autodiff Variational Inference](https://docs.pymc.io/notebooks/variational_api_quickstart.html).
# Here we're finding the Gaussian approximation to the posterior that minimizes the [KL divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence):

# +
np.random.seed(10863087)
with model:
    approx = pm.fit(
        n=20000,
        method="fullrank_advi",
        obj_optimizer=pm.adagrad(learning_rate=1e-1),
    )
    approx_trace = approx.sample(3000)

approx_period_samples = approx_trace["period"]
plt.hist(
    period_samples, 25, histtype="step", color="k", density=True, label="MCMC"
)
plt.hist(
    approx_period_samples,
    25,
    histtype="step",
    color="C1",
    linestyle="dashed",
    density=True,
    label="VI",
)
plt.legend()
plt.yticks([])
plt.xlabel("rotation period [days]")
_ = plt.ylabel("posterior density")
# -

# In this case, the periods inferred with both methods are consistent and variational inference was significantly faster.

# ## Citations
#
# As described in the [citation tutorial](https://docs.exoplanet.codes/en/stable/tutorials/citation/), we can use [`citations.get_citations_for_model`](https://docs.exoplanet.codes/en/stable/user/api/#exoplanet.citations.get_citations_for_model) to construct an acknowledgement and BibTeX listing that includes the relevant citations for this model.

with model:
    txt, bib = xo.citations.get_citations_for_model()
print(txt)

print("\n".join(bib.splitlines()[:10]) + "\n...")
