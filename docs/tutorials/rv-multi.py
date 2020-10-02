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

# %matplotlib inline

# %run notebook_setup

# # RVs with multiple instruments
#
# In this case study, we will look at how we can use exoplanet and PyMC3 to combine datasets from different RV instruments to fit the orbit of an exoplanet system.
# Before getting started, I want to emphasize that the exoplanet code doesn't have strong opinions about how your data are collected, it only provides extensions that allow PyMC3 to evaluate some astronomy-specific functions.
# This means that you can build any kind of observation model that PyMC3 supports, and support for multiple instruments isn't really a *feature* of exoplanet, even though it is easy to implement.
#
# For the example, we'll use public observations of Pi Mensae which hosts two planets, but we'll ignore the inner planet because the significance of the RV signal is small enough that it won't affect our results.
# The datasets that we'll use are from the Anglo-Australian Planet Search (AAT) and the HARPS archive.
# As is commonly done, we will treat the HARPS observations as two independent datasets split in June 2015 when the HARPS hardware was upgraded.
# Therefore, we'll consider three datasets that we will allow to have different instrumental parameters (RV offset and jitter), but shared orbital parameters and stellar variability.
# In some cases you might also want to have a different astrophyscial variability model for each instrument (if, for example, the observations are made in very different bands), but we'll keep things simple for this example.
#
# The AAT data are available from [The Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) and the HARPS observations can be downloaded from the [ESO Archive](http://archive.eso.org/wdb/wdb/adp/phase3_spectral/form).
# For the sake of simplicity, we have extracted the HARPS RVs from the archive in advance using [Megan Bedell's harps_tools library](https://github.com/megbedell/harps_tools).
#
# To start, download the data and plot them with a (very!) rough zero point correction.

# +
import numpy as np
import pandas as pd
from astropy.io import ascii

aat = ascii.read(
    "https://exoplanetarchive.ipac.caltech.edu/data/ExoData/0026/0026394/data/UID_0026394_RVC_001.tbl"
)
harps = pd.read_csv(
    "https://raw.githubusercontent.com/exoplanet-dev/case-studies/master/data/pi_men_harps_rvs.csv",
    skiprows=1,
)
harps = harps.rename(lambda x: x.strip().strip("#"), axis=1)
harps_post = np.array(harps.date > "2015-07-01", dtype=int)

t = np.concatenate((aat["JD"], harps["bjd"]))
rv = np.concatenate((aat["Radial_Velocity"], harps["rv"]))
rv_err = np.concatenate((aat["Radial_Velocity_Uncertainty"], harps["e_rv"]))
inst_id = np.concatenate((np.zeros(len(aat), dtype=int), harps_post + 1))

inds = np.argsort(t)
t = np.ascontiguousarray(t[inds], dtype=float)
rv = np.ascontiguousarray(rv[inds], dtype=float)
rv_err = np.ascontiguousarray(rv_err[inds], dtype=float)
inst_id = np.ascontiguousarray(inst_id[inds], dtype=int)

inst_names = ["aat", "harps_pre", "harps_post"]
num_inst = len(inst_names)

for i, name in enumerate(inst_names):
    m = inst_id == i
    plt.errorbar(
        t[m], rv[m] - np.min(rv[m]), yerr=rv_err[m], fmt=".", label=name
    )

plt.legend(fontsize=10)
plt.xlabel("BJD")
_ = plt.ylabel("radial velocity [m/s]")
# -

# Then set up the probabilistic model.
# Most of this is similar to the model in the :ref:`rv` tutorial, but there are a few changes to highlight:
#
# 1. Instead of a polynomial model for trends, stellar varaiability, and inner planets, we're using a Gaussian process here. This won't have a big effect here, but more careful consideration should be performed when studying lower signal-to-noise systems.
# 2. There are three radial velocity offests and three jitter parameters (one for each instrument) that will be treated independently. This is the key addition made by this case study.

# +
import pymc3 as pm
import exoplanet as xo
import theano.tensor as tt

import pymc3_ext as pmx
from celerite2.theano import terms, GaussianProcess

with pm.Model() as model:

    # Parameters describing the orbit
    K = pm.Lognormal("K", mu=np.log(300), sigma=10)
    P = pm.Lognormal("P", mu=np.log(2093.07), sigma=10)

    ecs = pmx.UnitDisk("ecs", testval=np.array([0.7, -0.3]))
    ecc = pm.Deterministic("ecc", tt.sum(ecs ** 2))
    omega = pm.Deterministic("omega", tt.arctan2(ecs[1], ecs[0]))
    phase = pmx.UnitUniform("phase")
    tp = pm.Deterministic("tp", 0.5 * (t.min() + t.max()) + phase * P)

    orbit = xo.orbits.KeplerianOrbit(
        period=P, t_periastron=tp, ecc=ecc, omega=omega
    )

    # Noise model parameters
    sigma_gp = pm.Lognormal("sigma_gp", mu=np.log(10), sigma=50)
    rho_gp = pm.Lognormal("rho_gp", mu=np.log(50), sigma=50)

    # Per instrument parameters
    means = pm.Normal(
        "means",
        mu=np.array([np.median(rv[inst_id == i]) for i in range(num_inst)]),
        sigma=200,
        shape=num_inst,
    )
    sigmas = pm.HalfNormal("sigmas", sigma=10, shape=num_inst)

    # Compute the RV offset and jitter for each data point depending on its instrument
    mean = tt.zeros(len(t))
    diag = tt.zeros(len(t))
    for i in range(len(inst_names)):
        mean += means[i] * (inst_id == i)
        diag += (rv_err ** 2 + sigmas[i] ** 2) * (inst_id == i)
    pm.Deterministic("mean", mean)
    pm.Deterministic("diag", diag)
    resid = rv - mean

    def rv_model(x):
        return orbit.get_radial_velocity(x, K=K)

    kernel = terms.SHOTerm(sigma=sigma_gp, rho=rho_gp, Q=1.0 / 3)
    gp = GaussianProcess(kernel, t=t, diag=diag, mean=rv_model)
    gp.marginal("obs", observed=resid)
    pm.Deterministic("gp_pred", gp.predict(resid, include_mean=False))

    map_soln = model.test_point
    map_soln = pmx.optimize(map_soln, [means])
    map_soln = pmx.optimize(map_soln, [means, phase])
    map_soln = pmx.optimize(map_soln, [means, phase, K])
    map_soln = pmx.optimize(map_soln, [means, tp, K, P, ecs])
    map_soln = pmx.optimize(map_soln, [sigmas, sigma_gp, rho_gp])
    map_soln = pmx.optimize(map_soln)
# -

# After fitting for the parameters that maximize the posterior probability, we can plot this model to make sure that things are looking reasonable:

# +
t_pred = np.linspace(t.min() - 400, t.max() + 400, 5000)
with model:
    plt.plot(t_pred, xo.eval_in_model(rv_model(t_pred), map_soln), "k", lw=0.5)

detrended = rv - map_soln["mean"] - map_soln["gp_pred"]
plt.errorbar(t, detrended, yerr=rv_err, fmt=",k")
plt.scatter(
    t, detrended, c=inst_id, s=8, zorder=100, cmap="tab10", vmin=0, vmax=10
)
plt.xlim(t_pred.min(), t_pred.max())
plt.xlabel("BJD")
plt.ylabel("radial velocity [m/s]")
_ = plt.title("map model", fontsize=14)
# -

# That looks fine, so now we can run the MCMC sampler:

np.random.seed(39091)
with model:
    trace = pmx.sample(
        tune=3500, draws=3000, start=map_soln, chains=2, cores=2
    )

# Then we can look at some summaries of the trace and the constraints on some of the key parameters:

# +
import corner

corner.corner(
    pm.trace_to_dataframe(trace, varnames=["P", "K", "tp", "ecc", "omega"])
)

with model:
    summary = pm.summary(
        trace, var_names=["P", "K", "tp", "ecc", "omega", "means", "sigmas"]
    )
summary
# -

# And finally we can plot the phased RV curve and overplot our posterior inference:

# +
mu = np.mean(trace["mean"] + trace["gp_pred"], axis=0)
mu_var = np.var(trace["mean"], axis=0)
jitter_var = np.median(trace["diag"], axis=0)
period = np.median(trace["P"])
tp = np.median(trace["tp"])

detrended = rv - mu
folded = ((t - tp + 0.5 * period) % period) / period
plt.errorbar(folded, detrended, yerr=np.sqrt(mu_var + jitter_var), fmt=",k")
plt.scatter(
    folded,
    detrended,
    c=inst_id,
    s=8,
    zorder=100,
    cmap="tab10",
    vmin=0,
    vmax=10,
)
plt.errorbar(
    folded + 1, detrended, yerr=np.sqrt(mu_var + jitter_var), fmt=",k"
)
plt.scatter(
    folded + 1,
    detrended,
    c=inst_id,
    s=8,
    zorder=100,
    cmap="tab10",
    vmin=0,
    vmax=10,
)

t_phase = np.linspace(-0.5, 0.5, 5000)
with model:
    func = xo.get_theano_function_for_var(
        rv_model(model.P * t_phase + model.tp)
    )
    for point in xo.get_samples_from_trace(trace, 100):
        args = xo.get_args_for_theano_function(point)
        x, y = t_phase + 0.5, func(*args)
        plt.plot(x, y, "k", lw=0.5, alpha=0.5)
        plt.plot(x + 1, y, "k", lw=0.5, alpha=0.5)
plt.axvline(1, color="k", lw=0.5)
plt.xlim(0, 2)
plt.xlabel("phase")
plt.ylabel("radial velocity [m/s]")
_ = plt.title("posterior inference", fontsize=14)
# -

# ## Citations
#
# As described in the :ref:`citation` tutorial, we can use :func:`exoplanet.citations.get_citations_for_model` to construct an acknowledgement and BibTeX listing that includes the relevant citations for this model.

with model:
    txt, bib = xo.citations.get_citations_for_model()
print(txt)

print("\n".join(bib.splitlines()[:10]) + "\n...")
