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

import lightkurve as lk

# %matplotlib inline

# %run notebook_setup

# # Fitting light curves from multiple instruments
#
# In the :ref:`rv-multi` case study, we discussed fitting the radial velocity curve for a planetary system observed using multiple instruments.
# You might also want to fit data from multiple instruments when fitting the light curve of a transiting planet and that's what we work through in this example.
# This is a somewhat more complicated example than the radial velocity case because some of the physical properties of the system can vary as as function of the instrument.
# Specifically, the transit depth (or the effective raduis of the planet) will be a function of the filter or effective wavelength of the observations.
# This is the idea behind transit spectroscopy and the method used in this case study could (and should!) be extended to that use case.
# In this case, we'll combine the light curves from the Kepler and TESS missions for the planet host HAT-P-11.
#
# ## A brief aside on dataset "weighting"
#
# Before getting into the details of this case study, let's spend a minute talking about a topic that comes up a lot when discussing combining observations from different instruments or techniques.
# To many people, it seems intuitive that one should (and perhaps must) "weight" how much each dataset contributes to the likelihood based on how much they "trust" those data.
# For example, you might be worried that a dataset with more datapoints will have a larger effect on the the results than you would like.
# While this might seem intuitive, it's wrong: **the only way to combine datasets is to multiply their likelihood functions**.
# Instead, it is useful to understand what you actually mean when you say that you don't "trust" a dataset as much as another.
# **What you're really saying is that you don't believe the observation model that you wrote down**.
# For example, you might think that the quoted error bars are underestimated or there might be correlated noise that an uncorrelated normal observation model can't capture.
# The benefit of thinking about it this way is that it suggests a solution to the problem: incorporate a more flexible observation model that can capture these issues.
# In this case study, the 4 years of (long-cadence) Kepler observations only include about two times as many data points as one month of TESS observations.
# But, as you can see in the figure below, these two datasets have different noise properties (both in terms of photon noise and correlated noise) so we will fit using a different flexible Gaussian process noise model for each data set that will take these different properties into account.

# +
import numpy as np
import lightkurve as lk
from collections import OrderedDict

kepler_lcfs = lk.search_lightcurvefile(
    "HAT-P-11", mission="Kepler"
).download_all()
kepler_lc = kepler_lcfs.PDCSAP_FLUX.stitch().remove_nans()
kepler_t = np.ascontiguousarray(kepler_lc.time, dtype=np.float64)
kepler_y = np.ascontiguousarray(1e3 * (kepler_lc.flux - 1), dtype=np.float64)
kepler_yerr = np.ascontiguousarray(1e3 * kepler_lc.flux_err, dtype=np.float64)

hdr = kepler_lcfs[0].hdu[1].header
kepler_texp = hdr["FRAMETIM"] * hdr["NUM_FRM"]
kepler_texp /= 60.0 * 60.0 * 24.0

tess_lcfs = lk.search_lightcurvefile("HAT-P-11", mission="TESS").download_all()
tess_lc = tess_lcfs.PDCSAP_FLUX.stitch().remove_nans()
tess_t = np.ascontiguousarray(
    tess_lc.time + 2457000 - 2454833, dtype=np.float64
)
tess_y = np.ascontiguousarray(1e3 * (tess_lc.flux - 1), dtype=np.float64)
tess_yerr = np.ascontiguousarray(1e3 * tess_lc.flux_err, dtype=np.float64)

hdr = tess_lcfs[0].hdu[1].header
tess_texp = hdr["FRAMETIM"] * hdr["NUM_FRM"]
tess_texp /= 60.0 * 60.0 * 24.0

datasets = OrderedDict(
    [
        ("Kepler", [kepler_t, kepler_y, kepler_yerr, kepler_texp]),
        ("TESS", [tess_t, tess_y, tess_yerr, tess_texp]),
    ]
)

fig, axes = plt.subplots(1, len(datasets), sharey=True, figsize=(10, 5))

for i, (name, (t, y, _, _)) in enumerate(datasets.items()):
    ax = axes[i]
    ax.plot(t, y, "k", lw=0.75, label=name)
    ax.set_xlabel("time [KBJD]")
    ax.set_title(name, fontsize=14)

    x_mid = 0.5 * (t.min() + t.max())
    ax.set_xlim(x_mid - 10, x_mid + 10)
axes[0].set_ylim(-10, 10)
fig.subplots_adjust(wspace=0.05)
_ = axes[0].set_ylabel("relative flux [ppt]")
# -

# ## The probabilistic model
#
# This model is mostly the same as the one used in :ref:`quick-tess`, but we're allowing for different noise variances (both the white noise component and the GP amplitude), effective planet radii, and limb-darkening coeeficients for each dataset.
# For the purposes of demonstration, we're sharing the length scale of the GP between the two datasets, but this could just have well been a different parameter for each dataset without changing the results.
# The final change that we're using is to use the approximate transit depth `approx_depth` (the depth of the transit at minimum assuming the limb-darkening profile is constant under the disk of the planet) as a parameter instead of the radius ratio.
# This does not have a large effect on the performance or the results, but it can sometimes be a useful parameterization when dealing with high signal-to-noise transits because it reduces the covariance between the radius parameter and the limb darkening coefficients.
# As usual, we run a few iterations of sigma clipping and then find the maximum a posteriori parameters to check to make sure that everything is working:

# +
import pymc3 as pm
import pymc3_ext as pmx
import exoplanet as xo
import theano.tensor as tt
from functools import partial
from celerite2.theano import terms, GaussianProcess

# Period and reference transit time from the literature for initialization
lit_period = 4.887803076
lit_t0 = 124.8130808

# Find a reference transit time near the middle of the observations to avoid
# strong covariances between period and t0
x_min = min(np.min(x) for x, _, _, _ in datasets.values())
x_max = max(np.max(x) for x, _, _, _ in datasets.values())
x_mid = 0.5 * (x_min + x_max)
t0_ref = lit_t0 + lit_period * np.round((x_mid - lit_t0) / lit_period)

# Do several rounds of sigma clipping
for i in range(10):
    with pm.Model() as model:

        # Shared orbital parameters
        period = pm.Lognormal("period", mu=np.log(lit_period), sigma=1.0)
        t0 = pm.Normal("t0", mu=t0_ref, sigma=1.0)
        dur = pm.Lognormal("dur", mu=np.log(0.1), sigma=10.0)
        b = pmx.UnitUniform("b")
        ld_arg = 1 - tt.sqrt(1 - b ** 2)
        orbit = xo.orbits.KeplerianOrbit(
            period=period, duration=dur, t0=t0, b=b
        )

        # We'll also say that the timescale of the GP will be shared
        rho_gp = pm.InverseGamma(
            "rho_gp",
            testval=2.0,
            **pmx.estimate_inverse_gamma_parameters(1.0, 5.0),
        )

        # Loop over the instruments
        parameters = dict()
        lc_models = dict()
        gp_preds = dict()
        gp_preds_with_mean = dict()
        for n, (name, (x, y, yerr, texp)) in enumerate(datasets.items()):

            # We define the per-instrument parameters in a submodel so that we
            # don't have to prefix the names manually
            with pm.Model(name=name, model=model):
                # The flux zero point
                mean = pm.Normal("mean", mu=0.0, sigma=10.0)

                # The limb darkening
                u = xo.QuadLimbDark("u")
                star = xo.LimbDarkLightCurve(u)

                # The radius ratio
                approx_depth = pm.Lognormal(
                    "approx_depth", mu=np.log(4e-3), sigma=10
                )
                ld = 1 - u[0] * ld_arg - u[1] * ld_arg ** 2
                ror = pm.Deterministic("ror", tt.sqrt(approx_depth / ld))

                # Noise parameters
                med_yerr = np.median(yerr)
                std = np.std(y)
                sigma = pm.InverseGamma(
                    "sigma",
                    testval=med_yerr,
                    **pmx.estimate_inverse_gamma_parameters(
                        med_yerr, 0.5 * std
                    ),
                )
                sigma_gp = pm.InverseGamma(
                    "sigma_gp",
                    testval=0.5 * std,
                    **pmx.estimate_inverse_gamma_parameters(
                        med_yerr, 0.5 * std
                    ),
                )

                # Keep track of the parameters for optimization
                parameters[name] = [mean, u, approx_depth]
                parameters[f"{name}_noise"] = [sigma, sigma_gp]

            # The light curve model
            def lc_model(mean, star, ror, texp, t):
                return mean + 1e3 * tt.sum(
                    star.get_light_curve(orbit=orbit, r=ror, t=t, texp=texp),
                    axis=-1,
                )

            lc_model = partial(lc_model, mean, star, ror, texp)
            lc_models[name] = lc_model

            # The Gaussian Process noise model
            kernel = terms.SHOTerm(sigma=sigma_gp, rho=rho_gp, Q=1.0 / 3)
            gp = GaussianProcess(
                kernel, t=x, diag=yerr ** 2 + sigma ** 2, mean=lc_model
            )
            gp.marginal(f"{name}_obs", observed=y)
            gp_preds[name] = gp.predict(y, include_mean=False)
            gp_preds_with_mean[name] = gp_preds[name] + gp.mean_value

        # Optimize the model
        map_soln = model.test_point
        for name in datasets:
            map_soln = pmx.optimize(map_soln, parameters[name])
        for name in datasets:
            map_soln = pmx.optimize(map_soln, parameters[f"{name}_noise"])
            map_soln = pmx.optimize(map_soln, parameters[name] + [dur, b])
        map_soln = pmx.optimize(map_soln)

        # Do some sigma clipping
        num = dict((name, len(datasets[name][0])) for name in datasets)
        clipped = dict()
        masks = dict()
        for name in datasets:
            mdl = xo.eval_in_model(gp_preds_with_mean[name], map_soln)
            resid = datasets[name][1] - mdl
            sigma = np.sqrt(np.median((resid - np.median(resid)) ** 2))
            masks[name] = np.abs(resid - np.median(resid)) < 7 * sigma
            clipped[name] = num[name] - masks[name].sum()
            print(f"Sigma clipped {clipped[name]} {name} light curve points")

        if all(c < 10 for c in clipped.values()):
            break

        else:
            for name in datasets:
                datasets[name][0] = datasets[name][0][masks[name]]
                datasets[name][1] = datasets[name][1][masks[name]]
                datasets[name][2] = datasets[name][2][masks[name]]
# -

# Here are the two phased light curves (with the Gaussian process model removed).
# We can see the effect of exposure time integration and the difference in photometric precision, but everything should be looking good!

# +
dt = np.linspace(-0.2, 0.2, 500)

with model:
    trends = xo.eval_in_model([gp_preds[k] for k in datasets], map_soln)
    phase_curves = xo.eval_in_model(
        [lc_models[k](t0 + dt) for k in datasets], map_soln
    )

fig, axes = plt.subplots(2, sharex=True, sharey=True, figsize=(8, 6))

for n, name in enumerate(datasets):
    ax = axes[n]

    x, y = datasets[name][:2]

    period = map_soln["period"]
    folded = (x - map_soln["t0"] + 0.5 * period) % period - 0.5 * period
    m = np.abs(folded) < 0.2
    ax.plot(
        folded[m],
        (y - trends[n] - map_soln[f"{name}_mean"])[m],
        ".k",
        alpha=0.3,
        mec="none",
    )
    ax.plot(
        dt, phase_curves[n] - map_soln[f"{name}_mean"], f"C{n}", label=name
    )
    ax.annotate(
        name,
        xy=(1, 0),
        xycoords="axes fraction",
        va="bottom",
        ha="right",
        xytext=(-3, 3),
        textcoords="offset points",
        fontsize=14,
    )

axes[-1].set_xlim(-0.15, 0.15)
axes[-1].set_xlabel("time since transit [days]")
for ax in axes:
    ax.set_ylabel("relative flux [ppt]")
# -

# Then we run the MCMC:

np.random.seed(11)
with model:
    trace = pmx.sample(
        tune=2500,
        draws=2000,
        start=map_soln,
        cores=2,
        chains=2,
        initial_accept=0.5,
    )

# And check the convergence diagnostics:

with model:
    summary = pm.summary(trace)
summary

# Since we fit for a radius ratio in each band, we can see if the transit depth is different in Kepler compared to TESS.
# The plot below demonstrates that there is no statistically significant difference between the radii measured in these two bands:

plt.hist(
    trace["Kepler_ror"], 30, density=True, histtype="step", label="Kepler"
)
plt.hist(trace["TESS_ror"], 30, density=True, histtype="step", label="TESS")
plt.yticks([])
plt.xlabel("effective radius ratio")
_ = plt.legend(fontsize=12)

# We can also compare the inferred limb-darkening coefficients:

# +
import corner

fig = corner.corner(
    trace["TESS_u"], bins=40, color="C1", range=((0.5, 0.9), (-0.5, 0.1))
)
corner.corner(
    trace["Kepler_u"],
    bins=40,
    color="C0",
    fig=fig,
    labels=["$u_1$", "$u_2$"],
    range=((0.5, 0.9), (-0.5, 0.1)),
)
fig.axes[0].axvline(-1.0, color="C0", label="Kepler")
fig.axes[0].axvline(-1.0, color="C1", label="TESS")
_ = fig.axes[0].legend(
    fontsize=12, loc="center left", bbox_to_anchor=(1.1, 0.5)
)
# -

# ## Citations
#
# As described in the :ref:`citation` tutorial, we can use :func:`exoplanet.citations.get_citations_for_model` to construct an acknowledgement and BibTeX listing that includes the relevant citations for this model.

with model:
    txt, bib = xo.citations.get_citations_for_model()
print(txt)

print("\n".join(bib.splitlines()[:10]) + "\n...")
