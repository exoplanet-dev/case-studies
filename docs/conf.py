#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess

import sphinx_typlog_theme
from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution("exoplanet").version
except DistributionNotFound:
    __version__ = "unknown version"


def setup(app):
    app.add_stylesheet("css/exoplanet.css?v=2019-08-02")


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
]

autodoc_mock_imports = [
    "numpy",
    "scipy",
    "astropy",
    "pymc3",
    "theano",
    "tqdm",
    "rebound_pymc3",
]

# Convert the tutorials
if os.environ.get("READTHEDOCS", "False") == "True":
    subprocess.check_call("make tutorials", shell=True)

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "exoplanet": ("https://docs.exoplanet.codes/en/latest/", None),
}

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"

# General information about the project.
project = "case studies"
author = "Dan Foreman-Mackey"
copyright = "2018, 2019, 2020, " + author

version = __version__
release = __version__

exclude_patterns = ["_build"]
pygments_style = "sphinx"

# List of case studies
case_studies = [
    dict(
        slug="stellar-variability",
        title="Gaussian process models for stellar variability",
        figure="stellar-variability_10_0.png",
    ),
    dict(
        slug="together",
        title="Putting it all together",
        figure="together_35_0.png",
    ),
    dict(slug="tess", title="Fitting TESS data", figure="tess_29_0.png"),
    dict(
        slug="quick-tess",
        title="Quick fits for TESS light curves",
        figure="quick-tess_14_0.png",
    ),
    dict(slug="ttv", title="Fitting transit times", figure="ttv_19_0.png"),
    dict(
        slug="eb",
        title="Fitting a detached eclipsing binary",
        figure="eb_13_0.png",
    ),
    dict(
        slug="rv-multi",
        title="RVs with multiple instruments",
        figure="rv-multi_13_0.png",
    ),
]

# HTML theme
html_favicon = "_static/logo.png"
html_theme = "exoplanet"
html_theme_path = ["_themes", sphinx_typlog_theme.get_path()]
html_theme_options = {"logo": "logo.png"}
html_sidebars = {
    "**": ["logo.html", "globaltoc.html", "relations.html", "searchbox.html"]
}
html_static_path = ["_static"]
html_additional_pages = {"index": "index.html"}

# Get the git branch name
html_context = dict(
    this_branch="master",
    this_version=os.environ.get("READTHEDOCS_VERSION", "latest"),
    case_studies=case_studies,
)
