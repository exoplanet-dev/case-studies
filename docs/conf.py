#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution("exoplanet").version
except DistributionNotFound:
    __version__ = "unknown version"


# General stuff
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "myst_nb",
]

myst_enable_extensions = ["dollarmath", "colon_fence"]

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"

# General information about the project.
project = "exoplanet"
author = "Dan Foreman-Mackey"
copyright = "2018, 2019, 2020, 2021, " + author

version = __version__
release = __version__

exclude_patterns = ["_build"]

# HTML theme
html_theme = "sphinx_book_theme"
html_copy_source = True
html_show_sourcelink = True
html_sourcelink_suffix = ""
html_title = "exoplanet"
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.png"
html_static_path = ["_static"]
html_css_files = ["exoplanet.css"]
html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/exoplanet-dev/case-studies",
    "repository_branch": "main",
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "notebook_interface": "jupyterlab",
    },
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
}
jupyter_execute_notebooks = "auto"
execution_timeout = -1

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
    dict(slug="tess", title="Fitting TESS data", figure="tess_20_0.png"),
    dict(
        slug="quick-tess",
        title="Quick fits for TESS light curves",
        figure="quick-tess_12_0.png",
    ),
    dict(slug="ttv", title="Fitting transit times", figure="ttv_20_0.png"),
    dict(
        slug="eb",
        title="Fitting a detached eclipsing binary",
        figure="eb_12_0.png",
    ),
    dict(
        slug="rv-multi",
        title="RVs with multiple instruments",
        figure="rv-multi_13_0.png",
    ),
    dict(
        slug="lc-multi",
        title="Fitting light curves from multiple instruments",
        figure="lc-multi_7_0.png",
    ),
]
html_context = dict(
    case_studies=case_studies,
)
