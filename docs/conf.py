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
    "IPython.sphinxext.ipython_console_highlighting",
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

intersphinx_mapping = {
    "exoplanet": ("https://docs.exoplanet.codes/en/latest/", None),
}

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
nb_execution_mode = "off"
nb_execution_timeout = -1
