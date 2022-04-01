# -*- coding: utf-8 -*-
#
# conf.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'NEST-GPU'
copyright = '2022, JoseJVS, golosio, jhnnsnk'
author = 'JoseJVS, golosio, jhnnsnk'

# The full version, including alpha/beta/rc tags
release = '1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'nbsphinx'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'manni'
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'nest': ('https://nest-simulator.readthedocs.io/en/latest/', None),
    'nestml': ('https://nestml.readthedocs.io/en/latest/', None),
    'pynn': ('http://neuralensemble.org/docs/PyNN/', None),
    'elephant': ('https://elephant.readthedocs.io/en/latest/', None),
    'desktop': ('https://nest-desktop.readthedocs.io/en/latest/', None),
    'neuromorph': ('https://electronicvisions.github.io/hbp-sp9-guidebook/', None),
    'arbor': ('https://arbor.readthedocs.io/en/latest/', None),
    'tvb': ('http://docs.thevirtualbrain.org/', None),
    'extmod': ('https://nest-extension-module.readthedocs.io/en/latest/', None),
}


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = [
    'css/custom.css',
    'css/pygments.css',
]

html_logo =  '_static/img/nest_logo.png'
html_theme_options = {'logo_only': True,
                      'display_version': True}

#def setup(app):
#    app.add_css_file('css/custom.css')
#    app.add_css_file('css/pygments.css')
