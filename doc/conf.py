#
#  conf.py
#
# This file is part of NEST GPU.
#
# Copyright (C) 2021 The NEST Initiative
#
# NEST GPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST GPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST GPU.  If not, see <http://www.gnu.org/licenses/>.
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use str(Path().resolve()) to make it absolute.
#

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path().resolve()))

source_dir = Path(__file__).resolve().parent.resolve()
doc_build_dir = source_dir / "models"

print("doc_build_dir", str(doc_build_dir))
print("source_dir", str(source_dir))


# -- Project information -----------------------------------------------------

project = u'NEST GPU Documentation'
copyright = u'2004, nest-simulator'
author = u'nest-simulator'

# The full version, including alpha/beta/rc tags
release = '1'

source_suffix = '.rst'
master_doc = 'contents'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx_tabs.tabs',
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


# Extract documentation from header files in src/

from extractor_userdocs import relative_glob, ExtractUserDocs

def config_inited_handler(app, config):
    ExtractUserDocs(
        listoffiles=relative_glob("../src/*.h", basedir=source_dir),
        basedir=source_dir,
        outdir=str(doc_build_dir)
    )

def setup(app):
    # for events see
    # https://www.sphinx-doc.org/en/master/extdev/appapi.html#sphinx-core-events
    app.connect('config-inited', config_inited_handler)

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_show_sphinx = False
html_show_copyright = False
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = [
    'css/custom.css',
    'css/pygments.css',
]

html_logo =  'logo/nestgpu-logo.png'
html_theme_options = {'logo_only': True,
                      'display_version': True}

def setup(app):
    app.connect('config-inited', config_inited_handler)
#    app.add_css_file('css/custom.css')
#    app.add_css_file('css/pygments.css')