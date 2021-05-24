# -*- coding: utf-8 -*-
#
# librosa documentation build configuration file, created by
# sphinx-quickstart on Tue Jun 25 13:12:33 2013.
#
# This file is execfile()d with the current directory set to its containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

#
# To build multi-versions, run the following at the base directory
#   $ sphinx-multiversion docs/ build/html/
#

import os
import sys
from pathlib import Path
import sphinx

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# sys.path.insert(0, os.path.abspath("../"))

# This song and dance enables builds from outside the docs directory
srcpath = os.path.abspath(Path(os.path.dirname(__file__)) / '..')
sys.path.insert(0, srcpath)

# Purge old imports, trying to hack around sphinx-multiversion global environment
removals = [_ for _ in sys.modules if 'librosa' in _]
for _ in removals:
    del sys.modules[_]

# -- General configuration -----------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
if sphinx.__version__ < "2.0":
    raise RuntimeError("Sphinx 2.0 or newer is required")

needs_sphinx = "2.0"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.

from importlib.machinery import SourceFileLoader

librosa_version = SourceFileLoader(
    "librosa.version", os.path.abspath(Path(srcpath) / 'librosa' / 'version.py')
).load_module()

# The short X.Y version.
version = librosa_version.version
# The full version, including alpha/beta/rc tags.
release = librosa_version.version

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.autodoc",  # function indexing
    "sphinx.ext.autosummary",  # for older builds
    "sphinx.ext.viewcode",  # source linkage
    "sphinx.ext.intersphinx",  # cross-linkage
    "sphinx.ext.mathjax",
    "sphinx_gallery.gen_gallery",  # advanced examples
    "numpydoc",  # docstring examples
    "matplotlib.sphinxext.plot_directive",  # docstring examples
    "sphinxcontrib.inkscapeconverter",  # used for badge / logo conversion in tex 
    "sphinx_multiversion"  # historical builds
]

autosummary_generate = True

# --------
# Doctest
# --------

doctest_global_setup = """
import numpy as np
import scipy
import librosa
np.random.seed(123)
np.set_printoptions(precision=3, linewidth=64, edgeitems=2, threshold=200)
"""

numpydoc_show_class_members = False

# ------------------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------------------
plot_pre_code = (
    doctest_global_setup
    + """
import matplotlib
import librosa
import librosa.display
matplotlib.rcParams['figure.constrained_layout.use'] = librosa.__version__ >= '0.8'
"""
)
plot_include_source = True
plot_html_show_source_link = False
plot_formats = [("png", 100), ("pdf", 100)]
plot_html_show_formats = False

font_size = 12  # 13*72/96.0  # 13 px

plot_rcparams = {
    "font.size": font_size,
    "legend.loc": "upper right",
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "axes.xmargin": 0,
    "axes.ymargin": 0,
    "axes.titlesize": font_size,
    "axes.labelsize": font_size,
    "xtick.labelsize": font_size,
    "ytick.labelsize": font_size,
    "legend.fontsize": font_size,
    "figure.subplot.bottom": 0.2,
    "figure.subplot.left": 0.2,
    "figure.subplot.right": 0.9,
    "figure.subplot.top": 0.85,
    "figure.subplot.wspace": 0.4,
    "text.usetex": False,
}


def reset_mpl(gallery_conf, fname):
    global plot_rcparams

    import matplotlib
    import matplotlib.pyplot as plt
    import librosa

    matplotlib.rcParams.update(**plot_rcparams)

    # Only use constrained layout in 0.8 and above
    matplotlib.rcParams['figure.constrained_layout.use'] = librosa.__version__ >= '0.8'
    plt.close('all')


# Gallery
sphinx_gallery_conf = {
    "examples_dirs": "examples/",
    "gallery_dirs": "auto_examples",
    "backreferences_dir": None,
    "reference_url": {
        "sphinx_gallery": None,
        "librosa": None,
        "numpy": "https://numpy.org/doc/stable/",
        "np": "https://numpy.org/doc/stable/",
        "scipy": "https://docs.scipy.org/doc/scipy/reference/",
        "matplotlib": "https://matplotlib.org/",
        "sklearn": "https://scikit-learn.org/stable/",
        "resampy": "https://resampy.readthedocs.io/en/latest/",
        "pyrubberband": "https://pyrubberband.readthedocs.io/en/stable/",
        "samplerate": "https://python-samplerate.readthedocs.io/en/latest/",
        "pooch": "https://www.fatiando.org/pooch/latest/",
        "soxr": "https://github.com/dofuuz/python-soxr",
    },
    "reset_modules": (reset_mpl,),
    'capture_repr': ('_repr_html_',),
}

# Generate plots for example sections
numpydoc_use_plots = True


intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "np": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "resampy": ("https://resampy.readthedocs.io/en/latest/", None),
    "soundfile": ("https://pysoundfile.readthedocs.io/en/latest", None),
    "samplerate": ("https://python-samplerate.readthedocs.io/en/latest/", None),
    "pyrubberband": ("https://pyrubberband.readthedocs.io/en/stable/", None),
}


# Add any paths that contain templates here, relative to this directory.
templates_path = [
    "_templates",
]

html_sidebars = {'*': ["versions.html"]}


# The suffix of source filenames.
source_suffix = ".rst"

# The encoding of source files.
source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = "index"

# General information about the project.
project = u"librosa"
copyright = u"2013--2021, librosa development team"


# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
# language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
# today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build"]

# The reST default role (used for this markup: `text`) to use for all documents.
default_role = "autolink"

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []

# -- Options for HTML output -------------------------------------------------
import sphinx_rtd_theme

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

# If false, no module index is generated.
html_domain_indices = True

# If false, no index is generated.
html_use_index = True

html_use_modindex = True

# If true, the index is split into individual pages for each letter.
# html_split_index = False

# If true, links to the reST sources are added to the pages.
# html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
# html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
# html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
# html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
# html_file_suffix = None

# Output file base name for HTML help builder.
htmlhelp_basename = "librosadoc"

html_logo = 'img/librosa_logo_text.svg'

html_theme_options = {
    'logo_only': True,
    'style_nav_header_background': 'white',
    'analytics_id': 'UA-171031946-1',
}
html_static_path = ['_static']
html_css_files = [
    'css/custom.css',
]

# -- Options for LaTeX output --------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #'preamble': '',
    'maxlistdepth' : '12',
    'fontpkg': r'''
\usepackage[scaled]{helvet} % ss
\usepackage{courier} % tt
\usepackage{mathpazo} % math & rm
\linespread{1.05}        % Palatino needs more leading (space between lines)
\normalfont
\usepackage[T1]{fontenc}
''',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
    (
        "index",
        "librosa.tex",
        u"librosa Documentation",
        u"The librosa development team",
        "manual",
    )
]
latex_engine = 'xelatex'
# The name of an image file (relative to this directory) to place at the top of
# the title page.
latex_logo = 'img/librosa_logo_text.png'

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
# latex_use_parts = False

# If true, show page references after internal links.
# latex_show_pagerefs = False

# If true, show URL addresses after external links.
# latex_show_urls = False

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# If false, no module index is generated.
# latex_domain_indices = True


# -- Options for manual page output --------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ("index", "librosa", u"librosa Documentation", [u"The librosa development team"], 1)
]

# If true, show URL addresses after external links.
# man_show_urls = False


# -- Options for Texinfo output ------------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        "index",
        "librosa",
        u"librosa Documentation",
        u"The librosa development team",
        "librosa",
        "One line description of project.",
        "Miscellaneous",
    )
]

# Documents to append as an appendix to all manuals.
# texinfo_appendices = []

# If false, no module index is generated.
# texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
# texinfo_show_urls = 'footnote'

autodoc_member_order = "bysource"

smv_branch_whitelist = r"^(main)$"  # build main branch, and anything relating to documentation
smv_tag_whitelist = r"^((0\.6\.3)|(0\.7\.\d+)|(0\.[8]\.\d+))$"  # use this for final builds
smv_released_pattern = r'.*tags.*'
smv_remote_whitelist = None
smv_greatest_tag = True
smv_prefer_remote_refs = False
