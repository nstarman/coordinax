"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import importlib.metadata
from datetime import datetime

from typing import Any

import pytz
from docutils.nodes import Element, Node, reference
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment

# -- Project information -----------------------------------------------------

author = "Coordinax Developers"
project = "coordinax"
copyright = f"{datetime.now(pytz.timezone('UTC')).year}, {author}"
version = importlib.metadata.version("coordinax")

master_doc = "index"
language = "en"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_nb",  # Jupyter notebook support via MyST (includes myst_parser)
    "sphinx_design",
    "sphinx.ext.autodoc",  # TODO: replace with autodoc2
    "sphinx.ext.autosummary",  # TODO: replace with autodoc2
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx-prompt",
    "sphinxext.opengraph",
    # "sphinxext.rediraffe",  # Add redirects
    "sphinx_togglebutton",
    "sphinx_tippy",
]

# Wikipedia's REST API requires a User-Agent header; sphinx_tippy doesn't set
# one, so requests get a 403 with non-JSON body, causing a build warning.
# TODO: periodically check if this is fixed
tippy_enable_wikitips = False

python_use_unqualified_type_names = True

exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
    "guides/perf.md",  # Excluded: converted to perf.ipynb by jupytext
]

source_suffix = [".md", ".rst", ".ipynb"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    # Canonical URL (jax.readthedocs.io now redirects here)
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "jaxtyping": ("https://docs.kidger.site/jaxtyping/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "equinox": ("https://docs.kidger.site/equinox/", None),
    "plum": ("https://beartype.github.io/plum/", None),
    "quax": ("https://docs.kidger.site/quax/", None),
    "unxt": ("https://unxt.readthedocs.io/en/latest/", None),
}

# -- Autodoc settings ---------------------------------------------------

# Silence import failures for optional extension packages that are not
# installed in the docs build environment (coordinax.astro,
# coordinax.interop.astropy are optional extras).
suppress_warnings = ["autodoc.import_object"]

autodoc_typehints = "description"
autodoc_typehints_format = "short"

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}

always_document_param_types = True
typehints_use_signature = True


nitpick_ignore = [
    # typing module — sphinx_autodoc_typehints emits these from generated
    # signatures but they have no stable inventory target yet.
    # TODO: Revisit after upgrading Sphinx and/or sphinx-autodoc-typehints.
    ("py:data", "typing.Union"),
    ("py:data", "typing.Any"),
    ("py:data", "typing.ClassVar"),
    ("py:data", "typing.NoReturn"),
    ("py:data", "Ellipsis"),
    # ArrayLike is py:data in JAX but sphinx_autodoc_typehints emits py:class.
    ("py:class", "ArrayLike"),
    ("py:class", "jax.typing.ArrayLike"),
    # jax.Array is a type alias (py:data in JAX), not a class.
    ("py:class", "jax.Array"),
    ("py:class", "jax.lax.GatherScatterMode"),
    # JAX functions: intersphinx resolves these when network is available;
    # kept here as a fallback for restricted build environments.
    ("py:func", "jax.jit"),
    ("py:func", "jax.vmap"),
    ("py:func", "jax.grad"),
    ("py:func", "jax.jacobian"),
    ("py:func", "jax.hessian"),
    # unxt public API not exposed in unxt's Sphinx inventory
    ("py:class", "unxt.Angle"),
    ("py:class", "unxt.quantity.Quantity"),
    ("py:class", "AbstractQuantity"),
    ("py:class", "u.AbstractDimension"),
    ("py:class", "u.AbstractUnit"),
    # numpy types not in the numpy intersphinx inventory
    ("py:class", "numpy.bool"),
    ("py:class", "numpy.dtype"),
    ("py:class", "numpy.ndarray"),
    ("py:class", "numpy.number"),
    ("py:class", "jnp.ndarray"),
    # astropy internal paths not exposed in astropy's intersphinx
    ("py:class", "astropy.units.core.CompositeUnit"),
    ("py:class", "astropy.units.core.Unit"),
    ("py:class", "astropy.units.core.UnitBase"),
    ("py:class", "astropy.units.physical.PhysicalType"),
    ("py:class", "astropy.units.quantity.Quantity"),
    ("py:class", "astropy.coordinates.builtin_frames.icrs.ICRS"),
    ("py:class", "astropy.coordinates.builtin_frames.galactocentric.Galactocentric"),
    (
        "py:class",
        "astropy.coordinates.representation.cartesian.CartesianRepresentation",
    ),
    (
        "py:class",
        "astropy.coordinates.representation.cylindrical.CylindricalRepresentation",
    ),
    (
        "py:class",
        "astropy.coordinates.representation.spherical.SphericalRepresentation",
    ),
    (
        "py:class",
        "astropy.coordinates.representation.spherical.PhysicsSphericalRepresentation",
    ),
    # collections.abc — Python docs inventory uses py:class for these but
    # sphinx_autodoc_typehints sometimes resolves them as bare names.
    ("py:class", "collections.abc.Callable"),
    ("py:class", "collections.abc.Mapping"),
    ("py:class", "collections.abc.Sequence"),
    # equinox private types
    ("py:class", "equinox._module._better_abstract.AbstractVar"),
    ("py:class", "equinox._module._module.Module"),
    # quax — no Sphinx inventory (MkDocs)
    ("py:class", "quax.ArrayValue"),
    ("py:class", "quax._core.Value"),
    # plum
    ("py:exc", "plum.NotFoundLookupError"),
    # unxt_hypothesis — separate optional package
    ("py:func", "unxt_hypothesis.quantities"),
    # coordinax-astro (optional extension package, not always installed)
    ("py:class", "coordinax.astro.DistanceModulus"),
    ("py:class", "coordinax.astro.Parallax"),
    # coordinax internal public API re-exported via coordinax.api.frames
    ("py:obj", "coordinax.frames.act"),
    ("py:obj", "coordinax.frames.compose"),
    ("py:obj", "coordinax.frames.simplify"),
    # coordinax convenience singletons referenced in docstrings
    ("py:obj", "minkowski4d"),
    ("py:obj", "embed_tangent"),
    ("py:obj", "project_tangent"),
    ("py:obj", "same"),
    # coordinax internal private paths (via sphinx_autodoc_typehints)
    ("py:class", "CDict"),
    ("py:class", "OptUSys"),
    ("py:class", "coordinax.charts._src.base.CDictT"),
    ("py:class", "coordinax.distances._src.base.AbstractDistance"),
    ("py:class", "coordinax.internal.custom_types.Ks"),
    ("py:class", "coordinax.internal.custom_types.Ds"),
    ("py:obj", "coordinax.internal.custom_types.Ks"),
    ("py:obj", "coordinax.internal.custom_types.Ds"),
    # Private internal helpers from dependencies with no public docs
    ("py:class", "unxt._src.quantity.base._QuantityIndexUpdateHelper"),
    ("py:class", "dataclassish._src.converters.PassThroughTs"),
    ("py:class", "dataclassish._src.converters.ArgT"),
    ("py:class", "unxt._src.quantity.quantity.Quantity[PhysicalType('length')]"),
    ("py:class", "coordinax.vectors._src.core.Point"),
    ("py:class", "coordinax.representations._src.semantics.AbstractSemanticKind"),
    ("py:class", "coordinax.representations._src.geom.PointGeometry"),
    ("py:class", "coordinax.representations._src.basis.AbstractBasis"),
    ("py:class", "coordinax.charts._src.d3.LonLatSpherical3D"),
]

# TypedNdArray is a JAX-private type (jax._src.basearray) with no public docs.
# jax._src.* are private JAX implementation paths never in the public inventory.
nitpick_ignore_regex = [
    (r"py:class", r"jaxtyping\..*"),  # TODO: remove
    (r"py:class", r".*TypedNdArray.*"),
    (r"py:class", r"jax\._src\..*"),
    # Private implementation paths from unxt and coordinax itself
    (r"py:class", r"unxt\._src\..*"),
    (r"py:class", r"coordinax\..*\._src\..*"),
    (r"py:obj", r"coordinax\..*\._src\..*"),
    # Private equinox internals
    (r"py:class", r"equinox\._.*"),
    # Private jaxlib internals (e.g., jaxlib._jax.Device)
    (r"py:class", r"jaxlib\..*"),
    # quax private internals
    (r"py:class", r"quax\._.*"),
    # astropy internal representation paths
    (r"py:class", r"astropy\.coordinates\.representation\..*"),
    # coordinax internal custom types (not under ._src.)
    (r"py:class", r"coordinax\.internal\.custom_types\..*"),
    (r"py:obj", r"coordinax\.internal\.custom_types\..*"),
    # Bare names and garbage emitted by Plum dispatch auto-docstrings
    (r"py:class", r"^[a-zA-Z0-9]$"),        # Single character (TypeVars)
    (r"py:class", r"^'.*'$"),               # Quoted shape annotations (e.g. 'N N')
    (r"py:class", r"^[a-z][a-z ]+$"),       # lowercase words/phrases (no dots/caps)
    (r"py:class", r".*\s.*"),  # whitespace → not a valid class name
    (r"py:class", r"^[A-Z][a-z]+ [a-z].*"),  # Sentence-case phrases
    (r"py:class", r"^\d"),                  # Leading digit (e.g. '4')
    (r"py:class", r"^[A-Z][a-z]{1,4}$"),   # Short TypeVar names (e.g. Rep, Tau)
    (r"py:obj", r"typing\.Annotated\[.*"),  # Complex Annotated types
]

# -- MyST Setting -------------------------------------------------

myst_enable_extensions = [
    "amsmath",  # for direct LaTeX math
    "attrs_block",  # enable parsing of block attributes
    "attrs_inline",  # apply syntax highlighting to inline code
    "colon_fence",
    "deflist",
    "dollarmath",  # for $, $$
    # "linkify",  # identify “bare” web URLs and add hyperlinks:
    "smartquotes",  # convert straight quotes to curly quotes
    "substitution",  # substitution definitions
]
myst_heading_anchors = 3

# -- MyST-NB settings (Jupyter notebook support) --------------------------

nb_execution_mode = "cache"
nb_execution_cache_path = "_build/.jupyter_cache"
nb_execution_raise_on_error = True
nb_execution_timeout = 100

# myst_substitutions = {
#     "ArrayLike": "{obj}`jaxtyping.ArrayLike`",
#     "Any": "{obj}`typing.Any`",
# }


rst_prolog = """
.. py:module:: coordinax
.. py:module:: astropy
.. py:module:: plum
.. py:module:: JAX
.. py:module:: unxt-hypothesis
"""


# -- HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_title = "coordinax"
html_logo = "_static/favicon.png"  # TODO: an svg
html_copy_source = True
html_favicon = "_static/favicon.png"

html_static_path = ["_static"]
html_css_files = ["custom_toc.css", "custom_tooltip.css"]

html_theme_options: dict[str, Any] = {
    "home_page_in_toc": True,
    "repository_url": "https://github.com/GalacticDynamics/coordinax",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_edit_page_button": False,
    "use_issues_button": True,
    "show_toc_level": 2,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/GalacticDynamics/coordinax",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/coordinax/",
            "icon": "https://img.shields.io/pypi/v/coordinax",
            "type": "url",
        },
        {
            "name": "Zenodo",
            "url": "https://zenodo.org/doi/10.5281/zenodo.10850557",
            "icon": "fa fa-quote-right",
        },
    ],
}

# -- Missing-reference handler ----------------------------------------
# Plum's combined __doc__ emits bare short names (e.g. `PhysicalType`) inside
# ``.. py:function::`` RST directives. Intersphinx cannot match short names —
# only fully-qualified keys are in its inventory. This handler intercepts those
# unresolved references and returns real hyperlinks instead of suppressing them.

# Map bare short name → canonical documentation URL
_SHORT_NAME_URLS: dict[str, str] = {
    # astropy: not in Sphinx inventory under short name
    "PhysicalType": (
        "https://docs.astropy.org/en/stable/api/astropy.units.PhysicalType.html"
    ),
    # types.NoneType (Python 3.10+): intersphinx key is qualified; bare name
    # won't resolve
    "NoneType": "https://docs.python.org/3/library/types.html#types.NoneType",
    # numpy.ndarray: intersphinx key is qualified; bare name won't resolve
    "ndarray": "https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html",
    # jaxtyping.Array is an alias for jax.Array; jaxtyping has no Sphinx
    # inventory (MkDocs)
    "Array": "https://docs.jax.dev/en/latest/_autosummary/jax.Array.html",
}


def _resolve_short_names(
    app: Sphinx,
    env: BuildEnvironment,
    node: Element,
    contnode: Element,
) -> Node | None:
    """Resolve known bare short names to external documentation links."""
    if node.get("refdomain") != "py":
        return None
    target = node.get("reftarget", "")
    url = _SHORT_NAME_URLS.get(target)
    if url is None:
        return None
    return reference("", "", contnode, internal=False, refuri=url)


def setup(app: Sphinx, /) -> None:
    app.connect("missing-reference", _resolve_short_names)
