"""`coordinax.internal` — semi-public utilities.

.. warning::

    Everything in this module is **semi-public**.  The APIs exposed here
    are usable by downstream packages but are **not** covered by the
    same stability guarantees as the top-level ``coordinax`` API.  Names,
    signatures, and behaviour may change **at any time without warning**
    in minor or patch releases.  Pin to an exact version if you depend on
    anything here.

The heterogeneous-unit matrix machinery (``QuantityMatrix``, ``UnitsMatrix``,
``det``/``inv``, ``matmul``/``matvec``/``vecdot``/``vecmat``, ``cdict_units``)
lives in :mod:`unxts.linalg` as of unxt v2.0 and is imported directly from
there; it is intentionally **not** re-exported here.

Contents:

- ``pack_uniform_unit``
    Pack dict-of-quantities into an array, converting all entries to
    a common unit.

- ``tree_cast_int_bool_to_float``
    Tree-map over a PyTree, promoting integer and boolean leaves to the
    default floating-point dtype (``jax.dtypes.canonicalize_dtype(jnp.float_)``).
    Existing float and complex leaves are left unchanged.  Useful for
    satisfying ``jax.jacfwd``'s requirement of real-floating inputs.

- ``structured``
    Decorator for transparent argument and return value processing.
    This helps pushing the logic for packing/unpacking inside a JIT.

"""

# The heterogeneous-unit matrix machinery lives in ``unxts.linalg`` as of
# unxt v2.0; import it directly from there. ``QuantityMatrix`` is imported here
# only for the ``short_name`` display side-effect below — it is deliberately not
# re-exported from ``coordinax.internal``.
from unxts.linalg import QuantityMatrix

from . import custom_types  # noqa: F401
from .dtype_utils import *
from .pack_utils import *
from .wl_utils import *

# ``QuantityMatrix`` ships without a short name upstream. Give it one so it
# prints as ``QM(...)`` (matching the ``QM`` alias) under coordinax's
# ``use_short_name`` repr/str config (see ``[tool.unxts.unxt]`` in
# ``pyproject.toml``).
QuantityMatrix.short_name = "QM"  # ty: ignore[unresolved-attribute]
