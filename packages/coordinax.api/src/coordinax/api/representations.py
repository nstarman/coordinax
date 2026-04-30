"""Representations."""

__all__ = (
    "add",
    "cconvert",
    "change_basis",
    "guess_basis_kind",
    "guess_geometry_kind",
    "guess_rep",
    "guess_semantic_kind",
    "subtract",
)

from typing import Any

import plum


@plum.dispatch.abstract
def change_basis(*args: Any, **kwargs: Any) -> Any:
    """Change the basis of a tangent vector's components.

    Examples
    --------
    >>> import coordinax.representations as cxr
    >>> import coordinax.charts as cxc
    >>> v = {"x": 1.0, "y": 0.0}
    >>> at = {"x": 1.0, "y": 0.0}
    >>> cxr.change_basis(v, cxc.cart2d, cxr.coord_basis, cxr.phys_basis, at=at)
    {'x': 1.0, 'y': 0.0}
    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def tangent_map(*args: Any, **kwargs: Any) -> Any:
    """Compute the tangent map (Jacobian) of a chart transition.

    Pushes a tangent vector ``v`` (attached at base point ``at`` in
    ``from_chart``) forward to ``to_chart`` via the Jacobian of the chart
    transition map.

    This is an abstract API definition. See the main coordinax package for
    concrete implementations.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr

    **Same-chart identity** - when ``from_chart`` and ``to_chart`` are the
    same object the function returns ``v`` unchanged:

    >>> v  = {"x": jnp.array(1.0), "y": jnp.array(2.0), "z": jnp.array(3.0)}
    >>> at = {"x": jnp.array(1.0), "y": jnp.array(0.0), "z": jnp.array(0.0)}
    >>> cxr.tangent_map(v, cxc.cart3d, cxr.coord_disp, cxc.cart3d, at=at)
    {'x': Array(1., dtype=float64, ...), 'y': Array(2., dtype=float64, ...), 'z': Array(3., dtype=float64, ...)}

    **Cartesian → spherical** - pushes a radial tangent vector at
    ``(x=1, y=0, z=0)`` into spherical coordinate components.  At this base
    point the only non-zero component is ``dr``:

    >>> v  = {"x": jnp.array(1.0), "y": jnp.array(0.0), "z": jnp.array(0.0)}
    >>> at = {"x": jnp.array(1.0), "y": jnp.array(0.0), "z": jnp.array(0.0)}
    >>> cxr.tangent_map(v, cxc.cart3d, cxr.coord_disp, cxc.sph3d, at=at)
    {'r': Array(1., dtype=float64, ...), 'theta': Array(0., dtype=float64, ...), 'phi': Array(0., dtype=float64, ...)}

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def cconvert(*args: Any, **kwargs: Any) -> Any:
    """Transform the current vector to the target chart.

    This is an abstract API definition. See the main coordinax package for
    concrete implementations.

    Examples
    --------
    >>> import coordinax.representations as cxr
    >>> import coordinax.charts as cxc

    Define a point in Cartesian coordinates:

    >>> p = {"x": 1.0, "y": 2.0, "z": 3.0}

    Convert it to spherical coordinates:

    >>> cxr.cconvert(p, cxc.cart3d, cxr.point, cxc.sph3d, cxr.point)
    {'r': Array(3.74165739, dtype=float64, ...),
     'theta': Array(0.64052231, dtype=float64),
     'phi': Array(1.10714872, dtype=float64, ...)}

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def guess_basis_kind(*args: Any, **kwargs: Any) -> Any:
    """Guess the basis kind of the given data.

    This is an abstract API definition. See the main coordinax package for
    concrete implementations.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.representations as cxr

    >>> data = {"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")}
    >>> cxr.guess_basis_kind(data)
    NoBasis()

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def guess_geometry_kind(*args: Any, **kwargs: Any) -> Any:
    """Guess the geometry kind of the given data.

    This is an abstract API definition. See the main coordinax package for
    concrete implementations.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.representations as cxr

    >>> data = {"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")}
    >>> cxr.guess_geometry_kind(data)
    PointGeometry()

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def guess_rep(*args: Any, **kwargs: Any) -> Any:
    """Guess the representation of the given data.

    This is an abstract API definition. See the main coordinax package for
    concrete implementations.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.representations as cxr

    >>> data = {"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")}
    >>> cxr.guess_rep(data)
    Representation(geom_kind=PointGeometry(), basis=NoBasis(), semantic_kind=Location())

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def guess_semantic_kind(*args: Any, **kwargs: Any) -> Any:
    """Guess the semantic kind of the given data.

    This is an abstract API definition. See the main coordinax package for
    concrete implementations.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.representations as cxr

    >>> data = {"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")}
    >>> cxr.guess_semantic_kind(data)
    Location()
    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def add(*args: Any, **kwargs: Any) -> Any:
    """Add two coordinate data objects.

    This is an abstract API definition. See the main coordinax package for
    concrete implementations.
    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def subtract(*args: Any, **kwargs: Any) -> Any:
    """Subtract two coordinate data objects.

    This is an abstract API definition. See the main coordinax package for
    concrete implementations.
    """
    raise NotImplementedError  # pragma: no cover
