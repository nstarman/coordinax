"""Representation of coordinates in different systems."""

__all__: list[str] = []

from functools import partial
from typing import Any

import jax
from plum import dispatch

import quaxed.numpy as jnp

from .base import AbstractAcc2D, AbstractPos2D, AbstractVel2D
from .cartesian import CartesianAcc2D, CartesianPos2D, CartesianVel2D
from .polar import PolarAcc, PolarPos, PolarVel
from .spherical import TwoSphereAcc, TwoSpherePos, TwoSphereVel
from coordinax._src.vectors.api import AuxDict, OptAuxDict, OptUSys, ParamsDict
from coordinax._src.vectors.base import AbstractVector
from coordinax._src.vectors.private_api import (
    combine_aux,
    vconvert_parse_input,
    vconvert_parse_output,
)

###############################################################################
# Vector Transformation

# =============================================================================
# `vconvert_impl`


@dispatch
@partial(jax.jit, static_argnums=(0, 1), static_argnames=("units",))
def vconvert_impl(
    to_vector: type[AbstractPos2D],
    from_vector: type[AbstractPos2D],
    params: ParamsDict,
    /,
    *,
    in_aux: OptAuxDict = None,
    out_aux: OptAuxDict = None,
    units: OptUSys = None,
) -> tuple[ParamsDict, AuxDict]:
    """AbstractPos -> CartesianPos1D -> AbstractPos."""
    params, aux = vconvert_impl(
        CartesianPos2D, from_vector, params, in_aux=in_aux, out_aux=None, units=units
    )
    params, aux = vconvert_impl(
        to_vector, CartesianPos2D, params, in_aux=aux, out_aux=out_aux, units=units
    )
    return params, aux


@dispatch.multi(
    # Positions
    (type[CartesianPos2D], type[CartesianPos2D], ParamsDict),
    (type[PolarPos], type[PolarPos], ParamsDict),
    (type[TwoSpherePos], type[TwoSpherePos], ParamsDict),
    # Velocities
    (type[CartesianVel2D], type[CartesianVel2D], ParamsDict),
    (type[PolarVel], type[PolarVel], ParamsDict),
    (type[TwoSphereVel], type[TwoSphereVel], ParamsDict),
    # Accelerations
    (type[CartesianAcc2D], type[CartesianAcc2D], ParamsDict),
    (type[PolarAcc], type[PolarAcc], ParamsDict),
    (type[TwoSphereAcc], type[TwoSphereAcc], ParamsDict),
)
@partial(jax.jit, static_argnums=(0, 1), static_argnames=("units",), inline=True)
def vconvert_impl(
    to_vector: type[AbstractVector],
    from_vector: type[AbstractVector],
    params: ParamsDict,
    /,
    *,
    in_aux: OptAuxDict = None,
    out_aux: OptAuxDict = None,
    units: OptUSys = None,
) -> tuple[ParamsDict, AuxDict]:
    """Self transform."""
    return params, combine_aux(in_aux, out_aux)


@dispatch
@partial(jax.jit, static_argnums=(0, 1), static_argnames=("units",))
def vconvert_impl(
    to_vector: type[PolarPos],
    from_vector: type[CartesianPos2D],
    p: ParamsDict,
    /,
    *,
    in_aux: OptAuxDict = None,
    out_aux: OptAuxDict = None,
    units: OptUSys = None,
) -> tuple[ParamsDict, OptAuxDict]:
    """CartesianPos2D -> PolarPos.

    The `x` and `y` coordinates are converted to the radial coordinate `r` and
    the angular coordinate `phi`.

    """
    p = vconvert_parse_input(p, from_vector.dimensions, units)
    r = jnp.hypot(p["x"], p["y"])
    phi = jnp.atan2(p["y"], p["x"])
    outp = {"r": r, "phi": phi}
    outp = vconvert_parse_output(outp, to_vector.dimensions, units)
    return outp, combine_aux(in_aux, out_aux)


@dispatch
@partial(jax.jit, static_argnums=(0, 1), static_argnames=("units",))
def vconvert_impl(
    to_vector: type[CartesianPos2D],
    from_vector: type[PolarPos],
    p: ParamsDict,
    /,
    *,
    in_aux: OptAuxDict = None,
    out_aux: OptAuxDict = None,
    units: OptUSys = None,
) -> tuple[ParamsDict, OptAuxDict]:
    """PolarPos -> CartesianPos2D.

    The `r` and `phi` coordinates are converted to the `x` and `y` coordinates.

    """
    p = vconvert_parse_input(p, from_vector.dimensions, units)
    x = p["r"] * jnp.cos(p["phi"])
    y = p["r"] * jnp.sin(p["phi"])
    outp = {"x": x, "y": y}
    outp = vconvert_parse_output(outp, to_vector.dimensions, units)
    return outp, combine_aux(in_aux, out_aux)


# =============================================================================
# `vconvert`


@dispatch.multi(
    # Positions
    (type[CartesianPos2D], CartesianPos2D),
    (type[PolarPos], PolarPos),
    (type[TwoSpherePos], TwoSpherePos),
    # Velocities
    (type[CartesianVel2D], CartesianVel2D, AbstractPos2D),
    (type[CartesianVel2D], CartesianVel2D),  # q not needed
    (type[PolarVel], PolarVel, AbstractPos2D),
    # Accelerations
    (type[CartesianAcc2D], CartesianAcc2D, AbstractVel2D, AbstractPos2D),
    (type[CartesianAcc2D], CartesianAcc2D),  # q,p not needed
)
def vconvert(
    target: type[AbstractVector], current: AbstractVector, /, *args: Any, **kwargs: Any
) -> AbstractVector:
    """Self transform of 2D vectors."""
    return current


###############################################################################
# Corresponding Cartesian classes


@dispatch
def cartesian_vector_type(
    obj: type[AbstractPos2D] | AbstractPos2D, /
) -> type[CartesianPos2D]:
    """AbstractPos2D -> CartesianPos2D."""
    return CartesianPos2D


@dispatch
def cartesian_vector_type(
    obj: type[AbstractVel2D] | AbstractVel2D, /
) -> type[CartesianVel2D]:
    """AbstractVel2D -> CartesianVel2D."""
    return CartesianVel2D


@dispatch
def cartesian_vector_type(
    obj: type[AbstractAcc2D] | AbstractAcc2D, /
) -> type[CartesianAcc2D]:
    """AbstractPos -> CartesianAcc2D."""
    return CartesianAcc2D


###############################################################################
# Corresponding time derivative classes

# -----------------------------------------------
# Position -> Velocity


@dispatch
def time_derivative_vector_type(
    obj: type[CartesianPos2D] | CartesianPos2D, /
) -> type[CartesianVel2D]:
    """Return the corresponding time derivative class."""
    return CartesianVel2D


@dispatch
def time_derivative_vector_type(obj: type[PolarPos] | PolarPos, /) -> type[PolarVel]:
    """Return the corresponding time derivative class."""
    return PolarVel


@dispatch
def time_derivative_vector_type(
    obj: type[TwoSpherePos] | TwoSpherePos, /
) -> type[TwoSphereVel]:
    """Return the corresponding time derivative class."""
    return TwoSphereVel


# -----------------------------------------------
# Velocity -> Position


@dispatch
def time_antiderivative_vector_type(
    obj: type[CartesianVel2D] | CartesianVel2D, /
) -> type[CartesianPos2D]:
    """Return the corresponding time antiderivative class."""
    return CartesianPos2D


@dispatch
def time_antiderivative_vector_type(
    obj: type[PolarVel] | PolarVel, /
) -> type[PolarPos]:
    """Return the corresponding time antiderivative class."""
    return PolarPos


@dispatch
def time_antiderivative_vector_type(
    obj: type[TwoSphereVel] | TwoSphereVel, /
) -> type[TwoSpherePos]:
    """Return the corresponding time antiderivative class."""
    return TwoSpherePos


# -----------------------------------------------
# Velocity -> Acceleration


@dispatch
def time_derivative_vector_type(
    obj: type[CartesianVel2D] | CartesianVel2D, /
) -> type[CartesianAcc2D]:
    """Return the corresponding time derivative class."""
    return CartesianAcc2D


@dispatch
def time_derivative_vector_type(obj: type[PolarVel] | PolarVel, /) -> type[PolarAcc]:
    """Return the corresponding time derivative class."""
    return PolarAcc


@dispatch
def time_derivative_vector_type(
    obj: type[TwoSphereVel] | TwoSphereVel, /
) -> type[TwoSphereAcc]:
    """Return the corresponding time derivative class."""
    return TwoSphereAcc


# -----------------------------------------------
# Acceleration -> Velocity


@dispatch
def time_antiderivative_vector_type(
    obj: type[CartesianAcc2D] | CartesianAcc2D, /
) -> type[CartesianVel2D]:
    """Return the corresponding time antiderivative class."""
    return CartesianVel2D


@dispatch
def time_antiderivative_vector_type(
    obj: type[PolarAcc] | PolarAcc, /
) -> type[PolarVel]:
    """Return the corresponding time antiderivative class."""
    return PolarVel


@dispatch
def time_antiderivative_vector_type(
    obj: type[TwoSphereAcc] | TwoSphereAcc, /
) -> type[TwoSphereVel]:
    """Return the corresponding time antiderivative class."""
    return TwoSphereVel
