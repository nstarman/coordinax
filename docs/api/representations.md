# `coordinax.representations`

The `coordinax.representations` module defines representation descriptors for geometric data and the representation-aware conversion API.

## Overview

A representation is the triple $R = (K, B, S)$:

- $K$: geometry kind (what sort of geometric object the data represents)
- $B$: basis kind (how components are interpreted as basis components)
- $S$: semantic kind (what the object means physically/mathematically)

This is separate from charts and manifolds:

- Charts define coordinate systems.
- Manifolds define the underlying space.
- Representations define geometric meaning and transformation law category.

## Quick Start

```pycon
>>> import coordinax.charts as cxc
>>> import coordinax.representations as cxr
>>> import unxt as u

>>> # Canonical point representation.
>>> rep = cxr.point

>>> # Convert point data between charts while preserving representation.
>>> q_cart = {"x": 1.0, "y": 2.0, "z": 3.0}
>>> q_sph = cxr.cconvert(q_cart, cxc.cart3d, rep, cxc.sph3d, rep)
>>> q_sph
{'r': Array(3.74165739, dtype=float64, ...),
 'theta': Array(0.64052231, dtype=float64),
 'phi': Array(1.10714872, dtype=float64, ...)}

>>> # Build a reusable conversion map.
>>> to_sph = cxr.cmap(cxc.cart3d, cxr.point, cxc.sph3d)
>>> to_sph(q_cart)
{'r': Array(3.74165739, dtype=float64, ...),
 'theta': Array(0.64052231, dtype=float64),
 'phi': Array(1.10714872, dtype=float64, ...)}
```

For tangent data attached to a base point, `change_basis` changes only the basis convention in the current chart.

```pycon
>>> v = {"r": u.Q(1.0, "km/s"), "theta": u.Q(0.0, "rad/s"), "phi": u.Q(0.0, "rad/s")}
>>> at = {"r": u.Q(2.0, "km"), "theta": u.Q(3.0, "rad"), "phi": u.Q(4.0, "rad")}
>>> v2 = cxr.change_basis(v, cxc.sph3d, cxr.coord_basis, cxr.phys_basis, at=at)
>>> v2
{'r': Q(1., 'km / s'), 'theta': Q(0., 'km / s'), 'phi': Q(0., 'km / s')}
```

Use `tangent_map` to move tangent components between charts. It applies the chart Jacobian at the source base point, supplied with `at=...`:

```pycon
>>> v_cart = cxr.tangent_map(v, cxc.sph3d, cxr.coord_disp, cxc.cart3d, at=at)
>>> v_cart
{'x': Q(-0.09224219, 'km / s'), 'y': Q(-0.10679997, 'km / s'), 'z': Q(-0.9899925, 'km / s')}
```

## Available Objects

### Conversion Functions

- `cconvert`: convert data across charts/representations
- `change_basis`: change tangent basis without changing chart
- `tangent_map`: move tangent components between charts using the Jacobian
- `cmap`: build reusable conversion callables
- `guess_basis_kind`: infer a basis kind
- `guess_geometry_kind`: infer a geometry kind
- `guess_rep`: infer a full representation
- `guess_semantic_kind`: infer a semantic kind

### Representation Descriptor

- `Representation`: immutable descriptor `(geom_kind, basis, semantic_kind)`
- `point`: canonical point representation

`point` is equivalent to `(point_geom, no_basis, loc)`.

## Geometry Kind

- `AbstractGeometry`: base class for geometric kind descriptors
- `PointGeometry` / `point_geom`: point geometric kind

## Basis Kind

- `AbstractBasis`: base class for basis descriptors
- `NoBasis` / `no_basis`: basis kind used for affine point data
- `CoordinateBasis` / `coord_basis`: coordinate-basis tangent components
- `PhysicalBasis` / `phys_basis`: physical-basis tangent components

## Semantic Kind

- `AbstractSemanticKind`: base class for semantic descriptors
- `Location` / `loc`: location semantic kind

## Notes

- Representations are orthogonal to charts: chart choice and representation choice are independent concerns.
- The current built-in flow is point-first, centered on `(point_geom, no_basis, loc)`.
- `change_basis` currently supports tangent-data conversions between `CoordinateBasis` $\rightleftarrows$ `PhysicalBasis`, including Cartesian, spherical, and metric-based conversions.

```{eval-rst}

.. currentmodule:: coordinax.representations

.. automodule:: coordinax.representations
    :exclude-members: aval, default, materialise, enable_materialise

```
