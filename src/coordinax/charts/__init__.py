"""`coordinax.charts` Module.

A **chart** defines how point coordinates are represented:

- Component names (for example `x, y, z` or `r, theta, phi`)
- Coordinate dimensions (for example `length`, `angle`)
- Transition and realization behavior through the functional API

Use chart instances (for example `cart3d`, `sph3d`) when transforming concrete
coordinate data.

## Quick Start

Let's start by constructing some charts and see how they work.

>>> import coordinax.charts as cxc

We can create a 3D Cartesian chart like this:

>>> cart_chart = cxc.Cart3D()
>>> cart_chart == cxc.cart3d  # predefined instance
True

Charts have components and coordinate dimensions:

>>> cart_chart.components
('x', 'y', 'z')

>>> cart_chart.coord_dimensions
('length', 'length', 'length')

There are many different charts available, such as spherical coordinates:

>>> sph_chart = cxc.Spherical3D()
>>> sph_chart == cxc.sph3d  # predefined instance
True
>>> sph_chart.components
('r', 'theta', 'phi')
>>> sph_chart.coord_dimensions
('length', 'angle', 'angle')

Another example is a 4D spacetime chart:

>>> st_chart = cxc.SpaceTimeCT()
>>> st_chart.components
('ct', 'x', 'y', 'z')
>>> st_chart.coord_dimensions
('length', 'length', 'length', 'length')
>>> st_chart.time_chart
Time1D()
>>> st_chart.spatial_chart
Cart3D()

>>> cxc.SpaceTimeCT(cxc.sph3d)
SpaceTimeCT(spatial_chart=Spherical3D())

`SpaceTimeCT` is a special case of a Cartesian product chart. It has a fixed
time factor `time1d` and a user-selectable spatial factor and flattens its chart
factors into a "single" chart.

We can also build arbitrary Cartesian products of charts (without flattening)
using `CartesianProductChart`:

>>> prod_chart = cxc.CartesianProductChart((cxc.time1d, cxc.sph3d), ("t", "q"))
>>> prod_chart
CartesianProductChart(
    factors=(Time1D(), Spherical3D()), factor_names=('t', 'q')
)
>>> prod_chart.components
('t.t', 'q.r', 'q.theta', 'q.phi')

With charts we can transform point coordinates between different coordinate
systems.

>>> import unxt as u
>>> q = {"x": u.Q(1, "km"), "y": u.Q(2, "km"), "z": u.Q(3, "km")}
>>> q_sph = cxc.pt_map(q, cxc.cart3d, cxc.sph3d)
>>> q_sph
{'r': Q(3.74165739, 'km'), 'theta': Q(0.64052231, 'rad'),
 'phi': Q(1.10714872, 'rad')}

For same-manifold chart changes, `pt_map` and
`pt_map` agree:

>>> cxc.pt_map(q, cxc.cart3d, cxc.sph3d) == q_sph
True

## Functional API

- `cartesian_chart`: return a chart's canonical Cartesian chart
- `guess_chart`: infer a chart from keys or array/quantity trailing shape
- `cdict`: normalize inputs to component dictionaries
- `pt_map`: transform points between charts on the same manifold
- `pt_map`: general point map, including realization-style maps
- `realize_cartesian`: realize point coordinates in `chart.cartesian`

`pt_map` is the same-manifold specialization. `pt_map` is the more general interface.

## Available Objects

### Chart Families

The module exports both concrete chart classes and predefined singleton-style instances.

### 0D Charts

- `Cart0D` / `cart0d`: Zero-dimensional Cartesian (scalar)

### 1D Charts

- `Cart1D` / `cart1d`: 1D Cartesian
- `Radial1D` / `radial1d`: Radial distance
- `Time1D` / `time1d`: 1D time chart

### 2D Charts

- `Cart2D` / `cart2d`: 2D Cartesian
- `Polar2D` / `polar2d`: Polar coordinates
- `SphericalTwoSphere` / `sph2`: 2-sphere (`theta`, `phi`)
- `LonLatSphericalTwoSphere` / `lonlat_sph2`: 2-sphere (`lon`, `lat`)
- `LonCosLatSphericalTwoSphere` / `loncoslat_sph2`: 2-sphere (`lon_coslat`, `lat`)
- `MathSphericalTwoSphere` / `math_sph2`: mathematical 2-sphere convention

Intrinsic two-sphere charts do not have a global Cartesian 2D chart; requesting
`cartesian_chart(...)` on this family raises `NoGlobalCartesianChartError`.

### 3D Charts

- `Cart3D` / `cart3d`: 3D Cartesian
- `Cylindrical3D` / `cyl3d`: Cylindrical coordinates
- `Spherical3D` / `sph3d`: Spherical coordinates (physics convention)
- `LonLatSpherical3D` / `lonlat_sph3d`: Longitude/latitude spherical
- `LonCosLatSpherical3D` / `loncoslat_sph3d`: Lon/cos(lat) spherical
- `MathSpherical3D` / `math_sph3d`: Mathematical spherical convention
- `ProlateSpheroidal3D`: Prolate spheroidal chart with required `Delta` parameter

`ProlateSpheroidal3D` does not export a predefined instance because chart
instances depend on the focal parameter `Delta`.

### 6D Charts

- `PoincarePolar6D` / `poincarepolar6d`: 6D Poincare polar chart family

### N-D Charts

- `CartND` / `cartnd`: N-dimensional Cartesian
- `SpaceTimeCT` / `spacetimect`: spacetime chart with `ct` plus a spatial factor

### Product Charts

- `CartesianProductChart`: namespace-prefixed product chart with dot-delimited
  component keys (for example `q.x`, `q.y`, `p.x`, ...)
- `SpaceTimeCT`: flat-key product chart `time1d x spatial_chart`

Product-chart transitions are factorwise: each factor chart transforms
independently and then components are merged.

"""

__all__ = (
    # ===========================================
    "AbstractChart",
    "AbstractFixedComponentsChart",
    "AbstractDimensionalFlag",
    "DIMENSIONAL_FLAGS",
    "NoGlobalCartesianChartError",
    # -------------------------------------------
    "cartesian_chart",
    "guess_chart",
    "cdict",
    "jacobian_pt_map",
    "pt_map",
    "realize_cartesian",
    "pt_map",
    # ===========================================
    # R^n
    # - 0D --------------------------------------
    "Abstract0D",
    "Cart0D",
    "cart0d",
    # - 1D --------------------------------------
    "Abstract1D",
    "Cart1D",
    "cart1d",
    "Radial1D",
    "radial1d",
    "Time1D",
    "time1d",
    # - 2D --------------------------------------
    "Abstract2D",
    "Cart2D",
    "cart2d",
    "Polar2D",
    "polar2d",
    # - 3D --------------------------------------
    "Abstract3D",
    "Cart3D",
    "cart3d",
    "Cylindrical3D",
    "cyl3d",
    "AbstractSpherical3D",
    "Spherical3D",
    "sph3d",
    "LonLatSpherical3D",
    "lonlat_sph3d",
    "LonCosLatSpherical3D",
    "loncoslat_sph3d",
    "MathSpherical3D",
    "math_sph3d",
    "ProlateSpheroidal3D",  # Not exported as instance
    # - 6D --------------------------------------
    "Abstract6D",
    "PoincarePolar6D",
    "poincarepolar6d",
    # - N-D -------------------------------------
    "AbstractND",
    "CartND",
    "cartnd",
    # ===========================================
    # S^n
    "AbstractSphericalHyperSphere",
    "AbstractSphericalOneSphere",
    # - 1D --------------------------------------
    "CircularOneSphere",
    "sph1",
    # - 2D --------------------------------------
    "AbstractSphericalTwoSphere",
    "SphericalTwoSphere",
    "sph2",
    "LonLatSphericalTwoSphere",
    "lonlat_sph2",
    "LonCosLatSphericalTwoSphere",
    "loncoslat_sph2",
    "MathSphericalTwoSphere",
    "math_sph2",
    # ===========================================
    "AbstractCartesianProductChart",
    "AbstractFlatCartesianProductChart",
    "CartesianProductChart",
    "SpaceTimeCT",
    "spacetimect",
)

from ._setup_package import install_import_hook
from coordinax.internal import doc_patch_public_api

with install_import_hook("coordinax.charts"):
    from ._src import (
        DIMENSIONAL_FLAGS,
        Abstract0D,
        Abstract1D,
        Abstract2D,
        Abstract3D,
        Abstract6D,
        AbstractCartesianProductChart,
        AbstractChart,
        AbstractDimensionalFlag,
        AbstractFixedComponentsChart,
        AbstractFlatCartesianProductChart,
        AbstractND,
        AbstractSpherical3D,
        AbstractSphericalHyperSphere,
        AbstractSphericalOneSphere,
        AbstractSphericalTwoSphere,
        Cart0D,
        Cart1D,
        Cart2D,
        Cart3D,
        CartesianProductChart,
        CartND,
        CircularOneSphere,
        Cylindrical3D,
        LonCosLatSpherical3D,
        LonCosLatSphericalTwoSphere,
        LonLatSpherical3D,
        LonLatSphericalTwoSphere,
        MathSpherical3D,
        MathSphericalTwoSphere,
        NoGlobalCartesianChartError,
        PoincarePolar6D,
        Polar2D,
        ProlateSpheroidal3D,
        Radial1D,
        SpaceTimeCT,
        Spherical3D,
        SphericalTwoSphere,
        Time1D,
        cart0d,
        cart1d,
        cart2d,
        cart3d,
        cartnd,
        cyl3d,
        jacobian_pt_map,
        loncoslat_sph2,
        loncoslat_sph3d,
        lonlat_sph2,
        lonlat_sph3d,
        math_sph2,
        math_sph3d,
        poincarepolar6d,
        polar2d,
        radial1d,
        spacetimect,
        sph1,
        sph2,
        sph3d,
        time1d,
    )
    from coordinax.api.charts import (
        cartesian_chart,
        cdict,
        guess_chart,
        pt_map,
        realize_cartesian,
    )


del install_import_hook

doc_patch_public_api(set(__all__))
del doc_patch_public_api
