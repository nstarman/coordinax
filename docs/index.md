# 🚀 Get Started

`coordinax` enables working with coordinates and reference frames with [JAX][jax].

`coordinax` supports JAX's main features:

- JIT compilation via [`jax.jit`][jax.jit]
- vectorization via [`jax.vmap`][jax.vmap]
- auto-differentiation via [`jax.grad`][jax.grad], [`jax.jacobian`][jax.jacobian], and [`jax.hessian`][jax.hessian]
- GPU/TPU/multi-host acceleration

And best of all, `coordinax` doesn't force you to use special unit-compatible re-exports of JAX libraries. You can use `coordinax` with existing JAX code, and with one simple decorator ([`quax.quaxify`](https://docs.kidger.site/quax/)), JAX will work with `coordinax` objects.

---

## Installation

[![PyPI version][pypi-version]][pypi-link] [![PyPI platforms][pypi-platforms]][pypi-link]

=== "pip"

    ```bash
    pip install coordinax
    ```

=== "uv"

    ```bash
    uv add coordinax
    ```

=== "source, via uv"

    To install the latest development version of `coordinax` directly from the GitHub repository, use uv:

    ```bash
    uv add git+https://github.com/GalacticDynamics/coordinax.git@main
    ```

    You can customize the branch by replacing `main` with any other branch name.

=== "building from source"

    To build `coordinax` from source, clone the repository and install it with uv:

    ```bash
    cd /path/to/parent
    git clone https://github.com/GalacticDynamics/coordinax.git
    cd coordinax
    uv pip install -e .
    ```

## Quickstart

The `coordinax` package has powerful tools for representing, using, and transforming coordinate objects, such as:

- specific quantity subclasses like [Angle][coordinax.angles.Angle] and [Distance][coordinax.distances.Distance]
- and more!

This functionality is organized into submodules available under the top-level `coordinax` namespace. You can import them directly, or for many objects use the `coordinax.main` namespace to access them.

<!-- invisible-code-block: python
import importlib.util
import coordinax.angles
import coordinax.api
import coordinax.curveframes
import coordinax.charts
import coordinax.distances
import coordinax.frames
import coordinax.hypothesis
import coordinax.main
import coordinax.manifolds
import coordinax.representations
import coordinax.vectors
-->

<!-- skip: start if(importlib.util.find_spec('coordinax.astro') is None, reason="coordinax.astro not installed") -->

```pycon
>>> import coordinax
>>> import sys

>>> sorted(
...     name.removeprefix("coordinax.")
...     for name in sys.modules
...     if name.startswith("coordinax.") and name.count(".") == 1
... )
['angles', 'api', 'astro', 'charts', 'curveframes', 'distances', 'frames', 'hypothesis', 'internal', 'interop', 'main', 'manifolds', 'representations', 'transforms', 'vectors']
```

<!-- skip: end -->

We recommend importing as needed:

- `coordinax.main` as `cx` : probably everything you need!
- `coordinax.angles` as `cxa` : further angle-specific functionality.
- `coordinax.distances` as `cxd` : further distance-specific functionality.
- `coordinax.charts` as `cxc` : chart-specific functionality.
- `coordinax.frames` as `cxf` : frame-specific functionality.
- `coordinax.manifolds` as `cxm` : manifold-specific functionality.
- `coordinax.representations` as `cxr` : representation-specific functionality.
- `coordinax.transforms` as `cxt` : transform-specific functionality.
- `coordinax.vectors` as `cxv` : vector-specific functionality.

- `coordinax.astro` as `cxastro` : astronomy-specific functionality. Note that this package is an optional extra, so you may need to install it separately.
- `coordinax.hypothesis` as `cxst` : property-based testing strategies for `coordinax`. Note that this package is an optional extra, so you may need to install it separately.
- `coordinax.interop.astropy` as `cxapy` : interoperability with `astropy`. Note that this package is an optional extra, so you may need to install it separately.

### Angles and Distances

`coordinax` is built on top of [`unxt`](http://unxt.readthedocs.io), which provides quantity objects that pair array values with physical units. These quantity objects can be used throughout `coordinax`, and the library also provides specialized types with additional coordinate-aware behavior.

Let's start with angles, which are represented by [Angle][coordinax.angles.Angle]. This class enforces angular dimensionality and provides convenient utilities for working with branch cuts and wrapped ranges such as $[0, 2\pi)$ or $(-180^\circ, 180^\circ]$.

```pycon
>>> import coordinax.main as cx
>>> import unxt as u

>>> a = cx.Angle(370, "deg")
>>> a
Angle(370, 'deg')

>>> a.wrap_to(u.Q(0, "deg"), u.Q(360, "deg"))
Angle(10, 'deg')
```

Similarly, [Distance][coordinax.distances.Distance] represents distances in `coordinax`:

```pycon
>>> d = cx.Distance(10, "kpc")
>>> d
Distance(10, 'kpc')
```

Other distance-like objects can be represented with [`Parallax`][coordinax.astro.Parallax] and [`DistanceModulus`][coordinax.astro.DistanceModulus] from [`coordinax.astro`][coordinax.astro]. These classes validate their physical units and provide convenient conversions between distance representations.

<!-- skip: start if(importlib.util.find_spec('coordinax.astro') is None, reason="coordinax.astro not installed") -->

```pycon
>>> import coordinax.astro as cxastro
>>> import plum

>>> plum.convert(d, cxastro.Parallax)
Parallax(4.84813681e-10, 'rad')

>>> plum.convert(d, cxastro.DistanceModulus)
DistanceModulus(15., 'mag')
```

<!-- skip: end -->

## Ecosystem

### `coordinax`'s Dependencies

- [unxt][unxt]: Quantities in JAX.
- [Equinox][equinox]: one-stop JAX library, for everything that isn't already in core JAX.
- [Quax][quax]: JAX + multiple dispatch + custom array-ish objects.
- [Quaxed][quaxed]: pre-`quaxify`ed Jax.
- [plum][plum]: multiple dispatch in python

### `coordinax`'s Dependents

- [galax][galax]: Galactic dynamics in JAX.

<!-- LINKS -->

[unxt]: https://github.com/GalacticDynamics/unxt
[equinox]: https://docs.kidger.site/equinox/
[galax]: https://github.com/GalacticDynamics/galax
[jax]: https://jax.readthedocs.io/en/latest/
[plum]: https://pypi.org/project/plum-dispatch/
[quax]: https://github.com/patrick-kidger/quax
[quaxed]: https://quaxed.readthedocs.io/en/latest/
[pypi-link]: https://pypi.org/project/coordinax/
[pypi-platforms]: https://img.shields.io/pypi/pyversions/coordinax
[pypi-version]: https://img.shields.io/pypi/v/coordinax
