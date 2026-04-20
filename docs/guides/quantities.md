# Specialized Quantities

<!-- invisible-code-block: python

import unxt as u
import jax.numpy as jnp

-->

## Working with `Angle` Objects

The `Angle` class in `coordinax.angles` is a specialized quantity for representing angular measurements, similar to `unxt.quantity.Quantity` but with additional features and constraints tailored for angles.

!!! note

    `Angle` is a re-export of `unxt.Angle` with additional coordinax-specific functionality.

### Creating Angles

You can create an `Angle` just like a `unxt.quantity.Quantity`, by specifying a value and a unit with angular dimensions:

```pycon
>>> import coordinax.main as cx
>>> a = cx.Angle(45, "deg")
>>> a
Angle(45, 'deg')
```

Just like `unxt.quantity.Quantity`, you can flexibly create `Angle` objects using the `from_` constructor:

```pycon
>>> cx.Angle.from_(45, "deg")
Angle(45, 'deg')

>>> cx.Angle.from_([45, 90], "deg")
Angle([45, 90], 'deg')

>>> cx.Angle.from_(jnp.array([10, 15, 20]), "deg")
Angle([10, 15, 20], 'deg')

```

### Mathematical Operations

`Angle` objects support arithmetic operations, broadcasting, and most mathematical functions, just like `unxt.quantity.Quantity`:

```pycon
>>> b = cx.Angle(30, "deg")
>>> a + b
Angle(75, 'deg')
>>> 2 * a
Angle(90, 'deg')
>>> a.to("rad")
Angle(0.78539816, 'rad')
```

For more information on mathematical operations, see the unxt documentation.

### Enforced Dimensionality

Unlike a generic `unxt.quantity.Quantity`, the `Angle` class enforces that the unit must be angular (e.g., degrees, radians). Attempting to use a non-angular unit will raise an error:

```pycon
>>> try:
...     cx.Angle(1, "m")
... except ValueError as e:
...     print(e)
...
Angle must have units with angular dimensions.
```

### Wrapping Angles

A key feature of `Angle` is the ability to wrap values to a specified range, which is useful for keeping angles within a branch cut:

```pycon
>>> import unxt as u
>>> a = cx.Angle(370, "deg")
>>> a.wrap_to(u.Q(0, "deg"), u.Q(360, "deg"))
Angle(10, 'deg')
```

The `Angle.wrap_to` method has a function counterpart

```pycon
>>> import coordinax.angles as cxa
>>> cxa.wrap_to(a, u.Q(0, "deg"), u.Q(360, "deg"))
Angle(10, 'deg')
```

---

## Working with `Distance` Objects

The `Distance` class in `coordinax.distances` is a specialized quantity for representing physical distances, with enforced dimensionality and convenient conversions to and from other distance-like representations. Related classes, `coordinax.astro.Parallax` and `coordinax.astro.DistanceModulus`, are also provided for common astronomical use cases.

### Creating Distance Objects

You can create a `Distance` just like a `unxt.quantity.Quantity`, by specifying a value and a unit with length dimensions:

```pycon
>>> d = cx.Distance(10, "kpc")
>>> d
Distance(10, 'kpc')
```

### Creating Parallax and DistanceModulus Objects

`coordinax.astro.Parallax` and `coordinax.astro.DistanceModulus` are alternative representations of distance:

```pycon
>>> import coordinax.astro as cxastro
>>> p = cxastro.Parallax(0.1, "mas")
>>> p
Parallax(0.1, 'mas')

>>> dm = cxastro.DistanceModulus(15, "mag")
>>> dm
DistanceModulus(15, 'mag')
```

### Properties and Conversions

Each of these classes has a property to convert to `Distance`:

```pycon
>>> p.distance.uconvert("kpc")
Distance(10., 'kpc')

>>> dm.distance.uconvert("kpc")
Distance(10., 'kpc')
```

All these classes enforce that their units are appropriate for their physical meaning (e.g., `Distance` must have length units, `Parallax` must have angular units, and `DistanceModulus` must have magnitude units).

---

!!! info "See also"

    - [API Documentation for Angles](../api/angles.md)
    - [API Documentation for Distances](../api/distances.md)
