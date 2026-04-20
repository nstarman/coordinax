r"""Transform operators and transformation-group markers.

The `coordinax.transforms` module (commonly imported as `cxfm`) provides
transform operators, transform composition APIs, and transformation-group marker
classes.

## Overview

`coordinax.transforms` is the canonical transform namespace. `coordinax.frames`
depends on it to build frame-transition operators.

## Quick Start

```python import coordinax.frames as cxf import coordinax.transforms as cxfm
import coordinax.main as cx import unxt as u

op = cxfm.Rotate.from_euler("z", u.Q(90, "deg")) v = cx.Point.from_([1, 0, 0],
"m")

rotated = cxfm.act(op, None, v)

# frame transitions still come from coordinax.frames frame_op =
cxf.frame_transition(cxf.alice, cxf.alex) out = cxfm.act(frame_op, None, v) ```

## Functional API

- `act(transform, tau, x)`: apply a transform to data
- `simplify(transform)`: simplify transform structure
- `compose(*transforms)`: compose transforms into `Composed`
- `materialize_transform(transform, tau)`: materialize time-dependent transform
  parameters

## Transform Types

- `AbstractTransform`: base class for transforms
- `Identity`: null transform
- `Translate`: pure displacement
- `Rotate`: pure rotation
- `Reflect`: Householder hyperplane reflection
- `Scale`: Cartesian linear scaling
- `Shear`: Cartesian linear shear
- `Composed`: ordered transform composition
- `identity`: convenience singleton instance of `Identity`

## Transformation Group Classes (Markers)

Used for classification and dispatch; not instantiated directly:

- `AbstractTransformGroup`
- `IdentityGroup`
- `DiffeomorphismGroup`
- `AffineGroup`
- `EuclideanGroup`
- `OrthogonalGroup`
- `SpecialOrthogonalGroup`
- `LorentzGroup`
- `ProperOrthochronousLorentzGroup`
- `PoincareGroup`

"""

from importlib.metadata import entry_points

from collections.abc import Mapping
from typing import Final

from ._setup_package import install_import_hook
from coordinax.internal import doc_patch_public_api

__all__: tuple[str, ...] = (
    # API
    "act",
    "simplify",
    "compose",
    "materialize_transform",
    # Groups
    "AbstractTransformGroup",
    "IdentityGroup",
    "DiffeomorphismGroup",
    "AffineGroup",
    "EuclideanGroup",
    "OrthogonalGroup",
    "SpecialOrthogonalGroup",
    "PoincareGroup",
    "LorentzGroup",
    "ProperOrthochronousLorentzGroup",
    # Transformations
    "AbstractTransform",
    "AbstractCompositeTransform",
    "Identity",
    "Composed",
    "Translate",
    "Rotate",
    "Reflect",
    "Scale",
    "Shear",
    "identity",
)

with install_import_hook("coordinax.transforms"):
    from ._src.actions import (
        AbstractCompositeTransform,
        AbstractTransform,
        Composed,
        Identity,
        Reflect,
        Rotate,
        Scale,
        Shear,
        Translate,
        identity,
        materialize_transform,
    )
    from ._src.groups import (
        AbstractTransformGroup,
        AffineGroup,
        DiffeomorphismGroup,
        EuclideanGroup,
        IdentityGroup,
        LorentzGroup,
        OrthogonalGroup,
        PoincareGroup,
        ProperOrthochronousLorentzGroup,
        SpecialOrthogonalGroup,
    )
    from coordinax.api.transforms import act, compose, simplify


_TRANSFORM_EXPORTS_ENTRYPOINT_GROUP: Final = "coordinax.transforms"
_OPTIONAL_TRANSFORM_EXPORTS_STATE: dict[str, bool] = {"loading": False}


def _load_optional_transform_exports() -> None:
    """Load optional transform symbols.

    ``coordinax.transforms`` entry-point group.
    """
    if _OPTIONAL_TRANSFORM_EXPORTS_STATE["loading"]:
        return

    _OPTIONAL_TRANSFORM_EXPORTS_STATE["loading"] = True
    exported: dict[str, object] = {}
    export_owners: dict[str, str] = {}

    try:
        eps = sorted(
            entry_points(group=_TRANSFORM_EXPORTS_ENTRYPOINT_GROUP),
            key=lambda ep: ep.name,
        )
        for ep in eps:
            provider = ep.load()
            if not callable(provider):
                msg = (
                    f"Entry point {ep.name!r} in group "
                    f"'{_TRANSFORM_EXPORTS_ENTRYPOINT_GROUP}' "
                    "is not callable."
                )
                raise TypeError(msg)
            exports = provider()
            if not isinstance(exports, Mapping):
                msg = (
                    f"Entry point {ep.name!r} in group "
                    f"'{_TRANSFORM_EXPORTS_ENTRYPOINT_GROUP}' "
                    "must return a mapping."
                )
                raise TypeError(msg)
            for name, value in exports.items():
                if not isinstance(name, str):
                    msg = (
                        f"Entry point {ep.name!r} in group "
                        f"'{_TRANSFORM_EXPORTS_ENTRYPOINT_GROUP}' produced "
                        "a non-string export name."
                    )
                    raise TypeError(msg)
                if name in exported and exported[name] is not value:
                    msg = (
                        f"Conflicting transform export {name!r} from entry points "
                        f"{export_owners[name]!r} and {ep.name!r}."
                    )
                    raise RuntimeError(msg)
                exported[name] = value
                export_owners[name] = ep.name

        globals().update(exported)
    finally:
        _OPTIONAL_TRANSFORM_EXPORTS_STATE["loading"] = False


_load_optional_transform_exports()

doc_patch_public_api(set(__all__))
del doc_patch_public_api

del (
    install_import_hook,
    Final,
)
