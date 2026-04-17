"""`coordinax.vectors` Module."""

__all__ = (
    "cconvert",
    "AbstractVector",
    "Point",
    "ToUnitsOptions",
)

from ._setup_package import install_import_hook
from coordinax.internal import doc_patch_public_api

with install_import_hook("coordinax.vectors"):
    from ._src import (
        AbstractVector,
        Point,
        ToUnitsOptions,
    )
    from coordinax.api.representations import cconvert

del install_import_hook

doc_patch_public_api(set(__all__))
del doc_patch_public_api
