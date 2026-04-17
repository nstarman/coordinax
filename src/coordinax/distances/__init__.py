"""`coordinax.distances` module."""

__all__ = ("AbstractDistance", "Distance")

from ._setup_package import install_import_hook

with install_import_hook("coordinax.distances"):
    from ._src import AbstractDistance, Distance


del install_import_hook

from coordinax.internal import doc_patch_public_api  # noqa: E402

doc_patch_public_api(set(__all__))
del doc_patch_public_api
