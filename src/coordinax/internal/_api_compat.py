"""Documentation-compatibility helpers for ``coordinax``.

Provides utilities for patching ``__module__`` attributes so that Sphinx
autodoc reports public paths rather than private ``._src.`` paths when
``COORDINAX_BUILDING_DOCS=1``.

"""

__all__ = ["doc_public_api", "doc_patch_public_api"]

import contextlib
import inspect
import os

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

_BUILDING_DOCS: bool = os.environ.get("COORDINAX_BUILDING_DOCS", "0") == "1"


def _patch_orig_bases(cls: type, old_mod: str, new_mod: str, /) -> None:
    """Patch ``__module__`` on generic aliases in ``__orig_bases__`` of subclasses."""
    for subclass in cls.__subclasses__():
        for base in getattr(subclass, "__orig_bases__", ()):
            if getattr(base, "__module__", None) == old_mod:
                with contextlib.suppress(AttributeError, TypeError):
                    base.__module__ = new_mod
        _patch_orig_bases(subclass, old_mod, new_mod)


def _identity(obj: T, /) -> T:
    return obj


def doc_public_api(path: str, /) -> Callable[[T], T]:
    """Outermost decorator: rewrite ``__module__`` to *path* when building docs.

    No-op unless ``COORDINAX_BUILDING_DOCS=1``. Apply to public re-exports
    whose ``__module__`` points at a private ``._src.`` path so Sphinx shows
    the friendly public module path instead.

    Parameters
    ----------
    path
        The fully-qualified public module path to set (for example
        ``"coordinax.vectors"``).

    Examples
    --------
    ::

        from coordinax.internal import doc_public_api

        @doc_public_api("coordinax.vectors")
        class MyVector: ...

    """
    if not _BUILDING_DOCS:
        return _identity

    def decorator(obj: T, /) -> T:
        with contextlib.suppress(AttributeError, TypeError):
            obj.__module__ = path
        return obj

    return decorator


def doc_patch_public_api(names: set[str] | frozenset[str], /) -> None:
    """Patch ``__module__`` on named exports to the calling module's public path.

    No-op unless ``COORDINAX_BUILDING_DOCS=1``. Uses frame inspection to
    resolve *names* against the caller's globals and infer the public module
    path from the caller's ``__name__``.  Only objects whose ``__module__``
    contains ``._src.`` are touched, so re-exported helpers from other public
    packages are left unchanged.

    Also patches ``__module__`` on generic aliases stored in ``__orig_bases__``
    of every subclass of each patched class.

    Parameters
    ----------
    names
        Names to patch. Typically ``set(__all__)`` or
        ``set(__all__) - {"name_to_exclude"}``.

    Examples
    --------
    In a module's ``__init__.py``::

        from coordinax.internal import doc_patch_public_api

        doc_patch_public_api(set(__all__))
        del doc_patch_public_api

    """
    if not _BUILDING_DOCS:
        return

    # Inspect the caller so patched exports resolve to the public module that
    # re-exported them, rather than to their private implementation module.
    frame = inspect.currentframe()
    if frame is None or frame.f_back is None:
        return
    caller_globals = frame.f_back.f_globals
    module: str = caller_globals.get("__name__", "")
    for name in names:
        obj = caller_globals.get(name)
        if obj is None:
            continue

        # Only rewrite objects that still point at a private ``._src.`` path.
        mod = getattr(obj, "__module__", None)
        if mod is None or "._src." not in mod:
            continue

        # Some exported objects reject ``__module__`` reassignment; skip those
        # quietly so docs builds remain robust.
        with contextlib.suppress(AttributeError, TypeError):
            obj.__module__ = module

        # Keep generic aliases on subclasses aligned with the same public path.
        if isinstance(obj, type):
            _patch_orig_bases(obj, mod, module)
