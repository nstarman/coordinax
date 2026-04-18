"""Tests for documentation compatibility helpers."""

import importlib

from coordinax.internal import _api_compat


class _UnpatchableExport:
    @property
    def __module__(self) -> str:
        return "coordinax._src.fake"

    @__module__.setter
    def __module__(self, value: str) -> None:
        raise TypeError("cannot reassign __module__")


class _PatchableExport:
    pass


def test_doc_patch_public_api_patches_export_module(monkeypatch) -> None:
    """Public exports are rewritten to the caller module during docs builds."""
    monkeypatch.setenv("COORDINAX_BUILDING_DOCS", "1")
    api_compat = importlib.reload(_api_compat)

    globals()["PATCHABLE_EXPORT"] = _PatchableExport
    try:
        _PatchableExport.__module__ = "coordinax._src.fake"
        api_compat.doc_patch_public_api({"PATCHABLE_EXPORT"})
        assert _PatchableExport.__module__ == __name__
    finally:
        globals().pop("PATCHABLE_EXPORT", None)
        monkeypatch.setenv("COORDINAX_BUILDING_DOCS", "0")
        importlib.reload(api_compat)


def test_doc_patch_public_api_ignores_unpatchable_export(monkeypatch) -> None:
    """Exports that reject module reassignment are skipped without error."""
    monkeypatch.setenv("COORDINAX_BUILDING_DOCS", "1")
    api_compat = importlib.reload(_api_compat)

    globals()["UNPATCHABLE_EXPORT"] = _UnpatchableExport()
    try:
        api_compat.doc_patch_public_api({"UNPATCHABLE_EXPORT"})
    finally:
        globals().pop("UNPATCHABLE_EXPORT", None)
        monkeypatch.setenv("COORDINAX_BUILDING_DOCS", "0")
        importlib.reload(api_compat)
