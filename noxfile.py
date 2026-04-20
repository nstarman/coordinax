#!/usr/bin/env -S uv run --script
# /// script
#    dependencies = ["nox", "nox_uv"]
# ///
"""Nox setup."""

import argparse
import shutil
from enum import StrEnum, auto
from pathlib import Path

from typing import assert_never, final

import nox
from nox_uv import session

nox.needs_version = ">=2024.3.2"
nox.options.default_venv_backend = "uv"

DIR = Path(__file__).parent.resolve()


def _stage_docs_source() -> Path:
    """Materialize a concrete docs tree for Zensical.

    The authoritative package docs live inside each workspace package under
    ``packages/*/docs``. The repository-level ``docs/packages`` entries are
    routing bridges, but Zensical only indexes real files inside ``docs_dir``.
    Copy the docs tree into ``_build`` and follow those links so the staged tree
    contains the actual package pages without duplicating the source of truth in
    the repository.
    """
    staged_docs = DIR / "scratch" / "docs_src"
    staged_docs.parent.mkdir(parents=True, exist_ok=True)
    if staged_docs.exists():
        shutil.rmtree(staged_docs)
    shutil.copytree(DIR / "docs", staged_docs, symlinks=False)
    return staged_docs


def _stage_mkdocs_config(docs_dir: Path, /) -> Path:
    """Write a temporary MkDocs config that points at the staged docs tree."""
    staged_config = DIR / ".mkdocs.staged.yml"
    rel_docs_dir = docs_dir.relative_to(DIR).as_posix()
    config_text = (DIR / "mkdocs.yml").read_text()
    config_text = config_text.replace("docs_dir: docs", f"docs_dir: {rel_docs_dir}", 1)
    staged_config.write_text(config_text)
    return staged_config


@final
class PackageEnum(StrEnum):
    """Enum for package names."""

    @staticmethod
    def _generate_next_value_(name: str, *_: object, **__: object) -> str:
        return name

    def __repr__(self) -> str:
        return f"{self.value!r}"

    coordinax = auto()
    api = auto()
    astro = auto()
    curveframes = auto()
    hypothesis = auto()


# =============================================================================
# Comprehensive sessions


@session(
    uv_groups=["lint", "test", "docs"],
    uv_extras=["workspace"],
    reuse_venv=True,
    default=True,
)
def all(s: nox.Session, /) -> None:  # noqa: A001
    """Run all default sessions."""
    s.notify("lint")
    s.notify("test")
    s.notify("docs")


# =============================================================================
# Linting


@session(uv_groups=["lint"], reuse_venv=True)
def lint(s: nox.Session, /) -> None:
    """Run the linter."""
    s.notify("precommit")
    # s.notify("pylint") # TODO: re-enable after fixing lint errors
    s.notify("ty")


@session(uv_groups=["lint"], reuse_venv=True)
def precommit(s: nox.Session, /) -> None:
    """Run the linter."""
    s.run("pre-commit", "run", "--all-files", *s.posargs)


@session(uv_groups=["lint"], reuse_venv=True)
@nox.parametrize("package", list(PackageEnum))
def pylint(s: nox.Session, /, package: PackageEnum) -> None:
    """Run PyLint."""
    match package:
        case PackageEnum.coordinax:
            package_path = "src/coordinax"
        case PackageEnum.api:
            package_path = "packages/coordinax.api/"
        case PackageEnum.astro:
            package_path = "packages/coordinax.astro/"
        case PackageEnum.curveframes:
            package_path = "packages/coordinax.curveframes/"
        case PackageEnum.hypothesis:
            package_path = "packages/coordinax.hypothesis/"
        case _:
            assert_never(package)
    s.run("pylint", package_path, *s.posargs)


@session(uv_groups=["lint"], reuse_venv=True)
@nox.parametrize("package", list(PackageEnum))
def ty(s: nox.Session, /, package: PackageEnum) -> None:
    """Run ty."""
    package_paths: tuple[str, ...]
    match package:
        case PackageEnum.coordinax:
            package_paths = ("src/coordinax", "packages/coordinax.api/")
        case PackageEnum.api:
            package_paths = ("packages/coordinax.api/",)
        case PackageEnum.astro:
            package_paths = ("packages/coordinax.astro/",)
        case PackageEnum.curveframes:
            package_paths = ("packages/coordinax.curveframes/",)
        case PackageEnum.hypothesis:
            package_paths = ("packages/coordinax.hypothesis/",)
        case _:
            assert_never(package)
    s.run("ty", "check", *package_paths, *s.posargs)


# =============================================================================
# Testing


@session(uv_groups=["test"], reuse_venv=True, default=True)
def test(s: nox.Session, /) -> None:
    """Run the unit and regular tests."""
    s.notify("pytest", posargs=s.posargs)
    # s.notify("pytest_benchmark", posargs=s.posargs)


def _parse_pytest_paths(package: PackageEnum, /) -> list[str]:
    match package:
        case PackageEnum.coordinax:
            package_paths = ["README.md", "docs", "src/", "tests/"]
        case PackageEnum.api:
            package_paths = ["packages/coordinax.api/"]
        case PackageEnum.astro:
            package_paths = ["packages/coordinax.astro/"]
        case PackageEnum.curveframes:
            package_paths = ["packages/coordinax.curveframes/"]
        case PackageEnum.hypothesis:
            package_paths = ["packages/coordinax.hypothesis/"]
        case _:
            assert_never(package)

    return package_paths


@session(uv_groups=["test"], uv_extras=["workspace"], reuse_venv=True)
@nox.parametrize("package", list(PackageEnum))
def pytest(s: nox.Session, /, package: PackageEnum) -> None:
    """Run the unit and regular tests."""
    package_paths = _parse_pytest_paths(package)
    s.run("pytest", *package_paths, *s.posargs)


# =============================================================================
# Documentation


@session(uv_groups=["docs"], uv_extras=["workspace"], reuse_venv=True)
def docs(s: nox.Session, /) -> None:
    """Build the docs with Zensical. Pass "--serve" to preview locally."""
    # This is the canonical documentation entry point for the repository.
    # The old Sphinx pipeline has been fully replaced by MkDocs + Material.
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="Serve after building")
    parser.add_argument("--clean", action="store_true", help="Clean the Zensical cache")
    parser.add_argument(
        "-W",
        "--strict",
        action="store_true",
        help="Fail the session if the docs build emits warnings.",
    )
    args, posargs = parser.parse_known_args(s.posargs)

    # Build from a staged docs tree so the authoritative package docs remain in
    # their workspace packages while Zensical still sees a concrete docs_dir.
    staged_docs = _stage_docs_source()
    staged_config = _stage_mkdocs_config(staged_docs)
    command = [
        "zensical",
        "serve" if args.serve else "build",
        "-f",
        str(staged_config),
    ]
    if args.clean and not args.serve:
        # Only the build command supports a clean rebuild of generated output.
        command.append("--clean")
    command.extend(posargs)

    if not args.strict:
        s.run(*command)
        return

    # Zensical advertises a native strict flag, but it currently reports that
    # the mode is unsupported. Enforce warning-free builds here instead.
    output = s.run(*command, silent=True, success_codes=[0]) or ""
    print(output, end="")

    lowered = output.lower()
    warning_signatures = ("warning:", "warning ", "pluginerror:")
    if any(sig in lowered for sig in warning_signatures):
        s.error("Docs build emitted warnings in strict mode.")


# =============================================================================
# Packaging


@session(uv_groups=["build"])
def build(s: nox.Session, /) -> None:
    """Build an SDist and wheel."""
    build_path = DIR.joinpath("build")
    if build_path.exists():
        shutil.rmtree(build_path)
    s.run("python", "-m", "build")


################################################################################

if __name__ == "__main__":
    nox.main()
