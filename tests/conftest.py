"""Shared pytest helpers.

The tests run from a plain checkout: the repository root is put on ``sys.path`` so that
``starVLA`` can be imported without installing the package. Dependency-light modules
(pure helpers, scripts under ``tools/``) are loaded by file path so the tests do not
need torch.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def load_by_path():
    """Import a module by its repository-relative path, bypassing package ``__init__`` files."""
    cache = {}

    def _load(relative_path: str):
        if relative_path not in cache:
            path = REPO_ROOT / relative_path
            name = "bypath_" + "_".join(Path(relative_path).with_suffix("").parts)
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module  # dataclasses resolve their module through sys.modules
            spec.loader.exec_module(module)
            cache[relative_path] = module
        return cache[relative_path]

    return _load
