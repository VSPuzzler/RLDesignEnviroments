"""
Project-root .env autoloader.

Importing this module loads NeuroUI Judge's ``.env`` file (sitting at the
project root, two levels above this file) into ``os.environ``. Variables
already present in the environment are *not* overridden, so a shell
``export`` always wins.

Safe to import multiple times; the underlying ``python-dotenv`` call is
idempotent and ``load_dotenv`` is fast on cached files.
"""

from __future__ import annotations

import os
from pathlib import Path


def _project_root() -> Path:
    """Resolve the project root (the folder containing the ``services`` dir)."""
    return Path(__file__).resolve().parent.parent.parent


def load_project_env() -> Path | None:
    """
    Load ``<project_root>/.env`` if it exists. Returns the path that was
    loaded, or None if no .env is present / dotenv isn't installed.
    """
    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]
    except ImportError:
        return None
    env_path = _project_root() / ".env"
    if not env_path.is_file():
        return None
    load_dotenv(env_path, override=False)
    return env_path


# Load on import so any subsequent ``os.getenv(...)`` in the package sees it.
_LOADED_FROM = load_project_env()


def project_root() -> Path:
    """Public accessor for the project root."""
    return _project_root()


def loaded_from() -> Path | None:
    """Path of the ``.env`` that was loaded, or None."""
    return _LOADED_FROM
