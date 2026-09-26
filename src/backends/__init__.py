"""Backend auto-discovery.

Importing this package imports every ``*_backend.py`` module inside it, which
in turn populates ``RESTORE_BACKENDS`` and ``SR_BACKENDS`` via their
``register_*`` decorators. To add a new backend, drop a module here — no
edits to the pipeline or imports are required.
"""
import importlib
import pkgutil

import src.backends as _pkg

# Import every submodule ending in _backend (plus base) so their decorators run.
for _mod in pkgutil.iter_modules(_pkg.__path__):
    if _mod.name == "base":
        continue
    if _mod.name.endswith("_backend"):
        importlib.import_module(f"{_pkg.__name__}.{_mod.name}")


def available_backends():
    """Return {"restore": {...}, "sr": {...}} of registered backends.

    Each entry maps the backend name to its class (with name/default_weights/
    default_params metadata) — used by --list-backends and for validation.
    """
    from src.backends.base import RESTORE_BACKENDS, SR_BACKENDS
    return {"restore": RESTORE_BACKENDS, "sr": SR_BACKENDS}
