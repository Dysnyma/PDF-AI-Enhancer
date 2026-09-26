"""Backend base classes: shared contract + factory helpers.

A backend is a named, replaceable model that implements a single stage of the
pipeline. Two stage families exist today:

    restore  (document restoration)  -> RestoreBackend
    sr       (super resolution)      -> SRBackend

Every backend declares its own defaults (weights file, extra params) so the
pipeline does NOT need to know any backend-specific constructor signature.
The pipeline only passes the config sub-dict for that stage plus a weights
resolution callback.
"""
import logging

import numpy as np

log = logging.getLogger("pdfenhance")

# Shared registries. Each backend module registers into these via
# register_restore / register_sr; ``available_backends`` reads them back.
RESTORE_BACKENDS: dict = {}
SR_BACKENDS: dict = {}


def register_restore(name):
    def deco(cls):
        cls.name = name
        RESTORE_BACKENDS[name] = cls
        return cls
    return deco


def register_sr(name):
    def deco(cls):
        cls.name = name
        SR_BACKENDS[name] = cls
        return cls
    return deco


class BackendBase:
    """Common interface for all stage backends."""

    #: registry name (set automatically on registration)
    name = ""

    #: default weights file name relative to <project>/models/
    default_weights = None

    #: extra parameters and their defaults, e.g. {"scale": 2, "tile": 384}
    #: The pipeline merges the config sub-dict over these defaults.
    default_params: dict = {}

    #: whether this backend needs a weights file at all (e.g. "none" does not)
    needs_weights = True

    def process(self, img_bgr: np.ndarray) -> np.ndarray:
        """Run the stage on a BGR uint8 image, return a BGR uint8 image."""
        raise NotImplementedError

    @classmethod
    def create(cls, device, weights_path: str, params: dict, **extra):
        """Instantiate the backend.

        device        : torch.device
        weights_path  : resolved absolute path to weights (may be None)
        params        : merged param dict (default_params overridden by config)
        extra         : shared context (e.g. fp16) forwarded by the pipeline

        Subclasses override this to do the real construction; the default
        implementation forwards everything to __init__ for backends that keep
        a simple signature.
        """
        kwargs = dict(params)
        kwargs.update(extra)
        kwargs["device"] = device
        if cls.needs_weights:
            kwargs["weights_path"] = weights_path
        return cls(**kwargs)


class RestoreBackend(BackendBase):
    """Document restoration stage (e.g. DocRes)."""
    stage = "restore"


class SRBackend(BackendBase):
    """Super-resolution stage (e.g. Real-ESRGAN, SwinIR)."""
    stage = "sr"


def resolve_weights(project_root: str, filename: str) -> str | None:
    """Resolve a weights file relative to <project>/models/ (or absolute)."""
    if not filename:
        return None
    import os
    if os.path.isabs(filename):
        return filename
    return os.path.join(project_root, "models", filename)
