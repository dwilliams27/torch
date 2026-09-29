"""Diffusion engines.

Each engine is a module ``server.engines.<name>`` exposing ``create_engine(**cfg)``
that returns an object with ``name, model, width, height, device, warmup(),
process(image, prompt, strength, seed, negative=None)`` (see docs/DESIGN.md).
"""
from __future__ import annotations

import importlib
import pkgutil

# Tried in this order by ``--engine auto`` (override with --prefer or
# $HYPNAGOGIA_ENGINES="a,b,c"). Missing modules are skipped silently.
DEFAULT_PREFERENCE = ["coreml_turbo", "torch_turbo", "mlx_turbo"]


def available() -> list[str]:
    """Names of engine modules present in this package (not necessarily loadable)."""
    return sorted(m.name for m in pkgutil.iter_modules(__path__) if not m.name.startswith("_"))


def load(name: str, **cfg):
    """Import server.engines.<name> and call its create_engine(**cfg)."""
    mod = importlib.import_module(f"{__name__}.{name}")
    if not hasattr(mod, "create_engine"):
        raise AttributeError(f"server.engines.{name} has no create_engine()")
    return mod.create_engine(**{k: v for k, v in cfg.items() if v is not None})
