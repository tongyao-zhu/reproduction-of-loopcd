"""Independent reproduction of arXiv:2610.02185 inference-time guidance."""

from importlib import import_module as _import_module

__all__ = ["GuidanceConfig", "OuroGuidance", "adaptive_strength", "apply_guidance"]

__version__ = "0.1.0"

_EXPORT_MODULES = {
    "GuidanceConfig": ".guidance",
    "adaptive_strength": ".guidance",
    "apply_guidance": ".guidance",
    "OuroGuidance": ".ouro",
}


def __getattr__(name):
    """Load model dependencies only when an existing model API is requested."""
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_module(_EXPORT_MODULES[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
