try:
    from .nodes import comfy_entrypoint
except ImportError:
    if __package__:
        raise
    comfy_entrypoint = None

__all__ = ["comfy_entrypoint"]

