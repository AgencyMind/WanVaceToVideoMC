"""
WanVaceToVideoMC - Multi-Control Enhancement for WanVaceToVideo
A memory-efficient implementation that supports multiple control inputs in a single node
"""

# Only import if running as a custom node
try:
    import comfy.utils
except ImportError:
    pass
else:
    from .nodes import NODE_CLASS_MAPPINGS
    NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}
    print(f"[WanVaceToVideoMC] Successfully loaded with {len(NODE_CLASS_MAPPINGS)} nodes")
    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]