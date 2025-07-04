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
    try:
        from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        print(f"[WanVaceToVideoMC] Successfully loaded with {len(NODE_CLASS_MAPPINGS)} nodes")
    except ImportError as e:
        print(f"[WanVaceToVideoMC] Failed to import nodes: {e}")
        NODE_CLASS_MAPPINGS = {}
        NODE_DISPLAY_NAME_MAPPINGS = {}

__version__ = "1.0.0"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]