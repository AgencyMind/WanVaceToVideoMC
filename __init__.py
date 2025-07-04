"""
WanVaceToVideoMC - Multi-Control Enhancement for WanVaceToVideo
A memory-efficient implementation that supports multiple control inputs in a single node
"""

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError as e:
    print(f"[WanVaceToVideoMC] Failed to import nodes: {e}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__version__ = "1.0.0"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Debug print
print(f"[WanVaceToVideoMC] Loaded with {len(NODE_CLASS_MAPPINGS)} nodes")