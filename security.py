"""
Security validation utilities for WanVaceToVideoMultiControl
"""

import torch
from typing import Optional, Union, Tuple


class SecurityValidator:
    """Validates inputs to prevent security vulnerabilities"""
    
    @staticmethod
    def validate_tensor_shape(tensor: torch.Tensor, expected_dims: int, name: str) -> None:
        """Validate tensor has expected number of dimensions"""
        if tensor.ndim != expected_dims:
            raise ValueError(f"Security: {name} tensor has {tensor.ndim} dimensions, expected {expected_dims}")
    
    @staticmethod
    def validate_tensor_bounds(tensor: torch.Tensor, min_val: float, max_val: float, name: str) -> None:
        """Validate tensor values are within expected bounds"""
        if tensor.min() < min_val or tensor.max() > max_val:
            raise ValueError(f"Security: {name} tensor values out of bounds [{min_val}, {max_val}]")
    
    @staticmethod
    def validate_dimensions(width: int, height: int, length: int) -> None:
        """Validate video dimensions are reasonable"""
        MAX_DIM = 8192  # Reasonable maximum
        MIN_DIM = 16
        
        if not (MIN_DIM <= width <= MAX_DIM):
            raise ValueError(f"Security: Width {width} out of range [{MIN_DIM}, {MAX_DIM}]")
        if not (MIN_DIM <= height <= MAX_DIM):
            raise ValueError(f"Security: Height {height} out of range [{MIN_DIM}, {MAX_DIM}]")
        if not (1 <= length <= MAX_DIM):
            raise ValueError(f"Security: Length {length} out of range [1, {MAX_DIM}]")
        
        # Check total memory requirement
        total_pixels = width * height * length
        MAX_TOTAL_PIXELS = 4096 * 4096 * 256  # ~4GB at float32
        if total_pixels > MAX_TOTAL_PIXELS:
            raise ValueError(f"Security: Total pixel count {total_pixels} exceeds maximum {MAX_TOTAL_PIXELS}")
    
    @staticmethod
    def validate_batch_size(batch_size: int) -> None:
        """Validate batch size is reasonable"""
        if not (1 <= batch_size <= 64):
            raise ValueError(f"Security: Batch size {batch_size} out of range [1, 64]")
    
    @staticmethod
    def validate_strength(strength: float, name: str = "strength") -> None:
        """Validate strength parameter"""
        if not isinstance(strength, (int, float)):
            raise TypeError(f"Security: {name} must be numeric, got {type(strength)}")
        if not (0.0 <= strength <= 1000.0):
            raise ValueError(f"Security: {name} {strength} out of range [0.0, 1000.0]")
    
    @staticmethod
    def validate_control_mode(mode: str) -> None:
        """Validate control combination mode"""
        valid_modes = ["multiply", "add", "average", "max"]
        if mode not in valid_modes:
            raise ValueError(f"Security: Invalid control mode '{mode}', must be one of {valid_modes}")
    
    @staticmethod
    def validate_image_tensor(tensor: Optional[torch.Tensor], name: str) -> None:
        """Validate image tensor if provided"""
        if tensor is None:
            return
        
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Security: {name} must be a torch.Tensor")
        
        # Check dimensions (B, H, W, C) or (H, W, C)
        if tensor.ndim not in [3, 4]:
            raise ValueError(f"Security: {name} must have 3 or 4 dimensions, got {tensor.ndim}")
        
        # Check channels
        channels = tensor.shape[-1]
        if channels not in [1, 3, 4]:
            raise ValueError(f"Security: {name} must have 1, 3, or 4 channels, got {channels}")
        
        # Check value range (assuming normalized 0-1)
        if tensor.min() < -10.0 or tensor.max() > 10.0:
            raise ValueError(f"Security: {name} values seem out of normal range")
    
    @staticmethod
    def validate_mask_tensor(tensor: Optional[torch.Tensor], name: str) -> None:
        """Validate mask tensor if provided"""
        if tensor is None:
            return
        
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Security: {name} must be a torch.Tensor")
        
        # Check dimensions (B, H, W, 1) or (B, H, W) or (H, W)
        if tensor.ndim not in [2, 3, 4]:
            raise ValueError(f"Security: {name} must have 2, 3, or 4 dimensions, got {tensor.ndim}")
    
    @staticmethod
    def sanitize_conditioning(conditioning) -> None:
        """Validate conditioning input"""
        if not isinstance(conditioning, (list, tuple)):
            raise TypeError("Security: Conditioning must be a list or tuple")
        
        # Basic structure validation
        if len(conditioning) == 0:
            raise ValueError("Security: Empty conditioning provided")


class MemoryGuard:
    """Guards against excessive memory usage"""
    
    def __init__(self, max_gb: float = 48.0):
        self.max_bytes = int(max_gb * 1024 * 1024 * 1024)
    
    def check_allocation(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> None:
        """Check if tensor allocation would exceed memory limit"""
        element_size = torch.tensor([], dtype=dtype).element_size()
        total_elements = 1
        for dim in shape:
            total_elements *= dim
        total_bytes = total_elements * element_size
        
        if total_bytes > self.max_bytes:
            raise MemoryError(f"Security: Requested allocation {total_bytes / 1e9:.1f}GB exceeds limit {self.max_bytes / 1e9:.1f}GB")