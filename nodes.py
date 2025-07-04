"""
WanVaceToVideoMultiControl Node Implementation
Memory-efficient multi-control video generation node
"""

import torch
import comfy.utils
import comfy.latent_formats
import comfy.model_management
from nodes import MAX_RESOLUTION
from node_helpers import conditioning_set_values
from typing import Optional, Tuple, Dict, Any
from .security import SecurityValidator, MemoryGuard


class WanVaceToVideoMultiControl:
    """
    Enhanced WanVaceToVideo node with multi-control support and single VAE instance.
    Maintains full backward compatibility with the original node.
    """
    
    TITLE = "WAN VACE to Video (Multi-Control)"
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "vae": ("VAE", ),
                "width": ("INT", {"default": 832, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": MAX_RESOLUTION, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
            },
            "optional": {
                # Original single control (backward compatibility)
                "control_video": ("IMAGE", ),
                "control_masks": ("MASK", ),
                "reference_image": ("IMAGE", ),
                
                # Multi-control: Pose
                "control_video_pose": ("IMAGE", ),
                "control_masks_pose": ("MASK", ),
                "strength_video_pose": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "strength_mask_pose": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                
                # Multi-control: Depth
                "control_video_depth": ("IMAGE", ),
                "control_masks_depth": ("MASK", ),
                "strength_video_depth": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "strength_mask_depth": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                
                # Multi-control: Edge
                "control_video_edge": ("IMAGE", ),
                "control_masks_edge": ("MASK", ),
                "strength_video_edge": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "strength_mask_edge": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                
                # Control combination mode
                "multi_control_mode": (["multiply", "add", "average", "max"], {"default": "average"}),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "INT")
    RETURN_NAMES = ("positive", "negative", "latent", "trim_latent")
    FUNCTION = "encode"
    CATEGORY = "conditioning/video_models"

    def encode(self, positive, negative, vae, width, height, length, batch_size, strength,
               control_video=None, control_masks=None, reference_image=None,
               control_video_pose=None, control_masks_pose=None, 
               strength_video_pose=1.0, strength_mask_pose=1.0,
               control_video_depth=None, control_masks_depth=None,
               strength_video_depth=1.0, strength_mask_depth=1.0,
               control_video_edge=None, control_masks_edge=None,
               strength_video_edge=1.0, strength_mask_edge=1.0,
               multi_control_mode="average"):
        
        # Security validations
        try:
            SecurityValidator.validate_dimensions(width, height, length)
            SecurityValidator.validate_batch_size(batch_size)
            SecurityValidator.validate_strength(strength, "strength")
            SecurityValidator.validate_control_mode(multi_control_mode)
            SecurityValidator.sanitize_conditioning(positive)
            SecurityValidator.sanitize_conditioning(negative)
            
            # Validate all image inputs
            SecurityValidator.validate_image_tensor(control_video, "control_video")
            SecurityValidator.validate_image_tensor(reference_image, "reference_image")
            SecurityValidator.validate_image_tensor(control_video_pose, "control_video_pose")
            SecurityValidator.validate_image_tensor(control_video_depth, "control_video_depth")
            SecurityValidator.validate_image_tensor(control_video_edge, "control_video_edge")
            
            # Validate all mask inputs
            SecurityValidator.validate_mask_tensor(control_masks, "control_masks")
            SecurityValidator.validate_mask_tensor(control_masks_pose, "control_masks_pose")
            SecurityValidator.validate_mask_tensor(control_masks_depth, "control_masks_depth")
            SecurityValidator.validate_mask_tensor(control_masks_edge, "control_masks_edge")
            
            # Validate strength parameters
            SecurityValidator.validate_strength(strength_video_pose, "strength_video_pose")
            SecurityValidator.validate_strength(strength_mask_pose, "strength_mask_pose")
            SecurityValidator.validate_strength(strength_video_depth, "strength_video_depth")
            SecurityValidator.validate_strength(strength_mask_depth, "strength_mask_depth")
            SecurityValidator.validate_strength(strength_video_edge, "strength_video_edge")
            SecurityValidator.validate_strength(strength_mask_edge, "strength_mask_edge")
            
            # Memory guard check
            memory_guard = MemoryGuard(max_gb=48.0)  # Adjust based on available VRAM
            estimated_shape = (batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8)
            memory_guard.check_allocation(estimated_shape)
            
        except (ValueError, TypeError, MemoryError) as e:
            # Re-raise with cleaner error message
            raise RuntimeError(f"WanVaceToVideoMultiControl validation failed: {str(e)}")
        
        # Validate mutual exclusivity
        legacy_mode = control_video is not None
        multi_control_mode_active = any([
            control_video_pose is not None,
            control_video_depth is not None,
            control_video_edge is not None
        ])
        
        if legacy_mode and multi_control_mode_active:
            raise ValueError("Cannot use both legacy control_video and multi-control inputs simultaneously")
        
        # Legacy mode - use original implementation
        if legacy_mode:
            return self._encode_legacy(
                positive, negative, vae, width, height, length, batch_size, strength,
                control_video, control_masks, reference_image
            )
        
        # Multi-control mode
        return self._encode_multi_control(
            positive, negative, vae, width, height, length, batch_size, strength,
            reference_image,
            control_video_pose, control_masks_pose, strength_video_pose, strength_mask_pose,
            control_video_depth, control_masks_depth, strength_video_depth, strength_mask_depth,
            control_video_edge, control_masks_edge, strength_video_edge, strength_mask_edge,
            multi_control_mode
        )
    
    def _encode_legacy(self, positive, negative, vae, width, height, length, batch_size, 
                      strength, control_video, control_masks, reference_image):
        """Original WanVaceToVideo implementation for backward compatibility"""
        latent_length = ((length - 1) // 4) + 1
        
        if control_video is not None:
            control_video = comfy.utils.common_upscale(
                control_video[:length].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            if control_video.shape[0] < length:
                control_video = torch.nn.functional.pad(
                    control_video, (0, 0, 0, 0, 0, 0, 0, length - control_video.shape[0]), value=0.5
                )
        else:
            control_video = torch.ones((length, height, width, 3)) * 0.5

        if reference_image is not None:
            reference_image = comfy.utils.common_upscale(
                reference_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            reference_image = vae.encode(reference_image[:, :, :, :3])
            reference_image = torch.cat([
                reference_image, 
                comfy.latent_formats.Wan21().process_out(torch.zeros_like(reference_image))
            ], dim=1)

        if control_masks is None:
            mask = torch.ones((length, height, width, 1))
        else:
            mask = control_masks
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            mask = comfy.utils.common_upscale(mask[:length], width, height, "bilinear", "center").movedim(1, -1)
            if mask.shape[0] < length:
                mask = torch.nn.functional.pad(mask, (0, 0, 0, 0, 0, 0, 0, length - mask.shape[0]), value=1.0)

        control_video = control_video - 0.5
        inactive = (control_video * (1 - mask)) + 0.5
        reactive = (control_video * mask) + 0.5

        # Encode with VAE - the output should be latent representations
        inactive_encoded = vae.encode(inactive[:, :, :, :3])
        reactive_encoded = vae.encode(reactive[:, :, :, :3])
        
        # Based on the error, the model processes in chunks of 16 channels
        # We need to stack inactive/reactive differently, not concatenate channels
        # Create shape (2, frames, channels, height, width) where 2 is for inactive/reactive
        control_video_latent = torch.stack((inactive_encoded, reactive_encoded), dim=0)
        
        if reference_image is not None:
            control_video_latent = torch.cat((reference_image, control_video_latent), dim=2)

        vae_stride = 8
        height_mask = height // vae_stride
        width_mask = width // vae_stride
        mask = mask.view(length, height_mask, vae_stride, width_mask, vae_stride)
        mask = mask.permute(2, 4, 0, 1, 3)
        mask = mask.reshape(vae_stride * vae_stride, length, height_mask, width_mask)
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0), size=(latent_length, height_mask, width_mask), mode='nearest-exact'
        ).squeeze(0)

        trim_latent = 0
        if reference_image is not None:
            mask_pad = torch.zeros_like(mask[:, :reference_image.shape[2], :, :])
            mask = torch.cat((mask_pad, mask), dim=1)
            latent_length += reference_image.shape[2]
            trim_latent = reference_image.shape[2]

        mask = mask.unsqueeze(0)
        
        # Update conditioning with VACE parameters
        positive_out = conditioning_set_values(positive, {
            "vace_frames": control_video_latent,
            "vace_mask": mask,
            "vace_strength": strength
        })
        
        negative_out = conditioning_set_values(negative, {
            "vace_frames": control_video_latent,
            "vace_mask": mask,
            "vace_strength": strength
        })

        # WAN21_Vace uses 16 latent channels
        latent = torch.zeros([batch_size, 16, latent_length, height // 8, width // 8], 
                           device=comfy.model_management.intermediate_device())
        out_latent = {"samples": latent}
        
        return (positive_out, negative_out, out_latent, trim_latent)

    def _encode_multi_control(self, positive, negative, vae, width, height, length, batch_size, 
                            strength, reference_image,
                            control_video_pose, control_masks_pose, strength_video_pose, strength_mask_pose,
                            control_video_depth, control_masks_depth, strength_video_depth, strength_mask_depth,
                            control_video_edge, control_masks_edge, strength_video_edge, strength_mask_edge,
                            multi_control_mode):
        """Multi-control implementation with single VAE instance"""
        
        latent_length = ((length - 1) // 4) + 1
        
        # Process reference image once
        reference_latent = None
        if reference_image is not None:
            reference_image = comfy.utils.common_upscale(
                reference_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            reference_latent = vae.encode(reference_image[:, :, :, :3])
            reference_latent = torch.cat([
                reference_latent,
                comfy.latent_formats.Wan21().process_out(torch.zeros_like(reference_latent))
            ], dim=1)
        
        # Process each control type
        control_latents = []
        control_masks = []
        control_strengths = []
        
        for control_video, control_mask, video_strength, mask_strength in [
            (control_video_pose, control_masks_pose, strength_video_pose, strength_mask_pose),
            (control_video_depth, control_masks_depth, strength_video_depth, strength_mask_depth),
            (control_video_edge, control_masks_edge, strength_video_edge, strength_mask_edge)
        ]:
            if control_video is not None:
                # Process control video
                control_video = comfy.utils.common_upscale(
                    control_video[:length].movedim(-1, 1), width, height, "bilinear", "center"
                ).movedim(1, -1)
                
                if control_video.shape[0] < length:
                    control_video = torch.nn.functional.pad(
                        control_video, (0, 0, 0, 0, 0, 0, 0, length - control_video.shape[0]), value=0.5
                    )
                
                # Process mask
                if control_mask is None:
                    mask = torch.ones((length, height, width, 1))
                else:
                    mask = control_mask
                    if mask.ndim == 3:
                        mask = mask.unsqueeze(1)
                    mask = comfy.utils.common_upscale(mask[:length], width, height, "bilinear", "center").movedim(1, -1)
                    if mask.shape[0] < length:
                        mask = torch.nn.functional.pad(mask, (0, 0, 0, 0, 0, 0, 0, length - mask.shape[0]), value=1.0)
                
                # Apply mask strength
                mask = mask * mask_strength
                
                # Encode with VAE
                control_video = control_video - 0.5
                inactive = (control_video * (1 - mask)) + 0.5
                reactive = (control_video * mask) + 0.5
                
                try:
                    inactive_latent = vae.encode(inactive[:, :, :, :3])
                    reactive_latent = vae.encode(reactive[:, :, :, :3])
                except Exception as e:
                    raise RuntimeError(f"VAE encoding failed: {str(e)}")
                
                # Apply video strength
                inactive_latent = inactive_latent * video_strength
                reactive_latent = reactive_latent * video_strength
                
                # Stack inactive/reactive, not concatenate channels
                control_latent = torch.stack((inactive_latent, reactive_latent), dim=0)
                
                control_latents.append(control_latent)
                control_masks.append(mask)
                control_strengths.append(video_strength)
        
        # Combine multiple controls
        if not control_latents:
            # No controls provided, create default
            # Shape should be (2, frames, channels, height, width) where 2 is for inactive/reactive
            control_video_latent = torch.zeros((2, latent_length, 16, height // 8, width // 8), 
                                             device=comfy.model_management.intermediate_device())
            combined_mask = torch.ones((length, height, width, 1))
            combined_strength = strength
        else:
            # Combine control latents based on mode
            if multi_control_mode == "multiply":
                control_video_latent = control_latents[0]
                for latent in control_latents[1:]:
                    control_video_latent = control_video_latent * latent
            elif multi_control_mode == "add":
                control_video_latent = sum(control_latents)
            elif multi_control_mode == "average":
                control_video_latent = sum(control_latents) / len(control_latents)
            elif multi_control_mode == "max":
                control_video_latent = torch.stack(control_latents).max(dim=0)[0]
            
            # Combine masks
            combined_mask = sum(control_masks) / len(control_masks)
            combined_strength = sum(control_strengths) / len(control_strengths) * strength
        
        # Add reference image if provided
        if reference_latent is not None:
            control_video_latent = torch.cat((reference_latent, control_video_latent), dim=2)
        
        # Process mask for latent space
        vae_stride = 8
        height_mask = height // vae_stride
        width_mask = width // vae_stride
        combined_mask = combined_mask.view(length, height_mask, vae_stride, width_mask, vae_stride)
        combined_mask = combined_mask.permute(2, 4, 0, 1, 3)
        combined_mask = combined_mask.reshape(vae_stride * vae_stride, length, height_mask, width_mask)
        combined_mask = torch.nn.functional.interpolate(
            combined_mask.unsqueeze(0), size=(latent_length, height_mask, width_mask), mode='nearest-exact'
        ).squeeze(0)
        
        trim_latent = 0
        if reference_latent is not None:
            mask_pad = torch.zeros_like(combined_mask[:, :reference_latent.shape[2], :, :])
            combined_mask = torch.cat((mask_pad, combined_mask), dim=1)
            latent_length += reference_latent.shape[2]
            trim_latent = reference_latent.shape[2]
        
        combined_mask = combined_mask.unsqueeze(0)
        
        # Update conditioning with VACE parameters
        positive_out = conditioning_set_values(positive, {
            "vace_frames": control_video_latent,
            "vace_mask": combined_mask,
            "vace_strength": combined_strength
        })
        
        negative_out = conditioning_set_values(negative, {
            "vace_frames": control_video_latent,
            "vace_mask": combined_mask,
            "vace_strength": combined_strength
        })
        
        # Create output latent - use 16 channels to match model expectation
        latent = torch.zeros([batch_size, 16, latent_length, height // 8, width // 8],
                           device=comfy.model_management.intermediate_device())
        out_latent = {"samples": latent}
        
        return (positive_out, negative_out, out_latent, trim_latent)


# Node registration
NODE_CLASS_MAPPINGS = {
    "WanVaceToVideoMultiControl": WanVaceToVideoMultiControl
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVaceToVideoMultiControl": "WAN VACE to Video (Multi-Control)"
}