# WanVaceToVideoMC - Multi-Control Enhancement for ComfyUI

A memory-efficient enhancement of the WanVaceToVideo node that supports multiple control inputs (pose, depth, edge) in a single node, solving the VAE triple-loading memory crisis.

## Features

- **Single VAE Instance**: Uses one VAE for all control processing, saving ~20-30GB VRAM
- **Multi-Control Support**: Process pose, depth, and edge controls simultaneously
- **Granular Control**: Independent strength controls for video and mask per control type
- **Full Backward Compatibility**: Works seamlessly with existing WanVaceToVideo workflows
- **Multiple Combination Modes**: Choose how controls are combined (multiply, add, average, max)

## Installation

1. Navigate to your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes/
   ```

2. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/WanVaceToVideoMC.git
   ```

3. Restart ComfyUI

## Usage

### Multi-Control Mode

The node appears as "WAN VACE to Video (Multi-Control)" in the node menu under `conditioning/video_models`.

New inputs for multi-control:
- `control_video_pose` / `control_masks_pose` - Pose control inputs
- `control_video_depth` / `control_masks_depth` - Depth control inputs  
- `control_video_edge` / `control_masks_edge` - Edge control inputs
- `strength_video_*` - Control strength for video influence (0.0-10.0)
- `strength_mask_*` - Control strength for mask influence (0.0-10.0)
- `multi_control_mode` - How to combine controls: multiply, add, average, max

### Legacy Mode

For backward compatibility, you can still use the original inputs:
- `control_video` - Single control video input
- `control_masks` - Single control mask
- `reference_image` - Reference image for style

**Note**: You cannot use both legacy and multi-control inputs simultaneously.

## Memory Savings

Traditional approach (3 separate WanVaceToVideo nodes):
- VAE memory usage: ~30-45GB (3 × 10-15GB)

WanVaceToVideoMultiControl:
- VAE memory usage: ~10-15GB (single instance)
- **Savings: ~20-30GB VRAM**

## Control Combination Modes

- **multiply**: Most restrictive - all controls must agree
- **add**: Most permissive - any control can influence  
- **average**: Balanced combination (default)
- **max**: Strongest signal wins

## Requirements

- ComfyUI (latest version)
- PyTorch >= 2.0.0
- CUDA-capable GPU (tested on 3x A6000 setup)

## Credits

Developed at Zerospace for production workflows with WAN VACE 14B video generation.

## License

AGPL-3.0 License - See LICENSE file for details.