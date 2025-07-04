"""
Basic tests for WanVaceToVideoMultiControl node
"""

import unittest
from unittest.mock import Mock, MagicMock
import torch
import sys

# Mock ComfyUI modules for testing
sys.modules['comfy'] = MagicMock()
sys.modules['comfy.utils'] = MagicMock()
sys.modules['comfy.latent_formats'] = MagicMock()
sys.modules['comfy.model_management'] = MagicMock()
sys.modules['nodes'] = MagicMock()
sys.modules['comfy.node_helpers'] = MagicMock()

# Set MAX_RESOLUTION
sys.modules['nodes'].MAX_RESOLUTION = 8192

# Mock functions
sys.modules['comfy.model_management'].intermediate_device = Mock(return_value='cpu')
sys.modules['comfy.utils'].common_upscale = Mock(side_effect=lambda x, w, h, *args: x)
sys.modules['comfy.node_helpers'].conditioning_set_values = Mock(side_effect=lambda x, _: x)

# Import after mocking
from WanVaceToVideoMC.nodes import WanVaceToVideoMultiControl
from WanVaceToVideoMC.security import SecurityValidator, MemoryGuard


class TestWanVaceToVideoMultiControl(unittest.TestCase):
    
    def setUp(self):
        self.node = WanVaceToVideoMultiControl()
        # Mock VAE
        self.mock_vae = Mock()
        self.mock_vae.encode = Mock(side_effect=lambda x: torch.zeros(x.shape[0], 16, x.shape[1], x.shape[2]))
        
    def test_input_types(self):
        """Test that INPUT_TYPES returns correct structure"""
        input_types = WanVaceToVideoMultiControl.INPUT_TYPES()
        
        self.assertIn('required', input_types)
        self.assertIn('optional', input_types)
        
        # Check required inputs
        required = input_types['required']
        self.assertIn('positive', required)
        self.assertIn('negative', required)
        self.assertIn('vae', required)
        self.assertIn('width', required)
        self.assertIn('height', required)
        
        # Check optional multi-control inputs
        optional = input_types['optional']
        self.assertIn('control_video_pose', optional)
        self.assertIn('control_video_depth', optional)
        self.assertIn('control_video_edge', optional)
        self.assertIn('multi_control_mode', optional)
    
    def test_security_validation(self):
        """Test security validations"""
        # Test dimension validation
        with self.assertRaises(ValueError):
            SecurityValidator.validate_dimensions(10, 100, 100)  # Width too small
        
        with self.assertRaises(ValueError):
            SecurityValidator.validate_dimensions(10000, 100, 100)  # Width too large
        
        # Test batch size validation  
        with self.assertRaises(ValueError):
            SecurityValidator.validate_batch_size(0)
        
        with self.assertRaises(ValueError):
            SecurityValidator.validate_batch_size(100)
        
        # Test strength validation
        with self.assertRaises(ValueError):
            SecurityValidator.validate_strength(-1.0)
        
        with self.assertRaises(ValueError):
            SecurityValidator.validate_strength(1001.0)
        
        # Test control mode validation
        with self.assertRaises(ValueError):
            SecurityValidator.validate_control_mode("invalid_mode")
    
    def test_memory_guard(self):
        """Test memory guard functionality"""
        guard = MemoryGuard(max_gb=1.0)  # 1GB limit for test
        
        # Should pass
        guard.check_allocation((1, 16, 100, 100, 100))
        
        # Should fail
        with self.assertRaises(MemoryError):
            guard.check_allocation((100, 16, 1000, 1000, 1000))
    
    def test_mutual_exclusivity(self):
        """Test that legacy and multi-control modes are mutually exclusive"""
        positive = [({}, [])]
        negative = [({}, [])]
        
        # Create a control video tensor
        control_video = torch.rand(10, 100, 100, 3)
        
        # This should raise an error
        with self.assertRaises(ValueError) as context:
            self.node.encode(
                positive=positive,
                negative=negative,
                vae=self.mock_vae,
                width=128,
                height=128,
                length=10,
                batch_size=1,
                strength=1.0,
                control_video=control_video,  # Legacy input
                control_video_pose=control_video  # Multi-control input
            )
        
        self.assertIn("Cannot use both legacy", str(context.exception))
    
    def test_legacy_mode(self):
        """Test backward compatibility with legacy mode"""
        positive = [({}, [])]
        negative = [({}, [])]
        
        # Mock Wan21 format
        mock_wan21 = Mock()
        mock_wan21.process_out = Mock(side_effect=lambda x: x)
        sys.modules['comfy.latent_formats'].Wan21 = Mock(return_value=mock_wan21)
        
        result = self.node.encode(
            positive=positive,
            negative=negative,
            vae=self.mock_vae,
            width=128,
            height=128, 
            length=16,
            batch_size=1,
            strength=1.0,
            control_video=torch.rand(16, 128, 128, 3)
        )
        
        self.assertEqual(len(result), 4)
        self.assertIsNotNone(result[2]['samples'])  # Latent output
    
    def test_multi_control_mode(self):
        """Test multi-control mode with different combination modes"""
        positive = [({}, [])]
        negative = [({}, [])]
        
        # Create control tensors
        control_pose = torch.rand(16, 128, 128, 3)
        control_depth = torch.rand(16, 128, 128, 3)
        
        for mode in ['multiply', 'add', 'average', 'max']:
            result = self.node.encode(
                positive=positive,
                negative=negative,
                vae=self.mock_vae,
                width=128,
                height=128,
                length=16,
                batch_size=1,
                strength=1.0,
                control_video_pose=control_pose,
                control_video_depth=control_depth,
                multi_control_mode=mode
            )
            
            self.assertEqual(len(result), 4)
            self.assertIsNotNone(result[2]['samples'])


if __name__ == '__main__':
    unittest.main()