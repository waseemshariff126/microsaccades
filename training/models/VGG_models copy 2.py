"""
VGG Model with Optical Flow Estimation Branch
Modified from https://github.com/pytorch/vision.git
"""

import math
import torch
import torch.nn as nn
import torch.nn.init as init
from models.layers import *  # Ensure this module provides: SpikeModule, LIFSpike, tdBatchNorm, SeqToANNContainer, add_dimention

__all__ = [
    'VGG', 'vgg11', 'vgg13', 'vgg16', 'VGGWithFlow', 'vgg16_with_flow'
]

# -----------------------------------------------------------------------------
# Configuration for different VGG variants
# -----------------------------------------------------------------------------
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 512, 512],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 512, 512],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 512, 512, 512],
}

# -----------------------------------------------------------------------------
# VGG Base Class
# -----------------------------------------------------------------------------
# class VGG(nn.Module):
#     """
#     VGG model.
#     """
#     def __init__(self, cfg, num_classes=10, batch_norm=True, in_c=3, **lif_parameters):
#         super(VGG, self).__init__()
#         self.features, out_c = make_layers(cfg, batch_norm, in_c, **lif_parameters)
#         self.out_channels = out_c  # Save output channels for later use
#         self.avgpool = SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
#         self.classifier = nn.Sequential(
#             SeqToANNContainer(nn.Linear(out_c, num_classes)),
#         )
#         # Initialize convolutional layers
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#         # Optionally add a time dimension if required by your spiking modules.
#         # This lambda assumes that self.T is defined elsewhere (or you can set it in __init__)
#         self.add_dim = lambda x: add_dimention(x, self.T) if hasattr(self, 'T') else x

#     def forward(self, x):
#         # If the input is missing a time dimension, add it.
#         x = self.add_dim(x) if len(x.shape) == 4 else x
#         x = self.features(x)
#         x = self.avgpool(x)
#         # Flatten differently depending on input dimensions.
#         x = torch.flatten(x, 1) if len(x.shape) == 4 else torch.flatten(x, 2)
#         x = self.classifier(x)
#         return x
class VGG(nn.Module):
    """
    VGG model.
    """
    def __init__(self, cfg, num_classes=10, batch_norm=True, in_c=3, **lif_parameters):
        super(VGG, self).__init__()
        self.features, out_c = make_layers(cfg, batch_norm, in_c, **lif_parameters)
        self.out_channels = out_c  # Save output channels for later use
        self.avgpool = SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        # Change classifier to a plain Linear layer:
        self.classifier = nn.Linear(out_c, num_classes)
        
        # Initialize convolutional layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
        # Add a time dimension if required by your spiking modules.
        self.add_dim = lambda x: add_dimention(x, self.T) if hasattr(self, 'T') else x

    def forward(self, x):
        # Add time dimension if missing
        x = self.add_dim(x) if len(x.shape) == 4 else x
        x = self.features(x)
        x = self.avgpool(x)
        # Assuming x now has shape [B, T, C, 1, 1], average over T:
        if x.dim() == 5:
            x = x.mean(1)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# -----------------------------------------------------------------------------
# Helper Function to Build VGG Layers
# -----------------------------------------------------------------------------
def make_layers(cfg_list, batch_norm=False, in_c=3, **lif_parameters):
    layers = []
    in_channels = in_c
    for v in cfg_list:
        if v == 'M':
            layers += [SpikeModule(nn.AvgPool2d(kernel_size=2, stride=2))]
        else:
            conv2d = SpikeModule(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))
            lif = LIFSpike(**lif_parameters)
            if batch_norm:
                bn = tdBatchNorm(v)
                layers += [conv2d, bn, lif]
            else:
                layers += [conv2d, lif]
            in_channels = v
    return nn.Sequential(*layers), in_channels

# -----------------------------------------------------------------------------
# Factory Functions for VGG Variants
# -----------------------------------------------------------------------------
def vgg11(*args, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization."""
    return VGG(cfg['A'], *args, **kwargs)

def vgg13(*args, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization."""
    return VGG(cfg['B'], *args, **kwargs)

def vgg16(*args, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization."""
    return VGG(cfg['D'], *args, **kwargs)

# -----------------------------------------------------------------------------
# VGG with Optical Flow Estimation Branch
# -----------------------------------------------------------------------------
# class VGGWithFlow(VGG):
#     def __init__(self, cfg, num_classes=10, batch_norm=True, in_c=3, flow_channels=2, **lif_parameters):
#         super(VGGWithFlow, self).__init__(cfg, num_classes, batch_norm, in_c, **lif_parameters)
#         # Build a flow head to predict a two-channel optical flow field (horizontal and vertical)
#         self.flow_head = nn.Sequential(
#             nn.Conv2d(self.out_channels, self.out_channels // 2, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(self.out_channels // 2, flow_channels, kernel_size=3, padding=1)
#         )
    
#     def forward(self, x, return_flow=False):
#         # Process input through the shared backbone
#         x = self.add_dim(x) if len(x.shape) == 4 else x
#         features = self.features(x)
        
#         # Classification branch
#         x_class = self.avgpool(features)
#         x_class = torch.flatten(x_class, 1)
#         class_out = self.classifier(x_class)
        
#         if return_flow:
#             # Optical flow branch: predict flow from the features.
#             flow_out = self.flow_head(features)
#             return class_out, flow_out
#         else:
#             return class_out

# class VGGWithFlow(VGG):
#     def __init__(self, cfg, num_classes=10, batch_norm=True, in_c=3, flow_channels=2, **lif_parameters):
#         super(VGGWithFlow, self).__init__(cfg, num_classes, batch_norm, in_c, **lif_parameters)
#         self.flow_head = nn.Sequential(
#             nn.Conv2d(self.out_channels, self.out_channels // 2, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(self.out_channels // 2, flow_channels, kernel_size=3, padding=1)
#         )
    
#     def forward(self, x, return_flow=False):
#         # x is expected to be [B, T, C, H, W]
#         if len(x.shape) == 4:
#             x = self.add_dim(x)
#         features = self.features(x)  # shape: [B, T, C, H', W']
#         B, T, C, H, W = features.shape
        
#         # Classification Branch
#         features_pooled = features.view(B * T, C, H, W)
#         features_pooled = self.avgpool(features_pooled)  # shape: [B*T, C, 1, 1]
#         features_pooled = features_pooled.view(B, T, C, 1, 1)
#         features_avg = features_pooled.mean(1)  # shape: [B, C, 1, 1]
#         x_class = torch.flatten(features_avg, 1)  # shape: [B, C]
#         class_out = self.classifier(x_class)       # shape: [B, num_classes]
        
#         if return_flow:
#             flow_features = features.view(B * T, C, H, W)
#             flow_out = self.flow_head(flow_features)  # shape: [B*T, flow_channels, H, W]
#             flow_out = flow_out.view(B, T, -1, H, W)
#             return class_out, flow_out
#         else:
#             return class_out

class VGGWithFlow(VGG):
    def __init__(self, cfg, num_classes=7, batch_norm=True, in_c=2, flow_channels=2, **lif_parameters):
        super(VGGWithFlow, self).__init__(cfg, num_classes, batch_norm, in_c, **lif_parameters)
        self.flow_head = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels // 2, flow_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x, return_flow=True):
        if len(x.shape) == 4:
            x = self.add_dim(x)
        features = self.features(x)
        B, T, C, H, W = features.shape
        
        # Classification Branch
        features_pooled = features.view(B * T, C, H, W)
        features_pooled = self.avgpool(features_pooled)
        features_pooled = features_pooled.view(B, T, C, 1, 1)
        features_avg = features_pooled.mean(1)
        x_class = torch.flatten(features_avg, 1)
        class_out = self.classifier(x_class)
        
        # Flow Branch
        flow_features = features.view(B * T, C, H, W)
        flow_out = self.flow_head(flow_features)
        flow_out = flow_out.view(B, T, -1, H, W)
        
        if return_flow:
            return class_out, flow_out
        return class_out



def vgg16_with_flow(*args, **kwargs):
    """VGG 16-layer model with an Optical Flow Estimation branch."""
    return VGGWithFlow(cfg['D'], *args, **kwargs)

# -----------------------------------------------------------------------------
# Main Block for Testing
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # Test standard VGG16
    print("Testing VGG16:")
    model_vgg16 = vgg16(num_classes=10, batch_norm=True, in_c=3)
    print(model_vgg16)
    x = torch.rand(2, 3, 32, 32)
    y = model_vgg16(x)
    print("VGG16 output shape:", y.shape)
    
    # Test VGG with Optical Flow Estimation
    print("\nTesting VGGWithFlow:")
    model_vgg_flow = vgg16_with_flow(num_classes=10, batch_norm=True, in_c=3, flow_channels=2)
    print(model_vgg_flow)
    class_out, flow_out = model_vgg_flow(x, return_flow=True)
    print("Classification output shape:", class_out.shape)
    print("Optical flow output shape:", flow_out.shape)
