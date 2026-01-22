import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class AttentionGate(nn.Module):
    """
    Attention gate for skip connections.
    Critical: Helps the network focus on relevant encoder features that SAM missed.
    Without this, skip connections might propagate irrelevant information.
    """

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: Optional[int] = None):
        super().__init__()
        if inter_channels is None:
            inter_channels = skip_channels // 2

        self.W_gate = nn.Conv2d(gate_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.W_skip = nn.Conv2d(skip_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gate: Gating signal from decoder (lower resolution)
            skip: Skip connection from encoder (higher resolution)
        """
        # Upsample gate to match skip resolution if needed
        if gate.shape[2:] != skip.shape[2:]:
            gate = F.interpolate(gate, size=skip.shape[2:], mode='bilinear', align_corners=False)

        g = self.W_gate(gate)
        s = self.W_skip(skip)
        attention = self.sigmoid(self.psi(self.relu(g + s)))

        return skip * attention


class ConvBlock(nn.Module):
    """
    Double convolution block with normalization and residual connection.

    Critical choices:
    - GroupNorm over BatchNorm: More stable with small batches (common in segmentation)
    - Residual connections: Help gradient flow
    - Leaky ReLU: Prevents dead neurons
    """

    def __init__(self, in_channels: int, out_channels: int, use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual and (in_channels == out_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        self.activation = nn.LeakyReLU(0.1, inplace=True)

        if self.use_residual and in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.residual_conv = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.use_residual:
            if self.residual_conv is not None:
                identity = self.residual_conv(identity)
            out = out + identity

        out = self.activation(out)
        return out


class MaskRefinerUNet(nn.Module):
    """
    U-Net for refining SAM predictions.

    CRITICAL DESIGN PHILOSOPHY:
    - This network learns SMALL CORRECTIONS, not mask regeneration
    - Correction magnitude is constrained to prevent instability
    - Multi-scale features are essential (SAM encoder has them, decoder might miss them)
    - Boundary awareness is built-in through architecture

    Args:
        sam_feature_channels: Number of channels in SAM encoder feature
        base_channels: Base number of channels (will be scaled up in deeper layers)
        correction_scale: Maximum magnitude of correction (default: 0.3)
        use_attention: Whether to use attention gates in skip connections
        use_multi_scale: Whether to accept multi-scale SAM features
    """

    def __init__(
            self,
            sam_feature_channels: int = 256,  # SAM ViT-B uses 768, ViT-L uses 1024
            base_channels: int = 64,
            correction_scale: float = 0.3,
            use_attention: bool = True,
            use_multi_scale: bool = False,
            dropout_rate: float = 0.1
    ):
        super().__init__()

        self.correction_scale = correction_scale
        self.use_attention = use_attention
        self.use_multi_scale = use_multi_scale

        # Input: concatenated [SAM feature, SAM mask]
        input_channels = sam_feature_channels + 1  # +1 for mask channel

        # Encoder (downsampling path)
        self.enc1 = ConvBlock(input_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)

        # Bottleneck
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)

        # Decoder (upsampling path)
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(base_channels * 16, base_channels * 8)  # *16 because of skip connection

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        # Attention gates (optional but recommended)
        if use_attention:
            self.att4 = AttentionGate(base_channels * 8, base_channels * 8)
            self.att3 = AttentionGate(base_channels * 4, base_channels * 4)
            self.att2 = AttentionGate(base_channels * 2, base_channels * 2)
            self.att1 = AttentionGate(base_channels, base_channels)

        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout for regularization
        self.dropout = nn.Dropout2d(p=dropout_rate)

        # Final correction head
        # CRITICAL: We predict correction, not final mask
        self.correction_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(16, base_channels // 2), num_channels=base_channels // 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_channels // 2, 1, kernel_size=1),
            nn.Tanh()  # Output in [-1, 1], will be scaled
        )

        # Optional: Boundary detection head (auxiliary task)
        self.boundary_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(16, base_channels // 2), num_channels=base_channels // 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Critical: Proper initialization prevents gradient issues.
        Small initial weights for correction head prevent large early corrections.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Initialize correction head with small weights
        for m in self.correction_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
            self,
            sam_feature: torch.Tensor,
            sam_mask: torch.Tensor,
            return_boundary: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            sam_feature: Feature from SAM encoder [B, C, H, W]
            sam_mask: Predicted mask from SAM decoder [B, 1, H', W']
            return_boundary: Whether to return boundary prediction

        Returns:
            refined_mask: Refined segmentation mask [B, 1, H', W']
            boundary_pred: Optional boundary prediction [B, 1, H', W']

        CRITICAL ASSUMPTIONS:
        1. sam_mask should be in [0, 1] range (sigmoid output)
        2. sam_feature and sam_mask might have different resolutions
        3. Final output will match sam_mask resolution
        """
        original_mask_size = sam_mask.shape[2:]

        # Resize SAM mask to match feature map if needed
        if sam_mask.shape[2:] != sam_feature.shape[2:]:
            sam_mask_resized = F.interpolate(
                sam_mask,
                size=sam_feature.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        else:
            sam_mask_resized = sam_mask

        # Concatenate feature and mask
        x = torch.cat([sam_feature, sam_mask_resized], dim=1)

        # Encoder with skip connections
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc2 = self.dropout(enc2)

        enc3 = self.enc3(self.pool(enc2))
        enc3 = self.dropout(enc3)

        enc4 = self.enc4(self.pool(enc3))
        enc4 = self.dropout(enc4)

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        bottleneck = self.dropout(bottleneck)

        # Decoder with skip connections and optional attention
        dec4 = self.up4(bottleneck)
        if self.use_attention:
            enc4 = self.att4(dec4, enc4)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.up3(dec4)
        if self.use_attention:
            enc3 = self.att3(dec3, enc3)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.up2(dec3)
        if self.use_attention:
            enc2 = self.att2(dec2, enc2)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.up1(dec2)
        if self.use_attention:
            enc1 = self.att1(dec1, enc1)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        # Predict correction
        correction = self.correction_head(dec1)
        correction = correction * self.correction_scale  # Scale to [-correction_scale, +correction_scale]

        # Resize correction to match original mask size
        if correction.shape[2:] != original_mask_size:
            correction = F.interpolate(
                correction,
                size=original_mask_size,
                mode='bilinear',
                align_corners=False
            )

        # Apply correction with clamping
        # CRITICAL: Clamp to [0, 1] to ensure valid probability mask
        refined_mask = torch.clamp(sam_mask + correction, 0.0, 1.0)

        return refined_mask