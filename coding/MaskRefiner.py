import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ConvBlock(nn.Module):
    """
    Double convolution block with normalization and residual connection.
    Critical choices:
    - GroupNorm over BatchNorm: More stable with small batches
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


class DecoderBlock(nn.Module):
    """
    Decoder block: Upsample + ConvBlock with optional mask feature fusion.
    """

    def __init__(self, in_channels: int, out_channels: int, use_mask_fusion: bool = True):
        super().__init__()
        self.use_mask_fusion = use_mask_fusion

        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.norm = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        self.activation = nn.LeakyReLU(0.1, inplace=True)

        # If fusing with mask, input will be out_channels + mask_channels
        conv_in_channels = out_channels + 1 if use_mask_fusion else out_channels
        self.conv_block = ConvBlock(conv_in_channels, out_channels)

    def forward(self, x: torch.Tensor, mask_feature: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.upsample(x)
        x = self.norm(x)
        x = self.activation(x)

        # Fuse with mask features at this resolution if provided
        if self.use_mask_fusion and mask_feature is not None:
            # Ensure mask_feature matches spatial dimensions
            if mask_feature.shape[2:] != x.shape[2:]:
                mask_feature = F.interpolate(mask_feature, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, mask_feature], dim=1)

        x = self.conv_block(x)
        return x


class MaskRefinerDecoder(nn.Module):
    """
    Decoder-only architecture for mask refinement.

    CRITICAL DESIGN PHILOSOPHY:
    - No encoder needed - SAM embedding IS the encoded features
    - Dual-path: semantic (embedding) and spatial (mask) processed separately
    - Progressive upsampling with mask fusion at each scale
    - Predicts CORRECTIONS, not full masks

    Args:
        embed_channels: Channels in SAM embedding (768 for block 10)
        base_channels: Base decoder channels (will decrease as we upsample)
        num_upsample_stages: Number of 2x upsampling stages
        correction_scale: Maximum magnitude of correction
        use_mask_fusion: Fuse mask features at each decoder stage
        dropout_rate: Dropout for regularization
    """

    def __init__(
            self,
            embed_channels: int = 768,
            base_channels: int = 256,
            num_upsample_stages: int = 5,  # 64->128->256->512->1024->2048
            correction_scale: float = 0.3,
            use_mask_fusion: bool = True,
            dropout_rate: float = 0.1
    ):
        super().__init__()
        self.correction_scale = correction_scale
        self.num_upsample_stages = num_upsample_stages
        self.use_mask_fusion = use_mask_fusion

        # Project embedding to decoder base dimension
        self.embed_proj = nn.Sequential(
            nn.Conv2d(embed_channels, base_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(32, base_channels), num_channels=base_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Mask feature extractor (processes mask at multiple scales)
        # This creates a parallel path that preserves spatial detail
        self.mask_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1, bias=False)
        )

        # Progressive decoder stages
        self.decoder_stages = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        channels = base_channels
        for i in range(num_upsample_stages):
            out_channels = max(channels // 2, 32)  # Don't go below 32 channels

            self.decoder_stages.append(
                DecoderBlock(channels, out_channels, use_mask_fusion=use_mask_fusion)
            )
            self.dropout_layers.append(nn.Dropout2d(p=dropout_rate))

            channels = out_channels

        # Final correction head
        # CRITICAL: Small initialization for stable corrections
        self.correction_head = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(16, channels // 2), num_channels=channels // 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels // 2, 1, kernel_size=1),
            nn.Tanh()  # Output in [-1, 1]
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
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Initialize correction head with VERY small weights
        for m in self.correction_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
            self,
            sam_embedding: torch.Tensor,  # [B, 64, 64, 768]
            sam_mask: torch.Tensor,  # [B, 1, H, W] or [H, W]
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            sam_embedding: SAM block 10 embedding [B, 64, 64, 768]
            sam_mask: SAM predicted mask [B, 1, H, W] or [H, W]

        Returns:
            refined_mask: Refined mask [B, 1, H, W]
        """
        # Handle embedding dimension ordering
        if sam_embedding.shape[-1] == 768:  # [B, H, W, C]
            sam_embedding = sam_embedding.permute(0, 3, 1, 2)  # [B, C, H, W]

        # Handle mask dimensions
        if sam_mask.dim() == 2:
            sam_mask = sam_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        elif sam_mask.dim() == 3:
            sam_mask = sam_mask.unsqueeze(1)  # [B, 1, H, W]

        original_mask_size = sam_mask.shape[2:]

        # Project embedding to decoder dimension
        x = self.embed_proj(sam_embedding)  # [B, base_channels, 64, 64]

        # Extract mask features (this preserves spatial information)
        mask_features = self.mask_conv(sam_mask)  # [B, 1, H, W]

        # Progressive upsampling with mask fusion
        for i, (decoder_stage, dropout) in enumerate(zip(self.decoder_stages, self.dropout_layers)):
            x = decoder_stage(x, mask_features if self.use_mask_fusion else None)
            x = dropout(x)

        # Predict correction
        correction = self.correction_head(x)  # [B, 1, current_size, current_size]

        # Resize correction to match original mask size
        if correction.shape[2:] != original_mask_size:
            correction = F.interpolate(
                correction,
                size=original_mask_size,
                mode='bilinear',
                align_corners=False
            )

        # Scale correction and apply with clamping
        correction = correction * self.correction_scale
        refined_mask = torch.clamp(sam_mask + correction, 0.0, 1.0)

        return refined_mask.contiguous()