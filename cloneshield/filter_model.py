"""
FilterNet: A second-stage model that learns to prune raw adversarial noise
by generating a Sigmoid mask.
"""

import torch
import torch.nn as nn

from .model import ConvBlock, UpConvBlock, ResidualBlock
from .config import ModelConfig

class FilterNet(nn.Module):
    """
    Second-pass model that generates a pruning mask M in [0, 1].
    Takes original audio and raw noise to determine which noise can be safely removed
    while maintaining the adversarial effectiveness.
    """
    def __init__(self, config: ModelConfig = None):
        super().__init__()
        if config is None:
            config = ModelConfig()

        channels = config.channels  # e.g. [1, 32, 64, 128, 256]
        # We override input channels from 1 to 2 since we input [audio, noise] stacked.
        in_channels = [2] + channels[1:]
        ks = config.kernel_size
        st = config.stride

        # Encoder
        self.encoders = nn.ModuleList()
        for i in range(len(in_channels) - 1):
            self.encoders.append(ConvBlock(in_channels[i], in_channels[i + 1], ks, st))

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(in_channels[-1], ks),
            ResidualBlock(in_channels[-1], ks),
        )

        # Decoder (upsampling)
        self.decoders = nn.ModuleList()
        for i in range(len(in_channels) - 1, 1, -1):
            self.decoders.append(UpConvBlock(in_channels[i], in_channels[i - 1], ks, st))

        # Final upsampling
        self.final_up = nn.ConvTranspose1d(
            in_channels[1], in_channels[1], ks, stride=st,
            padding=ks // 2, output_padding=st - 1,
        )

        # Final convolution to produce Mask [0, 1]
        self.final_conv = nn.Sequential(
            nn.Conv1d(in_channels[1] + in_channels[0], 16, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(16, 1, kernel_size=7, padding=3),
            nn.Sigmoid()  # Mask output
        )

    def forward(self, original_audio: torch.Tensor, raw_noise: torch.Tensor) -> torch.Tensor:
        """
        Args:
            original_audio: (batch, 1, time)
            raw_noise: (batch, 1, time)
        Returns:
            mask: (batch, 1, time) in range [0, 1]
        """
        x = torch.cat([original_audio, raw_noise], dim=1) # [batch, 2, time]
        original_length = x.shape[-1]

        skips = [x]
        h = x
        for encoder in self.encoders:
            h = encoder(h)
            skips.append(h)

        h = self.bottleneck(h)

        for i, decoder in enumerate(self.decoders):
            skip_idx = len(skips) - 2 - i
            h = decoder(h, skips[skip_idx])

        h = self.final_up(h)
        if h.shape[-1] != original_length:
            diff = original_length - h.shape[-1]
            if diff > 0:
                h = nn.functional.pad(h, (0, diff))
            else:
                h = h[..., :original_length]

        h = torch.cat([h, x], dim=1)
        mask = self.final_conv(h)
        return mask
