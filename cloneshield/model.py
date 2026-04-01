"""
PerturbationGenerator: U-Net model that generates adversarial perturbations.

Output: protected_audio = original_audio + clamp(perturbation, -ε, +ε)
"""

import torch
import torch.nn as nn

from .config import ModelConfig


class ConvBlock(nn.Module):
    """Downsampling convolutional block."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, stride: int = 2):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpConvBlock(nn.Module):
    """Upsampling convolutional block with skip connection."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, stride: int = 2):
        super().__init__()
        padding = kernel_size // 2
        self.up = nn.ConvTranspose1d(
            in_ch, out_ch, kernel_size, stride=stride,
            padding=padding, output_padding=stride - 1,
        )
        # After concatenation with skip connection: out_ch + out_ch = 2*out_ch
        self.conv = nn.Sequential(
            nn.Conv1d(out_ch * 2, out_ch, kernel_size, stride=1, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle size mismatch from strided conv/deconv
        if x.shape[-1] != skip.shape[-1]:
            diff = skip.shape[-1] - x.shape[-1]
            x = nn.functional.pad(x, (0, diff))
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResidualBlock(nn.Module):
    """Residual block for the bottleneck."""

    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class PerturbationGenerator(nn.Module):
    """
    U-Net that generates an adversarial perturbation δ.

    protected_audio = original_audio + clamp(δ, -ε, +ε)

    The perturbation is small and imperceptible to humans,
    but disrupts XttsV2's speaker embedding extraction.
    """

    def __init__(self, config: ModelConfig = None):
        super().__init__()
        if config is None:
            config = ModelConfig()

        self.epsilon = config.epsilon
        channels = config.channels  # e.g. [1, 32, 64, 128, 256]
        ks = config.kernel_size
        st = config.stride

        # Encoder (downsampling)
        self.encoders = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.encoders.append(ConvBlock(channels[i], channels[i + 1], ks, st))

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(channels[-1], ks),
            ResidualBlock(channels[-1], ks),
        )

        # Decoder (upsampling) - reverse order
        self.decoders = nn.ModuleList()
        for i in range(len(channels) - 1, 1, -1):
            self.decoders.append(UpConvBlock(channels[i], channels[i - 1], ks, st))

        # Final upsampling (no skip connection from input needed differently)
        self.final_up = nn.ConvTranspose1d(
            channels[1], channels[1], ks, stride=st,
            padding=ks // 2, output_padding=st - 1,
        )

        # Final convolution to produce perturbation
        self.final_conv = nn.Sequential(
            nn.Conv1d(channels[1] + channels[0], 16, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(16, 1, kernel_size=7, padding=3),
            nn.AvgPool1d(kernel_size=5, stride=1, padding=2), # Smooth the perturbation vector
            nn.Tanh(),  # Output in [-1, 1], then scaled by epsilon
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input audio tensor, shape (batch, 1, time)

        Returns:
            perturbation: Clamped perturbation δ, shape (batch, 1, time)
        """
        original_length = x.shape[-1]

        # Encoder - collect skip connections
        skips = [x]
        h = x
        for encoder in self.encoders:
            h = encoder(h)
            skips.append(h)

        # Bottleneck
        h = self.bottleneck(h)

        # Decoder - use skip connections (in reverse, excluding last skip which is bottleneck input)
        for i, decoder in enumerate(self.decoders):
            skip_idx = len(skips) - 2 - i  # Skip connections from encoder
            h = decoder(h, skips[skip_idx])

        # Final upsample + concat with input
        h = self.final_up(h)
        if h.shape[-1] != original_length:
            diff = original_length - h.shape[-1]
            if diff > 0:
                h = nn.functional.pad(h, (0, diff))
            else:
                h = h[..., :original_length]

        h = torch.cat([h, x], dim=1)  # Concat with original input
        perturbation = self.final_conv(h)

        # Scale by epsilon: Tanh outputs [-1,1] * epsilon = [-ε, +ε]
        perturbation = perturbation * self.epsilon

        return perturbation

    def protect(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Generate protected audio.

        Args:
            audio: Input audio tensor, shape (batch, 1, time)

        Returns:
            protected: Protected audio, shape (batch, 1, time)
        """
        perturbation = self.forward(audio)
        protected = audio + perturbation
        return protected
