"""
Configuration for CloneShield training and inference.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """U-Net Perturbation Generator configuration."""
    # Input/output
    sample_rate: int = 22050
    segment_length: int = 22050 * 3  # 3 seconds of audio

    # U-Net architecture
    channels: list = field(default_factory=lambda: [1, 32, 64, 128, 256])
    kernel_size: int = 7
    stride: int = 2

    # Perturbation budget
    epsilon: float = 0.05  # Max perturbation amplitude [-ε, +ε]


@dataclass
class LossConfig:
    """Loss function weights."""
    lambda_quality: float = 1.0       # Audio quality preservation
    lambda_anti_clone: float = 15.0   # Anti-cloning objective (higher to force perturbation)
    mel_weight: float = 1.0           # Mel-spectrogram loss weight
    stft_weight: float = 0.5          # Multi-resolution STFT loss weight
    waveform_weight: float = 0.1      # Direct waveform L1 weight

    # Mel-spectrogram parameters
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 80


@dataclass
class TrainConfig:
    """Training configuration."""
    # Data
    data_root: str = "Small_LibriSpeech_Prepared"
    train_manifest: str = "train-clean-100_manifest.txt"
    val_manifest: str = "dev-clean_manifest.txt"
    test_manifest: str = "test-clean_manifest.txt"

    # Training
    batch_size: int = 8
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5  # Save checkpoint every N epochs
    log_every: int = 10  # Log every N steps

    # Device
    device: str = "auto"  # "auto", "cuda", or "cpu"

    # Model and loss configs
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    def get_device(self) -> str:
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device
