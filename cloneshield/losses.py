"""
Loss functions for CloneShield training.

- AudioQualityLoss: Keep protected audio perceptually close to original
- AntiCloningLoss: Push speaker embeddings apart
- CloneShieldLoss: Combined total loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from .config import LossConfig


class MelSpectrogramLoss(nn.Module):
    """L1 loss on mel-spectrograms."""

    def __init__(self, sample_rate: int = 22050, n_fft: int = 1024,
                 hop_length: int = 256, n_mels: int = 80):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            norm="slaney",
            mel_scale="slaney",
        )

    def forward(self, original: torch.Tensor, protected: torch.Tensor) -> torch.Tensor:
        """
        Args:
            original: (batch, 1, time)
            protected: (batch, 1, time)
        """
        # Move mel_transform to same device
        self.mel_transform = self.mel_transform.to(original.device)

        mel_orig = self.mel_transform(original.squeeze(1))
        mel_prot = self.mel_transform(protected.squeeze(1))

        # Log-magnitude mel spectrogram for perceptual relevance
        mel_orig = torch.log(mel_orig.clamp(min=1e-5))
        mel_prot = torch.log(mel_prot.clamp(min=1e-5))

        return F.l1_loss(mel_prot, mel_orig)


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss.
    Captures both fine and coarse spectral features.
    """

    def __init__(self, fft_sizes: list = None, hop_sizes: list = None,
                 win_sizes: list = None):
        super().__init__()
        if fft_sizes is None:
            fft_sizes = [512, 1024, 2048]
        if hop_sizes is None:
            hop_sizes = [128, 256, 512]
        if win_sizes is None:
            win_sizes = [512, 1024, 2048]

        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes

    def _stft_loss(self, original: torch.Tensor, protected: torch.Tensor,
                   n_fft: int, hop_length: int, win_length: int) -> torch.Tensor:
        """Single-resolution STFT loss (spectral convergence + log magnitude)."""
        window = torch.hann_window(win_length, device=original.device)

        orig_stft = torch.stft(
            original.squeeze(1), n_fft=n_fft, hop_length=hop_length,
            win_length=win_length, window=window, return_complex=True,
        )
        prot_stft = torch.stft(
            protected.squeeze(1), n_fft=n_fft, hop_length=hop_length,
            win_length=win_length, window=window, return_complex=True,
        )

        orig_mag = orig_stft.abs()
        prot_mag = prot_stft.abs()

        # Spectral convergence loss
        sc_loss = torch.norm(orig_mag - prot_mag, p="fro") / (torch.norm(orig_mag, p="fro") + 1e-8)

        # Log-magnitude loss
        log_loss = F.l1_loss(
            torch.log(prot_mag.clamp(min=1e-5)),
            torch.log(orig_mag.clamp(min=1e-5)),
        )

        return sc_loss + log_loss

    def forward(self, original: torch.Tensor, protected: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for n_fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
            loss += self._stft_loss(original, protected, n_fft, hop, win)
        return loss / len(self.fft_sizes)


class AudioQualityLoss(nn.Module):
    """
    Combined audio quality preservation loss.

    L_quality = waveform_weight * L1_waveform
              + mel_weight * L_mel
              + stft_weight * L_multi_stft
    """

    def __init__(self, config: LossConfig = None):
        super().__init__()
        if config is None:
            config = LossConfig()

        self.waveform_weight = config.waveform_weight
        self.mel_weight = config.mel_weight
        self.stft_weight = config.stft_weight

        self.mel_loss = MelSpectrogramLoss(
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
        )
        self.stft_loss = MultiResolutionSTFTLoss()

    def forward(self, original: torch.Tensor, protected: torch.Tensor) -> dict:
        """
        Returns dict with individual and total quality losses.
        """
        l1 = F.l1_loss(protected, original)
        mel = self.mel_loss(original, protected)
        stft = self.stft_loss(original, protected)

        total = (
            self.waveform_weight * l1
            + self.mel_weight * mel
            + self.stft_weight * stft
        )

        return {
            "quality_total": total,
            "quality_l1": l1,
            "quality_mel": mel,
            "quality_stft": stft,
        }


class AntiCloningLoss(nn.Module):
    """
    Anti-cloning loss: push both conditioning vectors (speaker embedding + GPT latents) apart.

    XttsV2 uses two distinct conditioning streams:
      1. speaker_embedding   (512-d d-vector from hifigan_decoder.speaker_encoder)
      2. gpt_cond_latents    ([1, 1024, T] spectrogram latents fed to the GPT)

    We minimise the cosine similarity of BOTH vectors, averaging them.
    A margin of 0.1 prevents over-optimisation once the signals are already far apart.
    """

    def forward(
        self,
        emb_original: torch.Tensor,
        emb_protected: torch.Tensor,
        gpt_orig: torch.Tensor = None,
        gpt_prot: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            emb_original: Speaker embedding from original audio (batch, dim)
            emb_protected: Speaker embedding from protected audio (batch, dim)
            gpt_orig:  GPT conditioning latents from original  (batch, 1024, T)  [optional]
            gpt_prot:  GPT conditioning latents from protected (batch, 1024, T)  [optional]
        """
        # -- Speaker embedding similarity (existing) --
        cos_sim_spk = F.cosine_similarity(emb_original, emb_protected, dim=-1)
        loss_spk = torch.clamp(cos_sim_spk, min=0.1).mean()

        # -- GPT latent similarity --
        if gpt_orig is not None and gpt_prot is not None:
            # Flatten time dimension and compute per-batch cosine similarity
            # gpt tensors: (batch, 1024, T) -> (batch, 1024*T)
            g_orig_flat = gpt_orig.reshape(gpt_orig.size(0), -1)
            g_prot_flat = gpt_prot.reshape(gpt_prot.size(0), -1)
            cos_sim_gpt = F.cosine_similarity(g_orig_flat, g_prot_flat, dim=-1)
            loss_gpt = torch.clamp(cos_sim_gpt, min=0.1).mean()
            anti_clone_loss = (loss_spk + loss_gpt) / 2.0
            gpt_sim_val = cos_sim_gpt.mean().item()
        else:
            anti_clone_loss = loss_spk
            gpt_sim_val = None

        result = {
            "anti_clone_loss": anti_clone_loss,
            "speaker_similarity": cos_sim_spk.mean().item(),
        }
        if gpt_sim_val is not None:
            result["gpt_similarity"] = gpt_sim_val
        return result


class CloneShieldLoss(nn.Module):
    """
    Combined CloneShield loss.

    L_total = λ₁ * L_quality - λ₂ * L_anti_clone
            = λ₁ * L_quality + λ₂ * (-cos_sim)

    We minimize quality loss (keep audio similar) and minimize
    negative cosine similarity (push embeddings apart).
    """

    def __init__(self, config: LossConfig = None):
        super().__init__()
        if config is None:
            config = LossConfig()

        self.lambda_quality = config.lambda_quality
        self.lambda_anti_clone = config.lambda_anti_clone

        self.quality_loss = AudioQualityLoss(config)
        self.anti_clone_loss = AntiCloningLoss()

    def forward(
        self,
        original: torch.Tensor,
        protected: torch.Tensor,
        emb_original: torch.Tensor,
        emb_protected: torch.Tensor,
        gpt_orig: torch.Tensor = None,
        gpt_prot: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            original: Original audio (batch, 1, time)
            protected: Protected audio (batch, 1, time)
            emb_original: Speaker embedding from original (batch, dim)
            emb_protected: Speaker embedding from protected (batch, dim)
            gpt_orig:  GPT conditioning latents from original  [optional]
            gpt_prot:  GPT conditioning latents from protected [optional]
        """
        quality = self.quality_loss(original, protected)
        anti_clone = self.anti_clone_loss(emb_original, emb_protected, gpt_orig, gpt_prot)

        # Total: minimize quality degradation + minimize speaker similarity
        total = (
            self.lambda_quality * quality["quality_total"]
            + self.lambda_anti_clone * anti_clone["anti_clone_loss"]
        )

        return {
            "total_loss": total,
            **quality,
            **anti_clone,
        }
