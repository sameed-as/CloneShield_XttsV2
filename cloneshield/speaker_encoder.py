"""
Speaker Encoder wrapper around XttsV2's frozen speaker conditioning encoder.

Extracts speaker embeddings from audio for the anti-cloning loss.
All parameters are frozen - gradients flow through audio only.
"""

import torch
import torch.nn as nn
import torchaudio

# We need safe-globals for XttsV2 model loading (PyTorch 2.6+)
try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import XttsAudioConfig
    torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig])
except Exception:
    pass


class SpeakerEncoder(nn.Module):
    """
    Wrapper around XttsV2's speaker encoder for extracting speaker embeddings.

    The encoder is frozen (no gradient updates). During training, gradients
    flow through the audio input to guide the perturbation generator.

    XttsV2 uses two conditioning signals:
    1. Speaker embedding from hifigan_decoder.speaker_encoder (d-vector, 512-d)
    2. GPT conditioning latents from mel spectrograms

    We attack the speaker embedding as the primary target.
    """

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device_str = device
        self._model = None
        self._loaded = False

    def _load_model(self):
        """Lazy-load XttsV2 model (it's large, so we load on first use)."""
        if self._loaded:
            return

        from TTS.api import TTS

        print("Loading XttsV2 model for speaker embedding extraction...")
        tts = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            gpu=(self.device_str == "cuda"),
        )
        self._model = tts.synthesizer.tts_model
        self._model.eval()

        # Freeze all parameters
        for param in self._model.parameters():
            param.requires_grad = False

        self._loaded = True
        print(f"XttsV2 loaded on {self.device_str}, all parameters frozen.")

    @property
    def model(self):
        self._load_model()
        return self._model

    def extract_speaker_embedding(self, audio: torch.Tensor,
                                   sample_rate: int = 22050) -> torch.Tensor:
        """
        Extract speaker embedding (d-vector) from audio.

        This is the main target for adversarial attack. The speaker encoder
        inside XttsV2 operates at 16kHz.

        Args:
            audio: Audio tensor, shape (batch, 1, time) or (batch, time)
            sample_rate: Sample rate of input audio

        Returns:
            Speaker embedding, shape (batch, 512)
        """
        model = self.model

        if audio.dim() == 3:
            audio = audio.squeeze(1)  # (batch, time)

        # Resample to 16kHz as XttsV2 speaker encoder expects
        if sample_rate != 16000:
            audio_16k = torchaudio.functional.resample(audio, sample_rate, 16000)
        else:
            audio_16k = audio

        audio_16k = audio_16k.to(self.device_str)

        # Extract through XttsV2's speaker encoder
        # Use .clone() to avoid in-place squeeze_ errors in XTTS code on leaf tensors
        speaker_embedding = model.hifigan_decoder.speaker_encoder.forward(
            audio_16k.clone(), l2_norm=True
        )

        # speaker_embedding shape: (batch, 512) or (1, 512)
        if speaker_embedding.dim() == 1:
            speaker_embedding = speaker_embedding.unsqueeze(0)

        return speaker_embedding

    def extract_gpt_latents(self, audio: torch.Tensor,
                            sample_rate: int = 22050) -> torch.Tensor:
        """
        Extract GPT conditioning latents (spectrogram features) from audio.
        Bypasses Xtts.get_gpt_cond_latents() to enable gradient flow.

        Args:
            audio: Audio tensor, shape (batch, 1, time) or (batch, time)
            sample_rate: Sample rate of input audio

        Returns:
            GPT latents, shape (batch, 1024, T)
        """
        model = self.model

        if audio.dim() == 3:
            audio = audio.squeeze(1)  # (batch, time)
            
        # Audio preprocessing equivalent to Xtts load_audio handling internally
        if audio.shape[-1] > 0:
            max_val = torch.abs(audio).max()
            if max_val > 0:
                audio = (audio / max_val) * 0.75
                
        audio = audio.to(self.device_str)

        # 1. Chunk audio explicitly like XTTS `get_conditioning_latents`
        # XTTS splits audio into `gpt_cond_chunk_len` chunks (default 132300 = 6 secs at 22050Hz)
        chunk_length = model.config.gpt_cond_chunk_len
        actual_chunk_len = chunk_length if chunk_length > 0 else audio.shape[-1]
        
        # 2. Extract style embeddings
        # To avoid massive CUDA OOM during backprop, instead of processing all 
        # chunks and averaging (as XTTS does during inference), we randomly sample
        # ONE chunk during training/protection. This provides the correct receptive
        # field (132300 samples) without blowing up the computation graph.
        
        if audio.shape[-1] <= actual_chunk_len:
            sampled_chunk = audio
        else:
            # Randomly select a start index for the chunk
            max_start = audio.shape[-1] - actual_chunk_len
            start_idx = torch.randint(0, max_start + 1, (1,)).item()
            sampled_chunk = audio[..., start_idx:start_idx + actual_chunk_len]
            
        # Prevent STFT crash on tiny trailing chunks
        if sampled_chunk.shape[-1] < 4096:
            sampled_chunk = torch.nn.functional.pad(sampled_chunk, (0, 4096 - sampled_chunk.shape[-1]))

        if not hasattr(self, '_mel_stft'):
            self._mel_stft = torchaudio.transforms.MelSpectrogram(
                n_fft=4096,
                hop_length=1024,
                win_length=4096,
                power=2,
                normalized=False,
                sample_rate=22050,
                f_min=0,
                f_max=8000,
                n_mels=80,
                norm="slaney",
            ).to(self.device_str)

        mel_stft = self._mel_stft
        mel_norms = model.mel_stats.to(self.device_str)

        # Manual wav_to_mel_cloning
        mel = mel_stft(sampled_chunk)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        mel = mel / mel_norms.unsqueeze(0).unsqueeze(-1)

        # Call GPT style encoder directly
        cond_latent = model.gpt.get_style_emb(mel, return_latent=False)
        final_cond_latent = cond_latent.transpose(1, 2)
            
        return final_cond_latent

    def forward(self, audio: torch.Tensor, sample_rate: int = 22050) -> dict:
        """
        Forward pass extracts both speaker embedding and GPT latents.
        
        Returns:
            dict containing "speaker_embedding" and "gpt_latents"
        """
        speaker_embedding = self.extract_speaker_embedding(audio, sample_rate)
        gpt_latents = self.extract_gpt_latents(audio, sample_rate)
        
        return {
            "speaker_embedding": speaker_embedding,
            "gpt_latents": gpt_latents
        }
