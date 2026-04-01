"""
Dataset loader for LibriSpeech prepared data.
"""

import os
import random

import torch
import torchaudio
from torch.utils.data import Dataset

from .config import ModelConfig


class LibriSpeechDataset(Dataset):
    """
    PyTorch Dataset for the prepared LibriSpeech data.

    Reads manifest files and loads WAV audio segments.
    Each item returns a fixed-length audio chunk and speaker ID.
    """

    def __init__(self, data_root: str, manifest_file: str,
                 config: ModelConfig = None):
        """
        Args:
            data_root: Root directory (e.g., "Small_LibriSpeech_Prepared")
            manifest_file: Manifest filename (e.g., "train-clean-100_manifest.txt")
            config: Model config with sample_rate and segment_length
        """
        if config is None:
            config = ModelConfig()

        self.sample_rate = config.sample_rate
        self.segment_length = config.segment_length

        # Read manifest - get the project root (parent of data_root)
        self.project_root = os.path.dirname(os.path.abspath(data_root))
        manifest_path = os.path.join(data_root, manifest_file)

        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        # Build file list and extract speaker IDs
        self.files = []
        self.speaker_ids = []
        self.speaker_to_idx = {}

        for line in lines:
            # Paths in manifest are relative like:
            # Small_LibriSpeech_Prepared\train-clean-100\LibriSpeech\...\{spk}-{chap}-{utt}.wav
            # Convert backslashes to forward slashes for cross-platform
            rel_path = line.replace("\\", "/")
            abs_path = os.path.join(self.project_root, rel_path)

            if not os.path.exists(abs_path):
                continue  # Skip missing files silently

            # Extract speaker ID from filename: {speaker_id}-{chapter}-{utterance}.wav
            filename = os.path.basename(rel_path)
            speaker_id = filename.split("-")[0]

            if speaker_id not in self.speaker_to_idx:
                self.speaker_to_idx[speaker_id] = len(self.speaker_to_idx)

            self.files.append(abs_path)
            self.speaker_ids.append(self.speaker_to_idx[speaker_id])

        print(f"Loaded {len(self.files)} files from {len(self.speaker_to_idx)} speakers")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with keys:
                - "audio": tensor of shape (1, segment_length)
                - "speaker_id": int
                - "file_path": str
        """
        audio_path = self.files[idx]
        speaker_id = self.speaker_ids[idx]

        # Load audio
        waveform, sr = torchaudio.load(audio_path)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Ensure shape is (1, time)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Normalize amplitude
        max_amp = waveform.abs().max()
        if max_amp > 0:
            waveform = waveform / max_amp

        # Pad or crop to segment_length
        current_length = waveform.shape[-1]
        if current_length >= self.segment_length:
            # Random crop
            start = random.randint(0, current_length - self.segment_length)
            waveform = waveform[:, start:start + self.segment_length]
        else:
            # Pad with zeros
            padding = self.segment_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        return {
            "audio": waveform,
            "speaker_id": speaker_id,
            "file_path": audio_path,
        }
