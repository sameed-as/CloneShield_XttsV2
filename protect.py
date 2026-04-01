"""
CloneShield Protection Script

Apply trained perturbation model to protect audio files from voice cloning.

Usage:
    python protect.py --input speaker.wav --output speaker_protected.wav --checkpoint checkpoints/cloneshield_best.pt
"""

import argparse
import os

import torch
import torchaudio

from cloneshield.config import ModelConfig
from cloneshield.model import PerturbationGenerator
from cloneshield.filter_model import FilterNet


def protect_audio(input_path: str, output_path: str, checkpoint_path: str,
                  device: str = "cpu", epsilon: float = None, filter_checkpoint: str = None):
    """
    Protect a single audio file by adding adversarial perturbation.

    Args:
        input_path: Path to input WAV file
        output_path: Path to save protected WAV file
        checkpoint_path: Path to trained model checkpoint
        device: Device to use
        epsilon: Override epsilon from checkpoint (optional)
        filter_checkpoint: Path to trained FilterNet checkpoint (optional)
    """
    # Load checkpoint
    print(f"Loading model from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Build model config from checkpoint
    config = ModelConfig()
    if "config" in ckpt:
        ckpt_cfg = ckpt["config"]
        if "epsilon" in ckpt_cfg:
            config.epsilon = ckpt_cfg["epsilon"]

    if epsilon is not None:
        config.epsilon = epsilon

    model = PerturbationGenerator(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded (ε={config.epsilon:.4f})")

    filter_model = None
    if filter_checkpoint and os.path.exists(filter_checkpoint):
        print(f"Loading FilterNet from {filter_checkpoint}...")
        f_ckpt = torch.load(filter_checkpoint, map_location=device, weights_only=False)
        filter_model = FilterNet(config).to(device)
        filter_model.load_state_dict(f_ckpt["filter_state_dict"])
        filter_model.eval()
        print(f"FilterNet loaded.")

    # Load audio
    print(f"Loading audio from {input_path}...")
    waveform, sr = torchaudio.load(input_path)

    # Resample to model sample rate if needed
    if sr != config.sample_rate:
        print(f"Resampling {sr} -> {config.sample_rate}")
        resampler = torchaudio.transforms.Resample(sr, config.sample_rate)
        waveform = resampler(waveform)
        sr = config.sample_rate

    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Normalize
    max_amp = waveform.abs().max()
    if max_amp > 0:
        waveform = waveform / max_amp

    original_length = waveform.shape[-1]
    print(f"Audio: {original_length} samples ({original_length/sr:.2f}s)")

    # Process in chunks to handle long audio
    segment_len = config.segment_length
    waveform_3d = waveform.unsqueeze(0).to(device)  # (1, 1, T)

    with torch.no_grad():
        if original_length <= segment_len:
            # Short audio: pad, process, trim
            if original_length < segment_len:
                padded = torch.nn.functional.pad(waveform_3d, (0, segment_len - original_length))
            else:
                padded = waveform_3d

            perturbation = model(padded)
            raw_perturbation = perturbation.clone()
            
            if filter_model is not None:
                mask = filter_model(padded, perturbation)
                perturbation = perturbation * mask
                print(f"Global mask applied, mean: {mask.mean().item():.4f}")
                
            raw_protected = padded + raw_perturbation
            raw_protected = raw_protected[:, :, :original_length]
            
            protected = padded + perturbation
            protected = protected[:, :, :original_length]
        else:
            # Long audio: process in non-overlapping chunks
            # Overlap-add destroys the adversarial perturbation through simple averaging
            hop = segment_len
            protected_chunks = []
            raw_protected_chunks = []

            for start in range(0, original_length, hop):
                end = min(start + segment_len, original_length)
                chunk = waveform_3d[:, :, start:end]

                # Pad if last chunk is shorter
                if chunk.shape[-1] < segment_len:
                    chunk = torch.nn.functional.pad(chunk, (0, segment_len - chunk.shape[-1]))

                perturbation = model(chunk)
                raw_perturbation = perturbation.clone()
                
                if filter_model is not None:
                    mask = filter_model(chunk, perturbation)
                    perturbation = perturbation * mask
                    
                raw_prot_chunk = chunk + raw_perturbation
                raw_prot_chunk = raw_prot_chunk[:, :, :end - start]
                
                prot_chunk = chunk + perturbation
                prot_chunk = prot_chunk[:, :, :end - start]

                protected_chunks.append((start, end, prot_chunk))
                raw_protected_chunks.append((start, end, raw_prot_chunk))

            # Overlap-add reconstruction
            protected = torch.zeros_like(waveform_3d)
            raw_protected = torch.zeros_like(waveform_3d)
            weight_sum = torch.zeros_like(waveform_3d)
            for i, (start, end, chunk) in enumerate(protected_chunks):
                protected[:, :, start:end] += chunk
                raw_protected[:, :, start:end] += raw_protected_chunks[i][2]
                weight_sum[:, :, start:end] += 1.0
            protected = protected / weight_sum.clamp(min=1.0)
            raw_protected = raw_protected / weight_sum.clamp(min=1.0)

    # Restore original amplitude
    protected = protected.squeeze(0) * max_amp
    if filter_model is not None:
        raw_protected = raw_protected.squeeze(0) * max_amp

    # Save
    if filter_model is not None:
        if "_protected" in output_path:
            cleaned_output_path = output_path.replace("_protected", "_clean_protected")
        else:
            base, ext = os.path.splitext(output_path)
            cleaned_output_path = f"{base}_clean{ext}"

        torchaudio.save(output_path, raw_protected.cpu(), sr)
        print(f"Raw protected audio saved to {output_path}")
        
        torchaudio.save(cleaned_output_path, protected.cpu(), sr)
        print(f"Cleaned protected audio saved to {cleaned_output_path}")
    else:
        torchaudio.save(output_path, protected.cpu(), sr)
        print(f"Protected audio saved to {output_path}")

    # Report stats
    delta = (protected.cpu() - waveform * max_amp)
    snr = 10 * torch.log10(
        (waveform * max_amp).pow(2).mean() / delta.pow(2).mean().clamp(min=1e-10)
    )
    print(f"Cleaned Perturbation SNR: {snr.item():.1f} dB")
    print(f"Cleaned Max perturbation: {delta.abs().max().item():.6f}")
    
    if filter_model is not None:
        raw_delta = (raw_protected.cpu() - waveform * max_amp)
        raw_snr = 10 * torch.log10(
            (waveform * max_amp).pow(2).mean() / raw_delta.pow(2).mean().clamp(min=1e-10)
        )
        print(f"Raw Perturbation SNR: {raw_snr.item():.1f} dB")


def main():
    parser = argparse.ArgumentParser(description="Protect audio from voice cloning")
    parser.add_argument("--input", "-i", required=True, help="Input WAV file")
    parser.add_argument("--output", "-o", required=True, help="Output protected WAV file")
    parser.add_argument("--checkpoint", "-c", required=True, default="checkpoints/cloneshield_best.pt", help="Model checkpoint path")
    parser.add_argument("--device", default="cpu", help="Device: cuda/cpu")
    parser.add_argument("--epsilon", type=float, default=None,
                        help="Override perturbation budget")
    parser.add_argument("--filter-checkpoint", "-f", default="checkpoints_filter/filter_best.pt",
                        help="Optional Path to trained FilterNet checkpoint")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return

    protect_audio(args.input, args.output, args.checkpoint, args.device, args.epsilon, args.filter_checkpoint)


if __name__ == "__main__":
    main()
