"""
CloneShield Evaluation Script

Evaluates the protection model by comparing:
1. Speaker similarity scores (original vs protected embeddings)
2. Audio quality metrics (SNR)
3. Optionally running XttsV2 cloning on both for manual comparison

Usage:
    python evaluate.py --checkpoint checkpoints/cloneshield_best.pt [--clone]
"""

import argparse
import os

import torch
import torchaudio

from cloneshield.config import ModelConfig, TrainConfig
from cloneshield.model import PerturbationGenerator
from cloneshield.dataset import LibriSpeechDataset
from cloneshield.speaker_encoder import SpeakerEncoder


@torch.no_grad()
def evaluate(checkpoint_path: str, data_root: str, manifest: str,
             device: str = "cpu", num_samples: int = 50, run_clone: bool = False):
    """
    Evaluate the protection model on test data.

    Args:
        checkpoint_path: Path to trained model checkpoint
        data_root: Data root directory
        manifest: Test manifest file
        device: Device to use
        num_samples: Number of samples to evaluate
        run_clone: Whether to run XttsV2 cloning comparison
    """
    # Load model
    print("Loading CloneShield model...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ModelConfig()
    if "config" in ckpt and "epsilon" in ckpt["config"]:
        config.epsilon = ckpt["config"]["epsilon"]

    model = PerturbationGenerator(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load speaker encoder
    print("Loading speaker encoder...")
    speaker_enc = SpeakerEncoder(device)

    # Load test data
    print("Loading test dataset...")
    dataset = LibriSpeechDataset(data_root, manifest, config)
    num_samples = min(num_samples, len(dataset))

    # Evaluation metrics
    similarities_before = []  # cos_sim(emb_A, emb_A') same audio different pass
    similarities_after = []   # cos_sim(emb_original, emb_protected)
    snr_values = []

    print(f"\nEvaluating on {num_samples} samples...")
    print("-" * 60)

    for i in range(num_samples):
        sample = dataset[i]
        audio = sample["audio"].unsqueeze(0).to(device)  # (1, 1, T)

        # Generate protected audio
        perturbation = model(audio)
        protected = audio + perturbation

        # Extract embeddings
        emb_orig = speaker_enc(audio, sample_rate=config.sample_rate)
        emb_prot = speaker_enc(protected, sample_rate=config.sample_rate)

        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(emb_orig, emb_prot, dim=-1)
        similarities_after.append(cos_sim.item())

        # SNR
        delta = perturbation
        signal_power = audio.pow(2).mean()
        noise_power = delta.pow(2).mean()
        snr = 10 * torch.log10(signal_power / noise_power.clamp(min=1e-10))
        snr_values.append(snr.item())

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{num_samples}] "
                  f"Cos Sim: {cos_sim.item():.4f} | "
                  f"SNR: {snr.item():.1f} dB")

    # Summary statistics
    avg_sim = sum(similarities_after) / len(similarities_after)
    min_sim = min(similarities_after)
    max_sim = max(similarities_after)
    avg_snr = sum(snr_values) / len(snr_values)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Samples evaluated:     {num_samples}")
    print(f"")
    print(f"Speaker Similarity (lower = better protection):")
    print(f"  Average:             {avg_sim:.4f}")
    print(f"  Min:                 {min_sim:.4f}")
    print(f"  Max:                 {max_sim:.4f}")
    print(f"")
    print(f"Audio Quality (higher = less distortion):")
    print(f"  Average SNR:         {avg_snr:.1f} dB")
    print(f"  Min SNR:             {min(snr_values):.1f} dB")
    print(f"")

    if avg_sim < 0.5:
        print("[OK] GOOD: Speaker similarity significantly reduced (< 0.5)")
    elif avg_sim < 0.7:
        print("~ MODERATE: Speaker similarity reduced but could be better")
    else:
        print("[!] WEAK: Speaker similarity still high, model needs more training")

    if avg_snr > 30:
        print("[OK] GOOD: Audio quality preserved (SNR > 30 dB)")
    elif avg_snr > 20:
        print("~ MODERATE: Slight quality degradation (SNR 20-30 dB)")
    else:
        print("[!] WEAK: Noticeable quality degradation (SNR < 20 dB)")

    # Optionally run cloning comparison
    if run_clone:
        print("\n=== Running XttsV2 Cloning Comparison ===")
        _run_clone_comparison(dataset, model, config, device)


def _run_clone_comparison(dataset, model, config, device):
    """Run XttsV2 cloning on original vs protected audio for manual comparison."""
    from TTS.api import TTS

    # Pick first sample
    sample = dataset[0]
    audio = sample["audio"].unsqueeze(0).to(device)

    # Save original and protected audio
    os.makedirs("eval_output", exist_ok=True)

    original_path = "eval_output/original_sample.wav"
    protected_path = "eval_output/protected_sample.wav"
    clone_orig_path = "eval_output/clone_from_original.wav"
    clone_prot_path = "eval_output/clone_from_protected.wav"

    torchaudio.save(original_path, audio.squeeze(0).cpu(), config.sample_rate)

    with torch.no_grad():
        protected = model.protect(audio)
    torchaudio.save(protected_path, protected.squeeze(0).cpu(), config.sample_rate)

    # Clone using XttsV2
    print("Cloning from original audio...")
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=(device == "cuda"))
    test_text = "This is a test of voice cloning protection."

    tts.tts_to_file(
        text=test_text, speaker_wav=original_path,
        language="en", file_path=clone_orig_path,
    )

    print("Cloning from protected audio...")
    tts.tts_to_file(
        text=test_text, speaker_wav=protected_path,
        language="en", file_path=clone_prot_path,
    )

    print(f"\nComparison files saved:")
    print(f"  Original audio:         {original_path}")
    print(f"  Protected audio:        {protected_path}")
    print(f"  Clone from original:    {clone_orig_path}")
    print(f"  Clone from protected:   {clone_prot_path}")
    print(f"\nListen to both clone outputs and compare!")


def main():
    parser = argparse.ArgumentParser(description="Evaluate CloneShield protection")
    parser.add_argument("--checkpoint", "-c", required=True, help="Model checkpoint")
    parser.add_argument("--data-root", default="Small_LibriSpeech_Prepared")
    parser.add_argument("--manifest", default="test-clean_manifest.txt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--clone", action="store_true",
                        help="Also run XttsV2 cloning comparison")
    args = parser.parse_args()

    evaluate(
        args.checkpoint, args.data_root, args.manifest,
        args.device, args.num_samples, args.clone,
    )


if __name__ == "__main__":
    main()
