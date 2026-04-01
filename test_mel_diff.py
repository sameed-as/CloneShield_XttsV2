import torch
import torchaudio
import matplotlib.pyplot as plt
import os

def visualize_mel_difference(original_path="speaker.wav", protected_path="speaker_protected.wav"):
    if not os.path.exists(original_path) or not os.path.exists(protected_path):
        print("Audio files not found.")
        return

    # Load audio
    orig, sr_orig = torchaudio.load(original_path)
    prot, sr_prot = torchaudio.load(protected_path)
    
    # Trim to shortest
    min_len = min(orig.shape[-1], prot.shape[-1])
    orig = orig[:, :min_len]
    prot = prot[:, :min_len]
    
    # 1. Mel Spectrogram computation matching CloneShield and XTTS settings
    mel_stft = torchaudio.transforms.MelSpectrogram(
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
    )
    
    # Compute Mels
    mel_orig = mel_stft(orig)
    mel_orig = torch.log(torch.clamp(mel_orig, min=1e-5))
    
    mel_prot = mel_stft(prot)
    mel_prot = torch.log(torch.clamp(mel_prot, min=1e-5))
    
    # Calculate difference
    mel_diff = torch.abs(mel_prot - mel_orig)
    
    print(f"Mean Mel difference: {mel_diff.mean().item():.5f}")
    print(f"Max Mel difference: {mel_diff.max().item():.5f}")
    
    # 2. Visualize
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    # Original Mel
    im0 = axs[0].imshow(mel_orig.squeeze().numpy(), origin="lower", aspect="auto", cmap="viridis")
    axs[0].set_title("Original Mel-Spectrogram (Log)")
    fig.colorbar(im0, ax=axs[0])
    
    # Protected Mel
    im1 = axs[1].imshow(mel_prot.squeeze().numpy(), origin="lower", aspect="auto", cmap="viridis")
    axs[1].set_title("Protected Mel-Spectrogram (Log)")
    fig.colorbar(im1, ax=axs[1])
    
    # Difference
    im2 = axs[2].imshow(mel_diff.squeeze().numpy(), origin="lower", aspect="auto", cmap="inferno")
    axs[2].set_title("Absolute Difference (Protected - Original)")
    fig.colorbar(im2, ax=axs[2])
    
    plt.tight_layout()
    plt.savefig("mel_diff_visualization.png")
    print("Saved visualization to mel_diff_visualization.png")
    
    # Check max volume of audio
    perturbation = torch.abs(prot - orig)
    print(f"Audio volume max (orig): {orig.abs().max().item():.5f}")
    print(f"Perturbation max amplitude applied: {perturbation.max().item():.5f}")

if __name__ == "__main__":
    visualize_mel_difference()
