import torchaudio
import torch

def compare():
    audio1, sr1 = torchaudio.load("output_protected_magnified.wav")
    audio2, sr2 = torchaudio.load("output_test.wav")
    
    # Trim to shortest
    min_len = min(audio1.shape[-1], audio2.shape[-1])
    a1 = audio1[:, :min_len]
    a2 = audio2[:, :min_len]
    
    diff = torch.abs(a1 - a2)
    print(f"Mean absolute difference (Magnified vs Original): {diff.mean().item():.5f}")
    print(f"Max absolute difference (Magnified vs Original): {diff.max().item():.5f}")

if __name__ == "__main__":
    compare()
