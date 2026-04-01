import torchaudio
import torch

def compare():
    audio1, sr1 = torchaudio.load("output_protected.wav")
    audio2, sr2 = torchaudio.load("output_test.wav")
    
    # Trim to shortest
    min_len = min(audio1.shape[-1], audio2.shape[-1])
    a1 = audio1[:, :min_len]
    a2 = audio2[:, :min_len]
    
    # Check if exactly equal
    is_equal = torch.allclose(a1, a2, atol=1e-5)
    print(f"Are the outputs exactly identical? {is_equal}")
    
    # Print the difference map summary
    diff = torch.abs(a1 - a2)
    print(f"Mean absolute difference: {diff.mean().item():.5f}")
    print(f"Max absolute difference: {diff.max().item():.5f}")

if __name__ == "__main__":
    compare()
