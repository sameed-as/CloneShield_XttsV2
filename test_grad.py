import torch
from cloneshield.speaker_encoder import SpeakerEncoder

def test_grad():
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    enc = SpeakerEncoder(device)
    
    # Dummy audio: requires_grad=True
    audio = torch.randn(1, 1, 22050, requires_grad=True, device=device)
    
    emb = enc(audio, sample_rate=22050)
    
    # Compute sum
    loss = emb.sum()
    loss.backward()
    
    if audio.grad is not None:
        print(f"Gradient flows! Gradient sum: {audio.grad.sum().item()}, max: {audio.grad.abs().max().item()}")
    else:
        print("GRADIENT BROKEN! audio.grad is None")

if __name__ == "__main__":
    test_grad()
