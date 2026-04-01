import torch
import torchaudio
from cloneshield.speaker_encoder import SpeakerEncoder

def test_gradient_flow():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on {device}")
    
    speaker_enc = SpeakerEncoder(device=device)
    
    # Create a dummy audio tensor with requires_grad=True
    # segment_length is 16000 * 4? No, let's just use some length
    audio = torch.randn(1, 1, 16000, requires_grad=True, device=device)
    
    # Extract latents
    # Note: extraction will move it to device internally
    outputs = speaker_enc(audio, sample_rate=16000)
    
    spk_emb = outputs["speaker_embedding"]
    gpt_latents = outputs["gpt_latents"]
    
    print(f"Speaker embedding shape: {spk_emb.shape}")
    print(f"GPT latents shape: {gpt_latents.shape}")
    
    # Check if speaker embedding has grad_fn
    print(f"Speaker embedding has grad_fn: {spk_emb.grad_fn is not None}")
    
    # Check if GPT latents have grad_fn
    print(f"GPT latents have grad_fn: {gpt_latents.grad_fn is not None}")
    
    # Try a backward pass
    loss = gpt_latents.sum()
    loss.backward()
    
    print(f"Audio grad max: {audio.grad.abs().max().item()}")
    
    if audio.grad.abs().max().item() > 0:
        print("SUCCESS: Gradients flow through GPT latents!")
    else:
        print("FAILURE: Gradients do NOT flow through GPT latents.")

if __name__ == "__main__":
    test_gradient_flow()
