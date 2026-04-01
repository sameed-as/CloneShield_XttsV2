import torch
from cloneshield.losses import CloneShieldLoss
from cloneshield.speaker_encoder import SpeakerEncoder

def evaluate_losses():
    # Load original and protected
    import torchaudio
    orig, sr = torchaudio.load("speaker.wav")
    prot, _ = torchaudio.load("speaker_protected.wav")
    
    # Sync length
    min_len = min(orig.shape[-1], prot.shape[-1])
    orig = orig[:, :min_len]
    prot = prot[:, :min_len]
    
    # Add batch dim
    orig = orig.unsqueeze(0)
    prot = prot.unsqueeze(0)
    
    # Init encoder and loss
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SpeakerEncoder(device=device)
    encoder._load_model()
    
    orig = orig[:, 0, :] if orig.dim() == 3 else orig
    prot = prot[:, 0, :] if prot.dim() == 3 else prot
    orig = orig.to(device)
    prot = prot.to(device)
    
    loss_fn = CloneShieldLoss()
    
    # Get embeddings
    print("Extracting embeddings for loss analysis...")
    with torch.no_grad():
        out_orig = encoder(orig, sr)
        out_prot = encoder(prot, sr)
        
        losses = loss_fn(
            orig.cpu(), prot.cpu(), 
            out_orig["speaker_embedding"].cpu(), out_prot["speaker_embedding"].cpu(),
            out_orig["gpt_latents"].cpu(), out_prot["gpt_latents"].cpu()
        )
    
    print("\n--- Loss Component Analysis ---")
    for k, v in losses.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {v.item():.4f}")
        else:
            print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    evaluate_losses()
