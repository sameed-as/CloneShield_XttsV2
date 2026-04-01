import torch
import torchaudio
from cloneshield.model import PerturbationGenerator
from cloneshield.config import ModelConfig
import os

def test_overlap_issue():
    device = "cpu"
    ckpt_path = "checkpoints/cloneshield_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ModelConfig()
    
    if "config" in ckpt and "epsilon" in ckpt["config"]:
        config.epsilon = ckpt["config"]["epsilon"]
        
    model = PerturbationGenerator(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    waveform, sr = torchaudio.load("test.wav")
    waveform = waveform.mean(dim=0, keepdim=True)
    max_amp = waveform.abs().max()
    waveform = waveform / max_amp

    original_length = waveform.shape[-1]
    segment_len = config.segment_length
    waveform_3d = waveform.unsqueeze(0).to(device)
    
    # EXACT LOGIC from protect.py
    with torch.no_grad():
        hop = segment_len // 2
        protected_chunks = []
        
        for start in range(0, original_length, hop):
            end = min(start + segment_len, original_length)
            chunk = waveform_3d[:, :, start:end]
            
            if chunk.shape[-1] < segment_len:
                chunk = torch.nn.functional.pad(chunk, (0, segment_len - chunk.shape[-1]))
                
            perturbation = model(chunk)
            print(f"Chunk {start}-{end} Perturbation max: {perturbation.abs().max().item()}")
            
            prot_chunk = chunk + perturbation
            prot_chunk = prot_chunk[:, :, :end - start]
            
            protected_chunks.append((start, end, prot_chunk))
            
        protected = torch.zeros_like(waveform_3d)
        weight_sum = torch.zeros_like(waveform_3d)
        
        for start, end, chunk in protected_chunks:
            protected[:, :, start:end] += chunk
            weight_sum[:, :, start:end] += 1.0
            
        protected = protected / weight_sum.clamp(min=1.0)
        
        delta = protected - waveform_3d
        print(f"Final Output Perturbation max: {delta.abs().max().item()}")

if __name__ == "__main__":
    test_overlap_issue()
