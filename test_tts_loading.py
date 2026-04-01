import torch
import torchaudio
from cloneshield.model import PerturbationGenerator
from cloneshield.config import ModelConfig
from TTS.api import TTS
import os

def test_gpt_latents():
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

    with torch.no_grad():
        waveform_chunk = waveform[:, :config.segment_length].unsqueeze(0)
        perturbation = model(waveform_chunk)
        protected_chunk = waveform_chunk + perturbation

    protected_audio = (protected_chunk.squeeze(0) * max_amp)
    torchaudio.save("out_protected.wav", protected_audio, sr)
    torchaudio.save("out_original.wav", waveform_chunk.squeeze(0) * max_amp, sr)

    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    
    # Extract conditioning latents from both
    gpt_orig, spk_orig = tts.synthesizer.tts_model.get_conditioning_latents("out_original.wav")
    gpt_prot, spk_prot = tts.synthesizer.tts_model.get_conditioning_latents("out_protected.wav")
    
    sim_spk = torch.nn.functional.cosine_similarity(spk_orig, spk_prot).item()
    
    # GPT latents shape is [1, 1024, T], so we need to flatten to compute similarity
    import torch.nn.functional as F
    gpt_orig_flat = gpt_orig.reshape(-1)
    gpt_prot_flat = gpt_prot.reshape(-1)
    # Ensure they are same length (if trimming changed length)
    min_len = min(gpt_orig_flat.shape[0], gpt_prot_flat.shape[0])
    
    sim_gpt = F.cosine_similarity(gpt_orig_flat[:min_len].unsqueeze(0), gpt_prot_flat[:min_len].unsqueeze(0)).item()
    
    print(f"Speaker Embedding Similarity (Targeted): {sim_spk}")
    print(f"GPT Conditioning Latents Similarity (Untargeted): {sim_gpt}")

if __name__ == "__main__":
    test_gpt_latents()
