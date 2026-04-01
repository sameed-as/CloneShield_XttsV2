import torch
import torchaudio
from cloneshield.model import PerturbationGenerator
from cloneshield.config import ModelConfig
from cloneshield.speaker_encoder import SpeakerEncoder
import os

def test_dtype_issue():
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
    
    # torchaudio.save automatically casts to 16-bit PCM by default for WAV
    # We will test saving with Float32 vs Int16 and seeing the impact on SpeakerEncoder
    
    torchaudio.save("out_int16.wav", protected_audio, sr, encoding="PCM_S", bits_per_sample=16)
    torchaudio.save("out_float32.wav", protected_audio, sr, encoding="PCM_F", bits_per_sample=32)
    torchaudio.save("out_default.wav", protected_audio, sr) # Default protect.py behavior
    
    # Extract
    speaker_enc = SpeakerEncoder(device)
    emb_orig = speaker_enc(waveform_chunk, sample_rate=config.sample_rate)
    
    wav_int16, _ = torchaudio.load("out_int16.wav")
    wav_float32, _ = torchaudio.load("out_float32.wav")
    wav_default, _ = torchaudio.load("out_default.wav")
    
    wav_int16 = wav_int16 / wav_int16.abs().max()
    wav_float32 = wav_float32 / wav_float32.abs().max()
    wav_default = wav_default / wav_default.abs().max()

    emb_int16 = speaker_enc(wav_int16, sample_rate=config.sample_rate)
    emb_float32 = speaker_enc(wav_float32, sample_rate=config.sample_rate)
    emb_default = speaker_enc(wav_default, sample_rate=config.sample_rate)
    
    sim_internal = torch.nn.functional.cosine_similarity(emb_orig, speaker_enc(protected_chunk, sample_rate=config.sample_rate)).item()
    sim_int16 = torch.nn.functional.cosine_similarity(emb_orig, emb_int16).item()
    sim_float32 = torch.nn.functional.cosine_similarity(emb_orig, emb_float32).item()
    sim_default = torch.nn.functional.cosine_similarity(emb_orig, emb_default).item()

    print(f"Internal Similarity: {sim_internal}")
    print(f"Similarity after saving/loading Int16: {sim_int16}")
    print(f"Similarity after saving/loading Float32: {sim_float32}")
    print(f"Similarity after saving/loading Default (protect.py): {sim_default}")

if __name__ == "__main__":
    test_dtype_issue()
