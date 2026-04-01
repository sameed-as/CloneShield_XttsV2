import torch
import torchaudio
from cloneshield.model import PerturbationGenerator
from cloneshield.config import ModelConfig
from cloneshield.speaker_encoder import SpeakerEncoder
from TTS.api import TTS
import os

def test_pipeline():
    device = "cpu"
    # 1. Load trained model
    ckpt_path = "checkpoints/cloneshield_best.pt"
    if not os.path.exists(ckpt_path):
        print("Model checkpoint not found!")
        return
        
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ModelConfig()
    if "config" in ckpt and "epsilon" in ckpt["config"]:
        config.epsilon = ckpt["config"]["epsilon"]
    model = PerturbationGenerator(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 2. Emulate training pipeline
    waveform, sr = torchaudio.load("test.wav")
    if sr != config.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, config.sample_rate)
    waveform = waveform.mean(dim=0, keepdim=True)
    max_amp = waveform.abs().max()
    waveform_norm = waveform / max_amp
    
    # Slice to 3s
    waveform_chunk = waveform_norm[:, :config.segment_length].unsqueeze(0)
    if waveform_chunk.shape[-1] < config.segment_length:
        waveform_chunk = torch.nn.functional.pad(waveform_chunk, (0, config.segment_length - waveform_chunk.shape[-1]))
    
    with torch.no_grad():
        perturbation = model(waveform_chunk)
        protected_chunk = waveform_chunk + perturbation

    # Save exactly as protect.py does
    protected_audio = (protected_chunk.squeeze(0) * max_amp)
    torchaudio.save("test_pipeline_out.wav", protected_audio, config.sample_rate)
    
    # 3. Emulate evaluation pipeline
    speaker_enc = SpeakerEncoder(device)
    emb_orig = speaker_enc(waveform_chunk, sample_rate=config.sample_rate)
    emb_prot = speaker_enc(protected_chunk, sample_rate=config.sample_rate)
    sim_internal = torch.nn.functional.cosine_similarity(emb_orig, emb_prot).item()
    print(f"Internal Similarity (Dataset pipeline): {sim_internal}")
    
    # 4. Emulate inference pipeline
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    
    # Load back the saved file
    loaded_wav, loaded_sr = torchaudio.load("test_pipeline_out.wav")
    loaded_orig, _ = torchaudio.load("test.wav")
    
    emb_loaded_prot = speaker_enc(loaded_wav, sample_rate=loaded_sr)
    emb_loaded_orig = speaker_enc(loaded_orig, sample_rate=loaded_sr)
    
    sim_saved = torch.nn.functional.cosine_similarity(emb_loaded_orig, emb_loaded_prot).item()
    print(f"Similarity after saving and loading: {sim_saved}")

if __name__ == "__main__":
    test_pipeline()
