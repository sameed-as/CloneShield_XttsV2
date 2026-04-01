import torch
from TTS.api import TTS
import torchaudio

def test_normalization():
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    
    # Load with XTTS's internal audio loader
    orig_xtts = tts.synthesizer.tts_model.load_audio("speaker.wav")
    prot_xtts = tts.synthesizer.tts_model.load_audio("speaker_protected.wav")
    prot_xtts = torch.tensor(prot_xtts)
    
    # Check max diff after XTTS loads it
    min_len = min(orig_xtts.shape[-1], prot_xtts.shape[-1])
    diff = torch.abs(orig_xtts[:min_len] - prot_xtts[:min_len])
    
    print(f"XTTS loaded orig max: {orig_xtts.abs().max().item()}")
    print(f"XTTS loaded prot max: {prot_xtts.abs().max().item()}")
    print(f"Max perturbation surviving XTTS load: {diff.max().item()}")
    
    # Compare to standard torchaudio load
    print("\n--- Standard Torchaudio Loading ---")
    orig_torch, _ = torchaudio.load("speaker.wav")
    prot_torch, _ = torchaudio.load("speaker_protected.wav")
    
    min_len = min(orig_torch.shape[-1], prot_torch.shape[-1])
    diff_torch = torch.abs(orig_torch[0, :min_len] - prot_torch[0, :min_len])
    print(f"Torchaudio loaded orig max: {orig_torch.abs().max().item()}")
    print(f"Torchaudio loaded prot max: {prot_torch.abs().max().item()}")
    print(f"Max perturbation surviving Torchaudio load: {diff_torch.max().item()}")

if __name__ == "__main__":
    test_normalization()
