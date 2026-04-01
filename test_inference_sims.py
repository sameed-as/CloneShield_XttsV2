import torch
from TTS.api import TTS
import torchaudio

def inspect_inference_latents():
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    model = tts.synthesizer.tts_model
    model.eval()

    # Get conditioning latents exactly as inference does
    with torch.inference_mode():
        gpt_orig, spk_orig = model.get_conditioning_latents(audio_path=["speaker.wav"])
        gpt_prot, spk_prot = model.get_conditioning_latents(audio_path=["speaker_protected.wav"])
        
    print(f"GPT Latents Orig Shape: {gpt_orig.shape}")
    print(f"GPT Latents Prot Shape: {gpt_prot.shape}")
    
    # Calculate similarity
    spk_sim = torch.nn.functional.cosine_similarity(spk_orig, spk_prot, dim=-1).mean().item()
    
    # For GPT, it's (1, 1024, T)
    gpt_orig_flat = gpt_orig.reshape(-1)
    gpt_prot_flat = gpt_prot.reshape(-1)
    gpt_sim = torch.nn.functional.cosine_similarity(gpt_orig_flat.unsqueeze(0), gpt_prot_flat.unsqueeze(0)).item()
    
    print(f"Speaker Similarity during ACTUAL inference: {spk_sim:.4f}")
    print(f"GPT Similarity during ACTUAL inference: {gpt_sim:.4f}")

if __name__ == "__main__":
    inspect_inference_latents()
