import torch
import torchaudio
from TTS.api import TTS

def test_inference_gap():
    device = "cpu"
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    
    # Load audio
    waveform, sr = torchaudio.load("test.wav")
    waveform_16k = torchaudio.functional.resample(waveform, sr, 16000).mean(dim=0)
    
    # Simulate padding (like in training)
    segment_len = 16000 * 3
    padded = torch.nn.functional.pad(waveform_16k, (0, max(0, segment_len - waveform_16k.shape[-1])))
    
    # Extract
    emb_original = tts.synthesizer.tts_model.hifigan_decoder.speaker_encoder.forward(
        waveform_16k.unsqueeze(0), l2_norm=True
    )
    emb_padded = tts.synthesizer.tts_model.hifigan_decoder.speaker_encoder.forward(
        padded.unsqueeze(0), l2_norm=True
    )
    
    sim = torch.nn.functional.cosine_similarity(emb_original, emb_padded)
    print(f"Similarity between original and 3s padded version: {sim.item()}")

if __name__ == "__main__":
    test_inference_gap()
