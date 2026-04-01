import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig  # NEW: This was missing

# Allowlist ALL required XTTS classes for PyTorch 2.6 security
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig])

from TTS.api import TTS

# Load XTTS v2 (CPU mode)
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

# Generate speech
tts.tts_to_file(
    text="Hello! This is Me, I am a Senior Software Engineer.",
    speaker_wav="sound_samples/input/speaker_protected_magnified.wav",  # 3-10s WAV required
    language="en",
    file_path="sound_samples/output/output_protected_magnified.wav"
)

tts.tts_to_file(
    text="Hello! This is Me, I am a Senior Software Engineer.",
    speaker_wav="sound_samples/input/speaker_protected.wav",  # 3-10s WAV required
    language="en",
    file_path="sound_samples/output/output_protected.wav"
)

tts.tts_to_file(
    text="Hello! This is Me, I am a Senior Software Engineer.",
    speaker_wav="sound_samples/input/speaker.wav",  # 3-10s WAV required
    language="en",
    file_path="sound_samples/output/output_test.wav"
)

tts.tts_to_file(
    text="Hello! This is Me, I am a Senior Software Engineer.",
    speaker_wav="sound_samples/input/speaker_clean_protected.wav",  # 3-10s WAV required
    language="en",
    file_path="sound_samples/output/output_clean_protected.wav"
)

print("""Done! Check sound_samples/output/output_protected_magnified.wav, \
        sound_samples/output/output_protected.wav and sound_samples/output/output_test.wav \
        and sound_samples/output/output_clean_protected.wav""")
