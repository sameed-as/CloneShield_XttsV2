import os
import time
import requests
import torch
import torchaudio
import pandas as pd
import io

# --- CONFIGURATION ---
API_URL = "https://fakeless-api.onrender.com/clone"
INPUT_FILE = r"sound_samples/input/speaker.wav" 
RESULTS = []

# We define the configurations using Sample Rate and Bit Depth
TEST_CONFIGS = [
    {"name": "Light", "sr": 8000, "precision": "PCM_16"},
    {"name": "Standard", "sr": 16000, "precision": "PCM_16"},
    {"name": "Medium", "sr": 44100, "precision": "PCM_16"},
    {"name": "Heavy", "sr": 48000, "precision": "PCM_32"}, # 32-bit significantly increases size
]

def create_variant_audio(input_path, output_path, target_sr, precision):
    """Generates audio variants using torchaudio instead of ffmpeg."""
    waveform, orig_sr = torchaudio.load(input_path)
    
    # 1. Resample if necessary
    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        waveform = resampler(waveform)
    
    # 2. Save with specific precision (bit depth)
    # precision can be 'PCM_16', 'PCM_32', or 'FLOAT'
    torchaudio.save(output_path, waveform, target_sr, encoding="PCM_S", bits_per_sample=int(precision.split('_')[1]))
    
    return os.path.getsize(output_path) / 1024  # Size in KB

def test_api(file_path):
    """Measures latency from request start to response received."""
    start_time = time.perf_counter()
    with open(file_path, "rb") as f:
        files = {"speaker": (os.path.basename(file_path), f, "audio/wav")}
        data = {"text": "Latency test."}
        try:
            # First call might be slow due to Render spin-up
            response = requests.post(API_URL, files=files, data=data, timeout=90)
            response.raise_for_status()
            latency = (time.perf_counter() - start_time) * 1000 
            return latency, "Success"
        except Exception as e:
            return 0, f"Error: {str(e)}"

# --- MAIN EXECUTION ---
print(f"Starting Latency Test (Using Torchaudio) on {API_URL}...\n")

# Pre-check: Ensure input exists
if not os.path.exists(INPUT_FILE):
    print(f"ERROR: Base file not found at {INPUT_FILE}")
    exit()

for cfg in TEST_CONFIGS:
    temp_filename = f"test_var_{cfg['sr']}_{cfg['precision']}.wav"
    
    print(f"Generating {cfg['name']} variant...")
    size_kb = create_variant_audio(INPUT_FILE, temp_filename, cfg['sr'], cfg['precision'])
    
    print(f"Testing {cfg['name']} | Size: {size_kb:.2f} KB")
    
    latencies = []
    for i in range(2): # Reduced to 2 trials for speed
        latency, status = test_api(temp_filename)
        if "Success" in status:
            latencies.append(latency)
            print(f"  Trial {i+1}: {latency:.2f}ms")
    
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    RESULTS.append({"Profile": cfg['name'], "Size (KB)": round(size_kb, 2), "Avg Latency (ms)": round(avg_latency, 2)})
    
    if os.path.exists(temp_filename):
        os.remove(temp_filename)

# Display Results
df = pd.DataFrame(RESULTS)
print("\n--- FINAL RESULTS ---")
print(df.to_string(index=False))