import subprocess
import os
import re
import sys

def run_evaluation():
    # Protection Levels Mapping: (Level ID, Filter Strength)
    levels = [
        (1, 1.00),   # Level 1: Max filter (Quiet noise, weak protection)
        (2, 0.75),   # Level 2: High filter
        (3, 0.50),   # Level 3: Medium filter
        (4, 0.25),   # Level 4: Weak filter
        (5, 0.00)    # Level 5: No filter (Raw noise, max protection)
    ]

    input_file = "sound_samples/input/speaker.wav"
    checkpoint = "checkpoints/cloneshield_best.pt"
    filter_checkpoint = "checkpoints_filter/filter_best.pt"
    
    # Optional parameters fallback
    os.makedirs("sound_samples/output", exist_ok=True)

    print("\nStarting Automated Protection Strength Evaluation...")
    print(f"Input Target: {input_file}")
    
    # Print Table Header
    print("\n" + "=" * 62)
    print(f"{'Level':<6} | {'Filter':<6} | {'SNR (dB)':<8} | {'Cosine':<6} | {'PESQ':<5} | {'STOI':<5}")
    print("-" * 62)

    for level, strength in levels:
        # We save each iteration linearly so you can physically listen to the differences
        output_file = f"sound_samples/output/speaker_protected_lvl{level}.wav"
        
        cmd = [
            sys.executable, "protect.py",
            "-i", input_file,
            "-o", output_file,
            "-c", checkpoint,
            "-f", filter_checkpoint,
            "-s", str(strength),
            "--device", "cuda"
        ]
        
        try:
            # Capture the offline terminal output
            result = subprocess.run(cmd, capture_output=True, text=True)
            out = result.stdout
            
            # Print stderr if something heavily crashed
            if result.returncode != 0:
                print(f"Error on Level {level}:\n{result.stderr}")
                continue
                
            # Regex aggressively grab our printed metrics
            snr_match = re.search(r"SNR \(dB\):\s+([\d\.\-]+)", out)
            cos_match = re.search(r"Cosine Sim:\s+([\d\.\-]+)", out)
            pesq_match = re.search(r"PESQ Score:\s+([\d\.\-]+)", out)
            stoi_match = re.search(r"STOI Score:\s+([\d\.\-]+)", out)
            
            # Mapping
            snr = snr_match.group(1) if snr_match else "N/A"
            cos = cos_match.group(1) if cos_match else "N/A"
            pesq = pesq_match.group(1) if pesq_match else "N/A"
            stoi = stoi_match.group(1) if stoi_match else "N/A"
            
            print(f"{level:<6} | {strength:<6.2f} | {snr:<8} | {cos:<6} | {pesq:<5} | {stoi:<5}")
            
        except Exception as e:
            print(f"Failed Level {level}: {str(e)}")

    print("=" * 62)
    print("Testing Complete! Check /sound_samples/output/ for the lvl1 -> lvl5 .wav files.\n")

if __name__ == "__main__":
    run_evaluation()
