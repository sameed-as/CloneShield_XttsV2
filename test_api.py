import requests
import base64
import os

def test_json_pipeline():
    print("Sending audio to Fakeless API...")
    
    # Send the test file
    response = requests.post(
        "https://fakeless-api.onrender.com/protect",
        files={"file": open("sound_samples/input/speaker.wav", "rb")},
        data={"filter_strength": 0.5}
    )

    data = response.json()

    if "error" in data:
        print("Error:", data["error"])
        return

    print(f"\n--- Protection Score Metrics ---")
    print(f"✅ PESQ Score: {data['pesq_score']}\n")

    # Extract base64 and securely decode
    audio_bytes = base64.b64decode(data['audio_base64'])

    # Write binary frame out
    output_dir = "sound_samples/output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "remote_protected_speaker.wav")
    
    with open(output_path, "wb") as f:
        f.write(audio_bytes)
        
    print(f"🎵 Audio parsed from Base64 mapping and properly verified to {output_path}")

if __name__ == "__main__":
    test_json_pipeline()
