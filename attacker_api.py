import os
import tempfile
import torch
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from TTS.api import TTS

app = FastAPI(title="Local XTTS Attacker API", description="Run XTTS natively for CloneShield verification.")

# Load Model
tts = None
device = "cuda" if torch.cuda.is_available() else "cpu"

@app.on_event("startup")
def load_models():
    global tts
    print(f"Loading XTTS v2 into {device}...")
    try:
        # Prevent TTS telemtry from halting the boot process
        os.environ["COQUI_TOS_AGREED"] = "1"
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        print("XTTS v2 is loaded and ready.")
    except Exception as e:
        print(f"Fatal failed to load TTS: {e}")

@app.post("/clone")
async def clone_endpoint(
    speaker: UploadFile = File(...),
    text: str = Form("Hello, this is a test clone.", description="Text for the voice clone")
):
    temp_dir = tempfile.mkdtemp()
    in_path = os.path.join(temp_dir, "speaker.wav")
    out_path = os.path.join(temp_dir, "cloned_output.wav")
    
    with open(in_path, "wb") as f:
        f.write(await speaker.read())
        
    try:
        if tts is None:
            return {"error": "TTS engine not loaded properly."}
            
        # Run inference natively on your GPU
        tts.tts_to_file(
            text=text,
            speaker_wav=in_path,
            language="en",
            file_path=out_path
        )
        
        # Cleanup routine
        def cleanup():
            try:
                if os.path.exists(in_path): os.remove(in_path)
                if os.path.exists(out_path): os.remove(out_path)
                if os.path.exists(temp_dir): os.rmdir(temp_dir)
            except Exception as e:
                print(f"Cleanup error: {e}")
                
        return FileResponse(
            out_path, 
            media_type="audio/wav", 
            filename="cloned_audio.wav",
            background=BackgroundTask(cleanup)
        )
        
    except Exception as e:
        if os.path.exists(in_path): os.remove(in_path)
        if os.path.exists(out_path): os.remove(out_path)
        if os.path.exists(temp_dir): os.rmdir(temp_dir)
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    # Make sure we bind directly to 127.0.0.1 for secure local use
    uvicorn.run("attacker_api:app", host="127.0.0.1", port=8000)
