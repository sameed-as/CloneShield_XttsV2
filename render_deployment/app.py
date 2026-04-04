import os
import io
import tempfile
import torch
import torchaudio
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from cloneshield.config import ModelConfig
from cloneshield.model import PerturbationGenerator
from cloneshield.filter_model import FilterNet

app = FastAPI(title="CloneShield API", description="Protect audio against voice cloning.")

# Global state to keep models in memory (CPU)
device = "cpu"
model = None
filter_model = None
config = None

@app.on_event("startup")
def load_models():
    global model, filter_model, config
    print("Loading models into memory...")
    
    ckpt_path = "checkpoints/cloneshield_best.pt"
    f_ckpt_path = "checkpoints/filter_best.pt"
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ModelConfig()
    
    if "config" in ckpt and "epsilon" in ckpt["config"]:
        config.epsilon = ckpt["config"]["epsilon"]
        
    model = PerturbationGenerator(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    if os.path.exists(f_ckpt_path):
        f_ckpt = torch.load(f_ckpt_path, map_location=device, weights_only=False)
        filter_model = FilterNet(config).to(device)
        filter_model.load_state_dict(f_ckpt["filter_state_dict"])
        filter_model.eval()
        print("Perturbation and Filter models loaded successfully.")
    else:
        print("WARNING: Filter model not found!")

@app.post("/protect")
async def protect_endpoint(
    file: UploadFile = File(...),
    filter_strength: float = Form(1.0, ge=0.0, le=1.0)
):
    # Create temp directory for incoming and outgoing
    # Using tempfile ensures isolation and automatic cleanup can be scheduled
    temp_dir = tempfile.mkdtemp()
    in_path = os.path.join(temp_dir, "input.wav")
    out_path = os.path.join(temp_dir, "protected_cleaned.wav")
    
    with open(in_path, "wb") as f:
        f.write(await file.read())
        
    try:
        import soundfile as sf
        import numpy as np
        
        # Load audio strictly with soundfile to bypass torchaudio FFmpeg dependency issues
        audio_data, sr = sf.read(in_path)
        
        if len(audio_data.shape) == 1:
            waveform = torch.from_numpy(audio_data).unsqueeze(0).float()
        else:
            waveform = torch.from_numpy(audio_data).transpose(0, 1).float()
        
        # Resample to model sample rate if needed
        if sr != config.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, config.sample_rate)
            waveform = resampler(waveform)
            sr = config.sample_rate
            
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        # Normalize
        max_amp = waveform.abs().max()
        if max_amp > 0:
            waveform = waveform / max_amp
            
        original_length = waveform.shape[-1]
        segment_len = config.segment_length
        waveform_3d = waveform.unsqueeze(0).to(device)
        
        with torch.no_grad():
            if original_length <= segment_len:
                # Short audio: pad, process, trim
                if original_length < segment_len:
                    padded = torch.nn.functional.pad(waveform_3d, (0, segment_len - original_length))
                else:
                    padded = waveform_3d
                    
                perturbation = model(padded)
                
                if filter_model is not None:
                    mask = filter_model(padded, perturbation)
                    blended_mask = 1.0 - filter_strength * (1.0 - mask)
                    perturbation = perturbation * blended_mask
                
                protected = padded + perturbation
                protected = protected[:, :, :original_length]
            else:
                # Long audio: process in non-overlapping chunks
                hop = segment_len
                protected_chunks = []
                
                for start in range(0, original_length, hop):
                    end = min(start + segment_len, original_length)
                    chunk = waveform_3d[:, :, start:end]
                    
                    if chunk.shape[-1] < segment_len:
                        chunk = torch.nn.functional.pad(chunk, (0, segment_len - chunk.shape[-1]))
                        
                    perturbation = model(chunk)
                    
                    if filter_model is not None:
                        mask = filter_model(chunk, perturbation)
                        blended_mask = 1.0 - filter_strength * (1.0 - mask)
                        perturbation = perturbation * blended_mask
                        
                    prot_chunk = chunk + perturbation
                    prot_chunk = prot_chunk[:, :, :end - start]
                    protected_chunks.append((start, end, prot_chunk))
                    
                protected = torch.zeros_like(waveform_3d)
                weight_sum = torch.zeros_like(waveform_3d)
                for start, end, chunk in protected_chunks:
                    protected[:, :, start:end] += chunk
                    weight_sum[:, :, start:end] += 1.0
                protected = protected / weight_sum.clamp(min=1.0)
                
        # Restore original amplitude
        protected = protected.squeeze(0) * max_amp
        
        # Save output to temp path
        audio_out = protected.cpu()
        if audio_out.shape[0] == 1:
            audio_out = audio_out.squeeze(0).numpy()
        else:
            audio_out = audio_out.transpose(0, 1).numpy()
            
        sf.write(out_path, audio_out, sr, subtype='PCM_16')
        
        # Cleanup routine
        def cleanup():
            try:
                if os.path.exists(in_path): os.remove(in_path)
                if os.path.exists(out_path): os.remove(out_path)
                if os.path.exists(temp_dir): os.rmdir(temp_dir)
            except Exception as e:
                print(f"Cleanup error: {e}")
            
        try:
            from pesq import pesq
            # Resample both tensors to 16k temporarily for standard scoring
            resampler_16k = torchaudio.transforms.Resample(config.sample_rate, 16000)
            orig_16k = resampler_16k(waveform.cpu()).squeeze(0).numpy()
            prot_16k = resampler_16k(protected.cpu()).squeeze(0).numpy()
            
            # Extract mono tracks solely for PESQ
            if len(orig_16k.shape) > 1: orig_16k = orig_16k[0]
            if len(prot_16k.shape) > 1: prot_16k = prot_16k[0]
                
            pesq_score = pesq(16000, orig_16k, prot_16k, 'wb')
        except Exception as e:
            print(f"Scoring framework exception: {e}")
            pesq_score = 0.0
            
        # Natively calculate PyTorch SNR and Perturbation metrics (Zero extra memory)
        try:
            # We scale the waveform back to its original amplitude for accurate comparison
            original_scaled = waveform.cpu() * max_amp
            delta = protected.cpu() - original_scaled
            
            snr_tensor = 10 * torch.log10(original_scaled.pow(2).mean() / delta.pow(2).mean().clamp(min=1e-10))
            snr_score = float(snr_tensor.item())
            max_perturbation = float(delta.abs().max().item())
        except Exception as e:
            print(f"PyTorch metric error: {e}")
            snr_score = 0.0
            max_perturbation = 0.0
            
        import base64
        with open(out_path, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode("utf-8")
            
        # Clean up files manually since we are not using BackgroundTask anymore
        try:
            if os.path.exists(in_path): os.remove(in_path)
            if os.path.exists(out_path): os.remove(out_path)
            if os.path.exists(temp_dir): os.rmdir(temp_dir)
        except Exception as e:
            print(f"Cleanup error: {e}")
            
        return {
            "pesq_score": round(float(pesq_score), 3),
            "snr_score": round(float(snr_score), 3),
            "max_perturbation": round(float(max_perturbation), 5),
            "audio_base64": encoded_string
        }
        
    except Exception as e:
        if os.path.exists(in_path): os.remove(in_path)
        if os.path.exists(out_path): os.remove(out_path)
        if os.path.exists(temp_dir): os.rmdir(temp_dir)
        return {"error": str(e)}

@app.post("/clone")
async def proxy_clone(
    speaker: UploadFile = File(...),
    text: str = Form("Hello, this is a test clone.", description="Text for the voice clone")
):
    import requests
    ngrok_url = os.environ.get("NGROK_URL")
    if not ngrok_url:
        return {"error": "NGROK_URL environment variable is not set. Please set it in the Render Dashboard."}
        
    ngrok_url = ngrok_url.rstrip("/")
    target_endpoint = f"{ngrok_url}/clone"
    
    temp_dir = tempfile.mkdtemp()
    in_path = os.path.join(temp_dir, "speaker.wav")
    out_path = os.path.join(temp_dir, "cloned_output.wav")
    
    # Save the incoming file to temp
    with open(in_path, "wb") as f:
        f.write(await speaker.read())
        
    try:
        # Construct a multipart payload to send back to the user's Desktop PC
        with open(in_path, "rb") as f:
            files = {"speaker": (speaker.filename or "speaker.wav", f, "audio/wav")}
            data = {"text": text}
            
            # Ngrok free tier securely blocks automated requests unless this specific bypass header is included
            headers = {"ngrok-skip-browser-warning": "true"}
            
            # Proxy the audio to the Desktop PC over ngrok
            response = requests.post(target_endpoint, files=files, data=data, headers=headers)
            
        if response.status_code != 200:
            return {"error": f"Local PC returned Error {response.status_code}: {response.text}"}
            
        # Secure the returned WAV file from the PC
        with open(out_path, "wb") as out_f:
            out_f.write(response.content)
            
        import base64
        with open(out_path, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode("utf-8")
            
        def cleanup():
            pass # BackgroundTask not used here anymore, run manual cleanup
            
        try:
            if os.path.exists(in_path): os.remove(in_path)
            if os.path.exists(out_path): os.remove(out_path)
            if os.path.exists(temp_dir): os.rmdir(temp_dir)
        except Exception as e:
            print(f"Cleanup error: {e}")
            
        return {
            "audio_base64": encoded_string
        }
        
    except Exception as e:
        if os.path.exists(in_path): os.remove(in_path)
        if os.path.exists(out_path): os.remove(out_path)
        if os.path.exists(temp_dir): os.rmdir(temp_dir)
        return {"error": f"Failed reaching Desktop PC: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    # Make sure we bind directly to 0.0.0.0 and dynamically assign the PORT variable as expected by Render
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
