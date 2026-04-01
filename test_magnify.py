import torch
import torchaudio

def magnify_perturbation():
    # Load original
    orig, sr_orig = torchaudio.load("speaker.wav")
    
    # Load protected
    prot, sr_prot = torchaudio.load("speaker_protected.wav")
    
    # Ensure same length
    min_len = min(orig.shape[-1], prot.shape[-1])
    orig = orig[:, :min_len]
    prot = prot[:, :min_len]
    
    # Isolate the perturbation
    perturbation = prot - orig
    
    # Multiply the perturbation by a large factor to force XTTS to hear it
    prot_magnified = orig + (perturbation * 20.0)
    
    # Clamp to avoid clipping distortion
    prot_magnified = torch.clamp(prot_magnified, min=-1.0, max=1.0)
    
    # Save
    torchaudio.save("speaker_protected_magnified.wav", prot_magnified, sr_orig)
    print("Saved speaker_protected_magnified.wav with 20x perturbation volume.")

if __name__ == "__main__":
    magnify_perturbation()
