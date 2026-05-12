# CloneShield - GPU Run Guide

## Prerequisites

- **Python**: 3.10.x (tested with 3.10.11)
- **GPU**: NVIDIA GPU with CUDA support (RTX 3060+ recommended, 8GB+ VRAM)
- **CUDA**: 12.1 or 12.4 (driver 530+)
- **OS**: Windows 10/11 or Linux (Ubuntu 20.04+)

---

## Step 1: Clone / Copy the Project

Copy the entire `xtts_project` folder to the GPU machine. The folder structure should look like:

```
xtts_project/
    cloneshield/
        __init__.py
        config.py
        model.py
        losses.py
        dataset.py
        speaker_encoder.py
    Small_LibriSpeech_Prepared/
        train-clean-100/
        dev-clean/
        test-clean/
        train-clean-100_manifest.txt
        dev-clean_manifest.txt
        test-clean_manifest.txt
    train.py
    protect.py
    evaluate.py
    testXttsv2.py
    requirements.txt
    gpu_run_guide.md
```

---

## Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3.10 -m venv venv
source venv/bin/activate
```

---

## Step 3: Install PyTorch with CUDA (FIRST)

Install PyTorch with GPU support BEFORE other packages:

```bash
# CUDA 12.1 (recommended)
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4 (alternative)
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
```

Verify GPU is detected:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX XXXX
```

---

## Step 4: Install Remaining Packages

```bash
pip install -r requirements.txt
```

This will install TTS (Coqui), librosa, transformers, and all other dependencies with locked versions.

---

## Step 5: Verify Setup

Run this quick check to verify everything works:

```bash
python -c "from cloneshield.model import PerturbationGenerator; import torch; m = PerturbationGenerator().cuda(); x = torch.randn(1,1,66150).cuda(); d = m(x); print(f'Model OK on GPU | Output shape: {d.shape} | Max perturbation: {d.abs().max():.4f}')"
```

Expected output:
```
Model OK on GPU | Output shape: torch.Size([1, 1, 66150]) | Max perturbation: 0.01xx
```

---

## Step 6: Train the Model

### Basic Training (recommended starting point)

```bash
python train.py --epochs 50 --batch-size 8 --device cuda
```

if you want to train on your own dataset
```bash
python train.py --epochs 50 --batch-size 8 --device cuda --data-root C:\Users\DotNet\Desktop\FYP\FYP-FakeLess\data\librispeech_prepared
```

This will:
- Load 4,280 training audio files from 251 speakers
- Train the perturbation generator for 50 epochs
- Save checkpoints to `checkpoints/` every 5 epochs
- Save the best model as `checkpoints/cloneshield_best.pt`

### Training with Custom Hyperparameters

```bash
# Stronger protection (may slightly reduce audio quality)
python train.py --epochs 100 --batch-size 8 --device cuda --lambda-anti-clone 1.0

# Larger perturbation budget (more aggressive protection)
python train.py --epochs 50 --batch-size 8 --device cuda --epsilon 0.05

# Higher learning rate + more epochs
python train.py --epochs 100 --batch-size 16 --lr 2e-4 --device cuda

# Resume from a checkpoint
python train.py --epochs 100 --batch-size 8 --device cuda --resume checkpoints/cloneshield_epoch050.pt
```

### What to Watch During Training

| Metric             | Good Sign                         | Bad Sign                 |
|--------------------|-----------------------------------|--------------------------|
| Speaker Similarity | Decreasing toward 0.0 or negative | Staying above 0.8        |
| Quality Loss       | Low and stable (< 0.1)            | Increasing significantly |
| Total Loss         | Decreasing                        | Exploding or NaN         |

Expected training time (approximate):

| GPU | Batch Size | Time per Epoch | Total (50 epochs) |
|---|---|---|---|
| RTX 3060 (12GB) | 8 | ~10 min | ~8 hours |
| RTX 3080 (10GB) | 8 | ~7 min | ~6 hours |
| RTX 4090 (24GB) | 16 | ~4 min | ~3 hours |
| A100 (80GB) | 32 | ~2 min | ~1.5 hours |

---

## Step 7: Protect Audio Files

After training, protect any WAV file:

```bash
python protect.py --input sound_samples/input/speaker.wav --output sound_samples/input/speaker_protected.wav --checkpoint checkpoints/cloneshield_best.pt --filter-checkpoint checkpoints_filter/filter_best.pt --filter-strength 1.0 --device cuda
```

Output will show:
- Perturbation SNR (higher = less audible, target: > 30 dB)
- Max perturbation amplitude

---

## Step 8: Evaluate the Model

### Metrics Only (fast)

```bash
python evaluate.py --checkpoint checkpoints/cloneshield_best.pt --device cuda --num-samples 50
```

### Full Evaluation with Cloning A/B Comparison

```bash
python evaluate.py --checkpoint checkpoints/cloneshield_best.pt --device cuda --clone
```

This creates 4 audio files in `eval_output/`:
- `original_sample.wav` - Original speaker audio
- `protected_sample.wav` - Protected audio (should sound identical to original)
- `clone_from_original.wav` - XttsV2 clone from original (should sound like speaker)
- `clone_from_protected.wav` - XttsV2 clone from protected (should sound degraded/different)

**Listen to all 4 files and compare!**

---

## Troubleshooting

### "CUDA out of memory"
Reduce batch size:
```bash
python train.py --batch-size 4 --device cuda
# or even smaller:
python train.py --batch-size 2 --device cuda
```

### "XttsV2 model not found"
The XttsV2 model will be auto-downloaded on first run (~1.8GB). Make sure you have internet access.

### "ModuleNotFoundError: No module named 'cloneshield'"
Make sure you're running from the `xtts_project/` directory:
```bash
cd xtts_project
python train.py ...
```

### Training loss is NaN
Try lowering the learning rate:
```bash
python train.py --lr 5e-5 --device cuda
```

---

## Project Overview

CloneShield adds tiny, inaudible perturbations to audio that disrupt XttsV2's speaker embedding extraction:

```
Normal audio  --> XttsV2 --> Accurate voice clone
Protected audio --> XttsV2 --> Degraded / different voice clone
```

The perturbation generator (U-Net, 3.5M parameters) is trained with two objectives:
1. **Keep audio quality** (mel-spectrogram + STFT + L1 loss)
2. **Break speaker cloning** (negative cosine similarity on speaker embeddings)
