"""
CloneShield Training Script

Trains the PerturbationGenerator to produce adversarial perturbations
that degrade XttsV2 voice cloning while preserving audio quality.

Usage:
    python train.py [--epochs N] [--batch-size N] [--lr LR] [--device DEVICE]
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from cloneshield.config import TrainConfig
from cloneshield.model import PerturbationGenerator
from cloneshield.losses import CloneShieldLoss
from cloneshield.dataset import LibriSpeechDataset
from cloneshield.speaker_encoder import SpeakerEncoder


def train_one_epoch(model, speaker_encoder, dataloader, loss_fn,
                    optimizer, device, epoch, config):
    """Train for one epoch."""
    model.train()
    total_losses = {
        "total_loss": 0.0,
        "quality_total": 0.0,
        "anti_clone_loss": 0.0,
        "speaker_similarity": 0.0,
        "gpt_similarity": 0.0,
    }
    num_batches = 0

    for step, batch in enumerate(dataloader):
        audio = batch["audio"].to(device)  # (B, 1, T)

        # Generate perturbation and protected audio
        perturbation = model(audio)
        protected = audio + perturbation

        # Extract speaker embeddings + GPT latents (frozen encoder, gradients flow through audio)
        with torch.no_grad():
            enc_orig = speaker_encoder(audio, sample_rate=config.model.sample_rate)

        # This one needs gradients to flow through protected audio
        enc_prot = speaker_encoder(protected, sample_rate=config.model.sample_rate)

        emb_original = enc_orig["speaker_embedding"]
        emb_protected = enc_prot["speaker_embedding"]
        gpt_orig = enc_orig.get("gpt_latents")
        gpt_prot = enc_prot.get("gpt_latents")

        # Compute combined loss
        losses = loss_fn(audio, protected, emb_original, emb_protected, gpt_orig, gpt_prot)

        # Backward pass
        optimizer.zero_grad()
        losses["total_loss"].backward()

        # Gradient clipping
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        optimizer.step()

        # Accumulate losses
        for key in total_losses:
            if key in losses:
                val = losses[key]
                total_losses[key] += val.item() if isinstance(val, torch.Tensor) else val
        num_batches += 1

        # Logging
        if (step + 1) % config.log_every == 0:
            avg_loss = total_losses["total_loss"] / num_batches
            avg_sim = total_losses["speaker_similarity"] / num_batches
            max_pert = perturbation.abs().max().item()
            gpt_sim_str = ""
            if "gpt_similarity" in total_losses:
                avg_gpt = total_losses["gpt_similarity"] / num_batches
                gpt_sim_str = f" | GPT Sim: {avg_gpt:.4f}"
            print(
                f"  [Epoch {epoch+1}, Step {step+1}/{len(dataloader)}] "
                f"Loss: {avg_loss:.4f} | "
                f"Speaker Sim: {avg_sim:.4f}"
                + gpt_sim_str +
                f" | Max |d|: {max_pert:.4f}"
            )

    # Average losses
    for key in total_losses:
        total_losses[key] /= max(num_batches, 1)

    return total_losses


@torch.no_grad()
def validate(model, speaker_encoder, dataloader, loss_fn, device, config):
    """Validate on dev set."""
    model.eval()
    total_losses = {
        "total_loss": 0.0,
        "quality_total": 0.0,
        "anti_clone_loss": 0.0,
        "speaker_similarity": 0.0,
        "gpt_similarity": 0.0,
    }
    num_batches = 0

    for batch in dataloader:
        audio = batch["audio"].to(device)

        perturbation = model(audio)
        protected = audio + perturbation

        enc_orig = speaker_encoder(audio, sample_rate=config.model.sample_rate)
        enc_prot = speaker_encoder(protected, sample_rate=config.model.sample_rate)

        emb_original = enc_orig["speaker_embedding"]
        emb_protected = enc_prot["speaker_embedding"]
        gpt_orig = enc_orig.get("gpt_latents")
        gpt_prot = enc_prot.get("gpt_latents")

        losses = loss_fn(audio, protected, emb_original, emb_protected, gpt_orig, gpt_prot)

        for key in total_losses:
            if key in losses:
                val = losses[key]
                total_losses[key] += val.item() if isinstance(val, torch.Tensor) else val
        num_batches += 1

    for key in total_losses:
        total_losses[key] /= max(num_batches, 1)

    return total_losses


def main():
    parser = argparse.ArgumentParser(description="Train CloneShield voice protection model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto/cuda/cpu")
    parser.add_argument("--data-root", type=str, default="Small_LibriSpeech_Prepared",
                        help="Data root directory")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--lambda-quality", type=float, default=None,
                        help="Quality loss weight")
    parser.add_argument("--lambda-anti-clone", type=float, default=None,
                        help="Anti-cloning loss weight")
    parser.add_argument("--epsilon", type=float, default=None,
                        help="Perturbation budget")
    args = parser.parse_args()

    # Build config
    config = TrainConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        data_root=args.data_root,
        checkpoint_dir=args.checkpoint_dir,
    )
    if args.epsilon is not None:
        config.model.epsilon = args.epsilon
    if args.lambda_quality is not None:
        config.loss.lambda_quality = args.lambda_quality
    if args.lambda_anti_clone is not None:
        config.loss.lambda_anti_clone = args.lambda_anti_clone

    device = config.get_device()
    print(f"Using device: {device}")

    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # ==== 1. Load Dataset ====
    print("\n=== Loading Dataset ===")
    train_dataset = LibriSpeechDataset(
        config.data_root, config.train_manifest, config.model
    )
    val_dataset = LibriSpeechDataset(
        config.data_root, config.val_manifest, config.model
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=0,
    )
    print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Val:   {len(val_dataset)} samples, {len(val_loader)} batches")

    # ==== 2. Build Models ====
    print("\n=== Building Models ===")
    model = PerturbationGenerator(config.model).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"PerturbationGenerator: {param_count:,} parameters")

    speaker_encoder = SpeakerEncoder(device)
    # Trigger lazy load
    print("Loading XttsV2 speaker encoder (this may take a moment)...")
    _ = speaker_encoder.model

    # ==== 3. Loss & Optimizer ====
    loss_fn = CloneShieldLoss(config.loss).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs, eta_min=1e-6,
    )

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    # ==== 4. Training Loop ====
    print(f"\n=== Starting Training ===")
    print(f"Epochs: {config.num_epochs}, Batch size: {config.batch_size}")
    print(f"L_quality: {config.loss.lambda_quality}, L_anti_clone: {config.loss.lambda_anti_clone}")
    print(f"Epsilon (perturbation budget): {config.model.epsilon}")
    print("=" * 60)

    best_val_loss = float("inf")

    for epoch in range(start_epoch, config.num_epochs):
        t0 = time.time()

        # Train
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        train_losses = train_one_epoch(
            model, speaker_encoder, train_loader, loss_fn,
            optimizer, device, epoch, config,
        )

        # Validate
        val_losses = validate(
            model, speaker_encoder, val_loader, loss_fn, device, config,
        )

        scheduler.step()
        elapsed = time.time() - t0

        # Print epoch summary
        print(
            f"  TRAIN | Loss: {train_losses['total_loss']:.4f} | "
            f"Quality: {train_losses['quality_total']:.4f} | "
            f"Spk Sim: {train_losses['speaker_similarity']:.4f} | "
            f"GPT Sim: {train_losses['gpt_similarity']:.4f}"
        )
        print(
            f"  VAL   | Loss: {val_losses['total_loss']:.4f} | "
            f"Quality: {val_losses['quality_total']:.4f} | "
            f"Spk Sim: {val_losses['speaker_similarity']:.4f} | "
            f"GPT Sim: {val_losses['gpt_similarity']:.4f}"
        )
        print(f"  Time: {elapsed:.1f}s | LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save checkpoint
        if (epoch + 1) % config.save_every == 0 or val_losses["total_loss"] < best_val_loss:
            ckpt_path = os.path.join(
                config.checkpoint_dir, f"cloneshield_epoch{epoch+1:03d}.pt"
            )
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "config": {
                    "epsilon": config.model.epsilon,
                    "lambda_quality": config.loss.lambda_quality,
                    "lambda_anti_clone": config.loss.lambda_anti_clone,
                },
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

            if val_losses["total_loss"] < best_val_loss:
                best_val_loss = val_losses["total_loss"]
                best_path = os.path.join(config.checkpoint_dir, "cloneshield_best.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "epsilon": config.model.epsilon,
                        "lambda_quality": config.loss.lambda_quality,
                        "lambda_anti_clone": config.loss.lambda_anti_clone,
                    },
                }, best_path)
                print(f"  * New best model saved!")

    print("\n=== Training Complete ===")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model: {os.path.join(config.checkpoint_dir, 'cloneshield_best.pt')}")


if __name__ == "__main__":
    main()
