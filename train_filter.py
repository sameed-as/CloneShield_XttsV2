import os
import argparse
import time
import torch
from torch.utils.data import DataLoader

from cloneshield.dataset import LibriSpeechDataset
from cloneshield.model import PerturbationGenerator
from cloneshield.filter_model import FilterNet 
from cloneshield.config import TrainConfig
from cloneshield.speaker_encoder import SpeakerEncoder
from cloneshield.losses import CloneShieldLoss

def train_one_epoch(model_gen, model_filter, speaker_encoder, dataloader, optimizer, loss_fn, device, config):
    model_gen.eval()
    model_filter.train()
    
    total_losses = {
        "total_loss": 0.0,
        "quality_total": 0.0,
        "anti_clone_loss": 0.0,
        "speaker_similarity": 0.0,
        "gpt_similarity": 0.0,
        "mask_mean": 0.0,
    }
    num_batches = 0

    for batch in dataloader:
        audio = batch["audio"].to(device)
        optimizer.zero_grad()
        
        with torch.no_grad():
            raw_noise = model_gen(audio)
            
        mask = model_filter(audio, raw_noise)
        refined_noise = raw_noise * mask
        protected = audio + refined_noise
        
        enc_orig = speaker_encoder(audio, sample_rate=config.model.sample_rate)
        enc_prot = speaker_encoder(protected, sample_rate=config.model.sample_rate)

        emb_original = enc_orig["speaker_embedding"]
        emb_protected = enc_prot["speaker_embedding"]
        gpt_orig = enc_orig.get("gpt_latents")
        gpt_prot = enc_prot.get("gpt_latents")

        losses = loss_fn(audio, protected, emb_original, emb_protected, gpt_orig, gpt_prot)
        
        # Explicitly penalize the mask area to force sparsity and prune unnecessary noise
        mask_penalty = 10.0 * mask.mean()
        loss = losses["total_loss"] + mask_penalty
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model_filter.parameters(), config.grad_clip)
        optimizer.step()

        for key in ["total_loss", "quality_total", "anti_clone_loss", "speaker_similarity", "gpt_similarity"]:
            if key in losses:
                val = losses[key]
                total_losses[key] += val.item() if isinstance(val, torch.Tensor) else val
                
        total_losses["mask_mean"] += mask.mean().item()
        num_batches += 1
        
        if num_batches % config.log_every == 0:
            print(f"  [Step {num_batches}/{len(dataloader)}] Loss: {loss.item():.4f} | Spk Sim: {losses['speaker_similarity']:.4f} | Mask Mean: {mask.mean().item():.4f}")

    for key in total_losses:
        total_losses[key] /= max(num_batches, 1)

    return total_losses

@torch.no_grad()
def validate(model_gen, model_filter, speaker_encoder, dataloader, loss_fn, device, config):
    model_gen.eval()
    model_filter.eval()
    total_losses = {
        "total_loss": 0.0,
        "quality_total": 0.0,
        "anti_clone_loss": 0.0,
        "speaker_similarity": 0.0,
        "gpt_similarity": 0.0,
        "mask_mean": 0.0,
    }
    num_batches = 0

    for batch in dataloader:
        audio = batch["audio"].to(device)
        raw_noise = model_gen(audio)
        mask = model_filter(audio, raw_noise)
        refined_noise = raw_noise * mask
        protected = audio + refined_noise

        enc_orig = speaker_encoder(audio, sample_rate=config.model.sample_rate)
        enc_prot = speaker_encoder(protected, sample_rate=config.model.sample_rate)

        emb_original = enc_orig["speaker_embedding"]
        emb_protected = enc_prot["speaker_embedding"]
        gpt_orig = enc_orig.get("gpt_latents")
        gpt_prot = enc_prot.get("gpt_latents")

        losses = loss_fn(audio, protected, emb_original, emb_protected, gpt_orig, gpt_prot)
        mask_penalty = 10.0 * mask.mean()
        losses["total_loss"] += mask_penalty
        
        for key in ["total_loss", "quality_total", "anti_clone_loss", "speaker_similarity", "gpt_similarity"]:
            if key in losses:
                val = losses[key]
                total_losses[key] += val.item() if isinstance(val, torch.Tensor) else val
        total_losses["mask_mean"] += mask.mean().item()
        num_batches += 1

    for key in total_losses:
        total_losses[key] /= max(num_batches, 1)

    return total_losses

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data-root", type=str, default="Small_LibriSpeech_Prepared")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_filter")
    parser.add_argument("--generator-ckpt", type=str, default="checkpoints/cloneshield_best.pt")
    args = parser.parse_args()

    config = TrainConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        data_root=args.data_root,
        checkpoint_dir=args.checkpoint_dir,
    )
    device = config.get_device()
    print(f"Using device: {device}")
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    print("\n=== Dataset ===")
    train_dataset = LibriSpeechDataset(config.data_root, config.train_manifest, config.model)
    val_dataset = LibriSpeechDataset(config.data_root, config.val_manifest, config.model)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    print("\n=== Models ===")
    model_gen = PerturbationGenerator(config.model).to(device)
    # Load frozen generator
    if os.path.exists(args.generator_ckpt):
        print(f"Loading base generator from {args.generator_ckpt}")
        ckpt_gen = torch.load(args.generator_ckpt, map_location=device)
        model_gen.load_state_dict(ckpt_gen["model_state_dict"])
    else:
        print(f"Base generator checkpoint {args.generator_ckpt} not found! Cannot train filter.")
        return
        
    model_gen.eval()
    for p in model_gen.parameters():
        p.requires_grad = False

    model_filter = FilterNet(config.model).to(device)
    
    # Load Xtts setup before printing anything to avoid output clutter
    _ = SpeakerEncoder(device)
    speaker_encoder = SpeakerEncoder(device)

    loss_fn = CloneShieldLoss(config.loss).to(device)
    optimizer = torch.optim.AdamW(model_filter.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=1e-6)

    best_val_loss = float("inf")

    print(f"\n=== Training FilterNet ===")
    for epoch in range(config.num_epochs):
        t0 = time.time()
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        
        train_losses = train_one_epoch(model_gen, model_filter, speaker_encoder, train_loader, optimizer, loss_fn, device, config)
        val_losses = validate(model_gen, model_filter, speaker_encoder, val_loader, loss_fn, device, config)
        
        scheduler.step()
        
        print(f"  TRAIN | Loss: {train_losses['total_loss']:.4f} | Mask: {train_losses['mask_mean']:.4f} | Spk Sim: {train_losses['speaker_similarity']:.4f}")
        print(f"  VAL   | Loss: {val_losses['total_loss']:.4f} | Mask: {val_losses['mask_mean']:.4f} | Spk Sim: {val_losses['speaker_similarity']:.4f}")
        print(f"  Time: {time.time()-t0:.1f}s | LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save latest
        torch.save({
            "epoch": epoch,
            "filter_state_dict": model_filter.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_losses["total_loss"],
        }, os.path.join(config.checkpoint_dir, f"filter_epoch{epoch+1:03d}.pt"))

        # Save best
        if val_losses["total_loss"] < best_val_loss:
            best_val_loss = val_losses["total_loss"]
            torch.save({
                "epoch": epoch,
                "filter_state_dict": model_filter.state_dict(),
                "val_loss": best_val_loss,
            }, os.path.join(config.checkpoint_dir, "filter_best.pt"))
            print("  * New best filter model saved!")

if __name__ == "__main__":
    main()
