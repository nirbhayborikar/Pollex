"""
SmolVLA Action Expert Head Fine-Tuning with Spatial Reach (X, Z) Priority Weighting.

Warm-starts from the previously calibrated expert checkpoint (outputs/smolvla_glass_expert),
masks out inactive dummy action dimensions (7:31), and heavily up-weights forward reach X (5.0)
and height Z (4.0) over secondary degrees of freedom, teaching the Flow-Matching head
to natively eliminate the ~3cm forward reach shortfall.
"""

import gc
import json
import time
import torch
import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, pad_vector
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.configs.types import PolicyFeature, FeatureType, NormalizationMode


DEFAULT_TASK_PROMPT = (
    "approach the glass on the table and grasp the cylinder and lift the cylinder "
    "and then bring down the cylinder to the table and then withdraw your hands"
)


class GlassDemonstrationDataset(Dataset):
    """Loads recorded RoboSuite episodes with action chunking horizon H=50."""
    def __init__(self, npz_paths: List[str], chunk_size: int = 50):
        self.chunk_size = chunk_size
        self.samples = []

        for p in npz_paths:
            path = Path(p)
            if not path.exists():
                print(f"[!] Warning: Episode file not found: {path}")
                continue

            data = np.load(path, allow_pickle=True)
            images = data["images"]         # (T, 256, 256, 3)
            actions = data["actions"]       # (T, 7)
            states = data["state_7d"]       # (T, 7)
            T = len(images)

            # Build action chunks
            for t in range(T):
                end_t = min(t + self.chunk_size, T)
                act_chunk = actions[t:end_t]

                # Pad chunk to 50 steps if near episode boundary
                if len(act_chunk) < self.chunk_size:
                    pad_len = self.chunk_size - len(act_chunk)
                    last_action = act_chunk[-1:]
                    padding = np.repeat(last_action, pad_len, axis=0)
                    act_chunk = np.vstack([act_chunk, padding])

                # Pad 7D actions to 32D action space
                chunk_32d = np.zeros((self.chunk_size, 32), dtype=np.float32)
                chunk_32d[:, :7] = act_chunk

                self.samples.append({
                    "image": images[t],
                    "state": states[t],
                    "action_chunk": chunk_32d,
                })

        print(f"[+] Loaded dataset from {len(npz_paths)} episodes: {len(self.samples)} total chunk transitions.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_np = item["image"].transpose(2, 0, 1)
        img_tensor = (torch.from_numpy(img_np).float() / 255.0) * 2.0 - 1.0
        st_tensor = torch.from_numpy(item["state"]).float()
        act_chunk = torch.from_numpy(item["action_chunk"]).float()

        return {
            "image": img_tensor,
            "state": st_tensor,
            "action_chunk": act_chunk,
        }


def initialize_weighted_training_policy(
    device: torch.device,
    warmstart_checkpoint: Optional[str] = "outputs/smolvla_glass_expert",
) -> Tuple[SmolVLAPolicy, SmolVLAConfig]:
    """Loads base SmolVLA and warm-starts from previous checkpoint if available."""
    print(f"[*] Initializing SmolVLA on {device}...")
    cfg_path = hf_hub_download(repo_id="lerobot/smolvla_base", filename="config.json")
    with open(cfg_path) as f:
        cfg_dict = json.load(f)

    cfg_dict.pop("type", None)
    cfg_dict["load_vlm_weights"] = False
    cfg_dict["freeze_vision_encoder"] = True
    cfg_dict["train_expert_only"] = True
    cfg_dict["train_state_proj"] = True
    cfg_dict["device"] = str(device)

    input_features = {
        k: PolicyFeature(type=FeatureType[v["type"]], shape=tuple(v["shape"]))
        for k, v in cfg_dict["input_features"].items()
    }
    output_features = {
        k: PolicyFeature(type=FeatureType[v["type"]], shape=tuple(v["shape"]))
        for k, v in cfg_dict["output_features"].items()
    }
    cfg_dict["input_features"] = input_features
    cfg_dict["output_features"] = output_features
    cfg_dict["normalization_mapping"] = {
        k: NormalizationMode[v] for k, v in cfg_dict["normalization_mapping"].items()
    }

    cfg = SmolVLAConfig(**cfg_dict)
    policy = SmolVLAPolicy(cfg)

    # Warm-start weights
    ckpt_file = Path(warmstart_checkpoint) / "model.safetensors" if warmstart_checkpoint else None
    if ckpt_file and ckpt_file.exists():
        print(f"[+] Warm-starting from previous fine-tuned checkpoint: {ckpt_file}...")
        state_dict = load_file(str(ckpt_file))
    else:
        print(f"[*] Warm-starting from base model (lerobot/smolvla_base)...")
        weights_path = hf_hub_download(repo_id="lerobot/smolvla_base", filename="model.safetensors")
        state_dict = load_file(weights_path)

    policy.load_state_dict(state_dict, strict=False)

    # Freeze vision encoder and VLM prefix backbone
    policy.model.vlm_with_expert.freeze_vision_encoder = True
    policy.model.vlm_with_expert.train_expert_only = True
    policy.model.vlm_with_expert.set_requires_grad()
    policy.model.set_requires_grad()

    policy.to(device)
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    print(f"[+] Trainable parameters: {sum(p.numel() for p in trainable_params) / 1e6:.2f} M.")

    return policy, cfg


def train_reach_weighted_expert(
    dataset_dir: str = "data/lerobot_dataset/glass_pick_place",
    warmstart_checkpoint: str = "outputs/smolvla_glass_expert",
    out_dir: str = "outputs/smolvla_glass_expert_reach_weighted",
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 8e-5,
    w_x: float = 5.0,
    w_y: float = 1.0,
    w_z: float = 4.0,
    w_rot: float = 0.5,
    w_grip: float = 2.0,
):
    """Executes fine-tuning with spatial reach and active-dimension loss weighting."""
    dpath = Path(dataset_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("=" * 80)
    print("SmolVLA Spatial Reach & Height Priority Fine-Tuning")
    print(f"Device: {device} | Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    print(f"Weights -> X_reach: {w_x:.1f} | Y: {w_y:.1f} | Z_height: {w_z:.1f} | Rot: {w_rot:.1f} | Grip: {w_grip:.1f}")
    print("=" * 80)

    train_sources = [
        str(dpath / "Vid_0_episode.npz"),
        str(dpath / "Vid_2_episode.npz"),
    ]
    val_sources = [
        str(dpath / "Vid_5_episode.npz"),
    ]

    train_ds = GlassDemonstrationDataset(train_sources)
    val_ds = GlassDemonstrationDataset(val_sources)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    policy, cfg = initialize_weighted_training_policy(device, warmstart_checkpoint=warmstart_checkpoint)

    # Construct loss weight tensor: (32,)
    loss_weights = torch.zeros(32, device=device)
    loss_weights[0] = w_x
    loss_weights[1] = w_y
    loss_weights[2] = w_z
    loss_weights[3:6] = w_rot
    loss_weights[6] = w_grip
    weight_sum = loss_weights.sum()

    print(f"[+] Active Loss Weights (Dims 0:7): {loss_weights[:7].cpu().numpy()}")
    print(f"    Normalized gradient distribution: {np.round((loss_weights[:7] / weight_sum).cpu().numpy() * 100, 1)}%")

    # Tokenize language instruction
    tokens = policy.language_tokenizer(
        [DEFAULT_TASK_PROMPT + "\n"],
        padding="max_length",
        max_length=policy.config.tokenizer_max_length,
        return_tensors="pt",
    )
    lang_tokens_single = tokens["input_ids"].to(device)
    lang_masks_single = tokens["attention_mask"].to(device, dtype=torch.bool)

    # Optimizer & Scheduler
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    best_val_loss = float("inf")
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_loss_x": [],
        "train_loss_z": [],
        "train_loss_grip": [],
        "epoch_times": [],
    }

    t0_start = time.time()

    for ep in range(1, epochs + 1):
        t_ep_start = time.time()
        policy.train()
        train_losses = []
        train_losses_x = []
        train_losses_z = []
        train_losses_grip = []

        for batch in train_loader:
            b_imgs = batch["image"].to(device)
            b_st = batch["state"].to(device)
            b_actions = batch["action_chunk"].to(device)
            B = b_imgs.shape[0]

            img_masks = torch.ones(B, dtype=torch.bool, device=device)
            padded_st = pad_vector(b_st, policy.config.max_state_dim)

            lang_tokens = lang_tokens_single.expand(B, -1)
            lang_masks = lang_masks_single.expand(B, -1)

            optimizer.zero_grad()

            # Forward pass: returns per-element flow-matching squared error: (B, 50, 32)
            loss_tensor = policy.model(
                [b_imgs], [img_masks], lang_tokens, lang_masks, padded_st, b_actions
            )

            # Apply active dimension loss weighting
            weighted_per_step = (loss_tensor * loss_weights).sum(dim=-1) / weight_sum
            loss = weighted_per_step.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            train_losses_x.append(loss_tensor[:, :, 0].mean().item())
            train_losses_z.append(loss_tensor[:, :, 2].mean().item())
            train_losses_grip.append(loss_tensor[:, :, 6].mean().item())

        avg_train_loss = float(np.mean(train_losses))
        avg_loss_x = float(np.mean(train_losses_x))
        avg_loss_z = float(np.mean(train_losses_z))
        avg_loss_grip = float(np.mean(train_losses_grip))

        # Validation pass
        policy.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                b_imgs = batch["image"].to(device)
                b_st = batch["state"].to(device)
                b_actions = batch["action_chunk"].to(device)
                B = b_imgs.shape[0]

                img_masks = torch.ones(B, dtype=torch.bool, device=device)
                padded_st = pad_vector(b_st, policy.config.max_state_dim)
                lang_tokens = lang_tokens_single.expand(B, -1)
                lang_masks = lang_masks_single.expand(B, -1)

                loss_tensor = policy.model(
                    [b_imgs], [img_masks], lang_tokens, lang_masks, padded_st, b_actions
                )
                weighted_val = (loss_tensor * loss_weights).sum(dim=-1) / weight_sum
                val_losses.append(weighted_val.mean().item())

        avg_val_loss = float(np.mean(val_losses))
        ep_duration = time.time() - t_ep_start

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_loss_x"].append(avg_loss_x)
        history["train_loss_z"].append(avg_loss_z)
        history["train_loss_grip"].append(avg_loss_grip)
        history["epoch_times"].append(ep_duration)

        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            # Save best checkpoint
            ckpt_path = out_path / "model.safetensors"
            save_file(policy.state_dict(), str(ckpt_path))

        if ep % 5 == 0 or ep == 1 or ep == epochs:
            best_mark = " [*BEST]" if is_best else ""
            print(
                f"Epoch {ep:2d}/{epochs} | "
                f"Train Loss: {avg_train_loss:.5f} (X: {avg_loss_x:.5f}, Z: {avg_loss_z:.5f}, Grip: {avg_loss_grip:.5f}) | "
                f"Val Loss: {avg_val_loss:.5f} | "
                f"Time: {ep_duration:.1f}s{best_mark}"
            )

        if device.type == "mps":
            torch.mps.empty_cache()
            gc.collect()

    total_duration = time.time() - t0_start
    print(f"\n[+] Fine-tuning complete in {total_duration / 60.0:.2f} minutes.")
    print(f"[+] Best Validation Loss: {best_val_loss:.5f}")

    # Export config and history
    cfg_out = out_path / "config.json"
    with open(cfg_out, "w") as f:
        json.dump({
            "model_type": "smolvla_glass_expert_reach_weighted",
            "warmstart": str(warmstart_checkpoint),
            "epochs": epochs,
            "batch_size": batch_size,
            "final_train_loss": history["train_loss"][-1],
            "best_val_loss": best_val_loss,
            "total_time_s": total_duration,
            "prompt": DEFAULT_TASK_PROMPT,
            "weights": {
                "w_x": w_x,
                "w_y": w_y,
                "w_z": w_z,
                "w_rot": w_rot,
                "w_grip": w_grip,
            }
        }, f, indent=2)

    hist_out = out_path / "training_history.json"
    with open(hist_out, "w") as f:
        json.dump(history, f, indent=2)

    print(f"[+] Saved checkpoint to: {out_path / 'model.safetensors'}")
    print(f"[+] Saved history to: {hist_out}")


def main():
    parser = argparse.ArgumentParser(description="Train reach-weighted SmolVLA Action Expert")
    parser.add_argument("--dataset_dir", type=str, default="data/lerobot_dataset/glass_pick_place")
    parser.add_argument("--warmstart", type=str, default="outputs/smolvla_glass_expert")
    parser.add_argument("--out_dir", type=str, default="outputs/smolvla_glass_expert_reach_weighted")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--w_x", type=float, default=5.0)
    parser.add_argument("--w_y", type=float, default=1.0)
    parser.add_argument("--w_z", type=float, default=4.0)
    parser.add_argument("--w_rot", type=float, default=0.5)
    parser.add_argument("--w_grip", type=float, default=2.0)
    args = parser.parse_args()

    train_reach_weighted_expert(
        dataset_dir=args.dataset_dir,
        warmstart_checkpoint=args.warmstart,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        w_x=args.w_x,
        w_y=args.w_y,
        w_z=args.w_z,
        w_rot=args.w_rot,
        w_grip=args.w_grip,
    )


if __name__ == "__main__":
    main()
