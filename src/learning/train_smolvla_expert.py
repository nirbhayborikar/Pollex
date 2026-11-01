"""
Fine-Tune the SmolVLA Flow-Matching Action Expert on the validated
Iteration 10 GlassLift expert demonstration.

Dataset:
    RGB observations: (340, 256, 256, 3)
    Panda state:      (340, 8)
    Expert actions:   (340, 7)

The SigLIP/VLM backbone is frozen. The action expert and state projection
are trained using 50-step action chunks.

Output:
    outputs/smolvla_glass_expert_iteration10/
"""

import os
import gc
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, pad_vector
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.configs.types import PolicyFeature, FeatureType, NormalizationMode
from transformers import AutoTokenizer



DEFAULT_TASK_PROMPT = (
    "approach the glass on the table and grasp the cylinder and lift the cylinder "
    "and then bring down the cylinder to the table and then withdraw your hands"
)


class GlassDemonstrationDataset(Dataset):
    """
    PyTorch Dataset yielding synchronized (image, state, task, action_chunk) tuples.
    """

    def __init__(
        self,
        npz_paths: List[str],
        chunk_size: int = 50,
        max_action_dim: int = 32,
        task_prompt: str = DEFAULT_TASK_PROMPT,
    ):
        self.chunk_size = chunk_size
        self.max_action_dim = max_action_dim
        self.task_prompt = task_prompt

        self.samples = []
        for p in npz_paths:
            path = Path(p)
            if not path.exists():
                continue

            data = np.load(str(path), allow_pickle=False)

            images = np.asarray(data["images"], dtype=np.uint8)
            actions = np.asarray(data["actions"], dtype=np.float32)
            state_8d = np.asarray(data["state_8d"], dtype=np.float32)

            T = len(actions)

            if len(images) != T or len(state_8d) != T:
                raise ValueError(
                    f"Episode alignment error in {path}: "
                    f"images={len(images)}, states={len(state_8d)}, actions={T}"
                )

            if actions.shape != (T, 7):
                raise ValueError(f"Expected actions (T, 7), got {actions.shape}")

            if state_8d.shape != (T, 8):
                raise ValueError(f"Expected states (T, 8), got {state_8d.shape}")

            if not np.isfinite(actions).all():
                raise ValueError(f"Non-finite actions in {path}")

            if not np.isfinite(state_8d).all():
                raise ValueError(f"Non-finite states in {path}")

            for t in range(T):
                # Build target action chunk: [chunk_size, max_action_dim]
                chunk = np.zeros((chunk_size, max_action_dim), dtype=np.float32)
                future_len = min(chunk_size, T - t)
                chunk[:future_len, :7] = actions[t : t + future_len]
                # Repeat last action if sequence ends before chunk horizon
                if future_len < chunk_size:
                    chunk[future_len:, :7] = actions[t + future_len - 1]

                self.samples.append({
                    "image": images[t],
                    "state": state_8d[t],
                    "action_chunk": chunk,
                })

        print(f"[*] Loaded dataset with {len(self.samples)} transitions across {len(npz_paths)} episodes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        # Transpose image to (3, 256, 256) and normalize [-1, 1]
        img_np = item["image"].transpose(2, 0, 1)
        img_tensor = (torch.from_numpy(img_np).float() / 255.0) * 2.0 - 1.0

        st_tensor = torch.from_numpy(item["state"]).float()
        act_chunk = torch.from_numpy(item["action_chunk"]).float()

        return {
            "image": img_tensor,
            "state": st_tensor,
            "action_chunk": act_chunk,
        }


def initialize_smolvla_training_policy(device: torch.device) -> Tuple[SmolVLAPolicy, SmolVLAConfig]:
    """Loads base SmolVLA and sets requires_grad=True only on action expert head."""
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
    print(f"[+] Loaded model. Trainable parameters: {sum(p.numel() for p in trainable_params) / 1e6:.2f} M.")

    return policy, cfg


def train_expert(
    dataset_dir: str = "datasets/lerobot_dataset/glass_pick_place",
    out_dir: str = "outputs/smolvla_glass_expert_iteration10",
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-4,
):
    """Executes fine-tuning of the flow-matching action expert head."""
    dpath = Path(dataset_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        print(f"[+] CUDA device: {torch.cuda.get_device_name(0)}")

    print("=" * 80)
    print("SmolVLA Action Expert Head Fine-Tuning")
    print(f"Device: {device} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print("=" * 80)

    train_sources = [str(dpath / "cup_lifting_01_episode.npz")]
    train_ds = GlassDemonstrationDataset(train_sources)

    if len(train_ds) == 0:
        raise RuntimeError("No SmolVLA training samples were loaded.")
    train_loader = DataLoader(train_ds,batch_size=batch_size,shuffle=True,drop_last=False)
    val_loader = None

    policy, cfg = initialize_smolvla_training_policy(device)

    # Tokenize instruction once for batch
    # SmolVLA uses the SmolVLM tokenizer for language conditioning.
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")

    tokens = tokenizer(
        [DEFAULT_TASK_PROMPT + "\n"],
        padding="max_length",
        truncation=True,
        max_length=policy.config.tokenizer_max_length,
        return_tensors="pt",)
    lang_tokens_single = tokens["input_ids"].to(device)
    lang_masks_single = tokens["attention_mask"].to(device, dtype=torch.bool)

    print(f"[+] Language tokens: {tuple(lang_tokens_single.shape)}")

    # Optimizer & Cosine Annealing Scheduler
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    best_train_loss = float("inf")
    history = {"train_loss": [], "epoch_times": []}

    print(f"\n[*] Starting training loop ({epochs} epochs, {len(train_loader)} batches/epoch)...")
    t0_start = time.time()

    for ep in range(1, epochs + 1):
        t_ep_start = time.time()
        policy.train()
        train_losses = []

        for batch in train_loader:
            b_imgs = batch["image"].to(device)              # (B, 3, 256, 256)
            b_st = batch["state"].to(device)                # (B, 8)
            b_actions = batch["action_chunk"].to(device)    # (B, 50, 32)
            B = b_imgs.shape[0]

            img_masks = torch.ones(B, dtype=torch.bool, device=device)
            padded_st = pad_vector(b_st, policy.config.max_state_dim)

            lang_tokens = lang_tokens_single.expand(B, -1)
            lang_masks = lang_masks_single.expand(B, -1)

            optimizer.zero_grad()

            # Flow-matching loss
            loss_tensor = policy.model(
                [b_imgs],
                [img_masks],
                lang_tokens,
                lang_masks,
                padded_st,
                b_actions,
            )
            loss = loss_tensor.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())

        avg_train_loss = float(np.mean(train_losses))
        ep_duration = time.time() - t_ep_start

        history["train_loss"].append(avg_train_loss)
        history["epoch_times"].append(ep_duration)

        if ep % 5 == 0 or ep == 1 or ep == epochs:
            curr_lr = optimizer.param_groups[0]["lr"]
            print(
                f"[Epoch {ep:02d}/{epochs:02d}] "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"LR: {curr_lr:.2e} | "
                f"Time: {ep_duration:.1f}s"
            )

        # One-episode baseline: use training loss as checkpoint criterion
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            best_ckpt = out_path / "model.safetensors"
            save_file(policy.state_dict(), str(best_ckpt))

        # Only relevant if training on Apple MPS
        if device.type == "mps":
            torch.mps.empty_cache()
            gc.collect()

    total_training_time = time.time() - t0_start
    print(f"\n[+] Training completed in {total_training_time:.1f}s.")
    print(f"[+] Initial Train Loss: {history['train_loss'][0]:.4f} -> Final: {history['train_loss'][-1]:.4f} (-{(1 - history['train_loss'][-1]/history['train_loss'][0])*100:.1f}% drop)")
    print(f"[+] Best Training Loss: {best_train_loss:.4f}")

    # Save config and training metrics
    cfg_out = out_path / "config.json"
    with open(cfg_out, "w") as f:
        # Save clean serializable config
        json.dump({
            "model_type": "smolvla_glass_expert",
            "epochs": epochs,
            "batch_size": batch_size,
            "final_train_loss": history["train_loss"][-1],
            "best_train_loss": best_train_loss,
            "validation_mode": "none - one expert episode smoke/baseline",
            "total_time_s": total_training_time,
            "prompt": DEFAULT_TASK_PROMPT,
        }, f, indent=2)

    hist_out = out_path / "training_history.json"
    with open(hist_out, "w") as f:
        json.dump(history, f, indent=2)

    print(f"[+] Saved fine-tuned checkpoint & config -> {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune SmolVLA action expert head")
    parser.add_argument("--dataset_dir", type=str, default="datasets/lerobot_dataset/glass_pick_place")
    parser.add_argument("--out_dir", type=str, default="outputs/smolvla_glass_expert_iteration10")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train_expert(
        dataset_dir=args.dataset_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

""" 
verify:
python3 - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
PY

Run smoke test:
cd ~/Documents/Pollex && \
PYTHONPATH=. python3 src/learning/train_smolvla_expert.py \
  --dataset_dir datasets/lerobot_dataset/glass_pick_place \
  --out_dir outputs/smolvla_glass_expert_iteration10_smoke \
  --epochs 1 \
  --batch_size 1 \
  --lr 1e-4

"""