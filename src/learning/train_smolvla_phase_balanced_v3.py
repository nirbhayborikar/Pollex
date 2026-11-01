"""
Phase-Balanced SmolVLA V3 Fine-Tuning.

V1 learned grasp and lift but continuously over-lifted.
V2 learned approach, grasp, return, and release, but suppressed sustained lift.
V3 uses temporal object-motion boundaries instead of classifying every
post-grasp frame from the instantaneous dz sign.

Target sampling:
    hover                 5%
    approach             10%
    descent/pre-grasp    15%
    grasp closing        10%
    post-grasp lift      25%
    post-grasp lower     12%
    release              13%
    retract              10%

Output:
    outputs/smolvla_glass_phase_balanced_v3/
"""

import argparse
import gc
import json
import time
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoTokenizer

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, pad_vector


DEFAULT_TASK_PROMPT = (
    "approach the glass on the table and grasp the cylinder and lift the cylinder "
    "and then bring down the cylinder to the table and then withdraw your hands"
)

PHASE_TARGETS = {
    "hover": 0.05,
    "approach": 0.10,
    "descent_pregrasp": 0.15,
    "grasp_closing": 0.10,
    "postgrasp_lift": 0.25,
    "postgrasp_lower": 0.12,
    "release": 0.13,
    "retract": 0.10,
}

RELEASE_SETTLE_FRAMES = 12


def classify_phase(
    frame: int,
    hover_end: int,
    descent_start: int,
    grasp_trigger_frame: int,
    physical_grasp_frame: int,
    peak_lift_frame: int,
    release_frame: int,
    retract_frame: int,
) -> str:
    """Classify an expert frame using temporal task boundaries."""
    if frame < hover_end:
        return "hover"
    if frame < descent_start:
        return "approach"
    if frame < grasp_trigger_frame:
        return "descent_pregrasp"
    if frame < physical_grasp_frame:
        return "grasp_closing"
    if frame < peak_lift_frame:
        return "postgrasp_lift"
    if frame < release_frame:
        return "postgrasp_lower"
    if frame < retract_frame:
        return "release"
    return "retract"


class PhaseBalancedGlassDataset(Dataset):
    """RGB, 8D state, action chunk, frame index, and temporal phase."""

    def __init__(
        self,
        npz_paths: List[str],
        expert_replay_path: str,
        chunk_size: int = 50,
        max_action_dim: int = 32,
    ):
        self.chunk_size = chunk_size
        self.max_action_dim = max_action_dim
        self.samples = []

        replay_path = Path(expert_replay_path)
        if not replay_path.exists():
            raise FileNotFoundError(f"Expert replay not found: {replay_path}")

        with np.load(replay_path, allow_pickle=False) as replay:
            replay_actions = np.asarray(replay["executed_actions"], dtype=np.float32)
            replay_tumbler = np.asarray(replay["tumbler_positions"], dtype=np.float32)
            grasp_trigger_frame = int(np.asarray(replay["grasp_trigger_frame"]).item())
            physical_grasp_frame = int(np.asarray(replay["physical_grasp_latch_frame"]).item())

        if replay_actions.ndim != 2 or replay_actions.shape[1] != 7:
            raise ValueError(f"Expected replay actions (T,7), got {replay_actions.shape}")
        if replay_tumbler.shape != (len(replay_actions), 3):
            raise ValueError(f"Expected tumbler positions (T,3), got {replay_tumbler.shape}")
        if not np.isfinite(replay_actions).all() or not np.isfinite(replay_tumbler).all():
            raise ValueError("Expert replay contains NaN or infinity.")

        frame_indices = np.arange(len(replay_actions))
        postgrasp_open = np.where(
            (frame_indices > physical_grasp_frame) & (replay_actions[:, 6] <= 0.0)
        )[0]

        if len(postgrasp_open) == 0:
            raise RuntimeError("No release transition found after physical grasp.")

        release_frame = int(postgrasp_open[0])
        retract_frame = min(len(replay_actions), release_frame + RELEASE_SETTLE_FRAMES)

        if release_frame <= physical_grasp_frame:
            raise RuntimeError(
                f"Invalid release boundary: physical grasp={physical_grasp_frame}, "
                f"release={release_frame}"
            )

        lift_window = replay_tumbler[physical_grasp_frame:release_frame, 2]
        if len(lift_window) == 0:
            raise RuntimeError("No samples available between physical grasp and release.")

        peak_lift_frame = physical_grasp_frame + int(np.argmax(lift_window))

        if not physical_grasp_frame < peak_lift_frame < release_frame:
            raise RuntimeError(
                f"Invalid temporal boundaries: physical grasp={physical_grasp_frame}, "
                f"peak lift={peak_lift_frame}, release={release_frame}. "
                "Inspect the Iteration 11 tumbler trajectory before training."
            )

        self.boundaries = {
            "grasp_trigger_frame": grasp_trigger_frame,
            "physical_grasp_frame": physical_grasp_frame,
            "peak_lift_frame": peak_lift_frame,
            "release_frame": release_frame,
            "retract_frame": retract_frame,
        }

        initial_tumbler_z = float(replay_tumbler[0, 2])
        peak_tumbler_z = float(replay_tumbler[peak_lift_frame, 2])

        print(
            f"[*] Expert boundaries: grasp_trigger={grasp_trigger_frame}, "
            f"physical_grasp={physical_grasp_frame}, peak_lift={peak_lift_frame}, "
            f"release={release_frame}, retract={retract_frame}"
        )
        print(
            f"[*] Expert peak lift: "
            f"{(peak_tumbler_z - initial_tumbler_z) * 100.0:.2f} cm"
        )

        global_frame_offset = 0

        for npz_path in npz_paths:
            path = Path(npz_path)
            if not path.exists():
                raise FileNotFoundError(f"Training episode not found: {path}")

            with np.load(path, allow_pickle=False) as data:
                images = np.asarray(data["images"], dtype=np.uint8)
                actions = np.asarray(data["actions"], dtype=np.float32)
                states = np.asarray(data["state_8d"], dtype=np.float32)

            number_of_frames = len(actions)

            if images.shape != (number_of_frames, 256, 256, 3):
                raise ValueError(f"Expected images (T,256,256,3), got {images.shape}")
            if states.shape != (number_of_frames, 8):
                raise ValueError(f"Expected states (T,8), got {states.shape}")
            if actions.shape != (number_of_frames, 7):
                raise ValueError(f"Expected actions (T,7), got {actions.shape}")
            if not np.isfinite(states).all() or not np.isfinite(actions).all():
                raise ValueError(f"Non-finite states or actions in {path}")

            if len(npz_paths) == 1:
                if replay_actions.shape != actions.shape:
                    raise ValueError(
                        f"Training/replay shape mismatch: "
                        f"{actions.shape} != {replay_actions.shape}"
                    )

                if not np.allclose(actions, replay_actions, atol=1e-5):
                    max_difference = float(np.max(np.abs(actions - replay_actions)))
                    raise ValueError(
                        "Training episode does not contain the Iteration 11 actions. "
                        f"Maximum difference: {max_difference:.6f}"
                    )

            hover_end = min(40, number_of_frames)
            descent_start = min(86, number_of_frames)
            grasp_trigger = int(np.clip(grasp_trigger_frame, 0, number_of_frames))
            physical_grasp = int(np.clip(physical_grasp_frame, 0, number_of_frames))
            episode_peak = int(np.clip(peak_lift_frame, 0, number_of_frames))
            episode_release = int(np.clip(release_frame, 0, number_of_frames))
            episode_retract = int(np.clip(retract_frame, 0, number_of_frames))

            for frame in range(number_of_frames):
                future_length = min(chunk_size, number_of_frames - frame)
                action_chunk = np.zeros((chunk_size, max_action_dim), dtype=np.float32)
                action_chunk[:future_length, :7] = actions[frame:frame + future_length]

                if future_length < chunk_size:
                    action_chunk[future_length:, :7] = actions[
                        frame + future_length - 1
                    ]

                phase = classify_phase(
                    frame=frame,
                    hover_end=hover_end,
                    descent_start=descent_start,
                    grasp_trigger_frame=grasp_trigger,
                    physical_grasp_frame=physical_grasp,
                    peak_lift_frame=episode_peak,
                    release_frame=episode_release,
                    retract_frame=episode_retract,
                )

                self.samples.append({
                    "image": images[frame].copy(),
                    "state": states[frame].copy(),
                    "action_chunk": action_chunk,
                    "frame": global_frame_offset + frame,
                    "phase": phase,
                })

            global_frame_offset += number_of_frames

        if not self.samples:
            raise RuntimeError("No training samples were loaded.")

        self.phase_counts = Counter(sample["phase"] for sample in self.samples)

        missing_phases = [
            phase
            for phase, target in PHASE_TARGETS.items()
            if target > 0 and self.phase_counts.get(phase, 0) == 0
        ]
        if missing_phases:
            raise RuntimeError(f"Target phases contain no samples: {missing_phases}")

        probability_sum = sum(PHASE_TARGETS.values())
        if not np.isclose(probability_sum, 1.0):
            raise RuntimeError(f"PHASE_TARGETS must sum to 1.0, got {probability_sum}")

        print(
            f"[*] Loaded {len(self.samples)} transitions "
            f"across {len(npz_paths)} episode(s)."
        )
        print("[*] Natural temporal phase distribution:")

        for phase in PHASE_TARGETS:
            count = self.phase_counts.get(phase, 0)
            percentage = 100.0 * count / len(self.samples)
            print(f"    {phase:24s} {count:3d} frames ({percentage:5.2f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = sample["image"].transpose(2, 0, 1).copy()
        image = (torch.from_numpy(image).float() / 255.0) * 2.0 - 1.0

        return {
            "image": image,
            "state": torch.from_numpy(sample["state"]).float(),
            "action_chunk": torch.from_numpy(sample["action_chunk"]).float(),
            "frame": sample["frame"],
            "phase": sample["phase"],
        }

    def build_sampler(
        self,
        samples_per_epoch: int,
        seed: int,
    ) -> WeightedRandomSampler:
        if samples_per_epoch < 1:
            raise ValueError("samples_per_epoch must be at least 1.")

        weights = [
            PHASE_TARGETS[sample["phase"]] / self.phase_counts[sample["phase"]]
            for sample in self.samples
        ]

        generator = torch.Generator()
        generator.manual_seed(seed)

        return WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=samples_per_epoch,
            replacement=True,
            generator=generator,
        )


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def initialize_smolvla_training_policy(
    device: torch.device,
) -> Tuple[SmolVLAPolicy, SmolVLAConfig]:
    print(f"[*] Initializing SmolVLA on {device}...")

    config_path = hf_hub_download(
        repo_id="lerobot/smolvla_base",
        filename="config.json",
    )

    with open(config_path, encoding="utf-8") as file:
        config_dict = json.load(file)

    config_dict.pop("type", None)
    config_dict["load_vlm_weights"] = False
    config_dict["freeze_vision_encoder"] = True
    config_dict["train_expert_only"] = True
    config_dict["train_state_proj"] = True
    config_dict["device"] = str(device)

    config_dict["input_features"] = {
        key: PolicyFeature(
            type=FeatureType[value["type"]],
            shape=tuple(value["shape"]),
        )
        for key, value in config_dict["input_features"].items()
    }

    config_dict["output_features"] = {
        key: PolicyFeature(
            type=FeatureType[value["type"]],
            shape=tuple(value["shape"]),
        )
        for key, value in config_dict["output_features"].items()
    }

    config_dict["normalization_mapping"] = {
        key: NormalizationMode[value]
        for key, value in config_dict["normalization_mapping"].items()
    }

    config = SmolVLAConfig(**config_dict)
    policy = SmolVLAPolicy(config)

    weights_path = hf_hub_download(
        repo_id="lerobot/smolvla_base",
        filename="model.safetensors",
    )

    policy.load_state_dict(load_file(weights_path), strict=False)
    policy.model.vlm_with_expert.freeze_vision_encoder = True
    policy.model.vlm_with_expert.train_expert_only = True
    policy.model.vlm_with_expert.set_requires_grad()
    policy.model.set_requires_grad()
    policy.to(device)

    trainable_parameters = [
        parameter for parameter in policy.parameters() if parameter.requires_grad
    ]
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)

    print(f"[+] Loaded model. Trainable parameters: {trainable_count / 1e6:.2f} M.")
    return policy, config


def save_checkpoint(policy: SmolVLAPolicy, checkpoint_path: Path):
    state_dict = {
        key: value.detach().contiguous().cpu()
        for key, value in policy.state_dict().items()
    }
    save_file(state_dict, str(checkpoint_path))


def train_phase_balanced_v3(
    dataset_dir: str = "datasets/lerobot_dataset/glass_pick_place_iteration11",
    expert_replay: str = (
        "datasets/cotracker_iterations/iteration_11_full_mujoco/"
        "cup_lifting_01_replay.npz"
    ),
    out_dir: str = "outputs/smolvla_glass_phase_balanced_v3",
    epochs: int = 30,
    batch_size: int = 4,
    lr: float = 1e-4,
    seed: int = 6,
):
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if lr <= 0:
        raise ValueError("lr must be positive.")

    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = select_device()
    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        print(f"[+] CUDA device: {torch.cuda.get_device_name(0)}")

    print("=" * 80)
    print("Phase-Balanced SmolVLA V3 Temporal-Sequence Fine-Tuning")
    print(
        f"Device: {device} | Epochs: {epochs} | "
        f"Batch Size: {batch_size} | LR: {lr}"
    )
    print("=" * 80)

    episode_path = Path(dataset_dir) / "cup_lifting_01_episode.npz"

    train_dataset = PhaseBalancedGlassDataset(
        npz_paths=[str(episode_path)],
        expert_replay_path=expert_replay,
        chunk_size=50,
        max_action_dim=32,
    )

    samples_per_epoch = len(train_dataset)
    sampler = train_dataset.build_sampler(
        samples_per_epoch=samples_per_epoch,
        seed=seed,
    )

    expected_samples = {
        phase: round(probability * samples_per_epoch)
        for phase, probability in PHASE_TARGETS.items()
    }

    print("[*] Expected samples per epoch:")
    for phase in PHASE_TARGETS:
        print(
            f"    {phase:24s} ~{expected_samples3d} "
            f"({PHASE_TARGETS[phase] * 100:5.1f}%)"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    policy, _ = initialize_smolvla_training_policy(device)

    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    )

    tokens = tokenizer(
        [DEFAULT_TASK_PROMPT + "\n"],
        padding="max_length",
        truncation=True,
        max_length=policy.config.tokenizer_max_length,
        return_tensors="pt",
    )

    language_tokens_single = tokens["input_ids"].to(device)
    language_masks_single = tokens["attention_mask"].to(device, dtype=torch.bool)

    print(f"[+] Language tokens: {tuple(language_tokens_single.shape)}")

    trainable_parameters = [
        parameter for parameter in policy.parameters() if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=lr,
        weight_decay=1e-4,
    )

    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-6,
    )

    best_train_loss = float("inf")
    history = {
        "train_loss": [],
        "epoch_times": [],
        "sampled_phase_counts": [],
    }

    training_start = time.time()

    print(
        f"\n[*] Starting training loop "
        f"({epochs} epochs, {len(train_loader)} batches/epoch)..."
    )

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        policy.train()
        epoch_losses = []
        sampled_phase_counts = Counter()

        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            states = batch["state"].to(device, non_blocking=True)
            action_chunks = batch["action_chunk"].to(device, non_blocking=True)
            sampled_phase_counts.update(batch["phase"])
            current_batch_size = images.shape[0]

            image_masks = torch.ones(
                current_batch_size,
                dtype=torch.bool,
                device=device,
            )

            padded_states = pad_vector(
                states,
                policy.config.max_state_dim,
            )

            language_tokens = language_tokens_single.expand(
                current_batch_size,
                -1,
            )

            language_masks = language_masks_single.expand(
                current_batch_size,
                -1,
            )

            optimizer.zero_grad(set_to_none=True)

            loss_tensor = policy.model(
                [images],
                [image_masks],
                language_tokens,
                language_masks,
                padded_states,
                action_chunks,
            )

            loss = loss_tensor.mean()

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=1.0,
            )

            optimizer.step()
            scheduler.step()
            epoch_losses.append(float(loss.item()))

        average_train_loss = float(np.mean(epoch_losses))
        epoch_duration = time.time() - epoch_start

        history["train_loss"].append(average_train_loss)
        history["epoch_times"].append(epoch_duration)
        history["sampled_phase_counts"].append(dict(sampled_phase_counts))

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"[Epoch {epoch:02d}/{epochs:02d}] "
                f"Train Loss: {average_train_loss:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_duration:.1f}s"
            )

            print(
                "    Sampled phases: "
                + ", ".join(
                    f"{phase}={sampled_phase_counts.get(phase, 0)}"
                    for phase in PHASE_TARGETS
                )
            )

        if average_train_loss < best_train_loss:
            best_train_loss = average_train_loss
            save_checkpoint(
                policy,
                output_path / "model.safetensors",
            )

        if device.type == "mps":
            torch.mps.empty_cache()
            gc.collect()

    total_training_time = time.time() - training_start
    initial_loss = history["train_loss"][0]
    final_loss = history["train_loss"][-1]

    loss_drop = (
        100.0 * (1.0 - final_loss / initial_loss)
        if initial_loss > 0
        else 0.0
    )

    print(f"\n[+] Training completed in {total_training_time:.1f}s.")
    print(
        f"[+] Initial Train Loss: {initial_loss:.4f} -> "
        f"Final: {final_loss:.4f} ({loss_drop:.1f}% drop)"
    )
    print(f"[+] Best Training Loss: {best_train_loss:.4f}")

    config_output = {
        "model_type": "smolvla_glass_phase_balanced_v3",
        "base_model": "lerobot/smolvla_base",
        "dataset": str(episode_path),
        "expert_replay": expert_replay,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "seed": seed,
        "chunk_size": 50,
        "state_dim": 8,
        "action_dim": 7,
        "max_action_dim": 32,
        "sampling_mode": "temporal_sequence_phase_balanced_v3",
        "samples_per_epoch": samples_per_epoch,
        "natural_phase_counts": dict(train_dataset.phase_counts),
        "target_phase_probabilities": PHASE_TARGETS,
        "expert_boundaries": train_dataset.boundaries,
        "release_settle_frames": RELEASE_SETTLE_FRAMES,
        "action_normalization": "none",
        "validation_mode": "none - one expert episode",
        "initial_train_loss": initial_loss,
        "final_train_loss": final_loss,
        "best_train_loss": best_train_loss,
        "total_time_s": total_training_time,
        "prompt": DEFAULT_TASK_PROMPT,
    }

    with open(
        output_path / "config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config_output, file, indent=2)

    with open(
        output_path / "training_history.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(history, file, indent=2)

    print(f"[+] Saved V3 checkpoint and metadata: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase-balanced SmolVLA V3 temporal-sequence fine-tuning"
    )

    parser.add_argument(
        "--dataset_dir",
        default="datasets/lerobot_dataset/glass_pick_place_iteration11",
    )

    parser.add_argument(
        "--expert_replay",
        default=(
            "datasets/cotracker_iterations/iteration_11_full_mujoco/"
            "cup_lifting_01_replay.npz"
        ),
    )

    parser.add_argument(
        "--out_dir",
        default="outputs/smolvla_glass_phase_balanced_v3",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=6,
    )

    args = parser.parse_args()

    train_phase_balanced_v3(
        dataset_dir=args.dataset_dir,
        expert_replay=args.expert_replay,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()


"""
Syntax check:

cd ~/Documents/Pollex && \
python3 -m py_compile src/learning/train_smolvla_phase_balanced_v3.py


Smoke test:

cd ~/Documents/Pollex && \
PYTHONPATH=. python3 -m src.learning.train_smolvla_phase_balanced_v3 \
  --dataset_dir datasets/lerobot_dataset/glass_pick_place_iteration11 \
  --expert_replay datasets/cotracker_iterations/iteration_11_full_mujoco/cup_lifting_01_replay.npz \
  --out_dir outputs/smolvla_glass_phase_balanced_v3_smoke \
  --epochs 1 \
  --batch_size 4 \
  --lr 1e-4 \
  --seed 6


Full training:

cd ~/Documents/Pollex && \
PYTHONPATH=. python3 -m src.learning.train_smolvla_phase_balanced_v3 \
  --dataset_dir datasets/lerobot_dataset/glass_pick_place_iteration11 \
  --expert_replay datasets/cotracker_iterations/iteration_11_full_mujoco/cup_lifting_01_replay.npz \
  --out_dir outputs/smolvla_glass_phase_balanced_v3 \
  --epochs 30 \
  --batch_size 4 \
  --lr 1e-4 \
  --seed 6
"""