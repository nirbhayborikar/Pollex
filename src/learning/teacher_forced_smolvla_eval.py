"""
Teacher-forced SmolVLA action evaluation.

At each selected Iteration 10 frame:
    expert RGB[t] + expert state_8d[t] + task prompt
        -> SmolVLA predicted action chunk
        -> compare predicted first action with expert action[t]

This separates policy-learning error from closed-loop covariate shift.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, pad_vector


TASK_PROMPT = (
    "approach the glass on the table and grasp the cylinder and lift the cylinder "
    "and then bring down the cylinder to the table and then withdraw your hands"
)

DEFAULT_FRAMES = [
    0, 50, 100, 120, 140, 143, 146,
    150, 160, 180, 200, 250, 300, 339,
]

ACTION_NAMES = [
    "dx", "dy", "dz", "droll", "dpitch", "dyaw", "grip"
]


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_policy(
    checkpoint_dir: str,
    device: torch.device,
) -> SmolVLAPolicy:
    checkpoint = Path(checkpoint_dir) / "model.safetensors"

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    config_path = hf_hub_download(
        repo_id="lerobot/smolvla_base",
        filename="config.json",
    )

    with open(config_path, encoding="utf-8") as file:
        config_dict = json.load(file)

    config_dict.pop("type", None)
    config_dict["load_vlm_weights"] = False
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

    policy = SmolVLAPolicy(SmolVLAConfig(**config_dict))
    policy.load_state_dict(load_file(str(checkpoint)), strict=False)
    policy.to(device)
    policy.eval()

    print(f"[+] Loaded checkpoint: {checkpoint}")
    print(f"[+] Device: {device}")

    if device.type == "cuda":
        print(f"[+] GPU: {torch.cuda.get_device_name(0)}")

    return policy


def build_language_tokens(
    policy: SmolVLAPolicy,
    device: torch.device,
):
    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    )

    tokens = tokenizer(
        [TASK_PROMPT + "\n"],
        padding="max_length",
        truncation=True,
        max_length=policy.config.tokenizer_max_length,
        return_tensors="pt",
    )

    return (
        tokens["input_ids"].to(device),
        tokens["attention_mask"].to(device, dtype=torch.bool),
    )


def prepare_image(
    image: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    if image.shape != (256, 256, 3):
        raise ValueError(f"Expected image (256,256,3), got {image.shape}")

    tensor = torch.from_numpy(
        image.transpose(2, 0, 1).copy()
    ).float()

    tensor = (tensor / 255.0) * 2.0 - 1.0
    return tensor.to(device).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(
        description="Teacher-forced SmolVLA action comparison"
    )

    parser.add_argument(
        "--episode",
        default=(
            "datasets/lerobot_dataset/glass_pick_place/"
            "cup_lifting_01_episode.npz"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        default="outputs/smolvla_glass_expert_iteration10",
    )

    parser.add_argument(
        "--output_dir",
        default="datasets/smolvla_comparisons/teacher_forced",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=DEFAULT_FRAMES,
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.episode, allow_pickle=False) as data:
        images = np.asarray(data["images"], dtype=np.uint8)
        states = np.asarray(data["state_8d"], dtype=np.float32)
        expert_actions = np.asarray(data["actions"], dtype=np.float32)

    if len(images) != len(states) or len(states) != len(expert_actions):
        raise ValueError(
            f"Alignment error: images={len(images)}, "
            f"states={len(states)}, actions={len(expert_actions)}"
        )

    if states.shape[1] != 8:
        raise ValueError(f"Expected state dimension 8, got {states.shape}")

    if expert_actions.shape[1] != 7:
        raise ValueError(
            f"Expected action dimension 7, got {expert_actions.shape}"
        )

    device = select_device()
    policy = load_policy(args.checkpoint, device)

    language_tokens, language_mask = build_language_tokens(
        policy,
        device,
    )

    selected_frames = [
        frame for frame in args.frames
        if 0 <= frame < len(expert_actions)
    ]

    predicted_first_actions = []
    predicted_chunks = []
    rows = []

    print("\n" + "=" * 94)
    print("TEACHER-FORCED EXPERT vs SmolVLA ACTION COMPARISON")
    print("=" * 94)

    print(
        f"{'frame':>6} | "
        f"{'expert dx':>10} {'VLA dx':>10} | "
        f"{'expert dz':>10} {'VLA dz':>10} | "
        f"{'exp grip':>9} {'VLA grip':>9}"
    )

    for frame in selected_frames:
        image_tensor = prepare_image(images[frame], device)
        image_mask = torch.ones(
            1,
            dtype=torch.bool,
            device=device,
        )

        state_tensor = torch.from_numpy(
            states[frame]
        ).to(device).unsqueeze(0)

        padded_state = pad_vector(
            state_tensor,
            policy.config.max_state_dim,
        )

        with torch.no_grad():
            chunk = policy.model.sample_actions(
                [image_tensor],
                [image_mask],
                language_tokens,
                language_mask,
                padded_state,
            )

        chunk = chunk[0, :, :7].detach().cpu().numpy()
        prediction = chunk[0].copy()
        expert = expert_actions[frame].copy()
        error = prediction - expert

        predicted_chunks.append(chunk)
        predicted_first_actions.append(prediction)

        print(
            f"{frame:6d} | "
            f"{expert[0]:+10.4f} {prediction[0]:+10.4f} | "
            f"{expert[2]:+10.4f} {prediction[2]:+10.4f} | "
            f"{expert[6]:+9.3f} {prediction[6]:+9.3f}"
        )

        row = {"frame": frame}

        for index, name in enumerate(ACTION_NAMES):
            row[f"expert_{name}"] = float(expert[index])
            row[f"predicted_{name}"] = float(prediction[index])
            row[f"error_{name}"] = float(error[index])

        rows.append(row)

    predicted_first_actions = np.asarray(
        predicted_first_actions,
        dtype=np.float32,
    )

    predicted_chunks = np.asarray(
        predicted_chunks,
        dtype=np.float32,
    )

    selected_expert = expert_actions[selected_frames]

    mae = np.mean(
        np.abs(predicted_first_actions - selected_expert),
        axis=0,
    )

    sign_accuracy = {}

    for index in [0, 2]:
        nonzero = np.abs(selected_expert[:, index]) > 1e-4

        if np.any(nonzero):
            accuracy = np.mean(
                np.sign(predicted_first_actions[nonzero, index])
                == np.sign(selected_expert[nonzero, index])
            )
        else:
            accuracy = np.nan

        sign_accuracy[ACTION_NAMES[index]] = float(accuracy)

    expert_closed = selected_expert[:, 6] > 0
    predicted_closed = predicted_first_actions[:, 6] > 0
    grip_accuracy = float(np.mean(expert_closed == predicted_closed))

    print("\nSUMMARY")
    print("-" * 60)

    for index, name in enumerate(ACTION_NAMES):
        print(f"MAE {name:6s}: {mae[index]:.4f}")


    print(f"DX sign accuracy:    {sign_accuracy['dx'] * 100:.1f}%")
    print(f"DZ sign accuracy:    {sign_accuracy['dz'] * 100:.1f}%")
    print(f"Grip-state accuracy: {grip_accuracy * 100:.1f}%")
    grasp_frames = [
        frame for frame in selected_frames
        if 143 <= frame < 160
    ]

    if grasp_frames:
        indices = [
            selected_frames.index(frame)
            for frame in grasp_frames
        ]

        mean_teacher_grip = float(
            predicted_first_actions[indices, 6].mean()
        )

        print(
            f"Mean teacher-forced grip at frames 143-159: "
            f"{mean_teacher_grip:+.4f}"
        )

    csv_path = output_dir / "teacher_forced_actions.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    npz_path = output_dir / "teacher_forced_actions.npz"

    np.savez_compressed(
        npz_path,
        frames=np.asarray(selected_frames, dtype=np.int32),
        expert_actions=selected_expert,
        predicted_first_actions=predicted_first_actions,
        predicted_chunks=predicted_chunks,
        mae=mae.astype(np.float32),
    )

    summary = {
        "checkpoint": args.checkpoint,
        "episode": args.episode,
        "seed": args.seed,
        "frames": selected_frames,
        "mae": {
            name: float(mae[index])
            for index, name in enumerate(ACTION_NAMES)
        },
        "dx_sign_accuracy": sign_accuracy["dx"],
        "dz_sign_accuracy": sign_accuracy["dz"],
        "grip_state_accuracy": grip_accuracy,
    }

    json_path = output_dir / "teacher_forced_summary.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"\n[+] Saved CSV:     {csv_path}")
    print(f"[+] Saved NPZ:     {npz_path}")
    print(f"[+] Saved summary: {json_path}")


if __name__ == "__main__":
    main()

"""
cd ~/Documents/Pollex && \
python3 -m py_compile src/learning/teacher_forced_smolvla_eval.py

source /home/nvidia/Documents/Pollex/.venv/bin/activate && \
export PYTHONPATH="$HOME/Documents/Pollex:$PYTHONPATH" && \
cd ~/Documents/Pollex && \
python3 -m src.learning.teacher_forced_smolvla_eval \
  --episode datasets/lerobot_dataset/glass_pick_place/cup_lifting_01_episode.npz \
  --checkpoint outputs/smolvla_glass_expert_iteration10 \
  --output_dir datasets/smolvla_comparisons/teacher_forced \
  --seed 6 \
  --frames 0 50 100 120 140 143 146 150 160 180 200 250 300 339
  
"""