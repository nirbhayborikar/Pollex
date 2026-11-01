"""
End-to-End SmolVLA Experiment Sweep.

For each candidate:
    1. Configure temporal phase sampling and action horizon.
    2. Fine-tune SmolVLA from the same base checkpoint.
    3. Run a closed-loop GlassLift rollout from reset.
    4. Evaluate approach, contact, grasp, lift, lower, release, and return.
    5. Rank candidates and save the best configuration.

This script reuses:
    src.learning.train_smolvla_phase_balanced_v5
    src.simulation.rollout_smolvla_baseline

Important:
    - Training and inference use the same chunk horizon.
    - Execution always uses K <= training horizon.
    - A lift above 3 cm alone is not considered full task completion.
    - Excessive lift, no lowering, and no release are penalized.
"""

import argparse
import csv
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

import src.learning.train_smolvla_phase_balanced_v5 as trainer
import src.simulation.rollout_smolvla_baseline as rollout_module


DEFAULT_TASK_PROMPT = (
    "approach the glass on the table and grasp the cylinder and lift the cylinder "
    "and then bring down the cylinder to the table and then withdraw your hands"
)

EXPERT_LIFT_CM = 8.17
MIN_SUCCESS_LIFT_CM = 3.0
MAX_ACCEPTABLE_LIFT_CM = 15.0
MAX_FINAL_LIFT_CM = 2.0
MAX_APPROACH_DISTANCE_CM = 3.0
MIN_GRASP_RUN_FRAMES = 10
MIN_LOWERING_CM = 2.0

EXPERIMENTS = [
    {
        "name": "h50_l15_low10_rel05",
        "chunk_size": 50,
        "epochs": 10,
        "phase_targets": {
            "hover": 0.10,
            "approach": 0.15,
            "descent_pregrasp": 0.20,
            "grasp_closing": 0.15,
            "postgrasp_lift": 0.15,
            "postgrasp_lower": 0.10,
            "release": 0.05,
            "retract": 0.10,
        },
    },
    {
        "name": "h50_l18_low10_rel07",
        "chunk_size": 50,
        "epochs": 10,
        "phase_targets": {
            "hover": 0.08,
            "approach": 0.15,
            "descent_pregrasp": 0.20,
            "grasp_closing": 0.12,
            "postgrasp_lift": 0.18,
            "postgrasp_lower": 0.10,
            "release": 0.07,
            "retract": 0.10,
        },
    },
    {
        "name": "h50_l20_low10_rel08",
        "chunk_size": 50,
        "epochs": 10,
        "phase_targets": {
            "hover": 0.07,
            "approach": 0.15,
            "descent_pregrasp": 0.20,
            "grasp_closing": 0.10,
            "postgrasp_lift": 0.20,
            "postgrasp_lower": 0.10,
            "release": 0.08,
            "retract": 0.10,
        },
    },
    {
        "name": "h50_l22_low10_rel08",
        "chunk_size": 50,
        "epochs": 10,
        "phase_targets": {
            "hover": 0.05,
            "approach": 0.15,
            "descent_pregrasp": 0.20,
            "grasp_closing": 0.10,
            "postgrasp_lift": 0.22,
            "postgrasp_lower": 0.10,
            "release": 0.08,
            "retract": 0.10,
        },
    },
    {
        "name": "h50_l18_low15_rel08",
        "chunk_size": 50,
        "epochs": 10,
        "phase_targets": {
            "hover": 0.05,
            "approach": 0.14,
            "descent_pregrasp": 0.20,
            "grasp_closing": 0.10,
            "postgrasp_lift": 0.18,
            "postgrasp_lower": 0.15,
            "release": 0.08,
            "retract": 0.10,
        },
    },
    {
        "name": "h50_l20_low12_rel10",
        "chunk_size": 50,
        "epochs": 10,
        "phase_targets": {
            "hover": 0.05,
            "approach": 0.13,
            "descent_pregrasp": 0.20,
            "grasp_closing": 0.10,
            "postgrasp_lift": 0.20,
            "postgrasp_lower": 0.12,
            "release": 0.10,
            "retract": 0.10,
        },
    },
]

ACTION_NAMES = [
    "dx",
    "dy",
    "dz",
    "droll",
    "dpitch",
    "dyaw",
    "grip",
]


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def validate_experiment(experiment: Dict[str, Any], k_steps: int) -> None:
    required_phases = set(trainer.PHASE_TARGETS)
    supplied_phases = set(experiment["phase_targets"])

    if required_phases != supplied_phases:
        raise ValueError(
            f"{experiment['name']}: phase mismatch. "
            f"Expected={sorted(required_phases)}, "
            f"received={sorted(supplied_phases)}"
        )

    probability_sum = sum(experiment["phase_targets"].values())

    if not np.isclose(probability_sum, 1.0):
        raise ValueError(
            f"{experiment['name']}: phase probabilities sum to "
            f"{probability_sum}, expected 1.0"
        )

    horizon = int(experiment["chunk_size"])

    if horizon < 1:
        raise ValueError(
            f"{experiment['name']}: chunk_size must be positive"
        )

    if k_steps > horizon:
        raise ValueError(
            f"{experiment['name']}: K={k_steps} exceeds H={horizon}"
        )

    if int(experiment["epochs"]) < 1:
        raise ValueError(
            f"{experiment['name']}: epochs must be positive"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        while True:
            block = file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)

    return digest.hexdigest()


def configure_trainer(experiment: Dict[str, Any]) -> None:
    horizon = int(experiment["chunk_size"])

    trainer.CHUNK_SIZE = horizon
    trainer.PHASE_TARGETS.clear()
    trainer.PHASE_TARGETS.update(experiment["phase_targets"])

    print(f"[*] Trainer horizon: {trainer.CHUNK_SIZE}")
    print("[*] Trainer phase targets:")

    for phase, probability in trainer.PHASE_TARGETS.items():
        print(f"    {phase:24s} {probability * 100:5.1f}%")


def load_policy_with_horizon(
    checkpoint_dir: str,
    horizon: int,
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
    config_dict["chunk_size"] = int(horizon)
    config_dict["n_action_steps"] = int(horizon)

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

    if policy.config.chunk_size != horizon:
        raise RuntimeError(
            f"Policy chunk mismatch: {policy.config.chunk_size} != {horizon}"
        )

    if policy.config.n_action_steps != horizon:
        raise RuntimeError(
            f"Policy action-step mismatch: "
            f"{policy.config.n_action_steps} != {horizon}"
        )

    print(f"[+] Loaded checkpoint: {checkpoint}")
    print(f"[+] Inference horizon: {policy.config.chunk_size}")

    return policy


def longest_true_run(values: np.ndarray) -> int:
    best = 0
    current = 0

    for value in values:
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0

    return best


def analyze_rollout(results: Dict[str, Any]) -> Dict[str, Any]:
    actions = np.asarray(results["actions"], dtype=np.float32)
    tumbler = np.asarray(results["tumbler_positions"], dtype=np.float32)
    contact = np.asarray(results["contact_flags"], dtype=np.float32) > 0.5
    grasp = np.asarray(results["grasp_flags"], dtype=np.float32) > 0.5

    if len(actions) == 0:
        raise RuntimeError("Rollout produced no actions.")

    if len(tumbler) != len(actions):
        raise RuntimeError(
            f"Action/tumbler mismatch: {len(actions)} != {len(tumbler)}"
        )

    initial_z = float(tumbler[0, 2])
    relative_z_cm = (tumbler[:, 2] - initial_z) * 100.0
    peak_index = int(np.argmax(relative_z_cm))
    peak_lift_cm = float(relative_z_cm[peak_index])
    final_lift_cm = float(relative_z_cm[-1])

    post_peak = relative_z_cm[peak_index:]
    lowering_cm = (
        float(peak_lift_cm - np.min(post_peak))
        if len(post_peak)
        else 0.0
    )

    first_contact = int(np.where(contact)[0][0]) if np.any(contact) else -1
    first_grasp = int(np.where(grasp)[0][0]) if np.any(grasp) else -1
    last_grasp = int(np.where(grasp)[0][-1]) if np.any(grasp) else -1

    released_after_grasp = bool(
        first_grasp >= 0
        and last_grasp < len(grasp) - 1
        and np.any(~grasp[last_grasp + 1:])
    )

    gripper_closed = actions[:, 6] > 0.0
    dz_positive = actions[:, 2] > 0.01
    dz_negative = actions[:, 2] < -0.01

    grasp_run = longest_true_run(grasp)
    closed_run = longest_true_run(gripper_closed)
    lift_run = longest_true_run(gripper_closed & dz_positive)
    lower_run = longest_true_run(gripper_closed & dz_negative)

    approach_ok = results["min_d_xy_cm"] <= MAX_APPROACH_DISTANCE_CM
    contact_ok = bool(results["contact_occurred"])
    grasp_ok = bool(results["grasp_occurred"])
    lift_ok = (
        MIN_SUCCESS_LIFT_CM
        <= peak_lift_cm
        <= MAX_ACCEPTABLE_LIFT_CM
    )
    grasp_retention_ok = grasp_run >= MIN_GRASP_RUN_FRAMES
    lowering_ok = lowering_cm >= MIN_LOWERING_CM
    final_height_ok = abs(final_lift_cm) <= MAX_FINAL_LIFT_CM
    release_ok = released_after_grasp

    full_task_success = all([
        approach_ok,
        contact_ok,
        grasp_ok,
        lift_ok,
        grasp_retention_ok,
        lowering_ok,
        final_height_ok,
        release_ok,
    ])

    lift_error_cm = abs(peak_lift_cm - EXPERT_LIFT_CM)

    score = 0.0
    score += 10.0 if approach_ok else 0.0
    score += 15.0 if contact_ok else 0.0
    score += 20.0 if grasp_ok else 0.0
    score += 15.0 if grasp_retention_ok else 0.0
    score += 20.0 if lift_ok else 0.0
    score += 10.0 if lowering_ok else 0.0
    score += 5.0 if release_ok else 0.0
    score += 5.0 if final_height_ok else 0.0

    score -= min(lift_error_cm, 30.0) * 0.5

    if peak_lift_cm > MAX_ACCEPTABLE_LIFT_CM:
        score -= min(
            peak_lift_cm - MAX_ACCEPTABLE_LIFT_CM,
            50.0,
        )

    if not contact_ok:
        score -= 20.0

    if not grasp_ok:
        score -= 20.0

    return {
        "score": float(score),
        "full_task_success": bool(full_task_success),
        "approach_ok": bool(approach_ok),
        "contact_ok": bool(contact_ok),
        "grasp_ok": bool(grasp_ok),
        "lift_ok": bool(lift_ok),
        "grasp_retention_ok": bool(grasp_retention_ok),
        "lowering_ok": bool(lowering_ok),
        "release_ok": bool(release_ok),
        "final_height_ok": bool(final_height_ok),
        "first_contact_frame": first_contact,
        "first_grasp_frame": first_grasp,
        "last_grasp_frame": last_grasp,
        "peak_lift_frame": peak_index,
        "peak_lift_cm": peak_lift_cm,
        "final_lift_cm": final_lift_cm,
        "lowering_after_peak_cm": lowering_cm,
        "lift_error_cm": lift_error_cm,
        "min_d_xy_cm": float(results["min_d_xy_cm"]),
        "longest_grasp_run": int(grasp_run),
        "longest_closed_run": int(closed_run),
        "longest_lift_run": int(lift_run),
        "longest_lower_run": int(lower_run),
        "positive_dz_percent": float(np.mean(dz_positive) * 100.0),
        "negative_dz_percent": float(np.mean(dz_negative) * 100.0),
        "closed_gripper_percent": float(np.mean(gripper_closed) * 100.0),
    }


def save_rollout(
    results: Dict[str, Any],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rollout_path = output_dir / "rollout.npz"

    np.savez_compressed(
        rollout_path,
        sim_frames=results["sim_frames"],
        actions=results["actions"],
        eef_positions=results["eef_positions"],
        tumbler_positions=results["tumbler_positions"],
        glass_z=results["glass_z"],
        contact_flags=results["contact_flags"],
        grasp_flags=results["grasp_flags"],
        rewards=results["rewards"],
    )

    return rollout_path


def train_candidate(
    experiment: Dict[str, Any],
    args: argparse.Namespace,
    checkpoint_dir: Path,
) -> Dict[str, Any]:
    configure_trainer(experiment)

    training_start = time.time()

    trainer.train_phase_balanced_v5(
        dataset_dir=args.dataset_dir,
        expert_replay=args.expert_replay,
        out_dir=str(checkpoint_dir),
        epochs=int(experiment["epochs"]),
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )

    training_time = time.time() - training_start
    checkpoint = checkpoint_dir / "model.safetensors"

    if not checkpoint.exists():
        raise RuntimeError(
            f"Training did not create checkpoint: {checkpoint}"
        )

    return {
        "training_time_s": float(training_time),
        "checkpoint_sha256": sha256_file(checkpoint),
    }


def evaluate_candidate(
    experiment: Dict[str, Any],
    args: argparse.Namespace,
    checkpoint_dir: Path,
    rollout_dir: Path,
) -> Dict[str, Any]:
    device = select_device()
    policy = load_policy_with_horizon(
        checkpoint_dir=str(checkpoint_dir),
        horizon=int(experiment["chunk_size"]),
        device=device,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    rollout_start = time.time()

    results = rollout_module.run_baseline_rollout(
        policy=policy,
        task_prompt=DEFAULT_TASK_PROMPT,
        max_steps=args.rollout_steps,
        chunk_exec_steps=args.k_steps,
        device=device,
        render_gui=args.gui,
    )

    rollout_time = time.time() - rollout_start
    rollout_path = save_rollout(results, rollout_dir)
    metrics = analyze_rollout(results)

    metrics["rollout_time_s"] = float(rollout_time)
    metrics["rollout_path"] = str(rollout_path)

    del policy
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


def write_leaderboard(
    records: List[Dict[str, Any]],
    sweep_dir: Path,
) -> None:
    ranked = sorted(
        records,
        key=lambda item: (
            item["full_task_success"],
            item["score"],
        ),
        reverse=True,
    )

    json_path = sweep_dir / "leaderboard.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(ranked, file, indent=2)

    csv_path = sweep_dir / "leaderboard.csv"

    fieldnames = [
        "rank",
        "name",
        "status",
        "chunk_size",
        "epochs",
        "score",
        "full_task_success",
        "contact_ok",
        "grasp_ok",
        "peak_lift_cm",
        "final_lift_cm",
        "lowering_after_peak_cm",
        "release_ok",
        "min_d_xy_cm",
        "longest_grasp_run",
        "longest_lift_run",
        "longest_lower_run",
        "positive_dz_percent",
        "negative_dz_percent",
        "training_time_s",
        "rollout_time_s",
        "checkpoint_dir",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for rank, item in enumerate(ranked, start=1):
            row = {"rank": rank}

            for key in fieldnames:
                if key != "rank":
                    row[key] = item.get(key, "")

            writer.writerow(row)

    print("\n" + "=" * 100)
    print("FINAL LEADERBOARD")
    print("=" * 100)

    for rank, item in enumerate(ranked, start=1):
        print(
            f"{rank:2d}. {item['name']:20s} "
            f"score={item.get('score', float('-inf')):7.2f} "
            f"success={item.get('full_task_success', False)} "
            f"contact={item.get('contact_ok', False)} "
            f"grasp={item.get('grasp_ok', False)} "
            f"peak={item.get('peak_lift_cm', 0.0):6.2f}cm "
            f"final={item.get('final_lift_cm', 0.0):+6.2f}cm"
        )

    successful = [
        item
        for item in ranked
        if item.get("full_task_success", False)
    ]

    best = successful[0] if successful else ranked[0]

    best_path = sweep_dir / "best_experiment.json"

    with open(best_path, "w", encoding="utf-8") as file:
        json.dump(best, file, indent=2)

    print("\n[+] Best experiment:")
    print(json.dumps(best, indent=2))
    print(f"[+] CSV leaderboard:  {csv_path}")
    print(f"[+] JSON leaderboard: {json_path}")
    print(f"[+] Best result:      {best_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train, roll out, evaluate, and rank SmolVLA configurations"
    )

    parser.add_argument(
        "--dataset_dir",
        default="datasets/lerobot_dataset/glass_pick_place_iteration11",
    )

    parser.add_argument(
        "--expert_replay",
        default=(
            "datasets/cotracker_iterations/"
            "iteration_11_full_mujoco/"
            "cup_lifting_01_replay.npz"
        ),
    )

    parser.add_argument(
        "--sweep_dir",
        default="outputs/smolvla_glass_sweep",
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

    parser.add_argument(
        "--k_steps",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--rollout_steps",
        type=int,
        default=340,
    )

    parser.add_argument(
        "--gui",
        action="store_true",
    )

    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Reuse candidates that already have result.json.",
    )

    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for index, experiment in enumerate(EXPERIMENTS, start=1):
        validate_experiment(experiment, args.k_steps)

        name = experiment["name"]
        experiment_dir = sweep_dir / name
        checkpoint_dir = experiment_dir / "checkpoint"
        rollout_dir = experiment_dir / "rollout"
        result_path = experiment_dir / "result.json"

        print("\n" + "#" * 100)
        print(
            f"EXPERIMENT {index}/{len(EXPERIMENTS)}: {name} | "
            f"H={experiment['chunk_size']} | "
            f"epochs={experiment['epochs']}"
        )
        print("#" * 100)

        if args.skip_existing and result_path.exists():
            with open(result_path, encoding="utf-8") as file:
                record = json.load(file)

            print(f"[+] Reusing existing result: {result_path}")
            records.append(record)
            continue

        experiment_dir.mkdir(parents=True, exist_ok=True)

        record = {
            "name": name,
            "status": "running",
            "chunk_size": int(experiment["chunk_size"]),
            "epochs": int(experiment["epochs"]),
            "phase_targets": experiment["phase_targets"],
            "checkpoint_dir": str(checkpoint_dir),
        }

        try:
            training_info = train_candidate(
                experiment=experiment,
                args=args,
                checkpoint_dir=checkpoint_dir,
            )

            evaluation_info = evaluate_candidate(
                experiment=experiment,
                args=args,
                checkpoint_dir=checkpoint_dir,
                rollout_dir=rollout_dir,
            )

            record.update(training_info)
            record.update(evaluation_info)
            record["status"] = "completed"

        except Exception as error:
            record["status"] = "failed"
            record["error"] = str(error)
            record["score"] = -1e9
            record["full_task_success"] = False

            print(f"[!] Experiment failed: {name}")
            print(f"[!] {type(error).__name__}: {error}")

        with open(result_path, "w", encoding="utf-8") as file:
            json.dump(record, file, indent=2)

        records.append(record)

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not records:
        raise RuntimeError("No experiment results were generated.")

    write_leaderboard(records, sweep_dir)


if __name__ == "__main__":
    main()

"""
source /home/nvidia/Documents/Pollex/.venv/bin/activate && \
export PYTHONPATH="$HOME/Documents/vla/LIBERO:$HOME/Documents/Pollex:$PYTHONPATH" && \
unset MUJOCO_GL QT_DEBUG_PLUGINS QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH && \
cd ~/Documents/Pollex && \
python3 -m src.learning.run_smolvla_experiment_sweep \
  --dataset_dir datasets/lerobot_dataset/glass_pick_place_iteration11 \
  --expert_replay datasets/cotracker_iterations/iteration_11_full_mujoco/cup_lifting_01_replay.npz \
  --sweep_dir outputs/smolvla_glass_sweep_h50_coarse \
  --batch_size 4 \
  --lr 1e-4 \
  --seed 6 \
  --k_steps 5 \
  --rollout_steps 340

"""