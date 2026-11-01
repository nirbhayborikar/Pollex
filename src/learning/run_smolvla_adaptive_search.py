#!/usr/bin/env python3
"""
SmolVLA Adaptive Hyperparameter Search
=======================================
Three-stage pipeline:

  Coarse   — broad grid over H, K, lr, epochs, phase distributions
  Refine   — narrow perturbation of top-N coarse candidates
  Validate — multi-seed robustness testing of top-M refined candidates

Compatible with:
  - src.learning.train_smolvla_phase_balanced_v5  (trainer)
  - src.simulation.rollout_smolvla_baseline        (rollout)
"""

import argparse
import copy
import csv
import gc
import hashlib
import itertools
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

import src.learning.train_smolvla_phase_balanced_v5 as trainer
import src.simulation.rollout_smolvla_baseline as rollout_module
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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

PHASE_NAMES = [
    "hover", "approach", "descent_pregrasp", "grasp_closing",
    "postgrasp_lift", "postgrasp_lower", "release", "retract",
]

# ---------------------------------------------------------------------------
# Named phase-target distributions
# ---------------------------------------------------------------------------

PHASE_DISTRIBUTIONS = {
    "balanced": {
        "hover": 0.05, "approach": 0.10, "descent_pregrasp": 0.15,
        "grasp_closing": 0.10, "postgrasp_lift": 0.25,
        "postgrasp_lower": 0.12, "release": 0.13, "retract": 0.10,
    },
    "lift_heavy": {
        "hover": 0.04, "approach": 0.08, "descent_pregrasp": 0.12,
        "grasp_closing": 0.08, "postgrasp_lift": 0.30,
        "postgrasp_lower": 0.15, "release": 0.13, "retract": 0.10,
    },
    "grasp_heavy": {
        "hover": 0.04, "approach": 0.10, "descent_pregrasp": 0.18,
        "grasp_closing": 0.15, "postgrasp_lift": 0.22,
        "postgrasp_lower": 0.10, "release": 0.11, "retract": 0.10,
    },
    "lower_heavy": {
        "hover": 0.04, "approach": 0.08, "descent_pregrasp": 0.12,
        "grasp_closing": 0.08, "postgrasp_lift": 0.20,
        "postgrasp_lower": 0.20, "release": 0.18, "retract": 0.10,
    },
    "uniform_ish": {
        "hover": 0.08, "approach": 0.12, "descent_pregrasp": 0.12,
        "grasp_closing": 0.12, "postgrasp_lift": 0.15,
        "postgrasp_lower": 0.15, "release": 0.13, "retract": 0.13,
    },
}

# ---------------------------------------------------------------------------
# Coarse search grid
# ---------------------------------------------------------------------------

COARSE_GRID = {
    "chunk_size": [10, 20, 50],
    "k_steps": [3, 5],
    "lr": [5e-5, 1e-4],
    "epochs": [15],
    "phase_dist": ["balanced", "lift_heavy", "grasp_heavy"],
}

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_candidate(candidate: Dict[str, Any]) -> None:
    cs = candidate["chunk_size"]
    ks = candidate["k_steps"]
    ep = candidate["epochs"]
    lr_val = candidate["lr"]
    pt = candidate["phase_targets"]

    if cs < 1:
        raise ValueError(f"chunk_size must be >= 1, got {cs}")
    if ks < 1:
        raise ValueError(f"k_steps must be >= 1, got {ks}")
    if ks > cs:
        raise ValueError(f"k_steps ({ks}) must be <= chunk_size ({cs})")
    if ep < 1:
        raise ValueError(f"epochs must be >= 1, got {ep}")
    if lr_val <= 0:
        raise ValueError(f"lr must be > 0, got {lr_val}")

    missing = set(PHASE_NAMES) - set(pt.keys())
    extra = set(pt.keys()) - set(PHASE_NAMES)
    if missing:
        raise ValueError(f"Missing phases: {missing}")
    if extra:
        raise ValueError(f"Unknown phases: {extra}")

    total = sum(pt.values())
    if abs(total - 1.0) > 1e-4:
        raise ValueError(f"Phase targets sum to {total:.6f}, expected 1.0")


def normalize_phase_targets(pt: Dict[str, float]) -> Dict[str, float]:
    total = sum(pt.values())
    if total <= 0:
        raise ValueError("Phase targets sum to 0")
    return {k: v / total for k, v in pt.items()}


# ---------------------------------------------------------------------------
# Trainer / model helpers (mirrors run_smolvla_experiment_sweep patterns)
# ---------------------------------------------------------------------------


def configure_trainer(candidate: Dict[str, Any]) -> None:
    horizon = int(candidate["chunk_size"])
    trainer.CHUNK_SIZE = horizon
    trainer.PHASE_TARGETS.clear()
    trainer.PHASE_TARGETS.update(candidate["phase_targets"])

    print(f"[*] Trainer horizon: {trainer.CHUNK_SIZE}")
    print("[*] Trainer phase targets:")
    for phase, prob in trainer.PHASE_TARGETS.items():
        print(f"    {phase:24s} {prob * 100:5.1f}%")


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

    with open(config_path, encoding="utf-8") as f:
        config_dict = json.load(f)

    config_dict.pop("type", None)
    config_dict["load_vlm_weights"] = False
    config_dict["device"] = str(device)
    config_dict["chunk_size"] = int(horizon)
    config_dict["n_action_steps"] = int(horizon)

    config_dict["input_features"] = {
        key: PolicyFeature(
            type=FeatureType[val["type"]],
            shape=tuple(val["shape"]),
        )
        for key, val in config_dict["input_features"].items()
    }
    config_dict["output_features"] = {
        key: PolicyFeature(
            type=FeatureType[val["type"]],
            shape=tuple(val["shape"]),
        )
        for key, val in config_dict["output_features"].items()
    }
    config_dict["normalization_mapping"] = {
        key: NormalizationMode[val]
        for key, val in config_dict["normalization_mapping"].items()
    }

    policy = SmolVLAPolicy(SmolVLAConfig(**config_dict))
    policy.load_state_dict(load_file(str(checkpoint)), strict=False)
    policy.to(device)
    policy.eval()

    if policy.config.chunk_size != horizon:
        raise RuntimeError(
            f"Policy chunk mismatch: {policy.config.chunk_size} != {horizon}"
        )

    return policy


# ---------------------------------------------------------------------------
# Analysis / scoring (identical to sweep — kept in sync)
# ---------------------------------------------------------------------------


def longest_true_run(values: np.ndarray) -> int:
    best = 0
    current = 0
    for v in values:
        if bool(v):
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
        float(peak_lift_cm - np.min(post_peak)) if len(post_peak) else 0.0
    )

    first_contact = int(np.where(contact)[0][0]) if np.any(contact) else -1
    first_grasp = int(np.where(grasp)[0][0]) if np.any(grasp) else -1
    last_grasp = int(np.where(grasp)[0][-1]) if np.any(grasp) else -1

    released_after_grasp = bool(
        first_grasp >= 0
        and last_grasp < len(grasp) - 1
        and np.any(~grasp[last_grasp + 1 :])
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
    lift_ok = MIN_SUCCESS_LIFT_CM <= peak_lift_cm <= MAX_ACCEPTABLE_LIFT_CM
    grasp_retention_ok = grasp_run >= MIN_GRASP_RUN_FRAMES
    lowering_ok = lowering_cm >= MIN_LOWERING_CM
    final_height_ok = abs(final_lift_cm) <= MAX_FINAL_LIFT_CM
    release_ok = released_after_grasp

    full_task_success = all([
        approach_ok, contact_ok, grasp_ok, lift_ok,
        grasp_retention_ok, lowering_ok, final_height_ok, release_ok,
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
        score -= min(peak_lift_cm - MAX_ACCEPTABLE_LIFT_CM, 50.0)
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


def save_rollout(results: Dict[str, Any], output_dir: Path) -> Path:
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


# ---------------------------------------------------------------------------
# Train / evaluate a single candidate
# ---------------------------------------------------------------------------


def train_candidate(
    candidate: Dict[str, Any],
    args: argparse.Namespace,
    checkpoint_dir: Path,
) -> Dict[str, Any]:
    configure_trainer(candidate)

    t0 = time.time()

    trainer.train_phase_balanced_v5(
        dataset_dir=args.dataset_dir,
        expert_replay=args.expert_replay,
        out_dir=str(checkpoint_dir),
        epochs=int(candidate["epochs"]),
        batch_size=args.batch_size,
        lr=float(candidate["lr"]),
        seed=args.seed,
    )

    elapsed = time.time() - t0
    ckpt = checkpoint_dir / "model.safetensors"

    if not ckpt.exists():
        raise RuntimeError(f"Training did not create checkpoint: {ckpt}")

    return {
        "training_time_s": float(elapsed),
        "checkpoint_sha256": sha256_file(ckpt),
    }


def evaluate_candidate(
    candidate: Dict[str, Any],
    args: argparse.Namespace,
    checkpoint_dir: Path,
    rollout_dir: Path,
    seed: int,
) -> Dict[str, Any]:
    device = select_device()
    policy = load_policy_with_horizon(
        checkpoint_dir=str(checkpoint_dir),
        horizon=int(candidate["chunk_size"]),
        device=device,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    t0 = time.time()

    results = rollout_module.run_baseline_rollout(
        policy=policy,
        task_prompt=DEFAULT_TASK_PROMPT,
        max_steps=args.rollout_steps,
        chunk_exec_steps=int(candidate["k_steps"]),
        device=device,
        render_gui=args.gui,
    )

    elapsed = time.time() - t0
    rollout_path = save_rollout(results, rollout_dir)
    metrics = analyze_rollout(results)
    metrics["rollout_time_s"] = float(elapsed)
    metrics["rollout_path"] = str(rollout_path)

    del policy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


def generate_coarse_candidates() -> List[Dict[str, Any]]:
    candidates = []
    grid = COARSE_GRID
    combos = list(itertools.product(
        grid["chunk_size"],
        grid["k_steps"],
        grid["lr"],
        grid["epochs"],
        grid["phase_dist"],
    ))

    for cs, ks, lr_val, ep, dist_name in combos:
        if ks > cs:
            continue
        name = f"H{cs}_K{ks}_lr{lr_val:.0e}_ep{ep}_{dist_name}"
        candidates.append({
            "name": name,
            "chunk_size": cs,
            "k_steps": ks,
            "lr": lr_val,
            "epochs": ep,
            "phase_dist_name": dist_name,
            "phase_targets": copy.deepcopy(PHASE_DISTRIBUTIONS[dist_name]),
        })

    return candidates


def perturb_phase_targets(
    base: Dict[str, float],
    rng: np.random.Generator,
    magnitude: float = 0.03,
) -> Dict[str, float]:
    keys = sorted(base.keys())
    vals = np.array([base[k] for k in keys], dtype=np.float64)
    noise = rng.normal(0, magnitude, size=len(vals))
    perturbed = np.clip(vals + noise, 0.01, 1.0)
    perturbed /= perturbed.sum()
    return {k: float(v) for k, v in zip(keys, perturbed)}


def generate_refine_candidates(
    top_records: List[Dict[str, Any]],
    n_variants: int = 3,
    refine_seed: int = 42,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(refine_seed)
    candidates = []

    lr_multipliers = [0.5, 1.0, 2.0]
    epoch_offsets = [-5, 0, +5]

    for record in top_records:
        base_cs = record["chunk_size"]
        base_ks = record.get("k_steps", 5)
        base_lr = record.get("lr", 1e-4)
        base_ep = record.get("epochs", 15)
        base_pt = record.get("phase_targets", PHASE_DISTRIBUTIONS["balanced"])
        base_name = record["name"]

        for vi in range(n_variants):
            lr_mult = rng.choice(lr_multipliers)
            ep_off = int(rng.choice(epoch_offsets))
            new_lr = base_lr * lr_mult
            new_ep = max(5, base_ep + ep_off)
            new_pt = perturb_phase_targets(base_pt, rng)

            k_choices = [k for k in [base_ks - 1, base_ks, base_ks + 1]
                         if 1 <= k <= base_cs]
            new_ks = int(rng.choice(k_choices))

            name = f"refine_{base_name}_v{vi}"
            candidates.append({
                "name": name,
                "chunk_size": base_cs,
                "k_steps": new_ks,
                "lr": new_lr,
                "epochs": new_ep,
                "phase_targets": new_pt,
                "parent": base_name,
            })

    return candidates


# ---------------------------------------------------------------------------
# Run a stage (train + evaluate all candidates)
# ---------------------------------------------------------------------------


def run_stage(
    candidates: List[Dict[str, Any]],
    args: argparse.Namespace,
    stage_dir: Path,
    stage_name: str,
    seed: int,
) -> List[Dict[str, Any]]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for idx, candidate in enumerate(candidates, start=1):
        validate_candidate(candidate)

        name = candidate["name"]
        cand_dir = stage_dir / name
        ckpt_dir = cand_dir / "checkpoint"
        roll_dir = cand_dir / "rollout"
        result_path = cand_dir / "result.json"

        print(f"\n{'#' * 100}")
        print(
            f"[{stage_name}] {idx}/{len(candidates)}: {name} | "
            f"H={candidate['chunk_size']} K={candidate['k_steps']} "
            f"lr={candidate['lr']:.1e} ep={candidate['epochs']}"
        )
        print("#" * 100)

        if args.skip_existing and result_path.exists():
            with open(result_path, encoding="utf-8") as f:
                record = json.load(f)
            print(f"[+] Reusing existing result: {result_path}")
            records.append(record)
            continue

        cand_dir.mkdir(parents=True, exist_ok=True)

        record = {
            "name": name,
            "stage": stage_name,
            "status": "running",
            "chunk_size": int(candidate["chunk_size"]),
            "k_steps": int(candidate["k_steps"]),
            "lr": float(candidate["lr"]),
            "epochs": int(candidate["epochs"]),
            "phase_targets": candidate["phase_targets"],
            "checkpoint_dir": str(ckpt_dir),
        }

        try:
            train_info = train_candidate(candidate, args, ckpt_dir)
            eval_info = evaluate_candidate(
                candidate, args, ckpt_dir, roll_dir, seed,
            )

            record.update(train_info)
            record.update(eval_info)
            record["status"] = "completed"

        except Exception as err:
            record["status"] = "failed"
            record["error"] = str(err)
            record["score"] = -1e9
            record["full_task_success"] = False
            print(f"[!] Failed: {name}")
            print(f"[!] {type(err).__name__}: {err}")

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        records.append(record)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return records


# ---------------------------------------------------------------------------
# Multi-seed validation
# ---------------------------------------------------------------------------


def run_multi_seed_validation(
    candidates: List[Dict[str, Any]],
    source_records: List[Dict[str, Any]],
    args: argparse.Namespace,
    validate_dir: Path,
    seeds: List[int],
) -> List[Dict[str, Any]]:
    validate_dir.mkdir(parents=True, exist_ok=True)

    source_lookup = {r["name"]: r for r in source_records}
    aggregated = []

    for candidate in candidates:
        name = candidate["name"]
        source = source_lookup.get(name, {})
        ckpt_dir = Path(source.get("checkpoint_dir", ""))

        if not (ckpt_dir / "model.safetensors").exists():
            print(f"[!] No checkpoint for {name}, skipping validation")
            continue

        cand_dir = validate_dir / name
        cand_dir.mkdir(parents=True, exist_ok=True)
        result_path = cand_dir / "multi_seed_result.json"

        if args.skip_existing and result_path.exists():
            with open(result_path, encoding="utf-8") as f:
                agg = json.load(f)
            print(f"[+] Reusing multi-seed result: {result_path}")
            aggregated.append(agg)
            continue

        print(f"\n{'=' * 80}")
        print(f"[VALIDATE] {name} — {len(seeds)} seeds: {seeds}")
        print("=" * 80)

        seed_results = []

        for seed in seeds:
            roll_dir = cand_dir / f"rollout_seed{seed}"
            print(f"  [seed={seed}] Evaluating ...")

            try:
                metrics = evaluate_candidate(
                    candidate, args, ckpt_dir, roll_dir, seed,
                )
                metrics["seed"] = seed
                metrics["status"] = "completed"
            except Exception as err:
                metrics = {
                    "seed": seed,
                    "status": "failed",
                    "error": str(err),
                    "score": -1e9,
                    "full_task_success": False,
                }
                print(f"  [!] seed={seed} failed: {err}")

            seed_results.append(metrics)

        completed = [r for r in seed_results if r["status"] == "completed"]
        n_completed = len(completed)

        if n_completed == 0:
            agg = {
                "name": name,
                "n_seeds": len(seeds),
                "n_completed": 0,
                "mean_score": -1e9,
                "success_rate": 0.0,
                "full_task_success": False,
                "seed_results": seed_results,
            }
        else:
            scores = [r["score"] for r in completed]
            successes = [r["full_task_success"] for r in completed]
            peak_lifts = [r["peak_lift_cm"] for r in completed]

            agg = {
                "name": name,
                "stage": "validate",
                "n_seeds": len(seeds),
                "n_completed": n_completed,
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "min_score": float(np.min(scores)),
                "max_score": float(np.max(scores)),
                "success_rate": float(np.mean(successes)),
                "full_task_success": all(successes),
                "mean_peak_lift_cm": float(np.mean(peak_lifts)),
                "std_peak_lift_cm": float(np.std(peak_lifts)),
                "chunk_size": candidate["chunk_size"],
                "k_steps": candidate["k_steps"],
                "lr": candidate["lr"],
                "epochs": candidate["epochs"],
                "phase_targets": candidate["phase_targets"],
                "checkpoint_dir": str(ckpt_dir),
                "seed_results": seed_results,
            }

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2)

        aggregated.append(agg)

    return aggregated


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

LEADERBOARD_FIELDS = [
    "rank", "name", "stage", "status", "chunk_size", "k_steps", "lr",
    "epochs", "score", "full_task_success", "contact_ok", "grasp_ok",
    "peak_lift_cm", "final_lift_cm", "lowering_after_peak_cm", "release_ok",
    "min_d_xy_cm", "longest_grasp_run", "longest_lift_run",
    "longest_lower_run", "positive_dz_percent", "negative_dz_percent",
    "training_time_s", "rollout_time_s", "checkpoint_dir",
]


def rank_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        records,
        key=lambda r: (r.get("full_task_success", False), r.get("score", -1e9)),
        reverse=True,
    )


def write_leaderboard(
    records: List[Dict[str, Any]],
    output_dir: Path,
    label: str = "leaderboard",
) -> None:
    ranked = rank_records(records)

    json_path = output_dir / f"{label}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ranked, f, indent=2)

    csv_path = output_dir / f"{label}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LEADERBOARD_FIELDS)
        writer.writeheader()
        for rank, item in enumerate(ranked, start=1):
            row = {"rank": rank}
            for key in LEADERBOARD_FIELDS:
                if key != "rank":
                    row[key] = item.get(key, "")
            writer.writerow(row)

    print(f"\n{'=' * 100}")
    print(f"{label.upper()}")
    print("=" * 100)

    for rank, item in enumerate(ranked, start=1):
        print(
            f"{rank:2d}. {item.get('name', '?'):30s} "
            f"score={item.get('score', float('-inf')):7.2f} "
            f"success={item.get('full_task_success', False)} "
            f"contact={item.get('contact_ok', False)} "
            f"grasp={item.get('grasp_ok', False)} "
            f"peak={item.get('peak_lift_cm', 0.0):6.2f}cm "
            f"final={item.get('final_lift_cm', 0.0):+6.2f}cm"
        )

    successful = [r for r in ranked if r.get("full_task_success", False)]
    best = successful[0] if successful else (ranked[0] if ranked else {})

    best_path = output_dir / f"{label}_best.json"
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print(f"\n[+] JSON: {json_path}")
    print(f"[+] CSV:  {csv_path}")
    print(f"[+] Best: {best_path}")


def write_validation_leaderboard(
    aggregated: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    ranked = sorted(
        aggregated,
        key=lambda r: (
            r.get("full_task_success", False),
            r.get("success_rate", 0.0),
            r.get("mean_score", -1e9),
        ),
        reverse=True,
    )

    json_path = output_dir / "validation_leaderboard.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ranked, f, indent=2)

    print(f"\n{'=' * 100}")
    print("VALIDATION LEADERBOARD (multi-seed)")
    print("=" * 100)

    for rank, item in enumerate(ranked, start=1):
        print(
            f"{rank:2d}. {item.get('name', '?'):30s} "
            f"mean_score={item.get('mean_score', -1e9):7.2f} "
            f"(±{item.get('std_score', 0):5.2f}) "
            f"success_rate={item.get('success_rate', 0.0):.0%} "
            f"peak={item.get('mean_peak_lift_cm', 0.0):6.2f}cm"
        )

    print(f"\n[+] Saved: {json_path}")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "SmolVLA Adaptive Hyperparameter Search: "
            "coarse → refine → validate"
        ),
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
    parser.add_argument("--search_dir", default="outputs/smolvla_adaptive_search")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--rollout_steps", type=int, default=340)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")

    parser.add_argument(
        "--stage",
        choices=["all", "coarse", "refine", "validate"],
        default="all",
        help="Which stage(s) to run.",
    )
    parser.add_argument(
        "--top_n_coarse",
        type=int,
        default=5,
        help="Number of top coarse candidates to refine.",
    )
    parser.add_argument(
        "--refine_variants",
        type=int,
        default=3,
        help="Number of perturbation variants per top coarse candidate.",
    )
    parser.add_argument(
        "--top_n_refine",
        type=int,
        default=3,
        help="Number of top refined candidates for multi-seed validation.",
    )
    parser.add_argument(
        "--validation_seeds",
        type=int,
        nargs="+",
        default=[6, 42, 123, 7, 99],
        help="Seeds for multi-seed validation stage.",
    )

    args = parser.parse_args()

    search_dir = Path(args.search_dir)
    search_dir.mkdir(parents=True, exist_ok=True)

    run_coarse = args.stage in ("all", "coarse")
    run_refine = args.stage in ("all", "refine")
    run_validate = args.stage in ("all", "validate")

    search_meta = {
        "search_dir": str(search_dir),
        "dataset_dir": args.dataset_dir,
        "expert_replay": args.expert_replay,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "rollout_steps": args.rollout_steps,
        "coarse_grid": COARSE_GRID,
        "top_n_coarse": args.top_n_coarse,
        "refine_variants": args.refine_variants,
        "top_n_refine": args.top_n_refine,
        "validation_seeds": args.validation_seeds,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with open(search_dir / "search_config.json", "w", encoding="utf-8") as f:
        json.dump(search_meta, f, indent=2)

    # -------------------------------------------------------------------
    # Stage 1: Coarse
    # -------------------------------------------------------------------
    coarse_dir = search_dir / "coarse"
    coarse_records = []

    if run_coarse:
        print("\n" + "=" * 100)
        print("STAGE 1: COARSE SEARCH")
        print("=" * 100)

        coarse_candidates = generate_coarse_candidates()
        print(f"[*] Generated {len(coarse_candidates)} coarse candidates")

        coarse_records = run_stage(
            coarse_candidates, args, coarse_dir, "coarse", args.seed,
        )
        write_leaderboard(coarse_records, coarse_dir, "coarse_leaderboard")

    elif coarse_dir.exists():
        lb_path = coarse_dir / "coarse_leaderboard.json"
        if lb_path.exists():
            with open(lb_path, encoding="utf-8") as f:
                coarse_records = json.load(f)
            print(f"[+] Loaded {len(coarse_records)} coarse results from disk")

    # -------------------------------------------------------------------
    # Stage 2: Refine
    # -------------------------------------------------------------------
    refine_dir = search_dir / "refine"
    refine_records = []

    if run_refine:
        if not coarse_records:
            raise RuntimeError(
                "No coarse results available. Run coarse stage first."
            )

        print("\n" + "=" * 100)
        print("STAGE 2: REFINEMENT")
        print("=" * 100)

        ranked_coarse = rank_records(coarse_records)
        top_coarse = ranked_coarse[: args.top_n_coarse]
        print(f"[*] Top {len(top_coarse)} coarse candidates selected:")
        for r in top_coarse:
            print(f"    {r['name']} — score={r.get('score', -1e9):.2f}")

        refine_candidates = generate_refine_candidates(
            top_coarse,
            n_variants=args.refine_variants,
            refine_seed=args.seed + 100,
        )
        print(f"[*] Generated {len(refine_candidates)} refinement candidates")

        refine_records = run_stage(
            refine_candidates, args, refine_dir, "refine", args.seed,
        )

        all_stage2 = coarse_records + refine_records
        write_leaderboard(all_stage2, refine_dir, "refine_leaderboard")

    elif refine_dir.exists():
        lb_path = refine_dir / "refine_leaderboard.json"
        if lb_path.exists():
            with open(lb_path, encoding="utf-8") as f:
                refine_records = json.load(f)
            print(f"[+] Loaded {len(refine_records)} refine results from disk")

    # -------------------------------------------------------------------
    # Stage 3: Multi-seed validation
    # -------------------------------------------------------------------
    validate_dir = search_dir / "validate"

    if run_validate:
        all_candidates_records = coarse_records + refine_records
        if not all_candidates_records:
            raise RuntimeError(
                "No results available. Run coarse/refine stages first."
            )

        print("\n" + "=" * 100)
        print("STAGE 3: MULTI-SEED VALIDATION")
        print("=" * 100)

        ranked_all = rank_records(all_candidates_records)
        top_for_validation = ranked_all[: args.top_n_refine]

        val_candidates = []
        for rec in top_for_validation:
            val_candidates.append({
                "name": rec["name"],
                "chunk_size": rec["chunk_size"],
                "k_steps": rec.get("k_steps", 5),
                "lr": rec.get("lr", 1e-4),
                "epochs": rec.get("epochs", 15),
                "phase_targets": rec.get(
                    "phase_targets", PHASE_DISTRIBUTIONS["balanced"]
                ),
            })

        aggregated = run_multi_seed_validation(
            val_candidates,
            all_candidates_records,
            args,
            validate_dir,
            args.validation_seeds,
        )

        write_validation_leaderboard(aggregated, validate_dir)

        final_records = all_candidates_records + [
            {**a, "score": a.get("mean_score", -1e9)}
            for a in aggregated
        ]
        write_leaderboard(
            rank_records(all_candidates_records),
            search_dir,
            "final_leaderboard",
        )
        write_validation_leaderboard(aggregated, search_dir)

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    search_meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(search_dir / "search_config.json", "w", encoding="utf-8") as f:
        json.dump(search_meta, f, indent=2)

    print("\n" + "=" * 100)
    print("ADAPTIVE SEARCH COMPLETE")
    print(f"Results in: {search_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()


"""
Usage:
------

source /home/nvidia/Documents/Pollex/.venv/bin/activate && \\
export PYTHONPATH="$HOME/Documents/vla/LIBERO:$HOME/Documents/Pollex:$PYTHONPATH" && \\
unset MUJOCO_GL QT_DEBUG_PLUGINS QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH && \\
export DISPLAY=:10 && \\
export XAUTHORITY="$HOME/.Xauthority" && \\
cd ~/Documents/Pollex && \\
python3 -m src.learning.run_smolvla_adaptive_search \\
  --search_dir outputs/smolvla_adaptive_search \\
  --batch_size 4 \\
  --seed 6 \\
  --skip_existing \\
  --stage all

Run only coarse:
  python3 -m src.learning.run_smolvla_adaptive_search --stage coarse

Resume from coarse, run refine + validate:
  python3 -m src.learning.run_smolvla_adaptive_search --stage refine --skip_existing
  python3 -m src.learning.run_smolvla_adaptive_search --stage validate --skip_existing
"""
