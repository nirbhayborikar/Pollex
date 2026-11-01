#!/usr/bin/env python3
"""
Inference-Time Gain Sweep
==========================
Sweeps dual-axis gain (γ_x, γ_z) on a trained SmolVLA checkpoint to find
the optimal forward-reach and vertical scaling that maximizes full-task success.

Uses the same scoring/analysis as run_smolvla_experiment_sweep.py.
"""

import argparse
import csv
import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from src.simulation.rollout_smolvla_calibrated import run_calibrated_rollout

# ---------------------------------------------------------------------------
# Constants (same as experiment sweep)
# ---------------------------------------------------------------------------

EXPERT_LIFT_CM = 8.17
MIN_SUCCESS_LIFT_CM = 3.0
MAX_ACCEPTABLE_LIFT_CM = 15.0
MAX_FINAL_LIFT_CM = 2.0
MAX_APPROACH_DISTANCE_CM = 3.0
MIN_GRASP_RUN_FRAMES = 10
MIN_LOWERING_CM = 2.0

# ---------------------------------------------------------------------------
# Gain grid
# ---------------------------------------------------------------------------

DEFAULT_GAMMA_X = [1.0, 1.25]
DEFAULT_GAMMA_Z = [1.0, 1.25]
DEFAULT_GAMMA_LIFT = [1.5, 2.0, 2.5, 3.0]


def load_policy(checkpoint_dir: str, horizon: int, device: torch.device) -> SmolVLAPolicy:
    checkpoint = Path(checkpoint_dir) / "model.safetensors"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    config_path = hf_hub_download(
        repo_id="lerobot/smolvla_base", filename="config.json",
    )
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    cfg.pop("type", None)
    cfg["load_vlm_weights"] = False
    cfg["device"] = str(device)
    cfg["chunk_size"] = int(horizon)
    cfg["n_action_steps"] = int(horizon)

    cfg["input_features"] = {
        k: PolicyFeature(type=FeatureType[v["type"]], shape=tuple(v["shape"]))
        for k, v in cfg["input_features"].items()
    }
    cfg["output_features"] = {
        k: PolicyFeature(type=FeatureType[v["type"]], shape=tuple(v["shape"]))
        for k, v in cfg["output_features"].items()
    }
    cfg["normalization_mapping"] = {
        k: NormalizationMode[v] for k, v in cfg["normalization_mapping"].items()
    }

    policy = SmolVLAPolicy(SmolVLAConfig(**cfg))
    policy.load_state_dict(load_file(str(checkpoint)), strict=False)
    policy.to(device)
    policy.eval()
    return policy


def longest_true_run(values: np.ndarray) -> int:
    best = current = 0
    for v in values:
        if bool(v):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def analyze(results: Dict[str, Any]) -> Dict[str, Any]:
    actions = np.asarray(results["actions"], dtype=np.float32)
    tumbler = np.asarray(results["tumbler_positions"], dtype=np.float32)
    contact = np.asarray(results["contact_flags"], dtype=np.float32) > 0.5
    grasp = np.asarray(results["grasp_flags"], dtype=np.float32) > 0.5

    initial_z = float(tumbler[0, 2])
    rel_z_cm = (tumbler[:, 2] - initial_z) * 100.0
    peak_idx = int(np.argmax(rel_z_cm))
    peak_lift_cm = float(rel_z_cm[peak_idx])
    final_lift_cm = float(rel_z_cm[-1])

    post_peak = rel_z_cm[peak_idx:]
    lowering_cm = float(peak_lift_cm - np.min(post_peak)) if len(post_peak) else 0.0

    first_contact = int(np.where(contact)[0][0]) if np.any(contact) else -1
    first_grasp = int(np.where(grasp)[0][0]) if np.any(grasp) else -1
    last_grasp = int(np.where(grasp)[0][-1]) if np.any(grasp) else -1

    released = bool(
        first_grasp >= 0 and last_grasp < len(grasp) - 1
        and np.any(~grasp[last_grasp + 1:])
    )

    gripper_closed = actions[:, 6] > 0.0
    dz_pos = actions[:, 2] > 0.01
    dz_neg = actions[:, 2] < -0.01

    grasp_run = longest_true_run(grasp)
    lift_run = longest_true_run(gripper_closed & dz_pos)
    lower_run = longest_true_run(gripper_closed & dz_neg)

    approach_ok = results["min_d_xy_cm"] <= MAX_APPROACH_DISTANCE_CM
    contact_ok = bool(results["contact_occurred"])
    grasp_ok = bool(results["grasp_occurred"])
    lift_ok = MIN_SUCCESS_LIFT_CM <= peak_lift_cm <= MAX_ACCEPTABLE_LIFT_CM
    retention_ok = grasp_run >= MIN_GRASP_RUN_FRAMES
    lowering_ok = lowering_cm >= MIN_LOWERING_CM
    height_ok = abs(final_lift_cm) <= MAX_FINAL_LIFT_CM
    release_ok = released

    full_success = all([
        approach_ok, contact_ok, grasp_ok, lift_ok,
        retention_ok, lowering_ok, height_ok, release_ok,
    ])

    lift_err = abs(peak_lift_cm - EXPERT_LIFT_CM)
    score = 0.0
    score += 10.0 if approach_ok else 0.0
    score += 15.0 if contact_ok else 0.0
    score += 20.0 if grasp_ok else 0.0
    score += 15.0 if retention_ok else 0.0
    score += 20.0 if lift_ok else 0.0
    score += 10.0 if lowering_ok else 0.0
    score += 5.0 if release_ok else 0.0
    score += 5.0 if height_ok else 0.0
    score -= min(lift_err, 30.0) * 0.5
    if peak_lift_cm > MAX_ACCEPTABLE_LIFT_CM:
        score -= min(peak_lift_cm - MAX_ACCEPTABLE_LIFT_CM, 50.0)
    if not contact_ok:
        score -= 20.0
    if not grasp_ok:
        score -= 20.0

    return {
        "score": float(score),
        "full_task_success": full_success,
        "approach_ok": approach_ok,
        "contact_ok": contact_ok,
        "grasp_ok": grasp_ok,
        "lift_ok": lift_ok,
        "grasp_retention_ok": retention_ok,
        "lowering_ok": lowering_ok,
        "release_ok": release_ok,
        "final_height_ok": height_ok,
        "peak_lift_cm": peak_lift_cm,
        "final_lift_cm": final_lift_cm,
        "lowering_after_peak_cm": lowering_cm,
        "min_d_xy_cm": float(results["min_d_xy_cm"]),
        "longest_grasp_run": int(grasp_run),
        "longest_lift_run": int(lift_run),
        "longest_lower_run": int(lower_run),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Sweep inference-time gains (γ_x, γ_z) on a trained checkpoint",
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/smolvla_glass_sweep_h50_coarse/h50_l15_low10_rel05/checkpoint",
    )
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--k_steps", type=int, default=5)
    parser.add_argument("--rollout_steps", type=int, default=340)
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--out_dir", default="outputs/gain_sweep")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--gamma_x_values", type=float, nargs="+", default=DEFAULT_GAMMA_X,
    )
    parser.add_argument(
        "--gamma_z_values", type=float, nargs="+", default=DEFAULT_GAMMA_Z,
    )
    parser.add_argument(
        "--gamma_lift_values", type=float, nargs="+", default=DEFAULT_GAMMA_LIFT,
    )

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")
    print(f"[*] Checkpoint: {args.checkpoint}")
    print(f"[*] Horizon: {args.horizon}, K: {args.k_steps}")

    combos = [
        (gx, gz, gl)
        for gx in args.gamma_x_values
        for gz in args.gamma_z_values
        for gl in args.gamma_lift_values
    ]
    print(f"[*] Testing {len(combos)} gain combinations")

    records = []

    for idx, (gx, gz, gl) in enumerate(combos, start=1):
        name = f"gx{gx:.2f}_gz{gz:.2f}_gl{gl:.2f}"
        result_path = out_dir / f"{name}_result.json"

        print(f"\n{'=' * 80}")
        print(f"[{idx}/{len(combos)}] γ_x={gx:.2f}  γ_z={gz:.2f}  γ_lift={gl:.2f}")
        print("=" * 80)

        if args.skip_existing and result_path.exists():
            with open(result_path, encoding="utf-8") as f:
                record = json.load(f)
            print(f"[+] Reusing: {result_path}")
            records.append(record)
            continue

        policy = load_policy(args.checkpoint, args.horizon, device)

        try:
            t0 = time.time()
            results = run_calibrated_rollout(
                policy=policy,
                gamma_x=gx,
                gamma_z=gz,
                gamma_lift=gl,
                max_steps=args.rollout_steps,
                chunk_exec_steps=args.k_steps,
                device=device,
                render_gui=args.gui,
                seed=args.seed,
            )
            elapsed = time.time() - t0

            metrics = analyze(results)

            record = {
                "name": name,
                "gamma_x": gx,
                "gamma_z": gz,
                "gamma_lift": gl,
                "status": "completed",
                "rollout_time_s": elapsed,
                **metrics,
            }

            # Save rollout NPZ for the best candidates
            if metrics["grasp_ok"] and metrics["peak_lift_cm"] > 1.0:
                npz_path = out_dir / f"{name}_rollout.npz"
                np.savez_compressed(
                    npz_path,
                    actions=results["actions"],
                    eef_positions=results["eef_positions"],
                    tumbler_positions=results["tumbler_positions"],
                    glass_z=results["glass_z"],
                    contact_flags=results["contact_flags"],
                    grasp_flags=results["grasp_flags"],
                )
                record["rollout_npz"] = str(npz_path)

        except Exception as err:
            record = {
                "name": name,
                "gamma_x": gx,
                "gamma_z": gz,
                "status": "failed",
                "error": str(err),
                "score": -1e9,
                "full_task_success": False,
            }
            print(f"[!] Failed: {err}")

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        records.append(record)

        del policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Leaderboard
    # -----------------------------------------------------------------------
    ranked = sorted(
        records,
        key=lambda r: (r.get("full_task_success", False), r.get("score", -1e9)),
        reverse=True,
    )

    with open(out_dir / "gain_leaderboard.json", "w", encoding="utf-8") as f:
        json.dump(ranked, f, indent=2)

    fields = [
        "rank", "name", "gamma_x", "gamma_z", "gamma_lift", "score",
        "full_task_success", "contact_ok", "grasp_ok", "peak_lift_cm",
        "final_lift_cm", "lowering_after_peak_cm", "release_ok",
        "min_d_xy_cm", "longest_grasp_run",
    ]
    with open(out_dir / "gain_leaderboard.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rank, r in enumerate(ranked, start=1):
            row = {"rank": rank}
            for k in fields:
                if k != "rank":
                    row[k] = r.get(k, "")
            w.writerow(row)

    print(f"\n{'=' * 80}")
    print("GAIN SWEEP LEADERBOARD")
    print("=" * 80)
    for rank, r in enumerate(ranked, start=1):
        print(
            f"{rank:2d}. γ_x={r.get('gamma_x', '?'):5.2f} γ_z={r.get('gamma_z', '?'):5.2f} "
            f"γ_lift={r.get('gamma_lift', '?'):5.2f} "
            f"score={r.get('score', -1e9):7.2f} "
            f"success={r.get('full_task_success', False)} "
            f"grasp={r.get('grasp_ok', False)} "
            f"peak={r.get('peak_lift_cm', 0):6.2f}cm "
            f"final={r.get('final_lift_cm', 0):+6.2f}cm"
        )

    best = ranked[0]
    with open(out_dir / "best_gain.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print(f"\n[+] BEST: γ_x={best.get('gamma_x')} γ_z={best.get('gamma_z')} γ_lift={best.get('gamma_lift')}")
    print(f"[+] Score: {best.get('score'):.2f}, Peak lift: {best.get('peak_lift_cm', 0):.2f}cm")
    print(f"[+] Results in: {out_dir}")


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
python3 -m src.simulation.sweep_inference_gains \\
  --checkpoint outputs/smolvla_glass_sweep_h50_coarse/h50_l15_low10_rel05/checkpoint \\
  --horizon 50 \\
  --k_steps 5 \\
  --seed 6 \\
  --out_dir outputs/gain_sweep \\
  --skip_existing
"""
