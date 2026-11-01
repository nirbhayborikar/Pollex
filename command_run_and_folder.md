# Pollex — Commands, Inputs, and Outputs Reference

Every command used to produce the results in the README. Run all commands from the project root (`~/Documents/Pollex`).

## Environment Setup (run once)

The setup script creates `.venv`, clones [LeRobot](https://github.com/huggingface/lerobot) and [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) into sibling directories, and installs everything:

```bash
cd ~/Documents/Pollex
bash setup_env.sh          # clones LeRobot → ../lerobot, LIBERO → ../vla/LIBERO
source .venv/bin/activate
```

This installs:
- **Pollex** (`pip install -e .`) — this project
- **LeRobot** (`pip install -e ../lerobot`) — SmolVLA policy architecture
- **LIBERO** (`pip install -e ../vla/LIBERO`) — simulation environment runner

**Common environment variables** (needed for all simulation/rollout commands, Steps 3–8):

```bash
source /home/nvidia/Documents/Pollex/.venv/bin/activate
export PYTHONPATH="$HOME/Documents/vla/LIBERO:$HOME/Documents/Pollex:$PYTHONPATH"
export DISPLAY=:10
export XAUTHORITY="$HOME/.Xauthority"
unset MUJOCO_GL QT_DEBUG_PLUGINS QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH
```

---

## Step 1: Hand Tracking (MediaPipe)

**File**: `src/tracking_algorithm/hand_tracker.py`
**Input**: `datasets/egocentric_video/cup_lifting_01.mp4` (iPhone 1080p, 30fps, 11s)
**Command**:

```bash
python3 src/tracking_algorithm/hand_tracker.py \
    --video datasets/egocentric_video/cup_lifting_01.mp4
```

**Output**: `media/cup_lifting_01_annotated.gif` — annotated video with 21 landmarks, pinch distance, and gripper state overlay.

---

## Step 2: Dense Point Tracking (CoTracker3)

**File**: `src/tracking_algorithm/cotracker_dense_point_tracking.py`
**Input**: `datasets/egocentric_video/cup_lifting_01.mp4`
**Command**:

```bash
PYTHONPATH=. python3 src/tracking_algorithm/cotracker_dense_point_tracking.py \
    --video datasets/egocentric_video/cup_lifting_01.mp4 \
    --output_dir datasets/cotracker_trajectories
```

**Output directory**: `datasets/cotracker_trajectories/`

| File | Content |
|:---|:---|
| `cup_lifting_01_ct_trajectory.npz` | 37-point tracked trajectories (21 hand + 16 glass grid), co-motion scores, gripper states |
| `cup_lifting_01_ct_annotated.mp4` | Annotated video with tracked points, velocity vectors, and co-motion HUD |

---

## Step 3: Motion Retargeting (CoTracker → Panda 7D Actions)

**File**: `src/retargeting/cotracker_to_panda.py`
**Input**: `datasets/cotracker_trajectories/cup_lifting_01_ct_trajectory.npz`
**Command**:

```bash
PYTHONPATH=. python3 -m src.retargeting.cotracker_to_panda \
    --trajectory datasets/cotracker_trajectories/cup_lifting_01_ct_trajectory.npz \
    --output_dir datasets/cotracker_iterations/iteration_07_blended_feedback
```

**Output directory**: `datasets/cotracker_iterations/iteration_07_blended_feedback/`

| File | Content |
|:---|:---|
| `cup_lifting_01_ct_actions.npz` | 340-step 7D OSC action sequence (dx, dy, dz, droll, dpitch, dyaw, gripper) |

---

## Step 4: Expert Trajectory Replay in MuJoCo (Iteration 11)

**File**: `src/simulation/replay_trajectory.py`
**Input**: Retargeted actions from Step 3 + annotated video from Step 2
**Command**:

```bash
PYTHONPATH=. python3 -m src.simulation.replay_trajectory \
    --actions datasets/cotracker_iterations/iteration_07_blended_feedback/cup_lifting_01_ct_actions.npz \
    --out_video datasets/cotracker_iterations/iteration_11_full_mujoco/cup_lifting_01_iteration_11.mp4 \
    --out_dataset datasets/cotracker_iterations/iteration_11_full_mujoco/cup_lifting_01_replay.npz \
    --human_video datasets/cotracker_trajectories/cup_lifting_01_ct_annotated.mp4 \
    --task GlassLift \
    --closed_loop_pregrasp \
    --target_pad_x_offset -0.017 \
    --pregrasp_kp_x 4.0 \
    --pregrasp_max_correction 0.07 \
    --pregrasp_deadband_m 0.002 \
    --target_pad_z_offset 0.015 \
    --pregrasp_kp_z 3.0 \
    --pregrasp_max_z_correction 0.04 \
    --pregrasp_z_deadband_m 0.002 \
    --pregrasp_control_start_frac 0.45 \
    --grasp_ready_x_tolerance_m 0.003 \
    --grasp_ready_z_tolerance_m 0.002 \
    --grasp_ready_consecutive_frames 3 \
    --grasp_control_hold_frames 6
```

**Output directory**: `datasets/cotracker_iterations/iteration_11_full_mujoco/`

| File | Content |
|:---|:---|
| `cup_lifting_01_iteration_11.mp4` | Side-by-side video: human demo (left) vs MuJoCo replay (right) |
| `cup_lifting_01_replay.npz` | 340 transitions: RGB frames (256x256), 8D Panda state, 7D executed actions |
| `metrics.txt` | Peak lift, contact frame, grasp frame, final height |

**Result**: Expert demonstration with +8.17 cm pick-and-place success.

---

## Step 5: Export LeRobot Dataset

**File**: `src/learning/export_lerobot_dataset.py`
**Input**: Expert replay NPZ from Step 4
**Command**:

```bash
PYTHONPATH=. python3 -m src.learning.export_lerobot_dataset \
    --replay datasets/cotracker_iterations/iteration_11_full_mujoco/cup_lifting_01_replay.npz \
    --output_dir datasets/lerobot_dataset/glass_pick_place_iteration11
```

**Output directory**: `datasets/lerobot_dataset/glass_pick_place_iteration11/`

| File | Content |
|:---|:---|
| `cup_lifting_01_episode.npz` | LeRobot-compatible episode with images, states, actions |
| `train_episodes.hdf5` | HDF5 dataset for SmolVLA DataLoader |
| `dataset_info.json` | Metadata: episode count, observation/action dimensions |

---

## Step 6: Fine-Tune SmolVLA (Phase-Balanced v5)

**File**: `src/learning/train_smolvla_phase_balanced_v5.py`
**Input**: LeRobot dataset from Step 5 + expert replay NPZ
**Command**:

```bash
python3 -m src.learning.train_smolvla_phase_balanced_v5 \
    --dataset_dir datasets/lerobot_dataset/glass_pick_place_iteration11 \
    --expert_replay datasets/cotracker_iterations/iteration_11_full_mujoco/cup_lifting_01_replay.npz \
    --out_dir outputs/smolvla_glass_phase_balanced_v5 \
    --epochs 30 --batch_size 4 --lr 1e-4 --seed 6
```

**Output directory**: `outputs/smolvla_glass_phase_balanced_v5/`

| File | Content |
|:---|:---|
| `model.safetensors` | Best-loss SmolVLA checkpoint (99.88M trainable params) |
| `config.json` | Model config (chunk_size=20, max_state_dim=32) |
| `training_history.json` | Per-epoch loss, phase-balanced sampling stats |

**Key parameters**:
- Frozen SigLIP-400M vision backbone (16 layers)
- Action chunk horizon H = 50, padded to 32D
- 8-phase weighted random sampler
- Best-loss checkpoint selection

---

## Step 7: H=50 Phase Target Sweep (6 Experiments)

**File**: `src/learning/run_smolvla_experiment_sweep.py`
**Input**: LeRobot dataset from Step 5
**Command**:

```bash
python3 -m src.learning.run_smolvla_experiment_sweep \
    --sweep_dir outputs/smolvla_glass_sweep_h50_coarse \
    --batch_size 4 --lr 1e-4 --seed 6 --k_steps 5 --skip_existing
```

**Output directory**: `outputs/smolvla_glass_sweep_h50_coarse/`

| File / Directory | Content |
|:---|:---|
| `h50_l15_low10_rel05/checkpoint/` | Best checkpoint (score 76.5, peak 1.17 cm) |
| `h50_l15_low10_rel05/result.json` | Rollout metrics: contact, grasp, lift, score |
| `h50_l15_low10_rel05/rollout/` | Rollout frames and trajectory data |
| `h50_l20_low12_rel10/` | Second-best experiment (score 10.98) |
| `h50_l18_low10_rel07/` | Third experiment (score 5.91) |
| `h50_l20_low10_rel08/` | Fourth experiment (score 5.91) |
| `h50_l22_low10_rel08/` | Fifth experiment (score 5.91) |
| `h50_l18_low15_rel08/` | Sixth experiment (score 5.91) |
| `leaderboard.json` | Ranked results for all 6 experiments |
| `leaderboard.csv` | CSV version of leaderboard |
| `best_experiment.json` | Pointer to best config |

**Result**: Best = `h50_l15_low10_rel05` — score 76.5, peak lift 1.17 cm, contact + grasp but lift too weak.

---

## Step 8: Inference Gain Sweep (with Release Guard)

**File**: `src/simulation/sweep_inference_gains.py`
**Input**: Best checkpoint from Step 7
**Command**:

```bash
python3 -m src.simulation.sweep_inference_gains \
    --checkpoint outputs/smolvla_glass_sweep_h50_coarse/h50_l15_low10_rel05/checkpoint \
    --horizon 50 --k_steps 5 --seed 6 \
    --out_dir outputs/gain_sweep_v4_release \
    --gamma_x_values 1.0 \
    --gamma_z_values 1.25 \
    --gamma_lift_values 1.70 1.80 1.90 2.00
```

**Output directory**: `outputs/gain_sweep_v4_release/`

| File | Content |
|:---|:---|
| `gx1.00_gz1.25_gl2.00_result.json` | Best result: score 98.07, peak 4.31 cm, full success |
| `gx1.00_gz1.25_gl2.00_rollout.npz` | Full rollout data (frames, actions, glass Z, contact/grasp flags) |
| `gx1.00_gz1.25_gl1.90_result.json` | Score 97.85, peak 3.87 cm |
| `gx1.00_gz1.25_gl1.80_result.json` | Score 97.55, peak 3.27 cm |
| `gx1.00_gz1.25_gl1.70_result.json` | Score 97.63, peak 3.44 cm |
| `gain_leaderboard.json` | Ranked gain combinations |
| `gain_leaderboard.csv` | CSV version |
| `best_gain.json` | Best: γ_x=1.0, γ_z=1.25, γ_lift=2.0 |

**Result**: ALL 4 combinations achieve full-task success. Best = γ_lift=2.0, score 98.07/100.

---

## Step 9: Reproduce the Final Result (98.07/100)

**File**: `src/simulation/rollout_smolvla_calibrated.py`
**Input**: Best checkpoint from Step 7
**Command**:

```bash
python3 -c "
import torch
from src.simulation.rollout_smolvla_calibrated import run_calibrated_rollout
from src.simulation.sweep_inference_gains import load_policy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
policy = load_policy(
    'outputs/smolvla_glass_sweep_h50_coarse/h50_l15_low10_rel05/checkpoint', 50, device
)
results = run_calibrated_rollout(
    policy=policy, gamma_x=1.0, gamma_z=1.25, gamma_lift=2.0,
    max_steps=340, chunk_exec_steps=5, device=device, seed=6,
    enable_release_guard=True,
)
print(f'Score: {results[\"score\"]:.2f}/100')
print(f'Peak lift: {results[\"max_lift_cm\"]:.2f} cm')
print(f'Full success: {results[\"success\"]}')
"
```

**Output** (printed to terminal):

```
Score: 98.07/100
Peak lift: 4.31 cm
Full success: True
```

**8 success criteria**:
- Approach: 0.42 cm (threshold ≤ 3 cm)
- Contact: YES (frame 90)
- Grasp: YES (frame 90)
- Lift: 4.31 cm (threshold 3–15 cm)
- Grasp retention: 112 frames (threshold ≥ 10)
- Lowering: 5.27 cm (threshold ≥ 2 cm)
- Release: YES
- Stable finish: -0.91 cm (threshold ≤ 2 cm)

---

## Step 10: Generate Media (GIFs and Tri-Panel Video)

**File**: `src/tracking_algorithm/video_to_gif.py`
**Input**: Any iteration MP4 video
**Command** (example for one iteration):

```bash
python3 src/tracking_algorithm/video_to_gif.py \
    --video datasets/cotracker_iterations/iteration_11_full_mujoco/cup_lifting_01_iteration_11.mp4 \
    --output media/cup_lifting_01_iteration_11.gif \
    --fps 10 --scale 0.5 --max_width 400
```

**All generated media files** in `media/`:

| File | Source | Content |
|:---|:---|:---|
| `cup_lifting_01.gif` | `datasets/egocentric_video/cup_lifting_01.mp4` | Original iPhone video |
| `cup_lifting_01_annotated.gif` | MediaPipe output | Hand tracking with landmarks |
| `cup_lifting_01_ct_annotated.gif` | CoTracker3 output | Dense point tracking overlay |
| `cup_lifting_01_iteration_1.gif` | `iteration_01_raw/` | Iteration 1: raw baseline |
| `cup_lifting_01_iteration_2.gif` | `iteration_02_camera_roi/` | Iteration 2: 60 deg camera |
| `cup_lifting_01_iteration_3.gif` | `iteration_03_physics/` | Iteration 3: pad diagnostic |
| `cup_lifting_01_iteration_4.gif` | `iteration_04_gripper/` | Iteration 4: diluted offset |
| `cup_lifting_01_iteration_5.gif` | `iteration_05_goal_directed/` | Iteration 5: first grasp + lift |
| `cup_lifting_01_iteration_6.gif` | `iteration_06_wrist_squared/` | Iteration 6: wrist squaring |
| `cup_lifting_01_iteration_07.gif` | `iteration_07_blended_feedback/` | Iteration 7: expert demo |
| `cup_lifting_01_iteration_10.gif` | `iteration_10_smolvla_expert/` | SmolVLA baseline rollout |
| `cup_lifting_01_iteration_11.gif` | `iteration_11_full_mujoco/` | Expert replay (ground truth) |
| `smolvla_gain_calibrated_best.gif` | gain_sweep_v4_release best rollout | Final SmolVLA success |
| `smolvla_gain_calibrated_best.mp4` | Same source as above | Full video version |
| `pollex_pipeline_tri_panel.gif` | Composite of 3 sources | 3-way: iPhone + Expert + SmolVLA |
| `pollex_pipeline_tri_panel.mp4` | Same composite | Full video version |

---

## Directory Structure Summary

```
Pollex/
├── datasets/
│   ├── egocentric_video/
│   │   └── cup_lifting_01.mp4                    # INPUT: raw iPhone video
│   ├── cotracker_trajectories/
│   │   ├── cup_lifting_01_ct_trajectory.npz       # Step 2 output: tracked points
│   │   └── cup_lifting_01_ct_annotated.mp4        # Step 2 output: annotated video
│   ├── cotracker_iterations/
│   │   ├── iteration_01_raw/                      # Iteration 1 replay
│   │   ├── iteration_02_camera_roi/               # Iteration 2 replay
│   │   ├── iteration_03_physics/                  # Iteration 3 replay
│   │   ├── iteration_04_gripper/                  # Iteration 4 replay
│   │   ├── iteration_05_goal_directed/            # Iteration 5 replay (first lift)
│   │   ├── iteration_06_wrist_squared/            # Iteration 6 replay
│   │   ├── iteration_07_blended_feedback/         # Iteration 7 replay (expert demo)
│   │   ├── iteration_10_smolvla_expert/           # SmolVLA baseline rollout
│   │   └── iteration_11_full_mujoco/              # Final expert replay + dataset
│   │       ├── cup_lifting_01_iteration_11.mp4
│   │       └── cup_lifting_01_replay.npz          # Step 4 output: expert dataset
│   └── lerobot_dataset/
│       └── glass_pick_place_iteration11/          # Step 5 output: LeRobot format
│           ├── cup_lifting_01_episode.npz
│           ├── train_episodes.hdf5
│           └── dataset_info.json
├── outputs/
│   ├── smolvla_glass_phase_balanced_v5/           # Step 6 output: trained model
│   │   ├── model.safetensors
│   │   ├── config.json
│   │   └── training_history.json
│   ├── smolvla_glass_sweep_h50_coarse/            # Step 7 output: 6-experiment sweep
│   │   ├── h50_l15_low10_rel05/checkpoint/        # BEST checkpoint (used in Step 8)
│   │   ├── h50_l15_low10_rel05/result.json
│   │   ├── leaderboard.json
│   │   └── best_experiment.json
│   └── gain_sweep_v4_release/                     # Step 8 output: gain sweep results
│       ├── gx1.00_gz1.25_gl2.00_result.json       # BEST: score 98.07
│       ├── gx1.00_gz1.25_gl2.00_rollout.npz
│       ├── gain_leaderboard.json
│       └── best_gain.json
└── media/                                         # Step 10 output: README media
    ├── pollex_pipeline_tri_panel.gif               # Hero banner (3-way comparison)
    ├── smolvla_gain_calibrated_best.gif            # Final success rollout
    └── cup_lifting_01_iteration_*.gif              # Per-iteration GIFs
```




## Just for note
Base score: 8 binary criteria adding up to 100 points

Criterion	Points	Threshold	Our Result
Approach	10	min d_xy ≤ 3 cm	0.42 cm
Contact	15	MuJoCo contact detected	YES
Grasp	20	RoboSuite _check_grasp	YES
Lift	20	3 cm ≤ peak ≤ 15 cm	4.31 cm
Grasp Retention	15	≥ 10 consecutive grasp frames	112 frames
Lowering	10	≥ 2 cm drop from peak	5.27 cm
Release	5	Gripper opens after last grasp	YES
Stable Finish	5	|final lift| ≤ 2 cm	-0.91 cm
Subtotal	100		All pass
Then a penalty is subtracted:

Lift error penalty: abs(peak_lift - expert_lift) * 0.5 = abs(4.31 - 8.17) * 0.5 = -1.93 points
So: 100 - 1.93 = 98.07

The deduction is because the expert demonstration lifts 8.17 cm but the SmolVLA policy only achieves 4.31 cm — correct and successful, but not as high as the expert. A perfect 100 would mean matching the expert's 8.17 cm lift exactly. It's our own scoring function, not a standard benchmark.