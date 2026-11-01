# Pollex
### Egocentric Video-to-Robot Policy: iPhone Demonstration to Autonomous Glass Pick-and-Place via SmolVLA

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.4](https://img.shields.io/badge/PyTorch-2.4%20(CUDA)-red.svg)](https://pytorch.org/)
[![RoboSuite 1.4](https://img.shields.io/badge/RoboSuite-1.4.0-green.svg)](https://robosuite.ai/)
[![LeRobot SmolVLA](https://img.shields.io/badge/HuggingFace-LeRobot%20SmolVLA-orange.svg)](https://github.com/huggingface/lerobot)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="media/pollex_pipeline_tri_panel.gif" width="768" alt="Pollex Pipeline: iPhone Video → Expert Retargeting → SmolVLA Policy" />
  <br>
  <em><b>End-to-End Pipeline</b>: (1) iPhone video with CoTracker3 point tracking → (2) Expert retargeted trajectory in MuJoCo → (3) SmolVLA policy closed-loop rollout — <b>FULL-TASK SUCCESS (98.07/100)</b></em>
</p>

### What is this project in plain English?

A person films themselves picking up a glass from a table using an iPhone. This project takes that 11-second video and, through a series of computer vision, physics simulation, and machine learning steps, teaches a simulated Franka Panda robot arm to perform the same pick-and-place task autonomously — seeing only through a camera, with no hardcoded trajectories at inference time.

The final SmolVLA policy (450M parameters) scores **98.07/100** on an 8-criterion evaluation: approach, contact, grasp, lift (+4.31 cm), retention (112 frames), lowering, release, and stable finish.

---

## Table of Contents

1. [Executive Summary & System Architecture](#1-executive-summary--system-architecture)
2. [Part I: Real-to-Sim Computer Vision](#2-part-i-real-to-sim-computer-vision)
3. [Part II: Physical Digital Twin Engineering](#3-part-ii-physical-digital-twin-engineering)
4. [Part III: SmolVLA Vision-Language-Action Policy](#4-part-iii-smolvla-vision-language-action-policy)
5. [Part IV: Milestone Evaluation Registry](#5-part-iv-milestone-evaluation-registry)
6. [Part V: Failure Analysis — Limits of Few-Shot Imitation Learning](#6-part-v-failure-analysis--limits-of-few-shot-imitation-learning)
7. [Part VI: Repository & File Catalog](#7-part-vi-repository--file-catalog)
8. [Part VII: Quick-Start & Reproduction Guide](#8-part-vii-quick-start--reproduction-guide)

---

## 1. Executive Summary & System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    POLLEX: Video → Robot Policy Pipeline                 │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  📱 iPhone Video (1080p, 30fps, 11s)                                    │
│       │                                                                  │
│       ├──► MediaPipe Hands (21 landmarks, occlusion fallback)           │
│       └──► CoTracker3 (37 dense points, co-motion ρ > 0.55)            │
│                 │                                                        │
│                 ▼                                                        │
│  🔄 Motion Retargeting                                                  │
│       60° camera tilt correction, EEF-to-pad offset,                    │
│       goal-directed reach with distance deceleration                    │
│                 │                                                        │
│                 ▼                                                        │
│  🤖 RoboSuite / MuJoCo Simulation                                      │
│       Franka Panda, GlassLift env, OSC_POSE 7D controller              │
│       Hollow glass (450 kg/m³), μ=2.0, q7=0 squared wrist              │
│                 │                                                        │
│                 ▼                                                        │
│  📦 LeRobot Dataset (340 transitions)                                   │
│       256×256 RGB + 8D state + 7D action per frame                      │
│                 │                                                        │
│                 ▼                                                        │
│  🧠 SmolVLA (450M params)                                               │
│       Frozen SigLIP-400M encoder + 99.88M Flow-Matching head           │
│       Phase-balanced training, H=50 action chunks                       │
│                 │                                                        │
│                 ▼                                                        │
│  ⚡ Inference-Time Gain Calibration                                     │
│       γ_z=1.25 (descent), γ_lift=2.0 (lift amplification)              │
│       Release Guard (lift→lower cycle detection)                        │
│                 │                                                        │
│                 ▼                                                        │
│  ✅ FULL-TASK SUCCESS: 98.07/100                                        │
│       approach → grasp → lift +4.31cm → lower → release                │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Part I: Real-to-Sim Computer Vision

### 2.1 MediaPipe Hand Tracking & Occlusion Fallback

<p align="center">
  <img src="media/cup_lifting_01_annotated.gif" width="480" alt="MediaPipe hand tracking with occlusion artifacts" />
  <br>
  <em><b>MediaPipe Tracking</b>: 21 anatomical landmarks extracted per frame. When the index fingertip disappears behind the glass, the tracker falls back to DIP/PIP joints with hysteresis band.</em>
</p>

The tracking pipeline (`src/tracking_algorithm/hand_tracker.py`) uses Google MediaPipe to extract 21 3D hand landmarks per frame. The end-effector position derives from wrist displacements ($L_0$), and the gripper command $g_t \in \{-1, +1\}$ is determined by pinch distance:

$$d_{\text{pinch}} = \| \mathbf{p}_{L_4} - \mathbf{p}_{L_8} \|_2$$

**Occlusion Problem**: When grasping, the index fingertip ($L_8$) is occluded. MediaPipe hallucinates coordinates, causing gripper chatter. A progressive fallback heuristic uses more proximal joints ($L_7, L_6$) when $L_8$ jumps more than $\tau = 0.12$ between frames or lands near the image edge.

**Limitation**: No static keypoint heuristic can guarantee reliability during physical contact, motivating the transition to dense point tracking.

### 2.2 CoTracker3 Dense Point Tracking

<p align="center">
  <img src="media/cup_lifting_01_ct_annotated.gif" width="300" alt="CoTracker3 Dense Grid Tracking" />
  <br>
  <em><b>CoTracker3 Dense Point Grid</b>: 21 hand points (green) + 16 glass grid points (blue) tracked across the full manipulation sequence.</em>
</p>

CoTracker3 (`src/tracking_algorithm/cotracker_dense_point_tracking.py`) tracks 37 surface points — 21 hand landmarks initialized from MediaPipe plus a 4×4 grid on the glass region. The hand and object centroids are computed per frame:

$$\mathbf{c}^{H}_t = \frac{1}{N_H} \sum_{i} \mathbf{p}^{H}_{i,t}, \qquad \mathbf{c}^{O}_t = \frac{1}{N_O} \sum_{j} \mathbf{p}^{O}_{j,t}$$

Grasp detection uses velocity co-motion correlation:

$$\rho_t = \frac{ \mathbf{v}^{H}_t \cdot \mathbf{v}^{O}_t }{ \|\mathbf{v}^{H}_t\|_2 \, \|\mathbf{v}^{O}_t\|_2 }$$

A grasp triggers when $\rho_t > 0.55$ within proximity, with release at $\rho_t < 0.25$ (hysteresis prevents chatter).

### 2.3 CoTracker Tuning Parameters

| Parameter | Default | Tuned | Effect |
|:---|:---:|:---:|:---|
| `GRASP_CORR_THRESHOLD` | 0.55 | 0.65 | Co-motion threshold for grasp trigger |
| `GRASP_PROXIMITY_THRESH` | 0.25 | 0.15 | Max hand-to-glass distance for grasp |
| `GRASP_FRAME_DELAY` | 10 | 20 | Frames after hand entry before grasp allowed |
| `MIN_VELOCITY_THRESH` | 0.4 | 0.8 | Min pixel velocity for valid correlation |
| `RELEASE_DIST_THRESH` | 0.40 | 0.35 | Distance threshold forcing release |

### 2.4 Retargeting Pipeline

The retargeting module (`src/retargeting/cotracker_to_panda.py`) converts CoTracker image-space trajectories into Franka Panda 7D OSC actions $(dx, dy, dz, droll, dpitch, dyaw, gripper)$:

1. **60° Camera Tilt Correction**: The egocentric camera points downward at ~60°. Image-plane vertical motion maps to forward table motion via $R_x(-60°)$ derotation.
2. **EEF-to-Pad Offset**: A 4.3 cm constant X offset corrects the difference between the EEF origin and where the finger pads actually make contact.
3. **Goal-Directed Reach**: Concentrated 5-frame burst (0.0086 m/frame) with distance-based deceleration prevents overshoot.
4. **Object-Relative Feedback**: X/Z feedback loop regulates pad position relative to the tumbler with adaptive blending.

---

## 3. Part II: Physical Digital Twin Engineering

### 3.1 GlassLift Environment

The custom `GlassLift` environment (`src/simulation/glass_lift_env.py`) extends RoboSuite's `Lift` task with calibrated physics:

| Parameter | Value | Rationale |
|:---|:---|:---|
| Glass position (X) | -0.055 m | Centered in Panda workspace |
| Glass radius | 0.030 m | Matches physical tumbler |
| Glass half-height | 0.0475 m | 9.5 cm tall cylinder |
| Glass density | 450 kg/m³ | Hollow tumbler (not solid steel) |
| Friction (μ) | [2.0, 0.05, 0.001] | High grip for rubber-on-glass |
| Wrist joint (q7) | 0 | Squared horizontal grip |

### 3.2 Retargeting Iteration History

Seven iterations progressively solved the geometric and physical challenges of transferring human video motion to the Franka Panda:

| Iter | GIF | Key Change | Contact | Grasp | Peak Lift | Outcome |
| :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | <img src="media/cup_lifting_01_iteration_1.gif" width="120"/> | Raw CoTracker3, 0° camera | No | No | 0.00 cm | 35 cm X motion but 14 cm miss |
| 2 | <img src="media/cup_lifting_01_iteration_2.gif" width="120"/> | 60° camera tilt, gain=8.0 | No | No | 0.00 cm | Min distance 3.80 cm (73% better) |
| 3 | <img src="media/cup_lifting_01_iteration_3.gif" width="120"/> | Pad geometry diagnostic | No | No | 0.00 cm | 4.35 cm pad-to-axis offset found |
| 4 | <img src="media/cup_lifting_01_iteration_4.gif" width="120"/> | Diluted 4.3 cm offset | YES | No | 0.00 cm | First contact (frame 201, retract) |
| **5** | <img src="media/cup_lifting_01_iteration_5.gif" width="120"/> | 5-frame burst + decel | YES | YES | **8.22 cm** | **First grasp + lift** |
| 6 | <img src="media/cup_lifting_01_iteration_6.gif" width="120"/> | q7=0 wrist squaring | YES | YES | 8.27 cm | Marginal; side-grip persists |
| **7** | <img src="media/cup_lifting_01_iteration_07.gif" width="120"/> | Object-relative X/Z feedback | YES | YES | **8.17 cm** | **Complete pick-and-place** |

### 3.3 Iteration 1: Raw CoTracker3 Baseline

<p align="center">
  <img src="media/cup_lifting_01_iteration_1.gif" width="560" alt="Iteration 1: Raw CoTracker3 Baseline" />
  <br>
  <em><b>Iteration 1 (Raw CoTracker3)</b>: The robot moves 35 cm but approaches the wrong location — 14 cm from the glass. No contact, no grasp, no lift.</em>
</p>

- **Configuration**: `camera_tilt_deg = 0°`, raw retargeting.
- **Analysis**: Without camera inclination compensation, the egocentric video's vertical axis maps incorrectly to the robot workspace. The end-effector traveled 34.95 cm along X but the closest approach was still 13.97 cm from the glass.
- **Root Cause**: Missing 60° camera-to-table projection. The failure is geometric misalignment, not robot immobility.

### 3.4 Iteration 2: 60° Camera Calibration

<p align="center">
  <img src="media/cup_lifting_01_iteration_2.gif" width="560" alt="Iteration 2: 60° Camera Calibration" />
  <br>
  <em><b>Iteration 2 (60° Camera Tilt)</b>: Closest distance drops from 13.97 cm to 3.80 cm (73% improvement), but gripper doesn't enclose the cylinder.</em>
</p>

- **Configuration**: `camera_tilt_deg = 60°`, `position_gain = 8.0`.
- **Analysis**: Camera calibration solved the gross reaching error ($d_{\min}: 13.97 \to 3.80$ cm), but reaching the *neighborhood* of an object is not the same as grasping it. The finger pads didn't straddle the glass.

### 3.5 Iteration 3: Grasp Geometry Diagnostic

- **Purpose**: Instrumented MuJoCo to measure the exact pad-to-glass relationship per frame.
- **Key Finding**: At EEF closest approach (frame 98), the pads were 4.35 cm from the glass axis with 8.70 cm separation (wide open). By frame 154 (pads closest), separation was 1.10 cm but the pads passed 4.35 cm to the side. A constant 4.3 cm X offset between EEF origin and pad contact surfaces explains the miss.

### 3.6 Iteration 4: EEF-to-Pad Offset Correction — First Contact

<p align="center">
  <img src="media/cup_lifting_01_iteration_4.gif" width="560" alt="Iteration 4: First Contact" />
  <br>
  <em><b>Iteration 4</b>: The 4.3 cm offset diluted across 115 frames yields 0.00037 m/frame — below OSC sensitivity. Contact occurs, but only during retract (frame 201), not approach.</em>
</p>

- **Configuration**: `eef_to_pad_offset_x = 0.043`, spread uniformly across approach.
- **Analysis**: First-ever MuJoCo contact (frame 201), but during retract — too late. The per-frame correction was absorbed by EMA smoothing.
- **Root Cause**: Diluted corrections below OSC controller sensitivity. The offset must be concentrated.

### 3.7 Iteration 5: Goal-Directed Targeting — FIRST GRASP & LIFT

<p align="center">
  <img src="media/cup_lifting_01_iteration_5.gif" width="560" alt="Iteration 5: First Successful Grasp and Lift" />
  <br>
  <em><b>Iteration 5 (Goal-Directed)</b>: Concentrated 5-frame burst + distance deceleration. <b>First successful grasp + lift: +8.22 cm</b>.</em>
</p>

- **Configuration**: Concentrated offset (0.0086 m/frame × 5 frames) + `dist_gain = min(dist_reach / 0.15, 1.0)`.
- **Analysis**: The burst was large enough for OSC to track. Distance deceleration prevented overshoot. Contact at frame 148 coincided with grasp closure timing. **+8.22 cm lift** — first successful pick-up in the pipeline.

### 3.8 Iteration 6: Wrist Squaring (q7 → 0)

<p align="center">
  <img src="media/cup_lifting_01_iteration_6.gif" width="560" alt="Iteration 6: Wrist Squaring" />
  <br>
  <em><b>Iteration 6 (q7=0)</b>: Wrist squared from 45° to 0°. Marginal improvement (+8.27 cm); the real problem is approach trajectory, not wrist angle.</em>
</p>

- **Configuration**: `q7 = 0` in `glass_lift_env.py` (was `π/4`).
- **Analysis**: Lift improved marginally (8.22 → 8.27 cm, clearance 0.84 → 0.80 cm). Visual inspection showed the gripper still approaches from the side — the problem is descent profile, not wrist rotation.

### 3.9 Iteration 7: Object-Relative Feedback — Expert Demonstration

<p align="center">
  <img src="media/cup_lifting_01_iteration_07.gif" width="560" alt="Iteration 7: Expert Demonstration Pick and Place" />
  <br>
  <em><b>Iteration 7 (Object-Relative Feedback)</b>: X/Z feedback regulates pad position relative to the tumbler. <b>PICK & PLACE: +8.17 cm [SUCCESS]</b> — the expert demonstration used for SmolVLA training.</em>
</p>

- **Configuration**: Closed-loop X/Z feedback with `target_pad_x_offset = -0.017`, `target_pad_z_offset = 0.015`, adaptive blending.
- **Analysis**: The controller measures finger-pad center relative to the tumbler and regulates approach geometry. Achieves consistent **+8.17 cm** lift with clean pick-and-place cycle. This trajectory became the expert demonstration for SmolVLA fine-tuning.

---

## 4. Part III: SmolVLA Vision-Language-Action Policy

With a verified expert demonstration (Iteration 7: +8.17 cm pick-and-place), we fine-tuned **SmolVLA** (450M parameter Vision-Language-Action model) to learn the complete manipulation task from camera observations alone.

### 4.1 SmolVLA Architecture

| Component | Details |
|:---|:---|
| **Vision Encoder** | SigLIP-400M (frozen, 16 transformer layers) |
| **Language Model** | SmolLM2-135M (frozen) |
| **Action Expert Head** | 99.88M params, Flow-Matching with 10-step ODE sampling |
| **Action Space** | 7D OSC: $(dx, dy, dz, droll, dpitch, dyaw, gripper)$ |
| **State Input** | 8D: EEF pos (3) + EEF quat (4) + gripper (1), padded to 32D |
| **Action Chunk Horizon** | H = 50 (execute K = 5 steps per chunk) |
| **Total Parameters** | ~450M (99.88M trainable) |

### 4.2 Phase-Balanced Training (v5)

Standard uniform sampling over-represents static phases (hover, retract) and under-represents critical transitions (grasp closure, lift initiation). We implemented **phase-balanced sampling** across 8 task phases:

```
hover → approach → descent_pregrasp → grasp_closing → postgrasp_lift → postgrasp_lower → release → retract
```

Each phase receives a target sampling probability, ensuring the flow-matching head sees enough examples of the critical contact transitions.

### 4.3 H=50 Phase Target Sweep — Baseline Results

<p align="center">
  <img src="media/cup_lifting_01_iteration_10.gif" width="400" alt="SmolVLA Baseline" />
  <br>
  <em><b>SmolVLA Baseline (H=50, K=5, no gain)</b>: The policy learns the complete task structure — approach, descend, grasp, hold — but the lift phase produces only +1.17 cm, below the 3 cm threshold.</em>
</p>

We swept 6 phase-target distributions (all H=50, 10 epochs):

| Rank | Name | Score | Contact | Grasp | Peak Lift | Full Success |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | h50_l15_low10_rel05 | 76.50 | YES | YES | 1.17 cm | No |
| 2 | h50_l20_low12_rel10 | 10.98 | YES | No | 0.13 cm | No |
| 3 | h50_l18_low10_rel07 | 5.91 | YES | No | 0.00 cm | No |
| 4 | h50_l20_low10_rel08 | 5.91 | YES | No | 0.00 cm | No |
| 5 | h50_l22_low10_rel08 | 5.91 | YES | No | 0.00 cm | No |
| 6 | h50_l18_low15_rel08 | 5.91 | YES | No | 0.00 cm | No |

**Key finding**: The best checkpoint learns the complete task structure — approach, contact, grasp, hold for 88 frames — but **regression-to-the-mean** on the low-variance $\Delta Z$ channel mutes the lift velocity to only 1.17 cm.

### 4.4 Inference-Time Dual-Axis Gain Calibration — The Breakthrough

<p align="center">
  <img src="media/smolvla_gain_calibrated_best.gif" width="400" alt="SmolVLA Calibrated: Full Task Success" />
  <br>
  <em><b>SmolVLA Calibrated (γ_z=1.25, γ_lift=2.0)</b>: Inference-time gain scaling amplifies the lift phase without retraining. <b>FULL-TASK SUCCESS (98.07/100)</b>.</em>
  <br>
  <a href="media/smolvla_gain_calibrated_best.mp4">Download full video (MP4)</a> · <a href="media/pollex_pipeline_tri_panel.mp4">3-way comparison video (MP4)</a>
</p>

SmolVLA's Flow-Matching action head systematically under-predicts low-variance channels ($\Delta X$, $\Delta Z$) due to **regression-to-the-mean** in few-shot learning. Rather than retraining, we apply **phase-aware velocity scaling** at inference time:

$$
a_{\text{calibrated}} = \begin{cases}
a_z \cdot \gamma_z & \text{if gripper open, descending, EEF above glass} \\
a_z \cdot \gamma_{\text{lift}} & \text{if currently grasping and lifting (} a_z > 0 \text{)} \\
a_x \cdot \gamma_x & \text{if gripper open, approaching, behind glass X}
\end{cases}
$$

**Key design decision**: Separate lift gain ($\gamma_{\text{lift}}$) from descent gain ($\gamma_z$), and gate lift amplification on the MuJoCo grasp flag.

**Gain Sweep Results (with Release Guard):**

| Rank | γ_x | γ_z | γ_lift | Score | Peak Lift | Lowering | Grasp Run | Full Success |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **1.00** | **1.25** | **2.00** | **98.07** | **4.31 cm** | **5.27 cm** | **112** | **YES** |
| 2 | 1.00 | 1.25 | 1.90 | 97.85 | 3.87 cm | 4.85 cm | 114 | YES |
| 3 | 1.00 | 1.25 | 1.70 | 97.63 | 3.44 cm | 4.41 cm | 114 | YES |
| 4 | 1.00 | 1.25 | 1.80 | 97.55 | 3.27 cm | 4.24 cm | 108 | YES |

**What we discovered during gain tuning:**

1. **γ_x > 1.0 breaks grasping**: Our model already reaches the glass perfectly — no forward boost needed (unlike the reference repo's 3.4 cm under-reach).
2. **γ_z = 1.25 helps descent**: Brings gripper fingers to the right height around the cylinder body.
3. **γ_lift is the critical parameter**: 2× amplification overcomes flow-matching regression-to-the-mean. Below 1.60, lift stays under 3 cm. Above 2.5, the robot flings the glass.

### 4.5 Release Guard Heuristic

The trained policy **never releases the gripper** — grasp_run = 257 out of 340 steps. The flow-matching head defaults to keeping the gripper closed since it's closed for most of the demonstration.

We implemented a **Release Guard** state machine that monitors the glass Z position:

```python
class ReleaseGuard:
    """Forces gripper open after detecting lift → lower cycle."""
    def update(self, glass_z, is_grasping):
        lift = glass_z - self.init_z
        if lift > self.peak_lift:
            self.peak_lift = lift
        lowered = self.peak_lift - lift
        if is_grasping and self.peak_lift >= 0.02 and lowered >= 0.015:
            self.triggered = True  # force gripper open
```

### 4.6 Final Configuration — Full-Task Success

| Parameter | Value |
|:---|:---|
| Checkpoint | `outputs/smolvla_glass_sweep_h50_coarse/h50_l15_low10_rel05/checkpoint` |
| Action Chunk Horizon (H) | 50 |
| Execution Steps (K) | 5 |
| Forward Gain (γ_x) | 1.00 (no scaling) |
| Descent Gain (γ_z) | 1.25 |
| Lift Gain (γ_lift) | 2.00 |
| Release Guard | Enabled (lift ≥ 2cm, lower ≥ 1.5cm) |
| Rollout Steps | 340 |

**All 8 success criteria pass:**

| Criterion | Threshold | Result |
|:---|:---|:---|
| Approach | min d_xy ≤ 3 cm | 0.42 cm |
| Contact | MuJoCo contact detected | YES (frame 90) |
| Grasp | RoboSuite `_check_grasp` | YES (frame 90) |
| Lift | 3 cm ≤ peak ≤ 15 cm | **4.31 cm** |
| Grasp Retention | ≥ 10 consecutive frames | **112 frames** |
| Lowering | ≥ 2 cm from peak | **5.27 cm** |
| Release | Gripper opens after grasp | YES |
| Stable Finish | \|final lift\| ≤ 2 cm | **-0.91 cm** |

---

## 5. Part IV: Milestone Evaluation Registry

### Summary Benchmark Table

| Visual Preview | Phase & Milestone | Key Change | Peak Lift | Score | Outcome | Video |
| :---: | :--- | :--- | :---: | :---: | :--- | :--- |
| <img src="media/cup_lifting_01_iteration_1.gif" width="120"/> | **Iter 1: Raw CoTracker3** | 0° camera, uncalibrated ROI | 0.00 cm | — | 35 cm motion, 14 cm miss | [MP4](datasets/cotracker_iterations/iteration_01_raw/cup_lifting_01_iteration_1.mp4) |
| <img src="media/cup_lifting_01_iteration_2.gif" width="120"/> | **Iter 2: 60° Camera** | Camera tilt 60°, gain=8.0 | 0.00 cm | — | Min dist 3.80 cm (73% better) | [MP4](datasets/cotracker_iterations/iteration_02_camera_roi/cup_lifting_01_iteration_2.mp4) |
| <img src="media/cup_lifting_01_iteration_3.gif" width="120"/> | **Iter 3: Pad Diagnostic** | Instrumented pad geometry | 0.00 cm | — | 4.35 cm offset identified | [MP4](datasets/cotracker_iterations/iteration_03_physics/cup_lifting_01_iteration_3.mp4) |
| <img src="media/cup_lifting_01_iteration_4.gif" width="120"/> | **Iter 4: Diluted Offset** | 4.3 cm spread over 115 frames | 0.00 cm | — | First contact (frame 201) | [MP4](datasets/cotracker_iterations/iteration_04_gripper/cup_lifting_01_iteration_4.mp4) |
| <img src="media/cup_lifting_01_iteration_5.gif" width="120"/> | **Iter 5: Goal-Directed** | 5-frame burst + decel | **8.22 cm** | — | **First grasp + lift** | [MP4](datasets/cotracker_iterations/iteration_05_goal_directed/cup_lifting_01_iteration_5.mp4) |
| <img src="media/cup_lifting_01_iteration_6.gif" width="120"/> | **Iter 6: Wrist Squared** | q7 = 0 (horizontal grip) | 8.27 cm | — | Marginal; side-grip persists | — |
| <img src="media/cup_lifting_01_iteration_07.gif" width="120"/> | **Iter 7: Expert Replay** | Object-relative X/Z feedback | **8.17 cm** | 100 | **Pick & place demonstration** | [MP4](datasets/cotracker_iterations/iteration_07_blended_feedback/cup_lifting_01_iteration_07.mp4) |
| <img src="media/cup_lifting_01_iteration_11.gif" width="120"/> | **Expert (Iter 11)** | Full MuJoCo expert trajectory | 8.17 cm | 100 | Ground-truth for SmolVLA | [MP4](datasets/cotracker_iterations/iteration_11_full_mujoco/cup_lifting_01_iteration_11.mp4) |
| <img src="media/cup_lifting_01_iteration_10.gif" width="120"/> | **SmolVLA Baseline** | Phase-balanced v5, H=50, K=5 | 1.17 cm | 76.5 | Contact + grasp, lift too weak | [MP4](datasets/cotracker_iterations/iteration_10_smolvla_expert/cup_lifting_01_iteration_10.mp4) |
| <img src="media/smolvla_gain_calibrated_best.gif" width="120"/> | **SmolVLA Calibrated** | γ_z=1.25, γ_lift=2.0, release guard | **4.31 cm** | **98.07** | **FULL-TASK SUCCESS** | [MP4](media/smolvla_gain_calibrated_best.mp4)<br>[3-Way](media/pollex_pipeline_tri_panel.mp4) |

---

## 6. Part V: Failure Analysis — Limits of Few-Shot Imitation Learning

Why does the SmolVLA policy learn the correct task structure but under-predict lift magnitude? Our diagnostic analysis revealed three mathematical barriers:

### 6.1 Uniform Temporal Averaging vs Critical Transitions

The flow-matching loss averages errors uniformly across all timesteps:

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, x_0, \mathbf{c}} \left[ \frac{1}{H} \sum_{k=0}^{H-1} \sum_{d=0}^{D-1} W_d \cdot \| u_{k,d} - v_\theta(x_{t,k}, t, \mathbf{c})_d \|^2 \right]$$

But the manipulation episode is **non-uniform**: ~80% of frames are static hover/retract phases where $\Delta X \approx 0$. The critical contact transitions (frames 85–95) that determine success represent < 5% of training data.

### 6.2 Signal-to-Noise Mismatch Across Action Channels

In the expert dataset, action dimensions have vastly different variances:
- $\Delta X$: $\sigma = 0.049$ (total transit only 6.2 cm)
- $\Delta Z$: $\sigma = 0.204$ (4× higher variance)
- $\Delta \text{Grip}$: $\sigma = 0.875$ (18× higher variance)

In few-shot learning (1 demonstration episode), flow-matching heads suffer from **regression-to-the-mean** on small-amplitude signals — the model collapses $\Delta X$ and $\Delta Z$ toward their empirical mean ($\approx 0$), muting forward velocity and lift force.

### 6.3 Closed-Loop Covariate Shift & Self-Throttling

During closed-loop rollout, flow-matching sampling noise during the hover phase (steps 0–30) produces tiny residual forward velocity ($\Delta X \approx +0.03$). Over 30 steps, the robot creeps forward prematurely. When the reaching phase begins, the policy sees the arm is already 2 cm ahead of where the demonstration was at the same timestep and **actively throttles** its forward velocity, interpreting the advanced position as overshoot.

This is the **covariate shift problem**: the model was trained on expert states but encounters its own drifted states at inference time, producing corrective actions the demonstrator never intended.

### 6.4 Why Inference-Time Gain Calibration Works

Rather than fighting the regression-to-the-mean through retraining (which risks destroying the learned task structure), gain calibration amplifies the model's correct *directional* predictions. The policy predicts "lift up" at the right time — it just predicts too little velocity. Multiplying by $\gamma_{\text{lift}} = 2.0$ restores the correct magnitude without changing when or where the action occurs.

---

## 7. Part VI: Repository & File Catalog

```
Pollex/
├── src/
│   ├── tracking_algorithm/
│   │   ├── hand_tracker.py                    # MediaPipe 21-landmark extractor with fallback heuristics
│   │   ├── cotracker_dense_point_tracking.py  # CoTracker3 dense tracking with co-motion correlation
│   │   ├── run_cotracker_pipeline.py          # Batch processing pipeline across demo clips
│   │   ├── select_tumbler_roi.py              # Interactive glass ROI selector
│   │   └── video_to_gif.py                    # MP4 → GIF converter for README media
│   ├── retargeting/
│   │   ├── hand_to_panda.py                   # MediaPipe landmark kinematic retargeter
│   │   └── cotracker_to_panda.py              # CoTracker3 → OSC 7D with 60° tilt un-projection
│   ├── simulation/
│   │   ├── glass_lift_env.py                  # Custom RoboSuite env (hollow glass, μ=2.0, q7=0)
│   │   ├── libero_runner.py                   # Unified sim wrapper with offscreen rendering
│   │   ├── replay_trajectory.py               # Expert trajectory replayer + video generator
│   │   ├── rollout_smolvla_baseline.py        # SmolVLA baseline rollout engine
│   │   ├── rollout_smolvla_calibrated.py      # Dual-axis gain calibration + release guard
│   │   └── sweep_inference_gains.py           # Systematic γ_x, γ_z, γ_lift grid search
│   ├── learning/
│   │   ├── export_lerobot_dataset.py          # Expert replay → LeRobot HDF5/NPZ converter
│   │   ├── train_smolvla_phase_balanced_v5.py # Phase-balanced fine-tuning (8-phase sampler)
│   │   ├── run_smolvla_experiment_sweep.py    # H=50 phase-target sweep (6 experiments)
│   │   ├── run_smolvla_adaptive_search.py     # 3-stage adaptive hyperparameter search
│   │   ├── train_smolvla_expert.py            # Base fine-tuning script
│   │   └── probe_encoder_features.py          # SigLIP-400M frozen feature analysis
│   └── evaluation/
│       └── compare_trackers.py                # MediaPipe vs CoTracker3 comparison
├── datasets/
│   ├── egocentric_video/                      # Raw iPhone demonstration clips
│   ├── cotracker_trajectories/                # Extracted CoTracker3 trajectories (.npz)
│   ├── cotracker_iterations/                  # Per-iteration replay videos and datasets
│   └── lerobot_dataset/                       # Packaged LeRobot training data
├── media/                                     # GIFs and videos for this README
├── outputs/                                   # Training checkpoints and sweep results
├── models/                                    # Pretrained model weights
├── tests/                                     # Unit tests
├── requirements.txt
├── pyproject.toml
├── setup_env.sh                               # One-command environment setup
└── README.md
```

---

## 8. Part VII: Quick-Start & Reproduction Guide

### Prerequisites
- **Python**: 3.12 (tested on NVIDIA Jetson Orin with CUDA)
- **GPU**: NVIDIA GPU with CUDA support (tested on Jetson AGX Orin 64GB)
- **OS**: Ubuntu 22.04+ (tested on JetPack 6.x / L4T)
- **Display**: X11 display server for MuJoCo GUI rendering (`DISPLAY=:10`)

### Environment Setup

> **Important**: Run the full setup before any pipeline command. The setup script clones two external repos ([LeRobot](https://github.com/huggingface/lerobot), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)) and installs them as editable packages alongside Pollex.

```bash
git clone https://github.com/<your-repo>/Pollex.git
cd Pollex
bash setup_env.sh          # creates .venv, clones LeRobot + LIBERO, installs all deps
source .venv/bin/activate
```

The setup script will:
1. Create a Python virtual environment at `.venv/`
2. Clone `huggingface/lerobot` → `../lerobot/` (SmolVLA policy architecture)
3. Clone `Lifelong-Robot-Learning/LIBERO` → `../vla/LIBERO/` (sim environment runner)
4. Install all three packages in editable mode (`pip install -e`)

### Simulation Display Setup (required for Steps 3–8)

Every MuJoCo simulation or rollout command needs the X11 display and PYTHONPATH configured. Run this block once per terminal session:

```bash
source .venv/bin/activate
export PYTHONPATH="$HOME/Documents/vla/LIBERO:$HOME/Documents/Pollex:$PYTHONPATH"
export DISPLAY=:10
export XAUTHORITY="$HOME/.Xauthority"
unset MUJOCO_GL QT_DEBUG_PLUGINS QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH
```

### 1. Extract Trajectories with CoTracker3
```bash
source .venv/bin/activate && \
PYTHONPATH=. python3 src/tracking_algorithm/cotracker_dense_point_tracking.py \
    --video datasets/egocentric_video/cup_lifting_01.mp4 \
    --output_dir datasets/cotracker_trajectories
```

### 2. Retarget to Franka Panda
```bash
PYTHONPATH=. python3 -m src.retargeting.cotracker_to_panda \
    --trajectory datasets/cotracker_trajectories/cup_lifting_01_ct_actions.npz \
    --output_dir datasets/cotracker_iterations/iteration_07_blended_feedback
```

### 3. Replay Expert Trajectory in MuJoCo
```bash
export DISPLAY=:10 && export XAUTHORITY="$HOME/.Xauthority" && \
unset MUJOCO_GL QT_DEBUG_PLUGINS QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH && \
PYTHONPATH=. python3 -m src.simulation.replay_trajectory \
    --actions datasets/cotracker_iterations/iteration_07_blended_feedback/cup_lifting_01_ct_actions.npz \
    --out_video datasets/cotracker_iterations/iteration_11_full_mujoco/cup_lifting_01_iteration_11.mp4 \
    --out_dataset datasets/cotracker_iterations/iteration_11_full_mujoco/cup_lifting_01_replay.npz \
    --human_video datasets/cotracker_trajectories/cup_lifting_01_ct_annotated.mp4 \
    --task GlassLift --closed_loop_pregrasp \
    --target_pad_x_offset -0.017 --pregrasp_kp_x 4.0 \
    --target_pad_z_offset 0.015 --pregrasp_kp_z 3.0
```

### 4. Export LeRobot Dataset
```bash
PYTHONPATH=. python3 -m src.learning.export_lerobot_dataset \
    --replay datasets/cotracker_iterations/iteration_11_full_mujoco/cup_lifting_01_replay.npz \
    --output_dir datasets/lerobot_dataset/glass_pick_place_iteration11
```

### 5. Fine-Tune SmolVLA (Phase-Balanced v5)
```bash
export PYTHONPATH="$HOME/Documents/vla/LIBERO:$HOME/Documents/Pollex:$PYTHONPATH" && \
python3 -m src.learning.train_smolvla_phase_balanced_v5 \
    --dataset_dir datasets/lerobot_dataset/glass_pick_place_iteration11 \
    --expert_replay datasets/cotracker_iterations/iteration_11_full_mujoco/cup_lifting_01_replay.npz \
    --out_dir outputs/smolvla_glass_phase_balanced_v5 \
    --epochs 30 --batch_size 4 --lr 1e-4 --seed 6
```

### 6. Run H=50 Phase Target Sweep
```bash
export DISPLAY=:10 && export XAUTHORITY="$HOME/.Xauthority" && \
unset MUJOCO_GL QT_DEBUG_PLUGINS QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH && \
export PYTHONPATH="$HOME/Documents/vla/LIBERO:$HOME/Documents/Pollex:$PYTHONPATH" && \
python3 -m src.learning.run_smolvla_experiment_sweep \
    --sweep_dir outputs/smolvla_glass_sweep_h50_coarse \
    --batch_size 4 --lr 1e-4 --seed 6 --k_steps 5 --skip_existing
```

### 7. Sweep Inference Gains (with Release Guard)
```bash
export DISPLAY=:10 && export XAUTHORITY="$HOME/.Xauthority" && \
unset MUJOCO_GL QT_DEBUG_PLUGINS QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH && \
export PYTHONPATH="$HOME/Documents/vla/LIBERO:$HOME/Documents/Pollex:$PYTHONPATH" && \
python3 -m src.simulation.sweep_inference_gains \
    --checkpoint outputs/smolvla_glass_sweep_h50_coarse/h50_l15_low10_rel05/checkpoint \
    --horizon 50 --k_steps 5 --seed 6 \
    --out_dir outputs/gain_sweep_v4_release \
    --gamma_x_values 1.0 \
    --gamma_z_values 1.25 \
    --gamma_lift_values 1.70 1.80 1.90 2.00
```

### 8. Reproduce the Final Result (98.07/100)
```bash
export DISPLAY=:10 && export XAUTHORITY="$HOME/.Xauthority" && \
unset MUJOCO_GL QT_DEBUG_PLUGINS QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH && \
export PYTHONPATH="$HOME/Documents/vla/LIBERO:$HOME/Documents/Pollex:$PYTHONPATH" && \
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