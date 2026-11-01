# what was log and how to start
Jetson clock and power is in max

# 1 smoke test 
(.venv) nvidia@nvidiathor:~/Documents/Pollex$ cd ~/Documents/Pollex && \
PYTHONPATH=. python3 src/learning/train_smolvla_expert.py \
  --dataset_dir datasets/lerobot_dataset/glass_pick_place \
  --out_dir outputs/smolvla_glass_expert_iteration10_smoke \
  --epochs 1 \
  --batch_size 1 \
  --lr 1e-4
[+] CUDA device: NVIDIA Thor
================================================================================
SmolVLA Action Expert Head Fine-Tuning
Device: cuda | Epochs: 1 | Batch Size: 1 | LR: 0.0001
================================================================================
[*] Loaded dataset with 340 transitions across 1 episodes.
[*] Initializing SmolVLA on cuda...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Reducing the number of VLM layers to 16 ...
[+] Loaded model. Trainable parameters: 99.88 M.
[+] Language tokens: (1, 48)

[*] Starting training loop (1 epochs, 340 batches/epoch)...
[Epoch 01/01] Train Loss: 0.0762 | LR: 1.00e-06 | Time: 56.1s

[+] Training completed in 58.1s.
[+] Initial Train Loss: 0.0762 -> Final: 0.0762 (-0.0% drop)
[+] Best Validation Loss: 0.0762
[+] Saved fine-tuned checkpoint & config -> outputs/smolvla_glass_expert_iteration10_smoke
================================================================================

# 2 batch size : 4

cd ~/Documents/Pollex && \
PYTHONPATH=. python3 src/learning/train_smolvla_expert.py \
  --dataset_dir datasets/lerobot_dataset/glass_pick_place \
  --out_dir outputs/smolvla_glass_expert_iteration10 \
  --epochs 30 \
  --batch_size 4 \
  --lr 1e-4
[+] CUDA device: NVIDIA Thor
================================================================================
SmolVLA Action Expert Head Fine-Tuning
Device: cuda | Epochs: 30 | Batch Size: 4 | LR: 0.0001
================================================================================
[*] Loaded dataset with 340 transitions across 1 episodes.
[*] Initializing SmolVLA on cuda...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Reducing the number of VLM layers to 16 ...
[+] Loaded model. Trainable parameters: 99.88 M.
[+] Language tokens: (1, 48)

[*] Starting training loop (30 epochs, 85 batches/epoch)...
[Epoch 01/30] Train Loss: 0.0903 | LR: 9.97e-05 | Time: 13.6s
[Epoch 05/30] Train Loss: 0.0334 | LR: 9.34e-05 | Time: 13.7s
[Epoch 10/30] Train Loss: 0.0236 | LR: 7.53e-05 | Time: 13.9s
[Epoch 15/30] Train Loss: 0.0150 | LR: 5.05e-05 | Time: 13.8s
[Epoch 20/30] Train Loss: 0.0127 | LR: 2.57e-05 | Time: 13.2s
[Epoch 25/30] Train Loss: 0.0106 | LR: 7.63e-06 | Time: 13.4s
[Epoch 30/30] Train Loss: 0.0106 | LR: 1.00e-06 | Time: 13.2s

[+] Training completed in 424.7s.
[+] Initial Train Loss: 0.0903 -> Final: 0.0106 (-88.3% drop)
[+] Best Training Loss: 0.0099
[+] Saved fine-tuned checkpoint & config -> outputs/smolvla_glass_expert_iteration10
================================================================================

## 4.3 Closed-Loop Baseline Evaluation (how to do and logs)
SmolVLA checkpoint
        ↓
RGB + current 8D Panda state + language
        ↓
predict H=50 actions
        ↓
execute K=5
        ↓
observe environment again
        ↓
replan


## Phase 1: baseline
K        = 5
gamma_x  = 1.0
gamma_z  = 1.0
steps    = 340
feedback = NONE

source /home/nvidia/Documents/Pollex/.venv/bin/activate && \
export PYTHONPATH="$HOME/Documents/vla/LIBERO:$HOME/Documents/Pollex:$PYTHONPATH" && \
export DISPLAY=:10 && \
export XAUTHORITY="$HOME/.Xauthority" && \
unset MUJOCO_GL && \
unset QT_DEBUG_PLUGINS && \
unset QT_PLUGIN_PATH && \
unset QT_QPA_PLATFORM_PLUGIN_PATH && \
cd ~/Documents/Pollex && \
python3 -m src.simulation.rollout_smolvla_baseline \
  --checkpoint outputs/smolvla_glass_expert_iteration10 \
  --human_video datasets/cotracker_trajectories/cup_lifting_01_ct_annotated.mp4 \
  --expert_video datasets/cotracker_iterations/iteration_10_smolvla_expert/cup_lifting_01_iteration_10.mp4 \
  --chunk_exec_steps 5 \
  --steps 340 \
  --seed 6 \
  --out_dir datasets/smolvla_comparisons \
  --gui

---
# k=1
source /home/nvidia/Documents/Pollex/.venv/bin/activate && \
export PYTHONPATH="$HOME/Documents/vla/LIBERO:$HOME/Documents/Pollex:$PYTHONPATH" && \
export DISPLAY=:10 && \
export XAUTHORITY="$HOME/.Xauthority" && \
unset MUJOCO_GL && \
unset QT_DEBUG_PLUGINS && \
unset QT_PLUGIN_PATH && \
unset QT_QPA_PLATFORM_PLUGIN_PATH && \
cd ~/Documents/Pollex && \
python3 -m src.simulation.rollout_smolvla_baseline \
  --checkpoint outputs/smolvla_glass_expert_iteration10 \
  --human_video datasets/cotracker_trajectories/cup_lifting_01_ct_annotated.mp4 \
  --expert_video datasets/cotracker_iterations/iteration_10_smolvla_expert/cup_lifting_01_iteration_10.mp4 \
  --chunk_exec_steps 1 \
  --steps 340 \
  --seed 6 \
  --out_dir datasets/smolvla_comparisons/k_1 \
  --gui

  ---

# k=2
source /home/nvidia/Documents/Pollex/.venv/bin/activate && \
export PYTHONPATH="$HOME/Documents/vla/LIBERO:$HOME/Documents/Pollex:$PYTHONPATH" && \
export DISPLAY=:10 && \
export XAUTHORITY="$HOME/.Xauthority" && \
unset MUJOCO_GL && \
unset QT_DEBUG_PLUGINS && \
unset QT_PLUGIN_PATH && \
unset QT_QPA_PLATFORM_PLUGIN_PATH && \
cd ~/Documents/Pollex && \
python3 -m src.simulation.rollout_smolvla_baseline \
  --checkpoint outputs/smolvla_glass_expert_iteration10 \
  --human_video datasets/cotracker_trajectories/cup_lifting_01_ct_annotated.mp4 \
  --expert_video datasets/cotracker_iterations/iteration_10_smolvla_expert/cup_lifting_01_iteration_10.mp4 \
  --chunk_exec_steps 2 \
  --steps 340 \
  --seed 6 \
  --out_dir datasets/smolvla_comparisons/k_2 \
  --gui

# k=10
source /home/nvidia/Documents/Pollex/.venv/bin/activate && \
export PYTHONPATH="$HOME/Documents/vla/LIBERO:$HOME/Documents/Pollex:$PYTHONPATH" && \
export DISPLAY=:10 && \
export XAUTHORITY="$HOME/.Xauthority" && \
unset MUJOCO_GL && \
unset QT_DEBUG_PLUGINS && \
unset QT_PLUGIN_PATH && \
unset QT_QPA_PLATFORM_PLUGIN_PATH && \
cd ~/Documents/Pollex && \
python3 -m src.simulation.rollout_smolvla_baseline \
  --checkpoint outputs/smolvla_glass_expert_iteration10 \
  --human_video datasets/cotracker_trajectories/cup_lifting_01_ct_annotated.mp4 \
  --expert_video datasets/cotracker_iterations/iteration_10_smolvla_expert/cup_lifting_01_iteration_10.mp4 \
  --chunk_exec_steps 10 \
  --steps 340 \
  --seed 6 \
  --out_dir datasets/smolvla_comparisons/k_10 \
  --gui



  Milestone Evaluation Registry

This is now the first measured SmolVLA row in your table:

Visual Preview	Phase & Milestone	K	γx	γz	Max Reach X	Min XY Distance	Lowest EEF Z	Contact	Grasp	Net Lift	Outcomebaseline_k5.gif	Phase 1: Fine-Tuned Baseline	5	1.0	1.0	-0.0818 m	2.77 cm	1.0110 m	NO	NO	0.00 cm	Under-reach + upward drift; no physical interaction

---

## compare the k=10 , and iteration 10 somlvla expert
PYTHONPATH=. python3 src/learning/compare_expert_smolvla_actions.py \
  --expert datasets/cotracker_iterations/iteration_10_smolvla_expert/cup_lifting_01_replay.npz \
  --rollout datasets/smolvla_comparisons/k_5/cup_lifting_01_baseline_rollout.npz
============================================================
EXPERT vs SmolVLA ACTION DIAGNOSTIC
============================================================
Expert:   (340, 7)
SmolVLA:  (340, 7)

Approach [0:100]
------------------------------------------------------------
dx    | expert mean=+0.0246 min=+0.0000 max=+0.1192 | smolvla mean=+0.0171 min=-0.0214 max=+0.0565
dz    | expert mean=-0.0378 min=-0.3196 max=+0.0000 | smolvla mean=+0.0454 min=-0.1503 max=+0.2049
grip  | expert mean=-1.0000 min=-1.0000 max=-1.0000 | smolvla mean=-0.9654 min=-1.0000 max=-0.8927

Pre-grasp [100:143]
------------------------------------------------------------
dx    | expert mean=+0.0030 min=+0.0001 max=+0.0059 | smolvla mean=+0.0074 min=-0.0254 max=+0.0717
dz    | expert mean=-0.2369 min=-0.3200 max=-0.0369 | smolvla mean=+0.0746 min=-0.0232 max=+0.1780
grip  | expert mean=-1.0000 min=-1.0000 max=-1.0000 | smolvla mean=-0.9801 min=-1.0000 max=-0.9194

Grasp/contact [143:160]
------------------------------------------------------------
dx    | expert mean=+0.0002 min=+0.0000 max=+0.0013 | smolvla mean=+0.0059 min=-0.0200 max=+0.0238
dz    | expert mean=+0.0083 min=-0.0197 max=+0.1112 | smolvla mean=+0.0669 min=-0.0424 max=+0.1486
grip  | expert mean=+1.0000 min=+1.0000 max=+1.0000 | smolvla mean=-0.9620 min=-1.0000 max=-0.8671

Lift/return [160:340]
------------------------------------------------------------
dx    | expert mean=-0.0378 min=-0.0600 max=+0.0000 | smolvla mean=-0.0003 min=-0.0425 max=+0.0654
dz    | expert mean=+0.0929 min=-0.3965 max=+0.3908 | smolvla mean=+0.0909 min=-0.1579 max=+0.2409
grip  | expert mean=-0.4111 min=-1.0000 max=+1.0000 | smolvla mean=-0.9643 min=-1.0000 max=-0.7018

CRITICAL FRAMES
----------------------------------------------------------------------------
 frame |  expert dx     VLA dx |  expert dz     VLA dz |  exp grip  VLA grip
     0 |    +0.0000    -0.0088 |    +0.0000    +0.1211 |    -1.000    -0.918
    50 |    +0.0799    -0.0007 |    +0.0000    +0.0048 |    -1.000    -0.987
   100 |    +0.0040    +0.0717 |    -0.3198    +0.0302 |    -1.000    -1.000
   120 |    +0.0037    +0.0167 |    -0.3200    +0.0901 |    -1.000    -0.996
   140 |    +0.0029    +0.0028 |    -0.0624    +0.0152 |    -1.000    -0.976
   143 |    +0.0000    +0.0127 |    -0.0169    +0.0360 |    +1.000    -0.962
   146 |    +0.0000    -0.0022 |    -0.0197    +0.1112 |    +1.000    -1.000
   150 |    +0.0003    +0.0141 |    -0.0004    +0.1486 |    +1.000    -0.867
   160 |    +0.0000    +0.0148 |    +0.1513    -0.0293 |    +1.000    -0.994
   180 |    +0.0000    -0.0032 |    +0.2008    +0.0652 |    +1.000    -0.941
   200 |    +0.0000    -0.0066 |    -0.3965    +0.1426 |    +1.000    -0.995
   250 |    -0.0600    +0.0654 |    +0.1500    -0.1579 |    -1.000    -0.966
   300 |    -0.0600    -0.0189 |    +0.1500    +0.1463 |    -1.000    -0.996
   339 |    -0.0600    -0.0148 |    +0.1500    +0.1316 |    -1.000    -0.980

baseline:
Expert:
approach → DESCEND → CLOSE → LIFT → RETURN → RETRACT

SmolVLA:
weak approach → RISE → STAY OPEN → drift


## iteration somlvla again trained.

(.venv) nvidia@nvidiathor:~/Documents/Pollex$ source /home/nvidia/Documents/Pollex/.venv/bin/activate && \
export PYTHONPATH="$HOME/Documents/vla/LIBERO:$HOME/Documents/Pollex:$PYTHONPATH" && \
export DISPLAY=:10 && export XAUTHORITY="$HOME/.Xauthority" && \
unset MUJOCO_GL QT_DEBUG_PLUGINS QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH && \
cd ~/Documents/Pollex && \
python3 -m src.simulation.rollout_smolvla_baseline \
  --checkpoint outputs/smolvla_glass_phase_balanced \
  --human_video datasets/cotracker_trajectories/cup_lifting_01_ct_annotated.mp4 \
  --expert_video datasets/cotracker_iterations/iteration_11_full_mujoco/cup_lifting_01_iteration_11.mp4 \
  --chunk_exec_steps 5 \
  --steps 340 \
  --seed 6 \
  --out_dir datasets/smolvla_comparisons/phase_balanced/baseline_k5 \
  --gui
[robosuite WARNING] No private macro file found! (__init__.py:7)
WARNING:robosuite_logs:No private macro file found!
[robosuite WARNING] It is recommended to use a private macro file (__init__.py:8)
WARNING:robosuite_logs:It is recommended to use a private macro file
[robosuite WARNING] To setup, run: python /home/nvidia/Documents/Pollex/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py (__init__.py:9)
WARNING:robosuite_logs:To setup, run: python /home/nvidia/Documents/Pollex/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Reducing the number of VLM layers to 16 ...
[+] Loaded checkpoint: outputs/smolvla_glass_phase_balanced/model.safetensors
[+] Rollout device: cuda
[+] GPU: NVIDIA Thor
[*] Starting pure SmolVLA baseline rollout: K=5, steps=340
[*] RoboSuite action dimension: 7
[*] Action low:  [-1. -1. -1. -1. -1. -1. -1.]
[*] Action high: [1. 1. 1. 1. 1. 1. 1.]
[+] Loaded RoboSuite environment 'GlassLift' with Franka Panda.
[PHYSICS] Glass body name: glass_main
[PHYSICS] Glass body ID: 23
[PHYSICS] Glass collision geom: glass_g0 (id=83)
[PHYSICS] Glass radius: 0.0300 m
[PHYSICS] Left pad geom: gripper0_finger1_pad_collision (id=74) half_size=[0.008 0.004 0.008]
[PHYSICS] Right pad geom: gripper0_finger2_pad_collision (id=77) half_size=[0.008 0.004 0.008]
[PHYSICS] Gripper collision geom IDs: [73, 74, 76, 77]
[PHYSICS] Initial tumbler XYZ: [-0.055   0.      0.8575]
[PHYSICS] Tumbler position source: MuJoCo body_xpos
QFontDatabase: Cannot find font directory /home/nvidia/Documents/Pollex/.venv/lib/python3.12/site-packages/cv2/qt/fonts.
Note that Qt no longer ships fonts. Deploy some (from https://dejavu-fonts.github.io/ for example) or switch to fontconfig.
QFontDatabase: Cannot find font directory /home/nvidia/Documents/Pollex/.venv/lib/python3.12/site-packages/cv2/qt/fonts.
Note that Qt no longer ships fonts. Deploy some (from https://dejavu-fonts.github.io/ for example) or switch to fontconfig.
QFontDatabase: Cannot find font directory /home/nvidia/Documents/Pollex/.venv/lib/python3.12/site-packages/cv2/qt/fonts.
Note that Qt no longer ships fonts. Deploy some (from https://dejavu-fonts.github.io/ for example) or switch to fontconfig.
QFontDatabase: Cannot find font directory /home/nvidia/Documents/Pollex/.venv/lib/python3.12/site-packages/cv2/qt/fonts.
Note that Qt no longer ships fonts. Deploy some (from https://dejavu-fonts.github.io/ for example) or switch to fontconfig.
QFontDatabase: Cannot find font directory /home/nvidia/Documents/Pollex/.venv/lib/python3.12/site-packages/cv2/qt/fonts.
Note that Qt no longer ships fonts. Deploy some (from https://dejavu-fonts.github.io/ for example) or switch to fontconfig.
[ROLLOUT] step=000/340 x=-0.1028 z=+1.0110 d_xy=4.78cm grip=-0.778 contact=0 grasp=0
[ROLLOUT] step=010/340 x=-0.0995 z=+1.0110 d_xy=4.45cm grip=-0.912 contact=0 grasp=0
[ROLLOUT] step=020/340 x=-0.0949 z=+1.0052 d_xy=3.99cm grip=-0.917 contact=0 grasp=0
[ROLLOUT] step=030/340 x=-0.0919 z=+0.9959 d_xy=3.69cm grip=-0.955 contact=0 grasp=0
[ROLLOUT] step=040/340 x=-0.0897 z=+0.9905 d_xy=3.47cm grip=-0.976 contact=0 grasp=0
[ROLLOUT] step=050/340 x=-0.0855 z=+0.9717 d_xy=3.05cm grip=-0.959 contact=0 grasp=0
[ROLLOUT] step=060/340 x=-0.0836 z=+0.9442 d_xy=2.85cm grip=-0.880 contact=0 grasp=0
[ROLLOUT] step=070/340 x=-0.0811 z=+0.9038 d_xy=2.61cm grip=-1.000 contact=0 grasp=0
[ROLLOUT] step=080/340 x=-0.0780 z=+0.8766 d_xy=2.31cm grip=-0.098 contact=0 grasp=0
[ROLLOUT] step=090/340 x=-0.0790 z=+0.8717 d_xy=2.32cm grip=+1.000 contact=1 grasp=1
[ROLLOUT] step=100/340 x=-0.0777 z=+0.8858 d_xy=2.30cm grip=+0.450 contact=1 grasp=1
[ROLLOUT] step=110/340 x=-0.0785 z=+0.9091 d_xy=2.23cm grip=+1.000 contact=1 grasp=1
[ROLLOUT] step=120/340 x=-0.0802 z=+0.9422 d_xy=2.18cm grip=+1.000 contact=1 grasp=1
[ROLLOUT] step=130/340 x=-0.0820 z=+0.9607 d_xy=2.14cm grip=+0.981 contact=1 grasp=1
[ROLLOUT] step=140/340 x=-0.0819 z=+0.9753 d_xy=2.12cm grip=+1.000 contact=1 grasp=1
[ROLLOUT] step=150/340 x=-0.0853 z=+0.9819 d_xy=2.04cm grip=+1.000 contact=1 grasp=1
[ROLLOUT] step=160/340 x=-0.0865 z=+0.9862 d_xy=1.99cm grip=+1.000 contact=1 grasp=1
[ROLLOUT] step=170/340 x=-0.0870 z=+0.9976 d_xy=1.96cm grip=+0.811 contact=1 grasp=1
[ROLLOUT] step=180/340 x=-0.0898 z=+1.0131 d_xy=1.95cm grip=+1.000 contact=1 grasp=1
[ROLLOUT] step=190/340 x=-0.0933 z=+1.0243 d_xy=1.92cm grip=+1.000 contact=1 grasp=1
[ROLLOUT] step=200/340 x=-0.0947 z=+1.0285 d_xy=1.91cm grip=+0.845 contact=1 grasp=1
[ROLLOUT] step=210/340 x=-0.0969 z=+1.0561 d_xy=1.87cm grip=+1.000 contact=1 grasp=1
[ROLLOUT] step=220/340 x=-0.0986 z=+1.0873 d_xy=1.86cm grip=+0.911 contact=1 grasp=1
[ROLLOUT] step=230/340 x=-0.1007 z=+1.1240 d_xy=1.85cm grip=+1.000 contact=1 grasp=1
[ROLLOUT] step=240/340 x=-0.1030 z=+1.1565 d_xy=1.84cm grip=+0.694 contact=1 grasp=1
[ROLLOUT] step=250/340 x=-0.1036 z=+1.1842 d_xy=1.85cm grip=+0.947 contact=1 grasp=1
[ROLLOUT] step=260/340 x=-0.1044 z=+1.2184 d_xy=1.90cm grip=+1.000 contact=1 grasp=1
[ROLLOUT] step=270/340 x=-0.1077 z=+1.2625 d_xy=1.93cm grip=+1.000 contact=1 grasp=1
[ROLLOUT] step=280/340 x=-0.1095 z=+1.3037 d_xy=1.95cm grip=+0.891 contact=1 grasp=1
[ROLLOUT] step=290/340 x=-0.1127 z=+1.3526 d_xy=1.96cm grip=+0.918 contact=1 grasp=1
[ROLLOUT] step=300/340 x=-0.1158 z=+1.4068 d_xy=1.97cm grip=+0.754 contact=1 grasp=1
[ROLLOUT] step=310/340 x=-0.1189 z=+1.4589 d_xy=2.00cm grip=+0.694 contact=1 grasp=1
[ROLLOUT] step=320/340 x=-0.1215 z=+1.5057 d_xy=2.03cm grip=+0.851 contact=1 grasp=1
[ROLLOUT] step=330/340 x=-0.1236 z=+1.5600 d_xy=2.13cm grip=+0.897 contact=1 grasp=1

[+] Baseline Rollout Completed
    Contact:             True
    First contact:       83
    Grasp:               True
    First grasp:         86
    Max glass lift:      71.04 cm
    Final glass lift:    71.04 cm
    Min XY distance:     1.83 cm
    Max reach X:         -0.0774 m
    Lowest EEF Z:        0.8664 m
    Mean inference time: 0.2026 s
    Inference FPS:       4.93
    Lift success:        True
[+] Saved rollout data: datasets/smolvla_comparisons/phase_balanced/baseline_k5/cup_lifting_01_baseline_rollout.npz
[+] Saved rollout metrics: datasets/smolvla_comparisons/phase_balanced/baseline_k5/cup_lifting_01_baseline_metrics.json
[+] Saved side-by-side video: datasets/smolvla_comparisons/phase_balanced/baseline_k5/cup_lifting_01_smolvla_baseline_k5_side_by_side.mp4
[+] Saved looping GIF: datasets/smolvla_comparisons/phase_balanced/baseline_k5/cup_lifting_01_smolvla_baseline_k5_side_by_side.gif
[+] Saved tri-panel video: datasets/smolvla_comparisons/phase_balanced/baseline_k5/cup_lifting_01_smolvla_baseline_k5_tri_panel.mp4
(.venv) nvidia@nvidiathor:~/Documents/Pollex$ python3 -m src.simulation.rollout_smolvla_exact_state_handoff \
  --checkpoint outputs/smolvla_glass_phase_balanced \
  --expert_replay datasets/cotracker_iterations/iteration_11_full_mujoco/cup_lifting_01_replay.npz \
  --human_video datasets/cotracker_trajectories/cup_lifting_01_ct_annotated.mp4 \
  --expert_video datasets/cotracker_iterations/iteration_11_full_mujoco/cup_lifting_01_iteration_11.mp4 \
  --handoff_frame 140 \
  --chunk_exec_steps 5 \
  --steps 200 \
  --seed 6 \
  --out_dir datasets/smolvla_comparisons/phase_balanced/exact_handoff_140 \
  --gui
[robosuite WARNING] No private macro file found! (__init__.py:7)
WARNING:robosuite_logs:No private macro file found!
[robosuite WARNING] It is recommended to use a private macro file (__init__.py:8)
WARNING:robosuite_logs:It is recommended to use a private macro file
[robosuite WARNING] To setup, run: python /home/nvidia/Documents/Pollex/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py (__init__.py:9)
WARNING:robosuite_logs:To setup, run: python /home/nvidia/Documents/Pollex/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Reducing the number of VLM layers to 16 ...
[+] Loaded checkpoint: outputs/smolvla_glass_phase_balanced/model.safetensors
[+] Rollout device: cuda
[+] GPU: NVIDIA Thor
[*] Exact-state SmolVLA handoff: frame=140, K=5, rollout_steps=200
[*] RoboSuite action dimension: 7
[*] Action low:  [-1. -1. -1. -1. -1. -1. -1.]
[*] Action high: [1. 1. 1. 1. 1. 1. 1.]
[+] Loaded RoboSuite environment 'GlassLift' with Franka Panda.
[PHYSICS] Glass body name: glass_main
[PHYSICS] Glass body ID: 23
[PHYSICS] Glass collision geom: glass_g0 (id=83)
[PHYSICS] Glass radius: 0.0300 m
[PHYSICS] Left pad geom: gripper0_finger1_pad_collision (id=74) half_size=[0.008 0.004 0.008]
[PHYSICS] Right pad geom: gripper0_finger2_pad_collision (id=77) half_size=[0.008 0.004 0.008]
[PHYSICS] Gripper collision geom IDs: [73, 74, 76, 77]
[PHYSICS] Initial tumbler XYZ: [-0.055   0.      0.8575]
[PHYSICS] Tumbler position source: MuJoCo body_xpos
[*] Restoring Iteration 11 post-action state at frame 140
[VERIFY] expected EEF:   [-6.7741945e-02 -1.5188655e-04  8.6376876e-01]
[VERIFY] restored EEF:   [-6.7746371e-02 -1.5161016e-04  8.6379129e-01]
[VERIFY] EEF error:      0.023 mm
[VERIFY] expected glass: [-5.2808993e-02 -8.5503576e-05  8.4753752e-01]
[VERIFY] restored glass: [-5.2811183e-02 -8.5357613e-05  8.4754270e-01]
[VERIFY] glass error:    0.006 mm
QFontDatabase: Cannot find font directory /home/nvidia/Documents/Pollex/.venv/lib/python3.12/site-packages/cv2/qt/fonts.
Note that Qt no longer ships fonts. Deploy some (from https://dejavu-fonts.github.io/ for example) or switch to fontconfig.
QFontDatabase: Cannot find font directory /home/nvidia/Documents/Pollex/.venv/lib/python3.12/site-packages/cv2/qt/fonts.
Note that Qt no longer ships fonts. Deploy some (from https://dejavu-fonts.github.io/ for example) or switch to fontconfig.
QFontDatabase: Cannot find font directory /home/nvidia/Documents/Pollex/.venv/lib/python3.12/site-packages/cv2/qt/fonts.
Note that Qt no longer ships fonts. Deploy some (from https://dejavu-fonts.github.io/ for example) or switch to fontconfig.
QFontDatabase: Cannot find font directory /home/nvidia/Documents/Pollex/.venv/lib/python3.12/site-packages/cv2/qt/fonts.
Note that Qt no longer ships fonts. Deploy some (from https://dejavu-fonts.github.io/ for example) or switch to fontconfig.
QFontDatabase: Cannot find font directory /home/nvidia/Documents/Pollex/.venv/lib/python3.12/site-packages/cv2/qt/fonts.
Note that Qt no longer ships fonts. Deploy some (from https://dejavu-fonts.github.io/ for example) or switch to fontconfig.
[ROLLOUT] local=000/200 expert_frame≈141 x=-0.0677 z=+0.8638 d_xy=1.49cm dx=+0.012 dz=-0.117 grip=-0.080 contact=0 grasp=0
[ROLLOUT] local=010/200 expert_frame≈151 x=-0.0861 z=+0.8664 d_xy=3.21cm dx=+0.011 dz=-0.162 grip=+0.837 contact=0 grasp=0
[ROLLOUT] local=020/200 expert_frame≈161 x=-0.0859 z=+0.8671 d_xy=3.19cm dx=+0.027 dz=-0.101 grip=-0.134 contact=1 grasp=0
[ROLLOUT] local=030/200 expert_frame≈171 x=-0.0860 z=+0.8664 d_xy=3.19cm dx=+0.008 dz=+0.043 grip=-0.310 contact=0 grasp=0
[ROLLOUT] local=040/200 expert_frame≈181 x=-0.0850 z=+0.8685 d_xy=3.09cm dx=+0.022 dz=-0.097 grip=-1.000 contact=0 grasp=0
[ROLLOUT] local=050/200 expert_frame≈191 x=-0.0823 z=+0.8629 d_xy=2.90cm dx=-0.003 dz=-0.009 grip=+0.947 contact=0 grasp=0
[ROLLOUT] local=060/200 expert_frame≈201 x=-0.0844 z=+0.8732 d_xy=2.90cm dx=+0.042 dz=-0.029 grip=+0.356 contact=1 grasp=1
[ROLLOUT] local=070/200 expert_frame≈211 x=-0.0845 z=+0.8700 d_xy=2.91cm dx=+0.023 dz=+0.172 grip=+0.289 contact=1 grasp=1
[ROLLOUT] local=080/200 expert_frame≈221 x=-0.0863 z=+0.8810 d_xy=2.90cm dx=+0.038 dz=-0.052 grip=+0.961 contact=1 grasp=1
[ROLLOUT] local=090/200 expert_frame≈231 x=-0.0870 z=+0.8901 d_xy=2.90cm dx=+0.026 dz=+0.052 grip=+1.000 contact=1 grasp=1
[ROLLOUT] local=100/200 expert_frame≈241 x=-0.0850 z=+0.9078 d_xy=2.90cm dx=+0.058 dz=+0.018 grip=+0.344 contact=1 grasp=1
[ROLLOUT] local=110/200 expert_frame≈251 x=-0.0861 z=+0.9276 d_xy=2.87cm dx=-0.016 dz=+0.299 grip=+1.000 contact=1 grasp=1
[ROLLOUT] local=120/200 expert_frame≈261 x=-0.0886 z=+0.9530 d_xy=2.83cm dx=+0.024 dz=+0.080 grip=+0.828 contact=1 grasp=1
[ROLLOUT] local=130/200 expert_frame≈271 x=-0.0903 z=+0.9638 d_xy=2.81cm dx=-0.002 dz=+0.221 grip=+0.899 contact=1 grasp=1
[ROLLOUT] local=140/200 expert_frame≈281 x=-0.0897 z=+0.9673 d_xy=2.80cm dx=-0.000 dz=-0.008 grip=+1.000 contact=1 grasp=1
[ROLLOUT] local=150/200 expert_frame≈291 x=-0.0930 z=+0.9712 d_xy=2.74cm dx=+0.013 dz=+0.204 grip=+1.000 contact=1 grasp=1
[ROLLOUT] local=160/200 expert_frame≈301 x=-0.0952 z=+0.9777 d_xy=2.67cm dx=-0.005 dz=-0.136 grip=+0.842 contact=1 grasp=1
[ROLLOUT] local=170/200 expert_frame≈311 x=-0.0973 z=+0.9898 d_xy=2.64cm dx=-0.023 dz=+0.081 grip=+0.343 contact=1 grasp=1
[ROLLOUT] local=180/200 expert_frame≈321 x=-0.1007 z=+0.9970 d_xy=2.59cm dx=-0.022 dz=+0.060 grip=+0.787 contact=1 grasp=1
[ROLLOUT] local=190/200 expert_frame≈331 x=-0.1049 z=+0.9985 d_xy=2.55cm dx=-0.003 dz=-0.104 grip=+1.000 contact=1 grasp=1

[+] Exact-State Handoff Rollout Completed
    Handoff frame:             140
    Actual rollout steps:      200
    Contact:                   True
    First local contact:       12
    Grasp:                     True
    First local grasp:         55
    Continuation max lift:     12.06 cm
    Full-episode max lift:     12.10 cm
    Final continuation lift:   11.62 cm
    Minimum XY distance:       1.49 cm
    Maximum EEF X:             -0.0677 m
    Lowest EEF Z:              0.8625 m
    Mean inference time:       0.2097 s
    Inference FPS:             4.77
    Lift success:              True
[+] Saved rollout data: datasets/smolvla_comparisons/phase_balanced/exact_handoff_140/cup_lifting_01_exact_handoff_140_rollout.npz
[+] Saved rollout metrics: datasets/smolvla_comparisons/phase_balanced/exact_handoff_140/cup_lifting_01_exact_handoff_140_metrics.json
[+] Saved side-by-side video: datasets/smolvla_comparisons/phase_balanced/exact_handoff_140/cup_lifting_01_exact_handoff_140_k5_side_by_side.mp4
[+] Saved looping GIF: datasets/smolvla_comparisons/phase_balanced/exact_handoff_140/cup_lifting_01_exact_handoff_140_k5_side_by_side.gif
[+] Saved tri-panel video: datasets/smolvla_comparisons/phase_balanced/exact_handoff_140/cup_lifting_01_exact_handoff_140_k5_tri_panel.mp4
(.venv) nvidia@nvidiathor:~/Documents/Pollex$ python3 -m src.simulation.rollout_smolvla_exact_state_handoff \
  --checkpoint outputs/smolvla_glass_phase_balanced \
  --expert_replay datasets/cotracker_iterations/iteration_11_full_mujoco/cup_lifting_01_replay.npz \
  --human_video datasets/cotracker_trajectories/cup_lifting_01_ct_annotated.mp4 \
  --expert_video datasets/cotracker_iterations/iteration_11_full_mujoco/cup_lifting_01_iteration_11.mp4 \
  --handoff_frame 146 \
  --chunk_exec_steps 5 \
  --steps 200 \
  --seed 6 \
  --out_dir datasets/smolvla_comparisons/phase_balanced/exact_handoff_146 \
  --gui
[robosuite WARNING] No private macro file found! (__init__.py:7)
WARNING:robosuite_logs:No private macro file found!
[robosuite WARNING] It is recommended to use a private macro file (__init__.py:8)
WARNING:robosuite_logs:It is recommended to use a private macro file
[robosuite WARNING] To setup, run: python /home/nvidia/Documents/Pollex/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py (__init__.py:9)
WARNING:robosuite_logs:To setup, run: python /home/nvidia/Documents/Pollex/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Reducing the number of VLM layers to 16 ...
[+] Loaded checkpoint: outputs/smolvla_glass_phase_balanced/model.safetensors
[+] Rollout device: cuda
[+] GPU: NVIDIA Thor
[*] Exact-state SmolVLA handoff: frame=146, K=5, rollout_steps=200
[*] RoboSuite action dimension: 7
[*] Action low:  [-1. -1. -1. -1. -1. -1. -1.]
[*] Action high: [1. 1. 1. 1. 1. 1. 1.]
[+] Loaded RoboSuite environment 'GlassLift' with Franka Panda.
[PHYSICS] Glass body name: glass_main
[PHYSICS] Glass body ID: 23
[PHYSICS] Glass collision geom: glass_g0 (id=83)
[PHYSICS] Glass radius: 0.0300 m
[PHYSICS] Left pad geom: gripper0_finger1_pad_collision (id=74) half_size=[0.008 0.004 0.008]
[PHYSICS] Right pad geom: gripper0_finger2_pad_collision (id=77) half_size=[0.008 0.004 0.008]
[PHYSICS] Gripper collision geom IDs: [73, 74, 76, 77]
[PHYSICS] Initial tumbler XYZ: [-0.055   0.      0.8575]
[PHYSICS] Tumbler position source: MuJoCo body_xpos
[*] Restoring Iteration 11 post-action state at frame 146
[VERIFY] expected EEF:   [-6.8850055e-02 -7.9696329e-05  8.6641490e-01]
[VERIFY] restored EEF:   [-6.8847835e-02 -7.9707432e-05  8.6642319e-01]
[VERIFY] EEF error:      0.009 mm
[VERIFY] expected glass: [-5.3319197e-02 -4.8154550e-05  8.4825540e-01]
[VERIFY] restored glass: [-5.3319450e-02 -4.8167098e-05  8.4826291e-01]
[VERIFY] glass error:    0.008 mm
QFontDatabase: Cannot find font directory /home/nvidia/Documents/Pollex/.venv/lib/python3.12/site-packages/cv2/qt/fonts.
Note that Qt no longer ships fonts. Deploy some (from https://dejavu-fonts.github.io/ for example) or switch to fontconfig.
QFontDatabase: Cannot find font directory /home/nvidia/Documents/Pollex/.venv/lib/python3.12/site-packages/cv2/qt/fonts.
Note that Qt no longer ships fonts. Deploy some (from https://dejavu-fonts.github.io/ for example) or switch to fontconfig.
QFontDatabase: Cannot find font directory /home/nvidia/Documents/Pollex/.venv/lib/python3.12/site-packages/cv2/qt/fonts.
Note that Qt no longer ships fonts. Deploy some (from https://dejavu-fonts.github.io/ for example) or switch to fontconfig.
QFontDatabase: Cannot find font directory /home/nvidia/Documents/Pollex/.venv/lib/python3.12/site-packages/cv2/qt/fonts.
Note that Qt no longer ships fonts. Deploy some (from https://dejavu-fonts.github.io/ for example) or switch to fontconfig.
QFontDatabase: Cannot find font directory /home/nvidia/Documents/Pollex/.venv/lib/python3.12/site-packages/cv2/qt/fonts.
Note that Qt no longer ships fonts. Deploy some (from https://dejavu-fonts.github.io/ for example) or switch to fontconfig.
[ROLLOUT] local=000/200 expert_frame≈147 x=-0.0688 z=+0.8664 d_xy=1.55cm dx=+0.010 dz=+0.069 grip=+1.000 contact=1 grasp=1
[ROLLOUT] local=010/200 expert_frame≈157 x=-0.0926 z=+0.8802 d_xy=1.68cm dx=+0.008 dz=-0.119 grip=+1.000 contact=1 grasp=1
[ROLLOUT] local=020/200 expert_frame≈167 x=-0.0938 z=+0.8906 d_xy=1.71cm dx=+0.008 dz=+0.014 grip=+1.000 contact=1 grasp=1
[ROLLOUT] local=030/200 expert_frame≈177 x=-0.0940 z=+0.8971 d_xy=1.73cm dx=-0.008 dz=+0.275 grip=+1.000 contact=1 grasp=1
[ROLLOUT] local=040/200 expert_frame≈187 x=-0.0960 z=+0.9206 d_xy=1.73cm dx=+0.011 dz=+0.081 grip=+1.000 contact=1 grasp=1
[ROLLOUT] local=050/200 expert_frame≈197 x=-0.0959 z=+0.9250 d_xy=1.73cm dx=-0.012 dz=-0.042 grip=+1.000 contact=1 grasp=1
[ROLLOUT] local=060/200 expert_frame≈207 x=-0.0984 z=+0.9295 d_xy=1.74cm dx=+0.016 dz=+0.022 grip=+0.999 contact=1 grasp=1
[ROLLOUT] local=070/200 expert_frame≈217 x=-0.1001 z=+0.9251 d_xy=1.72cm dx=+0.019 dz=+0.186 grip=+0.958 contact=1 grasp=1
[ROLLOUT] local=080/200 expert_frame≈227 x=-0.1022 z=+0.9337 d_xy=1.73cm dx=+0.013 dz=-0.061 grip=+1.000 contact=1 grasp=1
[ROLLOUT] local=090/200 expert_frame≈237 x=-0.1056 z=+0.9383 d_xy=1.69cm dx=+0.002 dz=+0.026 grip=+1.000 contact=1 grasp=1
[ROLLOUT] local=100/200 expert_frame≈247 x=-0.1067 z=+0.9543 d_xy=1.70cm dx=+0.042 dz=-0.010 grip=+0.702 contact=1 grasp=1
[ROLLOUT] local=110/200 expert_frame≈257 x=-0.1088 z=+0.9661 d_xy=1.66cm dx=-0.023 dz=+0.197 grip=+1.000 contact=1 grasp=1
[ROLLOUT] local=120/200 expert_frame≈267 x=-0.1118 z=+0.9834 d_xy=1.63cm dx=+0.023 dz=+0.058 grip=+0.639 contact=1 grasp=1
[ROLLOUT] local=130/200 expert_frame≈277 x=-0.1140 z=+0.9935 d_xy=1.61cm dx=+0.002 dz=+0.332 grip=+1.000 contact=1 grasp=1
[ROLLOUT] local=140/200 expert_frame≈287 x=-0.1138 z=+1.0121 d_xy=1.61cm dx=-0.013 dz=+0.036 grip=+1.000 contact=1 grasp=1
[ROLLOUT] local=150/200 expert_frame≈297 x=-0.1183 z=+1.0253 d_xy=1.56cm dx=+0.003 dz=+0.283 grip=+1.000 contact=1 grasp=1
[ROLLOUT] local=160/200 expert_frame≈307 x=-0.1209 z=+1.0449 d_xy=1.51cm dx=-0.007 dz=+0.018 grip=+0.810 contact=1 grasp=1
[ROLLOUT] local=170/200 expert_frame≈317 x=-0.1235 z=+1.0785 d_xy=1.51cm dx=-0.023 dz=+0.243 grip=+0.556 contact=1 grasp=1
[ROLLOUT] local=180/200 expert_frame≈327 x=-0.1276 z=+1.1103 d_xy=1.50cm dx=-0.018 dz=+0.194 grip=+0.890 contact=1 grasp=1
[ROLLOUT] local=190/200 expert_frame≈337 x=-0.1318 z=+1.1359 d_xy=1.48cm dx=-0.001 dz=+0.060 grip=+0.949 contact=1 grasp=1

[+] Exact-State Handoff Rollout Completed
    Handoff frame:             146
    Actual rollout steps:      200
    Contact:                   True
    First local contact:       0
    Grasp:                     True
    First local grasp:         0
    Continuation max lift:     28.17 cm
    Full-episode max lift:     28.28 cm
    Final continuation lift:   28.17 cm
    Minimum XY distance:       1.45 cm
    Maximum EEF X:             -0.0688 m
    Lowest EEF Z:              0.8664 m
    Mean inference time:       0.2094 s
    Inference FPS:             4.78
    Lift success:              True
[+] Saved rollout data: datasets/smolvla_comparisons/phase_balanced/exact_handoff_146/cup_lifting_01_exact_handoff_146_rollout.npz
[+] Saved rollout metrics: datasets/smolvla_comparisons/phase_balanced/exact_handoff_146/cup_lifting_01_exact_handoff_146_metrics.json
[+] Saved side-by-side video: datasets/smolvla_comparisons/phase_balanced/exact_handoff_146/cup_lifting_01_exact_handoff_146_k5_side_by_side.mp4
[+] Saved looping GIF: datasets/smolvla_comparisons/phase_balanced/exact_handoff_146/cup_lifting_01_exact_handoff_146_k5_side_by_side.gif
[+] Saved tri-panel video: datasets/smolvla_comparisons/phase_balanced/exact_handoff_146/cup_lifting_01_exact_handoff_146_k5_tri_panel.mp4
(.venv) nvidia@nvidiathor:~/Documents/Pollex$ python3 -m src.simulation.rollout_smolvla_exact_state_handoff \
  --checkpoint outputs/smolvla_glass_phase_balanced \
  --expert_replay datasets/cotracker_iterations/iteration_11_full_mujoco/cup_lifting_01_replay.npz \
  --human_video datasets/cotracker_trajectories/cup_lifting_01_ct_annotated.mp4 \
  --expert_video datasets/cotracker_iterations/iteration_11_full_mujoco/cup_lifting_01_iteration_11.mp4 \
  --handoff_frame 160 \
  --chunk_exec_steps 5 \
  --steps 180 \
  --seed 6 \
  --out_dir datasets/smolvla_comparisons/phase_balanced/exact_handoff_160 \
  --gui
[robosuite WARNING] No private macro file found! (__init__.py:7)
WARNING:robosuite_logs:No private macro file found!
[robosuite WARNING] It is recommended to use a private macro file (__init__.py:8)
WARNING:robosuite_logs:It is recommended to use a private macro file
[robosuite WARNING] To setup, run: python /home/nvidia/Documents/Pollex/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py (__init__.py:9)
WARNING:robosuite_logs:To setup, run: python /home/nvidia/Documents/Pollex/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Reducing the number of VLM layers to 16 ...
[+] Loaded checkpoint: outputs/smolvla_glass_phase_balanced/model.safetensors
[+] Rollout device: cuda
[+] GPU: NVIDIA Thor
[*] Exact-state SmolVLA handoff: frame=160, K=5, rollout_steps=180
[*] RoboSuite action dimension: 7
[*] Action low:  [-1. -1. -1. -1. -1. -1. -1.]
[*] Action high: [1. 1. 1. 1. 1. 1. 1.]
[+] Loaded RoboSuite environment 'GlassLift' with Franka Panda.
[PHYSICS] Glass body name: glass_main
[PHYSICS] Glass body ID: 23
[PHYSICS] Glass collision geom: glass_g0 (id=83)
[PHYSICS] Glass radius: 0.0300 m
[PHYSICS] Left pad geom: gripper0_finger1_pad_collision (id=74) half_size=[0.008 0.004 0.008]
[PHYSICS] Right pad geom: gripper0_finger2_pad_collision (id=77) half_size=[0.008 0.004 0.008]
[PHYSICS] Gripper collision geom IDs: [73, 74, 76, 77]
[PHYSICS] Initial tumbler XYZ: [-0.055   0.      0.8575]
[PHYSICS] Tumbler position source: MuJoCo body_xpos
[*] Restoring Iteration 11 post-action state at frame 160
[VERIFY] expected EEF:   [-7.037137e-02 -7.294919e-05  8.698228e-01]
[VERIFY] restored EEF:   [-7.038468e-02 -7.293952e-05  8.698854e-01]
[VERIFY] EEF error:      0.064 mm
[VERIFY] expected glass: [-5.4278210e-02 -4.4671979e-05  8.5155135e-01]
[VERIFY] restored glass: [-5.4289613e-02 -4.4675522e-05  8.5161412e-01]
[VERIFY] glass error:    0.064 mm
QFontDatabase: Cannot find font directory /home/nvidia/Documents/Pollex/.venv/lib/python3.12/site-packages/cv2/qt/fonts.
Note that Qt no longer ships fonts. Deploy some (from https://dejavu-fonts.github.io/ for example) orswitch to fontconfig.
QFontDatabase: Cannot find font directory /home/nvidia/Documents/Pollex/.venv/lib/python3.12/site-packages/cv2/qt/fonts.
Note that Qt no longer ships fonts. Deploy some (from https://dejavu-fonts.github.io/ for example) orswitch to fontconfig.
QFontDatabase: Cannot find font directory /home/nvidia/Documents/Pollex/.venv/lib/python3.12/site-packages/cv2/qt/fonts.
Note that Qt no longer ships fonts. Deploy some (from https://dejavu-fonts.github.io/ for example) orswitch to fontconfig.
QFontDatabase: Cannot find font directory /home/nvidia/Documents/Pollex/.venv/lib/python3.12/site-packages/cv2/qt/fonts.
Note that Qt no longer ships fonts. Deploy some (from https://dejavu-fonts.github.io/ for example) orswitch to fontconfig.
QFontDatabase: Cannot find font directory /home/nvidia/Documents/Pollex/.venv/lib/python3.12/site-packages/cv2/qt/fonts.
Note that Qt no longer ships fonts. Deploy some (from https://dejavu-fonts.github.io/ for example) orswitch to fontconfig.
[ROLLOUT] local=000/180 expert_frame≈161 x=-0.0704 z=+0.8699 d_xy=1.61cm dx=+0.010 dz=+0.094 grip=+1.000 contact=1 grasp=1
[ROLLOUT] local=010/180 expert_frame≈171 x=-0.0931 z=+0.8831 d_xy=1.69cm dx=+0.006 dz=-0.106 grip=+1.000 contact=1 grasp=1
[ROLLOUT] local=020/180 expert_frame≈181 x=-0.0938 z=+0.8934 d_xy=1.72cm dx=+0.006 dz=+0.076 grip=+1.000 contact=1 grasp=1
[ROLLOUT] local=030/180 expert_frame≈191 x=-0.0953 z=+0.9035 d_xy=1.74cm dx=-0.011 dz=+0.299 grip=+1.000 contact=1 grasp=1
[ROLLOUT] local=040/180 expert_frame≈201 x=-0.0972 z=+0.9278 d_xy=1.74cm dx=+0.000 dz=+0.102 grip=+1.000 contact=1 grasp=1
[ROLLOUT] local=050/180 expert_frame≈211 x=-0.0980 z=+0.9322 d_xy=1.73cm dx=-0.010 dz=-0.028 grip=+1.000 contact=1 grasp=1
[ROLLOUT] local=060/180 expert_frame≈221 x=-0.1016 z=+0.9383 d_xy=1.73cm dx=+0.013 dz=+0.037 grip=+0.973 contact=1 grasp=1
[ROLLOUT] local=070/180 expert_frame≈231 x=-0.1039 z=+0.9385 d_xy=1.70cm dx=+0.019 dz=+0.186 grip=+0.944 contact=1 grasp=1
[ROLLOUT] local=080/180 expert_frame≈241 x=-0.1056 z=+0.9486 d_xy=1.69cm dx=+0.014 dz=-0.103 grip=+1.000 contact=1 grasp=1
[ROLLOUT] local=090/180 expert_frame≈251 x=-0.1079 z=+0.9482 d_xy=1.66cm dx=+0.000 dz=-0.003 grip=+1.000 contact=1 grasp=1
[ROLLOUT] local=100/180 expert_frame≈261 x=-0.1090 z=+0.9600 d_xy=1.66cm dx=+0.039 dz=-0.042 grip=+0.562 contact=1 grasp=1
[ROLLOUT] local=110/180 expert_frame≈271 x=-0.1110 z=+0.9678 d_xy=1.63cm dx=-0.023 dz=+0.209 grip=+1.000 contact=1 grasp=1
[ROLLOUT] local=120/180 expert_frame≈281 x=-0.1138 z=+0.9842 d_xy=1.59cm dx=+0.019 dz=+0.116 grip=+0.865 contact=1 grasp=1
[ROLLOUT] local=130/180 expert_frame≈291 x=-0.1160 z=+0.9987 d_xy=1.57cm dx=-0.007 dz=+0.309 grip=+0.893 contact=1 grasp=1
[ROLLOUT] local=140/180 expert_frame≈301 x=-0.1167 z=+1.0155 d_xy=1.57cm dx=-0.013 dz=+0.047 grip=+1.000 contact=1 grasp=1
[ROLLOUT] local=150/180 expert_frame≈311 x=-0.1214 z=+1.0298 d_xy=1.51cm dx=+0.005 dz=+0.313 grip=+1.000 contact=1 grasp=1
[ROLLOUT] local=160/180 expert_frame≈321 x=-0.1237 z=+1.0541 d_xy=1.47cm dx=-0.004 dz=+0.055 grip=+0.811 contact=1 grasp=1
[ROLLOUT] local=170/180 expert_frame≈331 x=-0.1263 z=+1.0924 d_xy=1.47cm dx=-0.026 dz=+0.263 grip=+0.478 contact=1 grasp=1

[+] Exact-State Handoff Rollout Completed
    Handoff frame:             160
    Actual rollout steps:      180
    Contact:                   True
    First local contact:       0
    Grasp:                     True
    First local grasp:         0
    Continuation max lift:     24.54 cm
    Full-episode max lift:     24.99 cm
    Final continuation lift:   24.54 cm
    Minimum XY distance:       1.45 cm
    Maximum EEF X:             -0.0704 m
    Lowest EEF Z:              0.8699 m
    Mean inference time:       0.2124 s
    Inference FPS:             4.71
    Lift success:              True
[+] Saved rollout data: datasets/smolvla_comparisons/phase_balanced/exact_handoff_160/cup_lifting_01_exact_handoff_160_rollout.npz
[+] Saved rollout metrics: datasets/smolvla_comparisons/phase_balanced/exact_handoff_160/cup_lifting_01_exact_handoff_160_metrics.json
[+] Saved side-by-side video: datasets/smolvla_comparisons/phase_balanced/exact_handoff_160/cup_lifting_01_exact_handoff_160_k5_side_by_side.mp4
[+] Saved looping GIF: datasets/smolvla_comparisons/phase_balanced/exact_handoff_160/cup_lifting_01_exact_handoff_160_k5_side_by_side.gif
[+] Saved tri-panel video: datasets/smolvla_comparisons/phase_balanced/exact_handoff_160/cup_lifting_01_exact_handoff_160_k5_tri_panel.mp4
(.venv) nvidia@nvidiathor:~/Documents/Pollex$ 

## iteration v2 
"""
Phase-Balanced SmolVLA Fine-Tuning on the GlassLift Expert Dataset.

This experiment keeps the baseline SmolVLA architecture, raw action space,
flow-matching objective, optimizer, state input, and chunk horizon unchanged.

Only the frame-sampling distribution changes:
    hover                       10%
    approach                    15%
    descent / pre-grasp         20%
    grasp closing               15%
    post-grasp closed lift      25%
    post-grasp closed lower     10%
    post-grasp closed static     2%
    release / retract            3%

Inputs:
    RGB observations: (340, 256, 256, 3)
    Panda state:      (340, 8)
    Expert actions:   (340, 7)

Output:
    outputs/smolvla_glass_phase_balanced/
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
    "hover": 0.10,
    "approach": 0.15,
    "descent_pregrasp": 0.20,
    "grasp_closing": 0.15,
    "postgrasp_closed_lift": 0.25,
    "postgrasp_closed_lower": 0.10,
    "postgrasp_closed_static": 0.02,
    "release_retract": 0.03,
}


def classify_phase(
    frame: int,
    action: np.ndarray,
    hover_end: int,
    descent_start: int,
    grasp_trigger_frame: int,
    physical_grasp_frame: int,
) -> str:
    if frame < hover_end:
        return "hover"
    if frame < descent_start:
        return "approach"
    if frame < grasp_trigger_frame:
        return "descent_pregrasp"
    if frame < physical_grasp_frame:
        return "grasp_closing"

    grip_closed = action[6] > 0.0
    dz = action[2]

    if not grip_closed:
        return "release_retract"
    if dz > 1e-4:
        return "postgrasp_closed_lift"
    if dz < -1e-4:
        return "postgrasp_closed_lower"
    return "postgrasp_closed_static"


class PhaseBalancedGlassDataset(Dataset):
    """RGB, 8D state, action chunk, and phase label dataset."""

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
            grasp_trigger_frame = int(np.asarray(replay["grasp_trigger_frame"]).item())
            physical_grasp_frame = int(
                np.asarray(replay["physical_grasp_latch_frame"]).item()
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

            T = len(actions)

            if images.shape != (T, 256, 256, 3):
                raise ValueError(f"Expected images (T,256,256,3), got {images.shape}")
            if states.shape != (T, 8):
                raise ValueError(f"Expected states (T,8), got {states.shape}")
            if actions.shape != (T, 7):
                raise ValueError(f"Expected actions (T,7), got {actions.shape}")
            if not np.isfinite(states).all():
                raise ValueError(f"Non-finite states in {path}")
            if not np.isfinite(actions).all():
                raise ValueError(f"Non-finite actions in {path}")

            if len(npz_paths) == 1 and replay_actions.shape != actions.shape:
                raise ValueError(
                    f"Training/replay action shape mismatch: "
                    f"{actions.shape} != {replay_actions.shape}"
                )

            if len(npz_paths) == 1 and not np.allclose(
                actions,
                replay_actions,
                atol=1e-5,
            ):
                max_difference = float(np.max(np.abs(actions - replay_actions)))
                raise ValueError(
                    "Training episode does not contain the Iteration 11 actions. "
                    f"Maximum difference: {max_difference:.6f}. "
                    "Regenerate the LeRobot episode from Iteration 11 first."
                )

            hover_end = min(40, T)
            descent_start = min(86, T)
            grasp_trigger = int(np.clip(grasp_trigger_frame, 0, T))
            physical_grasp = int(np.clip(physical_grasp_frame, 0, T))

            for frame in range(T):
                future_length = min(chunk_size, T - frame)
                action_chunk = np.zeros(
                    (chunk_size, max_action_dim),
                    dtype=np.float32,
                )
                action_chunk[:future_length, :7] = actions[
                    frame:frame + future_length
                ]

                if future_length < chunk_size:
                    action_chunk[future_length:, :7] = actions[
                        frame + future_length - 1
                    ]

                phase = classify_phase(
                    frame=frame,
                    action=actions[frame],
                    hover_end=hover_end,
                    descent_start=descent_start,
                    grasp_trigger_frame=grasp_trigger,
                    physical_grasp_frame=physical_grasp,
                )

                self.samples.append({
                    "image": images[frame].copy(),
                    "state": states[frame].copy(),
                    "action_chunk": action_chunk,
                    "frame": global_frame_offset + frame,
                    "phase": phase,
                })

            global_frame_offset += T

        self.phase_counts = Counter(
            sample["phase"] for sample in self.samples
        )

        missing_phases = [
            phase
            for phase, target in PHASE_TARGETS.items()
            if target > 0 and self.phase_counts.get(phase, 0) == 0
        ]

        if missing_phases:
            raise RuntimeError(
                f"Target phases have no samples: {missing_phases}"
            )

        print(
            f"[*] Loaded {len(self.samples)} transitions "
            f"across {len(npz_paths)} episode(s)."
        )
        print("[*] Natural phase distribution:")

        for phase in PHASE_TARGETS:
            count = self.phase_counts.get(phase, 0)
            percentage = 100.0 * count / len(self.samples)
            print(f"    {phase:30s} {count:3d} frames ({percentage:5.2f}%)")

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
        parameter
        for parameter in policy.parameters()
        if parameter.requires_grad
    ]

    trainable_count = sum(
        parameter.numel()
        for parameter in trainable_parameters
    )

    print(
        f"[+] Loaded model. Trainable parameters: "
        f"{trainable_count / 1e6:.2f} M."
    )

    return policy, config


def save_checkpoint(
    policy: SmolVLAPolicy,
    checkpoint_path: Path,
):
    state_dict = {
        key: value.detach().contiguous().cpu()
        for key, value in policy.state_dict().items()
    }

    save_file(state_dict, str(checkpoint_path))


def train_phase_balanced(
    dataset_dir: str = "datasets/lerobot_dataset/glass_pick_place_iteration11",
    expert_replay: str = (
        "datasets/cotracker_iterations/iteration_11_full_mujoco/"
        "cup_lifting_01_replay.npz"
    ),
    out_dir: str = "outputs/smolvla_glass_phase_balanced",
    epochs: int = 30,
    batch_size: int = 4,
    lr: float = 1e-4,
    seed: int = 6,
):
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
    print("Phase-Balanced SmolVLA Action Expert Fine-Tuning")
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

    sampler = train_dataset.build_sampler(
        samples_per_epoch=len(train_dataset),
        seed=seed,
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
    language_masks_single = tokens["attention_mask"].to(
        device,
        dtype=torch.bool,
    )

    print(f"[+] Language tokens: {tuple(language_tokens_single.shape)}")

    trainable_parameters = [
        parameter
        for parameter in policy.parameters()
        if parameter.requires_grad
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
            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            states = batch["state"].to(
                device,
                non_blocking=True,
            )

            action_chunks = batch["action_chunk"].to(
                device,
                non_blocking=True,
            )

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
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch}"
                )

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
        history["sampled_phase_counts"].append(
            dict(sampled_phase_counts)
        )

        if (
            epoch == 1
            or epoch % 5 == 0
            or epoch == epochs
        ):
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
        f"[+] Initial Train Loss: {initial_loss:.4f} "
        f"-> Final: {final_loss:.4f} "
        f"({loss_drop:.1f}% drop)"
    )
    print(f"[+] Best Training Loss: {best_train_loss:.4f}")

    config_output = {
        "model_type": "smolvla_glass_phase_balanced",
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
        "sampling_mode": "phase_balanced_weighted_random_sampler",
        "samples_per_epoch": len(train_dataset),
        "natural_phase_counts": dict(train_dataset.phase_counts),
        "target_phase_probabilities": PHASE_TARGETS,
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

    print(
        f"[+] Saved phase-balanced checkpoint and metadata: "
        f"{output_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Phase-balanced SmolVLA fine-tuning"
    )

    parser.add_argument(
        "--dataset_dir",
        default=(
            "datasets/lerobot_dataset/"
            "glass_pick_place_iteration11"
        ),
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
        "--out_dir",
        default="outputs/smolvla_glass_phase_balanced",
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

    train_phase_balanced(
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

cd ~/Documents/Pollex && \
PYTHONPATH=. python3 -m src.learning.train_smolvla_phase_balanced \
  --dataset_dir datasets/lerobot_dataset/glass_pick_place_iteration11 \
  --expert_replay datasets/cotracker_iterations/iteration_11_full_mujoco/cup_lifting_01_replay.npz \
  --out_dir outputs/smolvla_glass_phase_balanced_smoke \
  --epochs 1 \
  --batch_size 4 \
  --lr 1e-4 \
  --seed 6
"""


FULL-TASK SUCCESS on all 4 combos! Every single criterion passes — fails=[] across the board.

γ_lift	Score	Peak Lift	Lowering	Final Height	Grasp Held
2.00	98.07	4.31cm	5.27cm	-0.91cm	112 frames
1.90	97.85	3.87cm	4.85cm	-0.91cm	114 frames
1.70	97.63	3.44cm	4.41cm	-0.91cm	114 frames
1.80	97.55	3.27cm	4.24cm	-0.91cm	108 frames
The best configuration: γ_x=1.0, γ_z=1.25, γ_lift=2.0 — score 98.07/100.

The complete task cycle works:

Approach — reaches glass (min d_xy ≤ 3cm)
Contact — physical contact detected
Grasp — secure grasp held for 112 frames
Lift — 4.31cm peak lift (above 3cm threshold)
Lower — 5.27cm lowering from peak
Release — gripper opens after detecting lower cycle
Stable finish — glass settles at -0.91cm (within ±2cm)
From 1.17cm peak lift and 76.5 score to 4.31cm lift and 98.07 score — the SmolVLA policy now completes the full pick-and-place task, matching what the retargeted trajectory achieves.



