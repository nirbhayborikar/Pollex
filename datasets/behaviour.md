
Iteration 1:

Notice:
tilt = 0. The robot moving away from the cube , and might be it is hitting to table , and not picking the cube. The motion is it perform going forward, down , then clos gripper , and the up .


Iteration 2
angle 60
gain 8.0
what observed: the robot grippe area metal is hitting the robot cannot able to recognize the object still.

What we already know (verified, no guessing needed)

Iteration 3 Diagnostic — Complete Results
The Decisive Numbers
Metric	Value	What it means
EEF closest frame	98	EEF origin gets within 3.65 cm horizontally
Pad closest frame	154 (both pads)	Pads get closest 56 frames later than EEF
Min pad-to-center distance	4.36 cm	Pads never reach the glass
Pad-to-glass-axis horizontal dist	4.35 cm	Glass radius = 3.0 cm → pads are 1.35 cm from surface
Surface clearance	1.25 cm	~12.5 mm gap on each side
Pad separation @ EEF closest (98)	8.70 cm	Gripper is wide open — glass (6.0 cm) fits easily
Pad separation @ pad closest (154)	1.10 cm	Gripper nearly fully closed — fingers touch empty air
Min pad separation in episode	0.80 cm at frame 176	Gripper fully closed
Gripper close starts	frame 143	Gripper open command ends, close begins
Root Cause: Case A — Horizontal Approach Misalignment
The gripper closes through the glass location, but the pads pass ~4.3 cm to the side of the glass axis. Here's the exact geometry at frame 154:

Glass axis: X = -0.028 m
Left pad: X = -0.0711 m
Right pad: X = -0.0711 m
EEF origin: X = -0.0707 m
The EEF, both pads, and the entire gripper assembly are consistently ~4.3 cm too far in the -X direction from the glass. This is a constant lateral offset, not a timing or closing issue.

At frame 98 (EEF closest): EEF is 3.65 cm off in X, gripper is 8.7 cm open. The arm approaches, overshoots in -X, then the gripper closes on empty space at frames 143-176.

Iteration 3 Parameter
The retargeting needs a constant X offset of approximately +4.3 cm (or the equivalent correction in image-to-robot mapping) so the gripper pads center over the glass instead of passing to its side.

This is in cotracker_to_panda.py — the retargeting maps hand centroid to a position that's consistently ~4.3 cm off from where the pads end up. One parameter change: add the measured offset.


Iteration 4 Results
What I did
Added one parameter eef_to_pad_offset_x = 0.043 to cotracker_to_panda.py. During the approach phase, each frame's dx_robot gets an extra 0.043 / reach_span added, spreading the 4.3 cm correction evenly across the approach frames. This shifts the EEF target forward so the finger pads land on the glass instead of passing 4.3 cm to the side.

What changed vs Iteration 3
Metric	Iteration 3 (Iter2 actions)	Iteration 4 (corrected actions)
Contact occurred	NO	YES (frame 201)
First contact frame	N/A	201
Min EEF-object dist	3.80 cm	3.79 cm
Min pad-to-center dist	4.36 cm	4.32 cm
Surface clearance	1.25 cm	1.20 cm
Grasp occurred	NO	NO
Lift success	NO	NO
What this tells us
Contact happened for the first time — the gripper fingers now physically touch the glass. But _check_grasp still returns False because it requires both fingers to contact the object simultaneously, and the offset correction alone isn't enough for bilateral contact.

The pad distance barely moved (4.36 → 4.32 cm). The 4.3 cm offset helped but wasn't the full story — the approach still isn't perfectly aligned. The glass is being pushed by one finger rather than enclosed by both.

The real issue
Looking at the contact frames (201-209), the EEF is at X ≈ -0.069 and the glass is at X ≈ -0.029. That's still a 4.0 cm gap. The offset correction spread the 4.3 cm over the approach but the EMA smoothing and the reach dynamics absorbed most of it. The pads are hitting the glass from the side (single-finger contact), not wrapping around it.

The next parameter should be a larger, more direct offset — or the approach needs to target the glass position explicitly rather than just adding a constant spread. Want me to try increasing the offset, or should we look at the reference repo's approach more closely to see how they achieve bilateral contact?


###

source /home/nvidia/Documents/Pollex/.venv/bin/activate && \
export PYTHONPATH="$HOME/Documents/vla/LIBERO:$PYTHONPATH" && \
export DISPLAY=:10 && \
export XAUTHORITY="$HOME/.Xauthority" && \
unset MUJOCO_GL && \
unset QT_DEBUG_PLUGINS && \
unset QT_PLUGIN_PATH && \
unset QT_QPA_PLATFORM_PLUGIN_PATH && \
cd /home/nvidia/Documents/Pollex && \
python3 -m src.simulation.replay_trajectory \
    --actions datasets/cotracker_iterations/iteration_06_wrist_squared/cup_lifting_01_ct_actions.npz \
    --out_video datasets/cotracker_iterations/iteration_06_wrist_squared/cup_lifting_01_iteration_6.mp4 \
    --out_dataset datasets/cotracker_iterations/iteration_06_wrist_squared/cup_lifting_01_replay.npz \
    --human_video datasets/cotracker_trajectories/cup_lifting_01_ct_annotated.mp4 \
    --task GlassLift \
    --gui 2>&1


    ## iteration 17
                  Tumbler XYZ
                  │
        ┌─────────┴─────────┐
        │                   │
 pad-center X          pad-center Z
        │                   │
        ▼                   ▼
 target - relative X   target - relative Z
        │                   │
        └────────┬──────────┘
                 ▼
           geometry ready
          X ±3 mm, Z ±2 mm
                 │
           3 consecutive
              frames
                 │
                 ▼
            close gripper
                 │
                 ▼
          physical grasp
                 │
                 ▼
                lift