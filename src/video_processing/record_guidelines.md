# Egocentric Video Recording Guidelines for Robot Learning

Follow these steps to record your manipulation dataset using your smartphone. This dataset will directly drive the Franka Emika Panda arm in the LIBERO simulation environment.

---

## 1. Physical Setup & Workspace

```
             [ Smartphone ] (Chest level, tilted ~45° downward)
                  │
                  ▼ Field of View
       ┌──────────────────────────────┐
       │   Clean, uncluttered table   │
       │                              │
       │    [Home]          [Target]  │
       │    (Rest)           (Goal)   │
       │      │                ▲      │
       │      └────► [Mug] ────┘      │
       │           (Object)           │
       └──────────────────────────────┘
```

1. **Table Surface**:
   * Use a plain, non-reflective desk or tabletop with minimal clutter.
   * If possible, use a neutral table (white, wood, or solid gray) to match standard simulated tabletop environments.

2. **Lighting**:
   * Ensure uniform, diffused lighting.
   * Avoid strong backlighting (e.g., sitting directly with a window behind the table) or harsh point shadows that confuse hand keypoint detectors.

3. **Target Object**:
   * Use an everyday object that mirrors LIBERO tasks:
     * **Option A (Recommended)**: A coffee mug or drinking cup.
     * **Option B**: A small can, spice jar, or wooden block.
   * Place a coaster, small sheet of paper, or small tray as the goal target.

---

## 2. Camera Positioning

1. **Egocentric Perspective**:
   * Position the phone at **chest or sternum height** (you can hold it against your chest with your non-dominant hand, wear a phone chest strap, or mount it on a desk tripod just behind your shoulder).
   * Angle the camera **~35° to 45° downward**, pointing towards the table.
2. **Field of View Framing**:
   * Both your dominant hand (starting from the bottom of the frame) and the target object must remain in view during the entire motion.
   * Avoid zooming in or using ultra-wide (0.5x fish-eye) distortion if possible; standard 1x wide lens is best.

---

## 3. Recording Protocol & Motion

Record **10 to 20 separate video clips** (5 to 8 seconds each) following this exact motion script:

1. **Rest Position (t = 0.0s - 0.5s)**:
   * Keep your hand still near the bottom-center of the frame with fingers slightly open.
2. **Reach (t = 0.5s - 2.0s)**:
   * Smoothly move your hand toward the object. Keep the motion continuous and deliberate (no rapid jerks).
3. **Pre-grasp & Pinch (t = 2.0s - 3.0s)**:
   * Form a clear two-finger pinch (thumb + index finger) or claw grasp around the object.
4. **Transport / Move (t = 3.0s - 5.0s)**:
   * Lift or slide the object to the target destination (e.g. onto the coaster).
5. **Release & Retract (t = 5.0s - 6.0s)**:
   * Open fingers, release the object, and retract your hand slightly backward.

---

## 4. Video Specifications

* **Format**: `.mp4` or `.mov` (H.264 / standard iPhone or Android camera).
* **Frame Rate**: 30 FPS (avoid 60 FPS or slow-motion; 30 FPS matches standard robot control frequency).
* **Resolution**: 1080p or 720p (the preprocessing script will automatically center-crop and resize to $256 \times 256$).
* **File Naming**: Save recordings into `data/raw_videos/` as:
  * `demo_01.mp4`, `demo_02.mp4`, ..., `demo_15.mp4`
