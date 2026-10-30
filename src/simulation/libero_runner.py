"""
Simulation Environment Runner & Wrapper for RoboSuite / LIBERO
Provides a unified interface for Franka Panda manipulation environments,
supporting both headless offscreen rendering and interactive on-screen visualization.

export PYTHONPATH="$HOME/Documents/vla/LIBERO:$PYTHONPATH"

export DISPLAY=:10
export XAUTHORITY="$HOME/.Xauthority"

unset MUJOCO_GL
unset QT_DEBUG_PLUGINS
unset QT_PLUGIN_PATH
unset QT_QPA_PLATFORM_PLUGIN_PATH

python3 -m src.simulation.libero_runner
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional

try:
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.simulation.glass_lift_env import register_glass_lift_env
    register_glass_lift_env()
except Exception:
    pass


class PandaSimEnvironment:
    """
    Standardized wrapper for Franka Emika Panda manipulation in RoboSuite / LIBERO.
    """

    def __init__(
        self,
        env_name: str = "GlassLift",
        has_renderer: bool = False,
        has_offscreen_renderer: bool = True,
        camera_name: str = "agentview",
        camera_height: int = 256,
        camera_width: int = 256,
        control_freq: int = 20,
    ):
        self.env_name = env_name
        self.camera_name = camera_name
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.control_freq = control_freq

        self._glass_body_id = None
        self._glass_body_name = None
        self._glass_geom_ids = set()
        self._gripper_contact_geom_ids = set()
        self._left_pad_geom_name = None
        self._right_pad_geom_name = None
        self._left_pad_geom_id = None
        self._right_pad_geom_id = None
        self._glass_geom_id = None
        self._glass_radius = 0.030
        self._left_pad_half_size = None
        self._right_pad_half_size = None
        self._physics_available = False

        try:
            import robosuite as suite
            from robosuite.controllers import load_controller_config

            controller_config = load_controller_config(default_controller="OSC_POSE")

            self.env = suite.make(
                env_name=self.env_name,
                robots="Panda",
                controller_configs=controller_config,
                has_renderer=has_renderer,
                has_offscreen_renderer=has_offscreen_renderer,
                render_camera=self.camera_name,
                camera_names=[self.camera_name],
                camera_heights=[self.camera_height],
                camera_widths=[self.camera_width],
                control_freq=self.control_freq,
                horizon=500,
                use_camera_obs=has_offscreen_renderer,
                initialization_noise=None,
            )

            action_low, action_high = self.env.action_spec
            print(f"[*] RoboSuite action dimension: {len(action_low)}")
            print(f"[*] Action low:  {action_low}")
            print(f"[*] Action high: {action_high}")

            self.is_libero = False
            print(f"[+] Loaded RoboSuite environment '{env_name}' with Franka Panda.")

        except ImportError as e:
            raise ImportError(f"Could not import robosuite.\nOriginal error: {e}")

    def reset(self) -> Dict[str, Any]:
        obs = self.env.reset()
        self._init_physics_ids()
        return self._format_obs(obs)

    def _init_physics_ids(self):
        """Cache MuJoCo body/geom IDs for glass and gripper after first reset."""
        if self._physics_available:
            return
        if not (hasattr(self.env, "cube") and hasattr(self.env, "sim")):
            print("[PHYSICS] No glass object in this environment — physics measurements disabled.")
            return

        try:
            model = self.env.sim.model
            self._glass_body_name = self.env.cube.root_body
            self._glass_body_id = model.body_name2id(self._glass_body_name)

            for i in range(model.ngeom):
                if model.geom_bodyid[i] == self._glass_body_id:
                    name = model.geom_id2name(i)
                    if name and "vis" not in name:
                        self._glass_geom_ids.add(i)

            gripper_collision_names = {
                "gripper0_finger1_collision",
                "gripper0_finger1_pad_collision",
                "gripper0_finger2_collision",
                "gripper0_finger2_pad_collision",
            }
            for i in range(model.ngeom):
                name = model.geom_id2name(i)
                if name in gripper_collision_names:
                    self._gripper_contact_geom_ids.add(i)

            # Finger pad geoms for grasp diagnostic (verified names from MuJoCo model dump)
            self._left_pad_geom_name = "gripper0_finger1_pad_collision"
            self._right_pad_geom_name = "gripper0_finger2_pad_collision"
            self._left_pad_geom_id = model.geom_name2id(self._left_pad_geom_name)
            self._right_pad_geom_id = model.geom_name2id(self._right_pad_geom_name)

            # Glass collision geom size (cylinder: [radius, half_height])
            self._glass_geom_id = sorted(self._glass_geom_ids)[0]
            self._glass_radius = float(model.geom_size[self._glass_geom_id][0])

            # Pad box half-extents for surface clearance calculation
            self._left_pad_half_size = model.geom_size[self._left_pad_geom_id].copy()
            self._right_pad_half_size = model.geom_size[self._right_pad_geom_id].copy()

            self._physics_available = True

            initial_pos = self.env.sim.data.body_xpos[self._glass_body_id].copy()
            print(f"[PHYSICS] Glass body name: {self._glass_body_name}")
            print(f"[PHYSICS] Glass body ID: {self._glass_body_id}")
            print(f"[PHYSICS] Glass collision geom: {model.geom_id2name(self._glass_geom_id)} (id={self._glass_geom_id})")
            print(f"[PHYSICS] Glass radius: {self._glass_radius:.4f} m")
            print(f"[PHYSICS] Left pad geom: {self._left_pad_geom_name} (id={self._left_pad_geom_id}) half_size={self._left_pad_half_size}")
            print(f"[PHYSICS] Right pad geom: {self._right_pad_geom_name} (id={self._right_pad_geom_id}) half_size={self._right_pad_half_size}")
            print(f"[PHYSICS] Gripper collision geom IDs: {sorted(self._gripper_contact_geom_ids)}")
            print(f"[PHYSICS] Initial tumbler XYZ: {initial_pos}")
            print(f"[PHYSICS] Tumbler position source: MuJoCo body_xpos")

        except Exception as e:
            print(f"[PHYSICS] Failed to initialize physics IDs: {e}")

    def _get_geom_xyz(self, geom_id: int) -> Optional[np.ndarray]:
        pos = self.env.sim.data.geom_xpos[geom_id].copy()
        if pos.shape == (3,) and np.all(np.isfinite(pos)):
            return pos
        return None

    def _get_tumbler_pos(self) -> Optional[np.ndarray]:
        if not self._physics_available:
            return None
        pos = self.env.sim.data.body_xpos[self._glass_body_id].copy()
        if pos.shape == (3,) and np.all(np.isfinite(pos)):
            return pos
        print(f"[PHYSICS] Invalid tumbler position: {pos}")
        return None

    def _get_tumbler_contact(self) -> bool:
        """True if any gripper finger/pad geom is in contact with the glass collision geom."""
        if not self._physics_available:
            return False
        for i in range(self.env.sim.data.ncon):
            c = self.env.sim.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if (g1 in self._glass_geom_ids and g2 in self._gripper_contact_geom_ids) or \
               (g2 in self._glass_geom_ids and g1 in self._gripper_contact_geom_ids):
                return True
        return False

    def _get_tumbler_grasped(self) -> bool:
        """Uses RoboSuite's built-in _check_grasp which tests that both fingers contact the object."""
        if not self._physics_available:
            return False
        try:
            return bool(self.env._check_grasp(self.env.robots[0].gripper, self.env.cube))
        except Exception:
            return False

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        assert len(action) == 7, f"Action must be 7-dimensional, got {len(action)}"
        obs, reward, done, info = self.env.step(action)
        return self._format_obs(obs), float(reward), bool(done), info

    def render(self):
        if hasattr(self.env, "render"):
            self.env.render()

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()

    def _format_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        img_key = f"{self.camera_name}_image"
        rgb = obs.get(img_key, None)
        if rgb is not None:
            rgb = np.flipud(rgb)

        tumbler_pos = self._get_tumbler_pos()
        tumbler_contact = self._get_tumbler_contact()
        tumbler_grasped = self._get_tumbler_grasped()

        left_pad_pos = None
        right_pad_pos = None
        if self._physics_available:
            left_pad_pos = self._get_geom_xyz(self._left_pad_geom_id)
            right_pad_pos = self._get_geom_xyz(self._right_pad_geom_id)

        return {
            "rgb": rgb,
            "robot_eef_pos": obs.get("robot0_eef_pos", None),
            "robot_eef_quat": obs.get("robot0_eef_quat", None),
            "robot_gripper_qpos": obs.get("robot0_gripper_qpos", None),
            "object_pos": tumbler_pos,
            "tumbler_pos": tumbler_pos,
            "tumbler_pos_source": f"body:{self._glass_body_name}" if self._physics_available else None,
            "tumbler_contact": tumbler_contact,
            "tumbler_grasped": tumbler_grasped,
            "left_finger_pad_pos": left_pad_pos,
            "right_finger_pad_pos": right_pad_pos,
            "left_finger_pad_geom": self._left_pad_geom_name,
            "right_finger_pad_geom": self._right_pad_geom_name,
            "glass_radius": self._glass_radius,
            "raw_obs": obs,
        }


if __name__ == "__main__":
    import time

    print("[*] Running Panda simulation with visible window...")
    try:
        sim = PandaSimEnvironment(env_name="Lift", has_renderer=True, has_offscreen_renderer=False)
        obs = sim.reset()
        print("[+] Environment reset successful.")
        print(f"    Observation keys: {list(obs.keys())}")
        print("[*] Running 200 steps — watch the simulation window...")

        for i in range(200):
            action = np.array([0.0, 0.0, -0.05, 0.0, 0.0, 0.0, 1.0])
            obs, r, d, _ = sim.step(action)
            sim.render()
            time.sleep(1.0 / 20)

        print("[+] Simulation steps completed successfully.")
        sim.close()
    except Exception as err:
        print(f"[-] Sim test could not run: {err}")
