"""
Simulation Environment Runner & Wrapper for RoboSuite / LIBERO
Provides a unified interface for Franka Panda manipulation environments,
supporting both headless offscreen rendering and interactive on-screen visualization.
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional

# Register GlassLift environment (steel cylinder) before robosuite is called
try:
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.simulation.glass_lift_env import register_glass_lift_env
    register_glass_lift_env()
except Exception:
    pass  # graceful fallback — GlassLift will just be unavailable


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
        """
        Args:
            env_name: RoboSuite task ("Lift", "PickPlaceCan", "Door") or LIBERO task name.
            has_renderer: Set True for interactive window on macOS.
            has_offscreen_renderer: Set True for saving RGB frames to video/dataset.
            camera_name: Primary camera ("agentview", "frontview", "robot0_eye_in_hand").
            camera_height: Rendered image height.
            camera_width: Rendered image width.
            control_freq: Policy step frequency in Hz (typically 20 Hz).
        """
        self.env_name = env_name
        self.camera_name = camera_name
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.control_freq = control_freq

        # Attempt to import robosuite
        try:
            import robosuite as suite
            from robosuite.controllers import load_composite_controller_config

            # Robosuite 1.5+ automatically loads the calibrated Panda OSC controller
            self.env = suite.make(
                env_name=self.env_name,
                robots="Panda",
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
            self.is_libero = False
            print(f"[+] Loaded RoboSuite environment '{env_name}' with Franka Panda.")

        except ImportError as e:
            raise ImportError(
                "Could not import robosuite. Install via: pip install mujoco robosuite\n"
                f"Original error: {e}"
            )

    def reset(self) -> Dict[str, Any]:
        """Resets the environment and returns the initial observation."""
        obs = self.env.reset()
        return self._format_obs(obs)

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Executes a 7D action: [dx, dy, dz, droll, dpitch, dyaw, gripper].
        Returns (obs, reward, done, info).
        """
        assert len(action) == 7, f"Action must be 7-dimensional, got {len(action)}"
        obs, reward, done, info = self.env.step(action)
        return self._format_obs(obs), float(reward), bool(done), info

    def render(self):
        """Renders live on-screen if has_renderer is True."""
        if hasattr(self.env, "render"):
            self.env.render()

    def close(self):
        """Cleans up simulation contexts."""
        if hasattr(self.env, "close"):
            self.env.close()

    def _format_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts and formats primary RGB frame and robot proprioception."""
        img_key = f"{self.camera_name}_image"
        rgb = obs.get(img_key, None)

        # RoboSuite camera images are rendered upside down due to OpenGL conventions
        if rgb is not None:
            rgb = np.flipud(rgb)

        return {
            "rgb": rgb,
            "robot_eef_pos": obs.get("robot0_eef_pos", None),
            "robot_eef_quat": obs.get("robot0_eef_quat", None),
            "robot_gripper_qpos": obs.get("robot0_gripper_qpos", None),
            "raw_obs": obs,
        }


if __name__ == "__main__":
    print("[*] Running Panda simulation smoke test...")
    try:
        sim = PandaSimEnvironment(env_name="Lift", has_renderer=False, has_offscreen_renderer=True)
        obs = sim.reset()
        print("[+] Environment reset successful.")
        print(f"    Observation keys: {list(obs.keys())}")
        if obs["rgb"] is not None:
            print(f"    Rendered camera shape: {obs['rgb'].shape}")

        for i in range(10):
            action = np.array([0.0, 0.0, -0.05, 0.0, 0.0, 0.0, 1.0])
            obs, r, d, _ = sim.step(action)

        print("[+] 10 simulation steps completed successfully.")
        sim.close()
    except Exception as err:
        print(f"[-] Sim test could not run: {err}")
