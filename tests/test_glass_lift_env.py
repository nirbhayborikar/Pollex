"""
Unit tests for GlassLiftEnv and simulation runner with steel cylinder.
"""

import pytest
import numpy as np
from src.simulation.glass_lift_env import GlassLiftEnv, register_glass_lift_env, GLASS_RADIUS, GLASS_HALF_HEIGHT
from src.simulation.libero_runner import PandaSimEnvironment


def test_glass_lift_env_registration():
    """Verify GlassLift is registered in robosuite environments."""
    register_glass_lift_env()
    from robosuite.environments.base import REGISTERED_ENVS
    assert "GlassLift" in REGISTERED_ENVS
    assert REGISTERED_ENVS["GlassLift"] is GlassLiftEnv


def test_glass_lift_sim_reset_and_dimensions():
    """Verify environment initializes with cylinder geometry and valid observations."""
    sim = PandaSimEnvironment(env_name="GlassLift", has_renderer=False, has_offscreen_renderer=True)
    obs = sim.reset()
    assert obs is not None
    assert "rgb" in obs
    assert obs["rgb"].shape == (256, 256, 3)
    assert "robot_eef_pos" in obs
    assert obs["robot_eef_pos"].shape == (3,)

    # Verify object is a cylinder
    from robosuite.models.objects import CylinderObject
    assert isinstance(sim.env.cube, CylinderObject)
    assert sim.env.cube.size[0] == GLASS_RADIUS

    sim.close()


def test_glass_lift_step_action():
    """Verify 7D action execution in GlassLift environment."""
    sim = PandaSimEnvironment(env_name="GlassLift", has_renderer=False, has_offscreen_renderer=True)
    obs = sim.reset()
    action = np.array([0.0, 0.0, -0.02, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    next_obs, reward, done, info = sim.step(action)
    assert next_obs["rgb"].shape == (256, 256, 3)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    sim.close()
