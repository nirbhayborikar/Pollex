"""
GlassLiftEnv — RoboSuite Lift environment with a steel glass cylinder.
Subclasses the standard Lift task and replaces the BoxObject (red cube)
with a CylinderObject (silver steel glass) whose geometry matches the
real 74mm-diameter steel glass in the recorded demonstration videos.

Register via: robosuite.environments.ALL_ENVIRONMENTS["GlassLift"] = GlassLiftEnv
"""

import numpy as np
from robosuite.environments.manipulation.lift import Lift
from robosuite.models.objects import CylinderObject
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.mjcf_utils import CustomMaterial


# Steel glass geometry: 60 mm diameter, 95 mm height (matches slim steel tumbler)
# This provides 10 mm clearance on each side of the 80 mm Franka parallel-jaw gripper
GLASS_RADIUS = 0.030    # metres  — 60 mm outer diameter
GLASS_HALF_HEIGHT = 0.0475  # metres  — 95 mm total height, half for MuJoCo

# Lift success threshold: glass CoM must rise this far above the tabletop
LIFT_HEIGHT_SUCCESS_M = 0.10   # 10 cm

# MuJoCo material properties for a hollow steel cylinder with high-friction contact
GLASS_FRICTION  = [2.0, 0.05, 0.001]     # high friction for reliable parallel-jaw grasp
GLASS_DENSITY   = 450.0                  # realistic hollow tumbler mass (~120g)
GLASS_RGBA      = [0.80, 0.82, 0.85, 1.0]  # steel-grey colour


class GlassLiftEnv(Lift):
    """
    Identical to the standard RoboSuite Lift task, except the object
    is a silver cylinder (steel glass) instead of a red wooden cube.

    The class intentionally keeps 'self.cube' as the attribute name
    so all inherited reward / success-detection methods that reference
    'self.cube' continue to work without modification.
    """

    def __init__(self, **kwargs):
        if "initialization_noise" not in kwargs:
            kwargs["initialization_noise"] = None
        super().__init__(**kwargs)

    def _load_model(self):
        """Override model loading to swap cube → glass cylinder."""
        # Call parent first — it sets up the table arena and robot.
        super()._load_model()

        # Square the Franka Panda wrist so fingers open along Y (perpendicular to table approach)
        # Default RoboSuite Panda has q7 = pi/4 (45° diagonal), causing the gripper to approach sideways
        if len(self.robots) > 0 and self.robots[0].name == "Panda":
            square_qpos = np.array([0, np.pi / 16.0, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.2, np.pi / 4])
            self.robots[0].init_qpos = square_qpos

        # Build a new CylinderObject for the glass with compliant contact.
        glass = CylinderObject(
            name="glass",
            size=[GLASS_RADIUS, GLASS_HALF_HEIGHT],
            rgba=GLASS_RGBA,
            density=GLASS_DENSITY,
            friction=GLASS_FRICTION,
            solref=[0.01, 1.0],
            solimp=[0.9, 0.95, 0.001],
            joints="default",
        )

        # Replace the parent's BoxObject ('cube') with our cylinder.
        # We keep the attribute name 'cube' for full compatibility with
        # all inherited methods (_check_success, reward, _check_grasp …).
        self.cube = glass

        # Rebuild the placement initializer around the new object.
        # Position the cylinder right in front of the robot gripper (-0.035m)
        glass_ref_pos = [self.table_offset[0] - 0.035, self.table_offset[1], self.table_offset[2]]
        self.placement_initializer = UniformRandomSampler(
            name="GlassObjectSampler",
            mujoco_objects=self.cube,
            x_range=[0.0, 0.0],
            y_range=[0.0, 0.0],
            rotation=None,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=glass_ref_pos,
            z_offset=0.01,
        )


        # Reconstruct the full ManipulationTask with the new object.
        from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
        from robosuite.models.tasks import ManipulationTask
        from robosuite.models.arenas import TableArena

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cube,
        )

    def _reset_internal(self):
        """Ensure Franka Panda wrist remains square across all resets."""
        if len(self.robots) > 0 and self.robots[0].name == "Panda":
            square_qpos = np.array([0, np.pi / 16.0, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.2, np.pi / 4])
            self.robots[0].init_qpos = square_qpos
        super()._reset_internal()

    def _check_success(self):
        """
        Success: glass CoM is at least LIFT_HEIGHT_SUCCESS_M above the table.
        Inherits the grasping check structure from Lift._check_success.
        """
        glass_height = self.sim.data.body_xpos[
            self.sim.model.body_name2id(self.cube.root_body)
        ][2]
        table_height = self.model.mujoco_arena.table_offset[2]
        return (glass_height - table_height) >= LIFT_HEIGHT_SUCCESS_M

    def get_object_z(self) -> float:
        """Returns the glass body Z position in world frame (for metrics)."""
        return float(
            self.sim.data.body_xpos[
                self.sim.model.body_name2id(self.cube.root_body)
            ][2]
        )


def register_glass_lift_env():
    """
    Register GlassLiftEnv with robosuite so suite.make('GlassLift', ...) works.
    Call this once at application startup.
    """
    from robosuite.environments.base import REGISTERED_ENVS
    if "GlassLift" not in REGISTERED_ENVS:
        REGISTERED_ENVS["GlassLift"] = GlassLiftEnv


# Auto-register when this module is imported
register_glass_lift_env()
