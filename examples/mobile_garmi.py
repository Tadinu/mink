from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from dm_control.viewer import user_input
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "garmi" / "garmi_scene.xml"


@dataclass
class KeyCallback:
    fix_base: bool = True
    pause: bool = False

    def __call__(self, key: int) -> None:
        if key == user_input.KEY_ENTER:
            self.fix_base = not self.fix_base
        elif key == user_input.KEY_SPACE:
            self.pause = not self.pause


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    configuration = mink.Configuration(model, data)

    left_end_effector_task = mink.RelativeFrameTask(
        frame_name="left_ee_site",
        frame_type="site",
        root_name="left_base",
        root_type="site",
        position_cost=5.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )

    right_end_effector_task = mink.RelativeFrameTask(
        frame_name="right_ee_site",
        frame_type="site",
        root_name="right_base",
        root_type="site",
        position_cost=5.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )

    arm_tasks = [left_end_effector_task, right_end_effector_task]

    home_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_key_id > -1:
        posture_cost = np.zeros((model.nv,))
        posture_cost[2] = 1e-3
        posture_task = mink.PostureTask(model, cost=posture_cost)
        arm_tasks += [posture_task]

    # When move the base, mainly focus on the motion on xy plane, minimize the rotation.
    immobile_base_cost = np.zeros((model.nv,))
    immobile_base_cost[:2] = 100
    immobile_base_cost[2] = 1e-3
    damping_task = mink.DampingTask(model, immobile_base_cost)
    full_body_tasks = arm_tasks #+ [damping_task]

    limits = [
        mink.ConfigurationLimit(model),
    ]

    # IK settings.
    solver = "quadprog"
    pos_threshold = 1e-4
    ori_threshold = 1e-4

    key_callback = KeyCallback()

    with (mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=key_callback,
    ) as viewer):
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        if home_key_id > -1:
            mujoco.mj_resetDataKeyframe(model, data, home_key_id)
            configuration.update(data.qpos)
            posture_task.set_target_from_configuration(configuration)
        mujoco.mj_forward(model, data)

        # Initialize the mocap target at the end-effector site.
        mink.move_mocap_to_frame(model, data, "left_target", "left_ee_site", "site")
        mink.move_mocap_to_frame(model, data, "right_target", "right_ee_site", "site")

        rate = RateLimiter(frequency=200.0, warn=False)
        while viewer.is_running():
            # Update task target.
            left_end_effector_task.set_target(
                configuration.get_transform(
                    "left_target", "body", "left_base", "site"
                )
            )
            right_end_effector_task.set_target(configuration.get_transform(
                    "right_target", "body", "right_base", "site"
                )
            )
            # Compute velocity and integrate into the next configuration.
            fix_base = key_callback.fix_base
            # Solve for left arm
            vel = mink.solve_ik(configuration, full_body_tasks if fix_base else arm_tasks,
                                rate.dt, solver, 1e-3)
            configuration.integrate_inplace(vel, rate.dt, kinematics_only=False)
            mujoco.mj_camlight(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
