from pathlib import Path
import os

from mujoco import mj_saveModel, mj_saveLastXML
from typing_extensions import Optional

import mujoco
import mujoco.viewer
from dm_control import mjcf
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_ARM_XML = _HERE / "kuka_iiwa_14" / "scene.xml"
_HAND_XML = _HERE / "wonik_allegro" / "left_hand.xml"

fingers = ["rf_tip", "mf_tip", "ff_tip", "th_tip"]
finger_colors = {
    fingers[0]: "0.9 0 0 1", # Red
    fingers[1]: "0 0.9 0 1", # Green
    fingers[2]: "0 0 0.9 1", # Blue
    fingers[3]: "0.9 0.9 0.9 1" # White
}
sites = {
    fingers[0]: "ball_s1",
    fingers[1]: "ball_s2",
    fingers[2]: "ball_s3",
    fingers[3]: "ball_s4"
}

# fmt: off
HOME_QPOS = [
    # iiwa.
    -0.0759329, 0.153982, 0.104381, -1.8971, 0.245996, 0.34972, -0.239115,
    # allegro.
    -0.0694123, 0.0551428, 0.986832, 0.671424,
    -0.186261, -0.0866821, 1.01374, 0.728192,
    -0.218949, -0.0318307, 1.25156, 0.840648,
    1.0593, 0.638801, 0.391599, 0.57284
]
# fmt: on

arm_dof = 7
palm_dof = 16
def construct_model():
    arm_mjcf = mjcf.from_path(_ARM_XML.as_posix())
    arm_mjcf.find("key", "home").remove()

    hand_mjcf = mjcf.from_path(_HAND_XML.as_posix())
    palm = hand_mjcf.worldbody.find("body", "palm")
    palm.quat = (1, 0, 0, 0)
    palm.pos = (0, 0, 0.095)
    attach_site = arm_mjcf.worldbody.find("site", "attachment_site")
    attach_site.attach(hand_mjcf)

    arm_mjcf.keyframe.add("key", name="home", qpos=HOME_QPOS)

    for finger in fingers:
        body = arm_mjcf.worldbody.add("body", name=f"{finger}_target", mocap=True)
        body.add(
            "geom",
            type="sphere",
            size=".02",
            contype="0",
            conaffinity="0",
            rgba=finger_colors[finger],
        )

    return mujoco.MjModel.from_xml_string(
        arm_mjcf.to_xml_string(), arm_mjcf.get_assets()
    )

def save_model(model: mujoco.MjModel, path: Optional[str]=""):
    mj_saveLastXML(path if path else f"{os.path.splitext(os.path.basename(__file__))[0]}.xml", model)

if __name__ == "__main__":
    model = construct_model()
    #save_model(model)

    configuration = mink.Configuration(model)

    end_effector_task = mink.FrameTask(
        frame_name="attachment_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )

    posture_task = mink.PostureTask(model=model, cost=5e-2)

    finger_tasks = []
    for finger in fingers:
        task = mink.RelativeFrameTask(
            frame_name=f"allegro_left/{finger}",
            frame_type="site",
            root_name="allegro_left/palm",
            root_type="body",
            position_cost=1.0,
            orientation_cost=0.0,
            lm_damping=1.0,
        )
        finger_tasks.append(task)

    tasks = [end_effector_task, posture_task, *finger_tasks]

    limits = [
        mink.ConfigurationLimit(model=model),
    ]

    # IK settings.
    solver = "quadprog"
    model = configuration.model
    data = configuration.data
    fingers_following_ball_sites = False

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)

        # Initialize the mocap target at the end-effector site.
        mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")
        for finger in fingers:
            mink.move_mocap_to_frame(
                model, data, f"{finger}_target", f"allegro_left/{finger}", "site"
            )

        T_eef_prev = configuration.get_transform_frame_to_world(
            "attachment_site", "site"
        )

        rate = RateLimiter(frequency=100.0, warn=False)
        while viewer.is_running():
            # Update kuka end-effector task, as [target]'s SE3
            T_wt = mink.SE3.from_mocap_name(model, data, "target")
            end_effector_task.set_target(T_wt)

            # Update finger tasks' targets, relative SE3 of [fingertip] relative to [palm]
            for finger, task in zip(fingers, finger_tasks):
                T_pm = configuration.get_transform(
                    f"{finger}_target", "body", "allegro_left/palm", "body"
                )
                task.set_target(T_pm)

                # Move [EE] -> also moving fingertip-target mocap-bodies
                # Calc [T], delta SE3 from current EE to prev EE (attachment_site)
                T_eef = configuration.get_transform_frame_to_world(
                    "attachment_site", "site"
                )
                dT = T_eef @ T_eef_prev.inverse()

                # Calc [T_w_mocap], current fingertip-target mocap-body's SE3
                if fingers_following_ball_sites:
                    site = sites[finger]
                    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
                    T_w_mocap = mink.SE3.from_rotation_and_translation(
                                   rotation=mink.SO3.from_matrix(data.site_xmat[site_id].reshape(3,3)),
                                   translation=data.site_xpos[site_id],
                                )
                else:
                    T_w_mocap = mink.SE3.from_mocap_name(model, data, f"{finger}_target")

                # Calc [T_w_mocap_new], new expected fingertip-target mocap-body's SE3
                # , moving them to new poses
                T_w_mocap_new = dT @ T_w_mocap
                data.mocap_pos[model.body(f"{finger}_target").mocapid[0]] = (
                    T_w_mocap_new.translation()
                )
                data.mocap_quat[model.body(f"{finger}_target").mocapid[0]] = (
                    T_w_mocap_new.rotation().wxyz
                )

            # Compute velocity and integrate into the next configuration.
            vel = mink.solve_ik(
                configuration, tasks, rate.dt, solver, 1e-3, limits=limits
            )
            kinematics = False
            if kinematics:
                configuration.integrate_inplace(vel, rate.dt)
            else:
                configuration.apply_ctrl(arm_dof, palm_dof, vel, rate.dt)
            mujoco.mj_camlight(model, data)

            T_eef_prev = T_eef.copy()

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
