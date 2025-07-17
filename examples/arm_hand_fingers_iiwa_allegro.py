from pathlib import Path

import mujoco as mj
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink

# mjmanip
from mjmanip.mj_utils import mj_spec_copy, mj_spec_add_spec, mj_model_joints_qpos_ids
from mjmanip.robot.allegro import Allegro

_HERE = Path(__file__).parent
_ARM_XML = _HERE / "kuka_iiwa_14" / "scene_target.xml"
_HAND_XML = _HERE / "wonik_allegro" / "left_hand.xml"

fingers = ["rf_tip", "mf_tip", "ff_tip", "th_tip"]

# fmt: off
HOME_QPOS = [
    # iiwa.
    -0.0759329, 0.153982, 0.104381, -1.8971, 0.245996, 0.34972, -0.239115,
    # allegro.
    -0.0694123, 0.0551428, 0.986832, 0.671424,
    -0.186261, -0.0866821, 1.01374, 0.728192,
    -0.218949, -0.0318307, 1.25156, 0.840648,
    1.0593, 0.638801, 0.391599, 0.57284,
]

# fmt: on
USE_GOAL_HAND = False
GOAL_HAND_BASE_NAME = None
GOAL_HAND_BASE_POS_OFFSET = (0, 0, 0.095)
GOAL_HAND_BASE_QUAT_OFFSET = (1, 0, 0, 0)
GOAL_HAND_FULL_JOINTS_NAMES = []
HAND_FULL_JOINTS_NAMES = []


def construct_model() -> mujoco.MjModel:
    global GOAL_HAND_BASE_NAME, GOAL_HAND_FULL_JOINTS_NAMES, HAND_FULL_JOINTS_NAMES
    arm = mujoco.MjSpec.from_file(_ARM_XML.as_posix())
    hand = mujoco.MjSpec.from_file(_HAND_XML.as_posix())
    HAND_FULL_JOINTS_NAMES = [f"{hand.modelname}/{jnt_name}" for jnt_name in Allegro.HAND_JOINTS_NAMES]

    palm = hand.body("palm")
    palm.quat[:] = GOAL_HAND_BASE_QUAT_OFFSET
    palm.pos[:] = GOAL_HAND_BASE_POS_OFFSET
    site = arm.site("attachment_site")
    arm.attach(hand, prefix="allegro_left/", site=site)

    home_key = arm.key("home")
    arm.delete(home_key)
    arm.add_key(name="home", qpos=HOME_QPOS)

    if USE_GOAL_HAND:
        goal_hand_spec = mj_spec_copy(hand, rem_visual=True, collision_enabled=False)
        goal_hand_base_spec = mj_spec_add_spec(arm, goal_hand_spec, prefix="goal/", with_free_joint=False)
        GOAL_HAND_BASE_NAME = goal_hand_base_spec.name
        GOAL_HAND_FULL_JOINTS_NAMES = [f"goal/{goal_hand_spec.modelname}/{jnt_name}" for jnt_name in
                                       Allegro.HAND_JOINTS_NAMES]
    else:
        for finger in fingers:
            body = arm.worldbody.add_body(name=f"{finger}_target", mocap=True)
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=(0.02,) * 3,
                contype=0,
                conaffinity=0,
                rgba=(0.6, 0.3, 0.3, 0.5),
            )

    # Obstacle
    obst = arm.worldbody.add_body(name="obstacle", mocap=True)
    obst.add_geom(name=obst.name,
                  type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                  size=(0.02, 0.1,),
                  pos=(0.8, 0, 0.5),
                  contype=0,
                  conaffinity=0,
                  rgba=(0.1, 0.3, 0.3, 0.5))
    return arm.compile()


if __name__ == "__main__":
    model = construct_model()

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
    if not USE_GOAL_HAND:
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

    collision_pairs = [
        (
            mink.get_subtree_geom_ids(model, model.body("allegro_left/palm").id),
            mink.get_subtree_geom_ids(model, model.body("obstacle").id),
        ),
    ]

    limits = [
        mink.ConfigurationLimit(model=model),
        mink.CollisionAvoidanceLimit(
            model=model,
            geom_pairs=collision_pairs,
            # minimum_distance_from_collisions=0.1,
            # collision_detection_distance=0.2,
        ),
    ]

    # IK settings.
    solver = "daqp"
    model = configuration.model
    data = configuration.data
    hand_jnt_ids = mj_model_joints_qpos_ids(model, HAND_FULL_JOINTS_NAMES)
    hand_qpos_ids = model.jnt_qposadr[hand_jnt_ids]

    with mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)
        if USE_GOAL_HAND:
            goal_hand_base = model.body(GOAL_HAND_BASE_NAME)
            goal_hand_jnt_ids = mj_model_joints_qpos_ids(model, GOAL_HAND_FULL_JOINTS_NAMES)
            goal_hand_qpos_ids = model.jnt_qposadr[goal_hand_jnt_ids]
            posture_task.target_q[hand_qpos_ids] = data.qpos[goal_hand_qpos_ids]

        # Initialize the mocap target at the end-effector site.
        mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")
        if not USE_GOAL_HAND:
            for finger in fingers:
                mink.move_mocap_to_frame(
                    model, data, f"{finger}_target", f"allegro_left/{finger}", "site"
                )

        T_palm_prev = configuration.get_transform_frame_to_world(
            "allegro_left/palm", "body"
        )

        hand_base_mocap = data.body("target")
        rate = RateLimiter(frequency=100.0, warn=False)
        while viewer.is_running():
            # Update kuka end-effector task.
            T_wt = mink.SE3.from_mocap_name(model, data, "target")
            end_effector_task.set_target(T_wt)

            # Update finger tasks
            if USE_GOAL_HAND:
                posture_task.target_q[hand_qpos_ids] = data.qpos[goal_hand_qpos_ids]
            else:
                for finger, task in zip(fingers, finger_tasks):
                    T_pm = configuration.get_transform(
                        f"{finger}_target", "body", "allegro_left/palm", "body"
                    )
                    task.set_target(T_pm)

            T_palm = configuration.get_transform_frame_to_world(
                "allegro_left/palm", "body"
            )
            T = T_palm @ T_palm_prev.inverse()
            T_palm_prev = T_palm.copy()
            if not USE_GOAL_HAND:
                for finger in fingers:
                    mocap_id = model.body(f"{finger}_target").mocapid[0]
                    T_w_mocap = mink.SE3.from_mocap_id(data, mocap_id)
                    T_w_mocap_new = T @ T_w_mocap
                    data.mocap_pos[mocap_id] = T_w_mocap_new.translation()
                    data.mocap_quat[mocap_id] = T_w_mocap_new.rotation().wxyz

            # Compute velocity and integrate into the next configuration.
            vel = mink.solve_ik(
                configuration, tasks, rate.dt, solver, damping=1e-3, limits=limits
            )
            q = configuration.integrate(vel, rate.dt)
            configuration.update(q, kinematics_only=True)

            # Move goal hand to hand-base-mocap
            if USE_GOAL_HAND:
                mj.mju_mulPose(goal_hand_base.pos, goal_hand_base.quat, hand_base_mocap.xpos, hand_base_mocap.xquat,
                               GOAL_HAND_BASE_POS_OFFSET, GOAL_HAND_BASE_QUAT_OFFSET)

            mujoco.mj_camlight(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
