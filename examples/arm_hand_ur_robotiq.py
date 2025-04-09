import argparse
from pathlib import Path

import numpy as np
import mujoco as mj
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink
from examples.grasp.utils import mj_add_mocap_body

_HERE = Path(__file__).parent

IDENTITY_WXYZ = np.array([1., 0., 0., 0.])
ZERO_XYZ = np.zeros(3)

class Ur10eRobotiq2f85:
    ARM_BODIES_NAMES = []
    HAND_BASE_NAME = "base_mount"
    HAND_MODEL_NAME = ""

    # ur10e
    ARM_HOME_QPOS = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0]

    # robotiq2f85
    HAND_HOME_QPOS = [0, 0, 0, 0,
                      0, 0, 0 ,0]
    HOME_QPOS = ARM_HOME_QPOS + HAND_HOME_QPOS
    ARM_DOFS_NO = len(ARM_HOME_QPOS)
    HAND_DOFS_NO = len(HAND_HOME_QPOS)

    def __init__(self):
        # Received from a client of this class, which is expected to have its own custom
        # [MjSpec] setting before compiling to [MjModel]
        self.model: mj.MjModel = None

        # Created in setup given an already compiled [MjModel]
        self.data: mj.MjData = None

        # DiffIK tasks
        self.ee_task: mink.FrameTask = None
        self.posture_task: mink.PostureTask = None
        self.T_ee_prev: mink.SE3 = None
        self.N_DOFS = Ur10eRobotiq2f85.ARM_DOFS_NO + Ur10eRobotiq2f85.HAND_DOFS_NO

        # 1- EE (hand base)
        self.hand_base = None # mujoco::python::MjModelBodyViews

        # 2- Targets
        self.targets_frame = 0
        # 2.1- EE target
        self.EE_TARGET_CENTER_DEFAULT = np.array([0.5, 0, 0.5])
        self.EE_TARGET_QUAT_DEFAULT = np.array([0, 1, 0, 0])
        self.EE_TARGET_MOVEMENT_RADIUS_DEFAULT = 0.1
        self.ee_target: str = "target"

    @classmethod
    def attach_prefix(cls) -> str:
        return f"{cls.HAND_MODEL_NAME}/"

    def setup(self, model: mj.MjModel) -> None:
        self.model = model

        # 1- EE (hand base)
        self.hand_base = self.model.body(f"{self.attach_prefix()}{self.HAND_BASE_NAME}")

        # 2- Create [mj.MjData] & configurations with tasks (frame, posture, etc.) & limits (collision, velocities, etc.)
        self._configure()

        # 3- Init targets for ee, fingertips, etc.
        self._init_targets()

    def _configure(self) -> None:
        # Robot kinematics
        self.configuration = mink.Configuration(model=self.model)
        self.data = self.configuration.data
        if self.HOME_QPOS:
            self.data.qpos = self.HOME_QPOS

        # Tasks
        self._config_tasks()

        # Limits (position/velocity, joints, collision, etc.)
        self._config_limits()
        
    def _config_tasks(self) -> None:
        # EE task
        self.ee_task = mink.FrameTask(
            frame_name="attachment_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )

        # Posture task
        self.posture_task = mink.PostureTask(model=self.model, cost=5e-2)
        self.tasks = [self.ee_task, self.posture_task]

    def _config_limits(self) -> None:
        # Enable collision avoidance between the following geoms
        collision_pairs = [
            #(["wrist_3_link"], ["floor", "wall"]),
            (["wrist_3_link"], ["floor"]),
        ]
        # Max velocities
        max_velocities = {
            "shoulder_pan_joint": np.pi,
            "shoulder_lift_joint": np.pi,
            "elbow_joint": np.pi,
            "wrist_1_joint": np.pi,
            "wrist_2_joint": np.pi,
            "wrist_3_joint": np.pi,
        }

        self.limits = [
            mink.ConfigurationLimit(model=self.model),
            mink.CollisionAvoidanceLimit(model=self.model, geom_pairs=collision_pairs),
            mink.VelocityLimit(self.model, max_velocities)
        ]

    def update_tasks(self) -> None:
        self._update_task_ee()

    def _update_task_ee(self) -> None:
        # Update kuka end-effector task, as [target]'s SE3
        T_wt = mink.SE3.from_mocap_name(self.model, self.data, self.ee_target)
        self.ee_task.set_target(T_wt)

    def _init_targets(self) -> None:
        # Init targets (ee_target + finger_targets)
        mink.utils.move_mocap_to_pose(self.model, self.data, self.ee_target,
                                      frame_pos=self.hand_base.pos,
                                      frame_quat=self.EE_TARGET_QUAT_DEFAULT)
        self.T_ee_prev = self.configuration.get_transform_frame_to_world(self.hand_base.name, "body")

    def update_targets(self) -> None:
        self.targets_frame += 1
        # Robot's [ee_target]
        delta = self.targets_frame / 360 * np.pi
        target_pos = (self.EE_TARGET_CENTER_DEFAULT +
                      np.array([np.cos(delta), np.sin(delta), 0]) * self.EE_TARGET_MOVEMENT_RADIUS_DEFAULT)
        mink.utils.move_mocap_to_pose(self.model, self.data, self.ee_target,
                                      frame_pos=target_pos,
                                      frame_quat=self.EE_TARGET_QUAT_DEFAULT)

class Ur10eRobotiq2f85DiffIK:
    DT: float = 0.01

    def __init__(self, arm_scene_xml: str, hand_xml: str):
        # Model building
        self.arm_scene_xml = arm_scene_xml
        self.hand_xml = hand_xml
        self.arm_spec: mj.MjSpec = None
        self.hand_spec: mj.MjSpec = None
        self.hand_base_spec: mj.MjsBody = None

        # Entities
        # Robot
        self.ur10e_robotiq_2f85: Ur10eRobotiq2f85 = None

        # Control loop
        self.rate: RateLimiter = None

        # Solver
        self.solver_name: str = "quadprog"

    def construct_robot_system_spec(self):
        # https://github.com/google-deepmind/mujoco/blob/main/python/mjspec.ipynb
        # https://mj.readthedocs.io/en/latest/python.html#construction
        self.arm_spec = mj.MjSpec.from_file(self.arm_scene_xml)
        print("SYSTEM MODEL NAME: ", self.arm_spec.modelname)
        Ur10eRobotiq2f85.ARM_BODIES_NAMES = [body.name for body in self.arm_spec.bodies]

        self.hand_spec = mj.MjSpec.from_file(self.hand_xml)
        Ur10eRobotiq2f85.HAND_MODEL_NAME = self.hand_spec.modelname
        self.hand_base_spec = self.hand_spec.worldbody.find_child(Ur10eRobotiq2f85.HAND_BASE_NAME)
        self.hand_base_spec.quat = IDENTITY_WXYZ
        #self.hand_base_spec.pos = (0, 0, 0.095)

        # Attach [hand_spec] to [arm_spec]
        attach_site = self.arm_spec.site("attachment_site")
        attach_site.attach_body(self.hand_spec.worldbody, Ur10eRobotiq2f85.attach_prefix())

        # TODO: Remove prev "home" key from arm_spec once MuJoCo releases [rem_key] API
        #self.arm_spec.add_key(name="home", qpos=self.HOME_QPOS)

        # NOTE: ADD OBJECT MOCAP, OTHERWISE WITH FREE JOINT, IT JUST MOVES ENDLESSLY UPON BEING DRAGGED BY MOUSE WRENCH
        mj_add_mocap_body(self.arm_spec, self.hand_base_spec, "target",
                          mocap_geom_type=mj.mjtGeom.mjGEOM_BOX,
                          mocap_size=np.array([0.03] * 3))

        # NOTE: The reason why [MjModel] is not compiled right away here-in is to leave it to the caller, which might
        # have extra settings to [arm_spec]
        return self.arm_spec

    def run(self):
        # Robot System
        ur10e_robotiq_2f85_spec = self.construct_robot_system_spec()
        # save_model_spec(ur10e_robotiq_2f85_spec)
        self.ur10e_robotiq_2f85 = Ur10eRobotiq2f85()
        self.ur10e_robotiq_2f85.setup(model = ur10e_robotiq_2f85_spec.compile())
        #self.ur10e_robotiq_2f85.OBSTACLE_NAMES = [self.BALL_NAME]

        model = self.ur10e_robotiq_2f85.model
        data = self.ur10e_robotiq_2f85.data
        with mj.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
            mj.mjv_defaultFreeCamera(model, viewer.cam)

            # Initialize to the home keyframe.
            self.ur10e_robotiq_2f85.configuration.update_from_keyframe("home")
            self.ur10e_robotiq_2f85.posture_task.set_target_from_configuration(self.ur10e_robotiq_2f85.configuration)

            # Initialize the mocap target at the end-effector site.
            mink.move_mocap_to_frame(model, data,
                                     "target", "attachment_site", "site")

            rate = RateLimiter(frequency=500.0, warn=False)
            while viewer.is_running():
                # Update task target.
                T_wt = mink.SE3.from_mocap_name(model, data, "target")
                self.ur10e_robotiq_2f85.ee_task.set_target(T_wt)

                # Compute velocity and integrate into the next configuration.
                vel = mink.solve_ik(
                    self.ur10e_robotiq_2f85.configuration, self.ur10e_robotiq_2f85.tasks, rate.dt, self.solver_name, 1e-3,
                    limits=self.ur10e_robotiq_2f85.limits
                )
                self.ur10e_robotiq_2f85.configuration.integrate_inplace(vel, rate.dt)
                mj.mj_camlight(model, data)

                # Note the below are optional: they are used to visualize the output of the
                # fromto sensor which is used by the collision avoidance constraint.
                mj.mj_fwdPosition(model, data)
                mj.mj_sensorPos(model, data)

                # Visualize at fixed FPS.
                viewer.sync()
                rate.sleep()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ## MODEL
    MJ_GRASP_DIR = "/home/tad/1_MUJOCO/MJ_GRASP"
    # MJ_GRASP_DIR="/media/ducthan/376b23a1-5a02-4960-b3ca-24b2fcef8f891/MUJOCO/MJ_GRASP"
    MODELS_DIR = f"{MJ_GRASP_DIR}/Models"

    ## ARM
    # [UR10]
    UR10E_NAME = "ur10e"
    UR10E_MODEL_DIR = f"{MODELS_DIR}/universal_robots_ur10e"
    UR10E_XML_PATH = f"{UR10E_MODEL_DIR}/{UR10E_NAME}.xml"

    # [MAIN ARM]
    ARM_NAME = UR10E_NAME
    ARM_MODEL_DIR = UR10E_MODEL_DIR
    ARM_SCENE_XML_PATH = f"{ARM_MODEL_DIR}/scene.xml"
    ARM_ATTACH_PREFIX = ARM_NAME

    # [ROBOTIQ_2F85]
    ROBOTIQ_2F85_NAME = "robotiq_2f85"
    ROBOTIQ_2F85_MODEL_DIR = f"{MODELS_DIR}/{ROBOTIQ_2F85_NAME}"

    # [MAIN GRIPPER]
    GRIPPER_NAME = ROBOTIQ_2F85_NAME
    GRIPPER_MODEL_DIR = ROBOTIQ_2F85_MODEL_DIR
    GRIPPER_SCENE_XML_PATH = f"{GRIPPER_MODEL_DIR}/scene.xml"
    GRIPPER_XML_PATH = f"{GRIPPER_MODEL_DIR}/{GRIPPER_NAME}.xml"
    GRIPPER_BASE_NAME = f"{GRIPPER_NAME}_base"

    diffIk = Ur10eRobotiq2f85DiffIK(arm_scene_xml=ARM_SCENE_XML_PATH, hand_xml=GRIPPER_XML_PATH)
    diffIk.run()
