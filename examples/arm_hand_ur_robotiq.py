import argparse
from typing_extensions import Callable, Optional
from pathlib import Path

import numpy as np
import mujoco as mj
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink
from examples.arm_hand_iiwa_allegro import HOME_QPOS
from examples.grasp.utils import mj_add_mocap_body, mj_set_body_tree_collision_enabled

_HERE = Path(__file__).parent

IDENTITY_WXYZ = np.array([1., 0., 0., 0.])
ZERO_XYZ = np.zeros(3)

class Ur10eRobotiq2f85:
    # ur10e
    ARM_HOME_QPOS = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.]
    ARM_HAND_ATTACHMENT_SITE_NAME = "attachment_site"
    ARM_DOFS_NO = len(ARM_HOME_QPOS)
    ARM_BODIES_NAMES = []

    # robotiq2f85
    # To be determined upon reading from xml in system construction
    HAND_MODEL_NAME = ""
    HAND_BASE_NAME = "base_mount"
    HAND_HOME_QPOS = [0., 0., 0., 0.,
                      0., 0., 0. ,0.]
    HAND_DOFS_NO = len(HAND_HOME_QPOS)

    # ee target
    EE_TARGET_MOCAP_NAME: str = "target"

    # Home qpos
    HOME_QPOS = ARM_HOME_QPOS + HAND_HOME_QPOS

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
        # mujoco::python::MjDataBodyViews -> NOTE: This holds runtime data, NOT [MjModelBodyViews]
        self.hand_base = None

        # 2- Targets
        self.targets_frame = 0
        # 2.1- EE target
        self.EE_TARGET_CENTER_DEFAULT = np.array([0.5, 0, 0.5])
        self.EE_TARGET_QUAT_DEFAULT = np.array([0, 1, 0, 0])
        self.EE_TARGET_MOVEMENT_RADIUS_DEFAULT = 0.1

    @classmethod
    def attach_prefix(cls) -> str:
        return f"{cls.HAND_MODEL_NAME}/"

    @classmethod
    def hand_base_full_name(cls) -> str:
        return f"{cls.attach_prefix()}{cls.HAND_BASE_NAME}"

    def setup(self, model: mj.MjModel, data: mj.MjData) -> None:
        self.model = model
        self.data = data

        # 1- Create configs with tasks (frame, posture, etc.) & limits (collision, velocities, etc.)
        self._configure()

        # 2- EE (hand base)
        self.hand_base = self.data.body(self.hand_base_full_name())

        # 3- Init targets for ee, fingertips, etc.
        self._init_targets()

    def _configure(self) -> None:
        # Robot kinematics
        self.configuration = mink.Configuration(model=self.model, data=self.data)
        if self.HOME_QPOS:
            self.data.qpos[:len(self.HOME_QPOS)] = self.HOME_QPOS

        # Tasks
        self._config_tasks()

        # Limits (position/velocity, joints, collision, etc.)
        self._config_limits()
        
    def _config_tasks(self) -> None:
        # EE task
        self.ee_task = mink.FrameTask(
            frame_name=self.ARM_HAND_ATTACHMENT_SITE_NAME,
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
        T_wt = mink.SE3.from_mocap_name(self.model, self.data, self.EE_TARGET_MOCAP_NAME)
        self.ee_task.set_target(T_wt)

    def _init_targets(self) -> None:
        # Init targets (ee_target + finger_targets)
        mink.utils.move_mocap_to_pose(self.model, self.data, self.EE_TARGET_MOCAP_NAME,
                                      frame_pos=self.hand_base.xpos,
                                      frame_quat=self.hand_base.xquat)
        self.T_ee_prev = self.configuration.get_transform_frame_to_world(self.hand_base.name, "body")

    def update_targets(self) -> None:
        self.targets_frame += 1
        # Robot's [ee_target]
        delta = self.targets_frame / 360 * np.pi
        target_pos = (self.EE_TARGET_CENTER_DEFAULT +
                      np.array([np.cos(delta), np.sin(delta), 0]) * self.EE_TARGET_MOVEMENT_RADIUS_DEFAULT)
        mink.utils.move_mocap_to_pose(self.model, self.data, self.EE_TARGET_MOCAP_NAME,
                                      frame_pos=target_pos,
                                      frame_quat=self.EE_TARGET_QUAT_DEFAULT)

class Ur10eRobotiq2f85DiffIK:
    DT: float = 0.01

    def __init__(self, arm_scene_xml: str, hand_xml: str):
        # Model building
        self.arm_scene_xml: str = arm_scene_xml
        self.hand_xml: str = hand_xml
        self.arm_spec: mj.MjSpec = None
        self.hand_spec: mj.MjSpec = None
        self.hand_base_spec: mj.MjsBody = None
        self.ghost_hand_base_spec: mj.MjsBody = None

        # Entities
        # Robot
        self.robot: Ur10eRobotiq2f85 = None

        # Control loop
        self.rate: RateLimiter = None

        # Solver
        self.solver_name: str = "quadprog" # "osqp"

    def construct_robot_system_spec(self) -> mj.MjSpec:
        # https://github.com/google-deepmind/mujoco/blob/main/python/mjspec.ipynb
        # https://mj.readthedocs.io/en/latest/python.html#construction
        self.arm_spec = mj.MjSpec.from_file(self.arm_scene_xml)
        print("SYSTEM MODEL NAME: ", self.arm_spec.modelname)
        Ur10eRobotiq2f85.ARM_BODIES_NAMES = [body.name for body in self.arm_spec.bodies]
        # Disable arm's bodies collision
        mj_set_body_tree_collision_enabled(self.arm_spec.bodies[1], False)

        self.hand_spec = mj.MjSpec.from_file(self.hand_xml)
        Ur10eRobotiq2f85.HAND_MODEL_NAME = self.hand_spec.modelname
        self.hand_base_spec = self.hand_spec.worldbody.find_child(Ur10eRobotiq2f85.HAND_BASE_NAME)
        self.hand_base_spec.quat = IDENTITY_WXYZ
        #self.hand_base_spec.pos = (0, 0, 0.01)

        # Attach [hand_spec] to [arm_spec]
        attach_site = self.arm_spec.site(Ur10eRobotiq2f85.ARM_HAND_ATTACHMENT_SITE_NAME)
        attach_site.attach_body(self.hand_spec.worldbody, Ur10eRobotiq2f85.attach_prefix())

        # TODO: Remove prev "home" key from arm_spec once MuJoCo releases [rem_key] API
        #self.arm_spec.add_key(name="home", qpos=self.HOME_QPOS)

        # Refetch [self.hand_base_spec], which seems to be just the same after attachment, in [self.arm_spec]
        self.hand_base_spec = self.arm_spec.body(Ur10eRobotiq2f85.hand_base_full_name())

        # EE Target mocap body (under [arm_spec]'s worldbody)
        mj_add_mocap_body(self.arm_spec, self.hand_base_spec, Ur10eRobotiq2f85.EE_TARGET_MOCAP_NAME,
                          mocap_geom_type=mj.mjtGeom.mjGEOM_BOX,
                          mocap_size=np.array([0.03] * 3))

        # Enabled [gravcomp]
        for arm_body in self.arm_spec.bodies:
            arm_body.gravcomp = 1

        # NOTE: The reason why [MjModel] is not compiled right away here-in is to leave it to the caller, which might
        # have extra settings to [arm_spec]
        return self.arm_spec

    def build_robot(self) -> None:
        # 1- Construct Robot System
        system_spec = self.construct_robot_system_spec()
        # mj_save_model(system_spec)

        # 2- Compile robot system model & Make data, also setting up robot configuration
        self.robot = Ur10eRobotiq2f85()
        self.robot.setup(model=system_spec.compile()) # NOTE: MjData is created here-in in robot's configuration
        # self.robot.OBSTACLE_NAMES = [self.BALL_NAME]

    def run(self, callback: Optional[Callable] = None, kinematics_only: bool = False) -> None:
        model = self.robot.model
        data = self.robot.data
        with mj.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
            mj.mjv_defaultFreeCamera(model, viewer.cam)

            # Initialize
            self.robot.configuration.update(q=np.array(Ur10eRobotiq2f85.HOME_QPOS))
            self.robot.posture_task.set_target_from_configuration(self.robot.configuration)

            # Initialize the mocap target at the end-effector site
            mink.move_mocap_to_frame(model, data,
                                     Ur10eRobotiq2f85.EE_TARGET_MOCAP_NAME,
                                     Ur10eRobotiq2f85.ARM_HAND_ATTACHMENT_SITE_NAME, "site")

            rate = RateLimiter(frequency=100.0, warn=False)
            while viewer.is_running():
                self.robot.update_targets()

                # Update task target
                T_wt = mink.SE3.from_mocap_name(model, data, Ur10eRobotiq2f85.EE_TARGET_MOCAP_NAME)
                self.robot.ee_task.set_target(T_wt)

                # Compute velocity and integrate into the next configuration.
                vel = mink.solve_ik(
                    self.robot.configuration, self.robot.tasks, rate.dt, self.solver_name, 1e-3,
                    limits=self.robot.limits
                )
                self.robot.configuration.integrate_inplace(vel, rate.dt)
                mj.mj_camlight(model, data)

                # Step [model, data]
                # Note the below are optional: they are used to visualize the output of the
                # fromto sensor which is used by the collision avoidance constraint.
                if kinematics_only:
                    mj.mj_fwdPosition(model, data)
                    mj.mj_comPos(model, data)
                    mj.mj_sensorPos(model, data)
                else:
                    print("[WARNING] DIFF-IK IS RUNNING IN PHYSICS MODE -> MAY NOT WORK ACCURATELY")
                    mj.mj_step(model, data)

                # Invoke [callback] whatever it is meant to do
                if callback:
                    callback(model, data, self)

                # Visualize at fixed FPS.
                viewer.sync()
                rate.sleep()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ## MODEL
    _CURRENT_DIR = Path(__file__).parent.as_posix()
    MODELS_DIR = f"{_CURRENT_DIR}/grasp/models"

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
    diffIk.build_robot()
    diffIk.run(kinematics_only=True)
