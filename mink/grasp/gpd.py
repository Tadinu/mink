from __future__ import annotations

import os
import time
import copy

from numpy import ndarray
from typing_extensions import Optional
import mediapy as media

from multiprocessing import cpu_count
# Set the number of threads to the number of cpu's that the multiprocessing module reports
CPU_NTHREAD = cpu_count()

import mujoco as mj
import mujoco.viewer as mj_viewer
from mujoco import rollout as mj_rollout, mjtObj
import numpy as np
from loop_rate_limiters import RateLimiter

from mink.grasp.utils import (in_notebook, obstacle_name, random_rgba, random_spawn_pose,
                              PclPoint, load_pointcloud, show_pointcloud, recenter_pointcloud, generate_pointcloud,
                              GraspType, GraspPose, LocalGrasp, load_grasps, write_out_grasps,
                              mj_reset_to_home, mj_get_states, mj_save_model_spec, mj_render_many,
                              mj_body_tree_body_names, mj_add_mocap_body, mj_move_mocap,
                              mj_set_body_tree_collision_enabled, mj_check_body_tree_overlapping)

# More legible printing from numpy
np.set_printoptions(precision=3, suppress=True, linewidth=100)

BLOCKDROP_MODE = False
SINGULAR_MODE = not BLOCKDROP_MODE
SINGULAR_MODE_OFFSET = np.array([0.1, 0.2, 0.5])
CLUTTERED_SCENE_PHYSICS_ENABLED = True
CLUTTERED_SCENE_OBSTACLES_NUM = 100

#MJ_GRASP_DIR="/home/tad/1_MUJOCO/MJ_GRASP"
MJ_GRASP_DIR="/media/ducthan/376b23a1-5a02-4960-b3ca-24b2fcef8f891/MUJOCO/MJ_GRASP"
GRASP_LOCOMO_DIR=f"{MJ_GRASP_DIR}/grasplocomo"
OBJECT_TYPE = "hammer"
OBJECT_NAME = f"GraspObject_{OBJECT_TYPE}"
OBJECT_MOCAP_NAME = f"{OBJECT_NAME}_mocap"
OBJECT_POINTCLOUD_FILEPATH_TXT = f"{GRASP_LOCOMO_DIR}/Clouds/{OBJECT_TYPE}_cloud.txt"
last_object_pose: np.ndarray = np.empty(7)
last_object_displacement_check_time = time.time()
OBJECT_DISPLACEMENT_CHECK_TIME_INVERVAL = 5 # sec

## DATA PREPARATION --
##
# 0- OBJECT MESH TO POINTCLOUD
O3D_GENERATE_POINTCLOUD = False
if O3D_GENERATE_POINTCLOUD:
  generate_pointcloud(f"{MJ_GRASP_DIR}/models/{OBJECT_TYPE}.obj",
                      out_txt_path=OBJECT_POINTCLOUD_FILEPATH_TXT)

# 1- Load pointcloud
VOXEL_SIZE = 0.003
OBJECT_POINTCLOUD = load_pointcloud(OBJECT_POINTCLOUD_FILEPATH_TXT)

# 1.1- Process pointcloud
O3D_RECENTER_TO_ORIGN_POINTCLOUD = False
if O3D_RECENTER_TO_ORIGN_POINTCLOUD:
  recenter_pointcloud(load_pointcloud(OBJECT_POINTCLOUD_FILEPATH_TXT, normal_as_quat=False),
                      out_txt_path=OBJECT_POINTCLOUD_FILEPATH_TXT)

# 1.2- Visualize pointcloud
O3D_VISUALIZE_POINTCLOUD = False
if O3D_VISUALIZE_POINTCLOUD:
  show_pointcloud(load_pointcloud(OBJECT_POINTCLOUD_FILEPATH_TXT, normal_as_quat=False))

# 2- Load grasps
cur_grasp_idx = 0
last_grasp_show_time = time.time()
GRASP_SHOW_TIME_INVERVAL = 1 # sec
GRASPS_ORIGINAL_FILEPATH = f"{GRASP_LOCOMO_DIR}/{OBJECT_TYPE}_grasp_results.txt"
GRASPS_RECALCULATED_FILEPATH = f"{os.path.splitext(GRASPS_ORIGINAL_FILEPATH)[0]}_recalculated.txt"
GRASPS_FILE_PATH = GRASPS_RECALCULATED_FILEPATH

def is_using_original_grasps():
  return GRASPS_FILE_PATH is GRASPS_ORIGINAL_FILEPATH

LOCAL_GRASPS: list[LocalGrasp] = load_grasps(GRASPS_FILE_PATH, post_processing=is_using_original_grasps())
GLOBAL_FREE_GRASP_POSES: list[GraspPose] = []

## OPERATIONS --
##
MODELS_DIR = f"{MJ_GRASP_DIR}/models"
MAIN_XML_PATH = f"{MODELS_DIR}/scene.xml"
SCHUNK_PG70_XML_PATH = f"{MODELS_DIR}/schunk/schunk_pg70.xml"
SCHUNK_PG70_NAME = os.path.splitext(os.path.basename(SCHUNK_PG70_XML_PATH))[0]
GRIPPER_XML_PATH = SCHUNK_PG70_XML_PATH
GRIPPER_XML_DIRNAME = os.path.dirname(GRIPPER_XML_PATH)
GRIPPER_NAME = SCHUNK_PG70_NAME
GRIPPER_BASE_NAME = f"{GRIPPER_NAME}_base"
def construct_model(save_to_xml:bool = False) -> tuple[mj.MjModel, mj.MjSpec, mj.MjsBody]:
  # https://github.com/google-deepmind/mujoco/blob/main/python/mjspec.ipynb
  # https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/specs_test.py
  main_spec = mj.MjSpec.from_file(MAIN_XML_PATH)
  if main_spec is None:
    print("Error: Main spec is failed being loaded from", MAIN_XML_PATH)
    main_spec = mj.MjSpec()
  #print(main_spec.modelname)
  #main_spec.option.timestep = 0.01
  #main_spec.option.gravity = [0,0,0]
  main_spec.option.solver = mj.mjtSolver.mjSOL_CG
  main_spec.option.enableflags |= mj.mjtEnableBit.mjENBL_MULTICCD
  main_spec.option.ccd_tolerance = 1e-6
  main_spec.option.ccd_iterations = 50

  worldbody = main_spec.worldbody
  #BODIES_NAMES = [body.name for body in main_spec.bodies]
  # [ENV]
  main_spec.lights[0].pos[2] = 2
  WALL_SIZE = [.5, .5, .05]
  worldbody.add_geom(name="plane+x", type=mj.mjtGeom.mjGEOM_PLANE, size=WALL_SIZE, zaxis=[1, 0, 0], pos=[-0.5, 0, -0.25],
                     contype=1, conaffinity=1)
  worldbody.add_geom(name="plane-x", type=mj.mjtGeom.mjGEOM_PLANE, size=WALL_SIZE, zaxis=[-1, 0, 0], pos=[0.5, 0, -0.25],
                     contype=1, conaffinity=1)
  worldbody.add_geom(name="plane+y", type=mj.mjtGeom.mjGEOM_PLANE, size=WALL_SIZE, zaxis=[0, 1, 0], pos=[0, -0.5, -0.25],
                     contype=1, conaffinity=1)
  worldbody.add_geom(name="plane-y", type=mj.mjtGeom.mjGEOM_PLANE, size=WALL_SIZE, zaxis=[0, -1, 0], pos=[0, 0.5, -0.25],
                     contype=1, conaffinity=1)

  # 1- [GRIPPER]
  for gripper_body in main_spec.bodies:
    gripper_body.gravcomp = 1
  gripper_base_spec = main_spec.bodies[1] # Idx 0 is worldbody
  gripper_base_spec.name = GRIPPER_BASE_NAME
  gripper_base_spec.pos[2] = 0.5
  # A free joint is required to move gripper freely around the scene
  # A mocap can be added if it needs to be control dynamically, instead of just teleportation
  gripper_base_spec.add_joint(name=f"{gripper_base_spec.name}_free_joint", type=mj.mjtJoint.mjJNT_FREE, align=True)
  # Disable gripper collision to avoid physical disturbance to scene upon overlapping check
  mj_set_body_tree_collision_enabled(gripper_base_spec, False)

  if BLOCKDROP_MODE:
    # 1.1- [GRIPPER POS/QUAT SENSORS]
    main_spec.add_sensor(name="gripper_pos", needstage=mj.mjtStage.mjSTAGE_POS,
                         type=mj.mjtSensor.mjSENS_FRAMEPOS,
                         objtype=mj.mjtObj.mjOBJ_BODY, objname=GRIPPER_BASE_NAME)
    main_spec.add_sensor(name="gripper_quat", needstage=mj.mjtStage.mjSTAGE_POS,
                         type=mj.mjtSensor.mjSENS_FRAMEQUAT,
                         objtype=mj.mjtObj.mjOBJ_BODY, objname=GRIPPER_BASE_NAME)

    # 1.2- [GRIPPER COLLISION SENSORS]
    # https://mujoco.readthedocs.io/en/latest/XMLreference.html#collision-sensors
    # Maximum distance up-to-which the collision distance can be reported
    # -> The larger it is, the larger space of collision-detecting realm is.
    # -> Must be > 0 to detect collision-free or non-penetrating state.
    DIST_MAX = 0.01
    for gripper_child_body_name in mj_body_tree_body_names(gripper_base_spec):
      main_spec.add_sensor(needstage=mj.mjtStage.mjSTAGE_POS,
                           type=mj.mjtSensor.mjSENS_GEOMDIST,
                           datatype=mj.mjtDataType.mjDATATYPE_REAL,
                           objtype=mj.mjtObj.mjOBJ_BODY, objname=gripper_child_body_name,
                           reftype=mj.mjtObj.mjOBJ_BODY, refname=OBJECT_NAME,
                           cutoff=DIST_MAX)
      for i in range(CLUTTERED_SCENE_OBSTACLES_NUM):
        main_spec.add_sensor(needstage=mj.mjtStage.mjSTAGE_POS,
                             type=mj.mjtSensor.mjSENS_GEOMDIST,
                             datatype=mj.mjtDataType.mjDATATYPE_REAL,
                             objtype=mj.mjtObj.mjOBJ_BODY, objname=gripper_child_body_name,
                             reftype=mj.mjtObj.mjOBJ_BODY, refname=obstacle_name(i),
                             cutoff=DIST_MAX)

      for affix in ["+x", "-x", "+y", "-y"]:
        main_spec.add_sensor(needstage=mj.mjtStage.mjSTAGE_POS,
                             type=mj.mjtSensor.mjSENS_GEOMDIST,
                             datatype=mj.mjtDataType.mjDATATYPE_REAL,
                             objtype=mj.mjtObj.mjOBJ_BODY, objname=gripper_child_body_name,
                             reftype=mj.mjtObj.mjOBJ_GEOM, refname=f"plane{affix}",
                             cutoff=DIST_MAX)

  # 2- [CLUTTERED SETTING]
  # 2.1- [TARGET OBJECT] AS A SINGLE PHYSICS-ENABLED BODY MADE FROM POINT CLOUD
  # NOTE: FIRST, OBJ MUST BE SPAWNED AT THE ORIGIN FOR [LOCAL_GRASPS] TO BE POST_PROCESSED
  target_obj = worldbody.add_body(name=OBJECT_NAME, pos=[0, 0, 0] if SINGULAR_MODE else [0, 0, 1],
                                  gravcomp=SINGULAR_MODE)
  for point in OBJECT_POINTCLOUD:
    target_obj.add_geom(type=mj.mjtGeom.mjGEOM_BOX, size=[VOXEL_SIZE] * 3, rgba=random_rgba(),
                        pos=point[0], quat=point[1])
  if CLUTTERED_SCENE_PHYSICS_ENABLED:
    target_obj.add_joint(name=f"{OBJECT_NAME}_free_joint", type=mj.mjtJoint.mjJNT_FREE, align=True)
  else:
    mj_set_body_tree_collision_enabled(target_obj, False)

  # 2.2- [OBSTACLES]
  if SINGULAR_MODE:
    # NOTE: ADD OBJECT MOCAP, OTHERWISE WITH FREE JOINT, IT JUST MOVES ENDLESSLY UPON BEING DRAGGED BY MOUSE WRENCH
    mj_add_mocap_body(main_spec, target_obj, OBJECT_MOCAP_NAME,
                      mocap_geom_type=mj.mjtGeom.mjGEOM_BOX,
                      mocap_size=np.array([0.03] * 3))
  else:
    obst_geom_types = [mj.mjtGeom.mjGEOM_BOX, mj.mjtGeom.mjGEOM_SPHERE, mj.mjtGeom.mjGEOM_CAPSULE, mj.mjtGeom.mjGEOM_CYLINDER]
    for i in range(CLUTTERED_SCENE_OBSTACLES_NUM):
      obj_i_pose = random_spawn_pose()
      obst_i = worldbody.add_body(name=obstacle_name(i), pos=obj_i_pose[0], quat=obj_i_pose[1])
      obst_i.add_geom(type=obst_geom_types[np.random.randint(low=0, high=len(obst_geom_types)-1)],
                      size=[0.05, 0.05, 0.05], rgba=random_rgba(alpha=0.3),
                      contype=1, conaffinity=1)
      if CLUTTERED_SCENE_PHYSICS_ENABLED:
        obst_i.add_joint(name=f"{obst_i.name}_free_joint", type=mj.mjtJoint.mjJNT_FREE, align=True)
      else:
        mj_set_body_tree_collision_enabled(obst_i, False)

  # COMPILE MODEL
  main_model = main_spec.compile()
  if save_to_xml:
    mj_save_model_spec(main_spec, f"{GRIPPER_XML_DIRNAME}/{main_spec.modelname}_gpd.xml")
  return main_model, main_spec, gripper_base_spec

def move_gripper(data: mj.MjData, gripper_base_body_spec: mj.MjsBody, pose: GraspPose):
  #print(euler_from_quaternion(quat))
  data.joint(gripper_base_body_spec.joints[0].name).qpos = np.concatenate([pose[0], pose[1]])

def get_global_grasp_pose(data: mj.MjData, grasp_idx: int, grasp_type: GraspType = GraspType.GRASP) -> GraspPose:
  obj = data.body(OBJECT_NAME)
  obj_pose = [obj.xpos, obj.xquat]
  local_grasp = LOCAL_GRASPS[grasp_idx].get_pose(grasp_type)
  assert local_grasp, f"grasp_idx: {grasp_idx} vs total no: {len(LOCAL_GRASPS)}"
  global_grasp_pos = np.empty(3)
  global_grasp_quat = np.empty(4)
  # NOTE: Original [local_grasp] is in Object's frame, so object is transformed in World frame first then comes the gripper
  mj.mju_mulPose(global_grasp_pos, global_grasp_quat,
                 obj_pose[0], obj_pose[1],
                 local_grasp[0], local_grasp[1])
  return [global_grasp_pos, global_grasp_quat]

def show_next_grasp(data: mj.MjData, gripper_base_body_spec: mj.MjsBody):
  # Manage show time interval
  global last_grasp_show_time
  cur_time = time.time()
  if (cur_time - last_grasp_show_time) < GRASP_SHOW_TIME_INVERVAL:
    return
  last_grasp_show_time = cur_time

  # Show gripper at the next grasp
  global cur_grasp_idx
  if SINGULAR_MODE:
    if cur_grasp_idx == len(LOCAL_GRASPS):
      cur_grasp_idx = 0
    next_grasp_pose = get_global_grasp_pose(data, cur_grasp_idx, GraspType.GRASP)
    move_gripper(data, gripper_base_body_spec, next_grasp_pose)
  else:
    if cur_grasp_idx == len(GLOBAL_FREE_GRASP_POSES):
      cur_grasp_idx = 0
    move_gripper(main_data, gripper_base_body_spec, GLOBAL_FREE_GRASP_POSES[cur_grasp_idx])
  cur_grasp_idx += 1

def init_viewer_option(data: mj.MjData, viewer: mj_viewer.Handle):
  pass

def modify_viewer_option(data: mj.MjData, viewer: mj_viewer.Handle):
  # Toggle contact points every two seconds.
  """
  with viewer.lock():
    viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
  """
  pass

def init_scene_visuals(data: mj.MjData, scene: mj.MjvScene):
  pass

def modify_scene_visuals(data: mj.MjData, scene: mj.MjvScene):
  pass

def recalculate_local_grasps(model: mj.MjModel, data: mj.MjData, gripper_base_body_spec: mj.MjsBody, out_txt_path: str):
  """
    RECALCULATE TRUE LOCAL GRASPS, ORIGINALLY RELATIVE TO [TARGET_OBJ]'s POINTCLOUD (from grasploco),
    -> TO BECOME RELATIVE TO [TARGET_OBJ] ITSELF
  """
  global LOCAL_GRASPS

  # CALCULATE TRUE LOCAL GRASPS
  # NOTE: Original [local_grasp] is in Object's frame, so object is transformed in World frame first then comes the gripper
  obj_body = data.body(OBJECT_NAME)
  gripper_body = data.body(gripper_base_body_spec.name)
  grasp_type = GraspType.GRASP
  for local_grasp in LOCAL_GRASPS:
    grasp_pose = local_grasp.get_pose(grasp_type)
    move_gripper(data, gripper_base_body_spec, grasp_pose)
    mj.mj_kinematics(model, data)
    # NOTE: [local_grasp] is of [LocalGrasp] type, thus mutable -> [LOCAL_GRASPS] is "inline modified" here also
    local_grasp.recalculate(obj_body, gripper_body, grasp_type)
  write_out_grasps(LOCAL_GRASPS, out_txt_path)

def run(model: mj.MjModel, data: mj.MjData,
        gripper_base_body_spec: mj.MjsBody,
        freq: float = 100.0,
        recalculate_grasps: bool = False, kinematics_only: bool = False):
  # RECALCULATE TRUE LOCAL GRASPS TO BE RELATIVE TO [TARGET_OBJ] ITSELF
  if recalculate_grasps:
    recalculate_local_grasps(model, data, gripper_base_body_spec,
                             out_txt_path=GRASPS_RECALCULATED_FILEPATH)

  # Move [target_obj] to an easy-to-view pos
  if SINGULAR_MODE:
    mj_move_mocap(model, data, OBJECT_MOCAP_NAME, pos=SINGULAR_MODE_OFFSET)

  # https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer
  with mj_viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
    mj.mjv_defaultFreeCamera(model, viewer.cam)
    mj_reset_to_home(model, data)

    # Init
    init_viewer_option(data, viewer)
    init_scene_visuals(data, viewer.user_scn)

    # Viewer loop
    rate = RateLimiter(frequency=freq, warn=False)
    while viewer.is_running():
      mj.mj_camlight(model, data)

      # Step [model, data]
      if kinematics_only:
        mj.mj_kinematics(model, data)
        mj.mj_comPos(model, data)
      else:
        mj.mj_step(model, data)

      # Upon object movement => Detect collision-free grasp poses
      if check_object_displacement(data):
        # Rollout on multi-LOCAL_GRASPS with collision check
        # NOTE: Technically, only need one step for collision check between gripper and env, but take 10 for rendering frames
        global GLOBAL_FREE_GRASP_POSES
        GLOBAL_FREE_GRASP_POSES = [*grasps_full_rollout(main_model, main_data)]

      # Show gripper at the next grasp
      show_next_grasp(data, gripper_base_body_spec)

      # Custom modify scene (Eg: drawing any debug graphics)
      modify_viewer_option(data, viewer)
      modify_scene_visuals(data, viewer.user_scn)

      # Visualize at fixed FPS.
      viewer.sync()
      rate.sleep()

def check_object_displacement(data: mj.MjData) -> bool:
  if SINGULAR_MODE:
    return False

  # Manage show time interval
  global last_object_displacement_check_time
  cur_time = time.time()
  if (cur_time - last_object_displacement_check_time) < OBJECT_DISPLACEMENT_CHECK_TIME_INVERVAL:
    return False
  last_object_displacement_check_time = cur_time

  # Update [last_object_pose]
  global last_object_pose
  object_pose = np.concatenate([np.array(data.body(OBJECT_NAME).xpos), np.array(data.body(OBJECT_NAME).xquat)])
  if not np.allclose(object_pose, last_object_pose, atol=0.1):
    print(f"!!{OBJECT_NAME} DISPLACEMENT DETECTED: {object_pose - last_object_pose} -> RECALCULATE GLOBAL NON-COLLIDING GRASPS")
    last_object_pose = object_pose
    return True
  return False

def rollout(top_model: mj.MjModel, top_data: mj.MjData, nstep: int,
            nsample: int = 1,
            initial_states: Optional[np.ndarray] = None,
            use_rollout_class: bool = False,
            reuse_thread_pools: bool = True,
            skip_checks: bool = True) -> tuple[list[mj.MjModel], list[mj.MjData], np.ndarray, np.ndarray]:
  if initial_states is None:
    initial_states = mj_get_states(top_model, top_data, nsample)

  # Run the rollout
  # https://github.com/google-deepmind/mujoco/blob/main/python/rollout.ipynb
  # [state]: nsample x nstep x nstate
  state = np.zeros((nsample, nstep, mj.mj_stateSize(top_model, mj.mjtState.mjSTATE_FULLPHYSICS))) if skip_checks else None
  # [sensordata]: nsample x nstep x nsensordata
  sensordata = np.zeros((nsample, nstep, top_model.nsensordata)) if skip_checks else None
  use_multi_models = nsample > 1
  # NOTE: model is compiled data, immutable (in MuJoCo sense), so no need for deep copy
  models = [top_model] * nsample
  if use_multi_models:
    datas = [copy.copy(top_data) for _ in range(nsample)]
  else:
    # Single model -> rollout on [top_data] directly
    datas = [top_data]

  # Start rollout
  start_rollout = time.time()
  print("- Start rollout... - Multimodels:", use_multi_models, "- Use Rollout class:", use_rollout_class,
        "- Skip checks:", skip_checks)
  if use_rollout_class:
    with mj_rollout.Rollout(nthread=CPU_NTHREAD) as rollout_instance:
      state, sensordata = rollout_instance.rollout(models, datas, initial_states, nstep=nstep, skip_checks=skip_checks)
  else:
    if skip_checks:
      mj_rollout.rollout(models, datas, initial_states,
                         nstep=nstep,
                         state=state, sensordata=sensordata,
                         skip_checks=True,
                         persistent_pool=reuse_thread_pools)
    else:
      state, sensordata = mj_rollout.rollout(models, datas, initial_states,
                                             nstep=nstep,
                                             state=state, sensordata=sensordata,
                                             persistent_pool=reuse_thread_pools)

  # End rollout
  if reuse_thread_pools:
    mj_rollout.shutdown_persistent_pool()

  end_rollout = time.time()
  print(f'- Rollout time {end_rollout-start_rollout:.1f} seconds')
  return models, datas, state, sensordata

def render(models: list[mj.MjModel], data: mj.MjData, state, sensordata, output_video: bool):
  # Render video
  start_render = time.time()
  framerate = 60
  cam = mj.MjvCamera()
  mj.mjv_defaultCamera(cam)
  cam.distance = 1
  cam.azimuth = 135
  cam.elevation = 2
  cam.lookat = [.2, -.2, 0.5]

  print("Start rendering... - Multimodels:", len(models) > 1)
  models[0].vis.global_.fovy = 60
  frames = mj_render_many(models, data, state, framerate, camera=cam)

  if output_video:
    video_filename = f"{OBJECT_NAME}.mp4"
    print("Writing video to:", video_filename)
    using_mediapy = True
    if using_mediapy:
      media.write_video(video_filename, frames, fps=framerate)
    else:
      # https://www.geeksforgeeks.org/saving-operated-video-from-a-webcam-using-opencv/
      import cv2
      # Define the codec and create VideoWriter object
      fourcc = cv2.VideoWriter_fourcc(*'XVID') if video_filename.endswith("avi") else cv2.VideoWriter_fourcc(*'mp4v')
      output = cv2.VideoWriter(video_filename, fourcc, framerate, (640, 480))

      i = 0
      show_frames = False
      while i < len(frames):
        frame_i = frames[i]
        # Write the frame to the output file
        # Convert frame to HSV color space
        hsv = cv2.cvtColor(frame_i, cv2.COLOR_RGB2HSV)
        output.write(hsv)

        if show_frames:
          # Show both original frame + operated video stream
          cv2.imshow(f'Frame_{i}', np.hstack((frame_i, hsv)))

          # Wait for 'a' key to stop the program
          if cv2.waitKey(1) & 0xFF == ord('a'):
            cv2.destroyWindow(f'Frame_{i}')
            i += 1
      output.release()
      cv2.destroyAllWindows()
  else:
    media.show_video(frames, fps=framerate)
  end_render = time.time()
  print(f'Rendering time {end_render-start_render:.1f} seconds')

def grasps_batch_rollout(model: mj.MjModel, data: mj.MjData,
                         grasp_type: GraspType,
                         batch_id_range: tuple[int, int],
                         nstep: int,
                         use_rollout_class: bool = False,
                         reuse_thread_pools: bool = True):
  nsample = batch_id_range[1] - batch_id_range[0]
  assert 0 < nsample < len(LOCAL_GRASPS)

  # Set the initial states, setting gripper poses
  mj_reset_to_home(model, data)
  initial_states = mj_get_states(model, data, nsample)
  for i in range(nsample):
    gripper_pose = get_global_grasp_pose(data, grasp_idx=batch_id_range[0] + i, grasp_type=grasp_type)

    # Note: For [mjSTATE_FULLPHYSICS] => first state is time, so qpos starting from index 1
    # Also, to make it easy, make sure gripper is the first body with free joint under [worldbody]
    # -> Refer to [construct_model()]
    initial_states[i, 1:8] = np.concatenate([gripper_pose[0], gripper_pose[1]])

  # Rollout
  print(f"[{grasp_type.name}] rollout - grasp indexes: [{batch_id_range[0]}, {batch_id_range[1]-1}]")
  return rollout(model, data, nstep, nsample,
                 initial_states=initial_states,
                 use_rollout_class=use_rollout_class,
                 reuse_thread_pools=reuse_thread_pools)

def grasps_full_rollout(model: mj.MjModel, data: mj.MjData) -> Optional[tuple[GraspPose, GraspPose, GraspPose]]:
  def grasp_type_rollout(grasp_type: GraspType, visualize_rollout_results = False) -> Optional[GraspPose]:
    BATCH_STEPS_NUM = 10 if visualize_rollout_results else 1
    BATCH_INTERVAL = 1000
    for batch_idx in range(int(len(LOCAL_GRASPS) / BATCH_INTERVAL)):
      # NOTE: Before rollout, this sets the initial states on each of [gr_datas](rolled-out grasp datas),
      # effectively moving gripper to a candidate grasp pose.
      # [model, data] are used as the original set, which are copied to rolled-out ones in prep for the rollout!
      batch_start = batch_idx * BATCH_INTERVAL
      batch_end = (batch_idx + 1) * BATCH_INTERVAL
      gr_models, gr_datas, gr_state, gr_sensordata = (
        grasps_batch_rollout(model, data,
                             grasp_type=grasp_type,
                             batch_id_range=(batch_start, batch_end), # NOTE: [batch_start, batch_end) -> open end
                             nstep=BATCH_STEPS_NUM))

      # Render [models] with aggregated batch [state] on [gdatas[0]]
      if visualize_rollout_results:
        render(gr_models, gr_datas[0], gr_state, gr_sensordata, output_video=not in_notebook())

      # Fetch collision-free grasp pos
      non_collision_grasp_pose = None
      for data_idx, gr_data in enumerate(gr_datas):
        # NOTE: BOTH gr_data.body(GRIPPER_BASE_NAME).xpos/xquat & gr_data.xpos/xquat are incorrect, so not usable here!
        # -> NEED TO READ FROM SENSOR DATA RETURNED BY ROLLOUT (AT THE LAST FRAME/STEP)!
        sensor_gripper_data = gr_sensordata[data_idx][BATCH_STEPS_NUM-1]
        # Refer to [construct_model()] for sensors adding order to infer about [sensor_gripper_data]
        sensor_gripper_pos = sensor_gripper_data[:3]
        sensor_gripper_quat = sensor_gripper_data[3:7]
        sensor_gripper_distance = np.min(sensor_gripper_data[7:])
        colliding = (sensor_gripper_distance < 0) or (sensor_gripper_pos[2] < 0)
        if not colliding:
          non_collision_grasp_pose = [sensor_gripper_pos, sensor_gripper_quat ]
          print(f"* Grasp idx[{batch_start + data_idx}]: "
                f"Found collision-free {grasp_type.name} pose:", non_collision_grasp_pose)
          break

      if non_collision_grasp_pose:
        return non_collision_grasp_pose
    return None

  # Global collision-free gripper pose
  free_pre = grasp_type_rollout(GraspType.PRE_GRASP)
  if free_pre:
    free = grasp_type_rollout(GraspType.GRASP)
    if free:
      free_post = grasp_type_rollout(GraspType.POST_GRASP)
      return free_pre, free, free_post
  return None

if __name__ == "__main__":
  main_model, main_spec, main_gripper_base_body_spec = construct_model()
  main_data = mj.MjData(main_model)
  mj_reset_to_home(main_model, main_data)

  if BLOCKDROP_MODE:
    # Rollout [main_model, main_data] physically (which is still faster than manual mj_step() x nstep)
    # Step num: just need to be large enough for cluttered scene to settle
    nstep = int(8 / main_model.opt.timestep) if CLUTTERED_SCENE_PHYSICS_ENABLED else 5
    print("Prepare the physics scene - Cluttered:", CLUTTERED_SCENE_PHYSICS_ENABLED)
    rollout(main_model, main_data, nstep)

  # Visualized BlockDrop or Singular mode
  run(main_model, main_data, main_gripper_base_body_spec,
      recalculate_grasps=is_using_original_grasps())
    #return [reward(model, data) for data in top_datas]