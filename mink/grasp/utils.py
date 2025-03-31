import os
from typing_extensions import Optional, Union
from dataclasses import dataclass
from enum import Enum
import mujoco as mj
import numpy as np
import open3d as o3d

## GENERAL --
##
def in_notebook() -> bool:
  try:
    from IPython import get_ipython
    if 'IPKernelApp' not in get_ipython().config:
      return False
  except ImportError:
    return False
  except AttributeError:
    return False
  return True

def obstacle_name(i: int) -> str:
  return f"obst_{i}"

def euler_from_quaternion(quat: np.ndarray) -> np.ndarray:
  from scipy.spatial.transform import Rotation
  # Convert to Euler angles (roll, pitch, yaw)
  return Rotation.from_quat(quat).as_euler('xyz', degrees=False)

def random_quaternion() -> np.ndarray:
  """Generate a random unit quaternion

    Uniformly distributed across the rotation space
    Ref: https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L261
  """
  r1, r2, r3 = np.random.random(3)

  q1 = np.sqrt(1.0 - r1) * (np.sin(2 * np.pi * r2))
  q2 = np.sqrt(1.0 - r1) * (np.cos(2 * np.pi * r2))
  q3 = np.sqrt(r1) * (np.sin(2 * np.pi * r3))
  q4 = np.sqrt(r1) * (np.cos(2 * np.pi * r3))

  return np.array([q1, q2, q3, q4])

def random_rgba(alpha: float = 1) -> np.ndarray:
  rgba = np.random.uniform(size=4)
  rgba[3] = alpha
  return rgba

def random_spawn_pose() -> list[np.ndarray]:
  pos = np.array([np.random.uniform(low=-0.3, high=0.3),
                  np.random.uniform(low=-0.3, high=0.3),
                  np.random.uniform(low=0., high=0.5)])
  quat = random_quaternion()
  return [pos, quat]

def draw_obj_pointcloud(scene: mj.MjvScene, point_cloud: list[list[np.ndarray]], voxel_size: float = 0.01,
                        pos_offset: Optional[np.ndarray] = None):
  if pos_offset is None:
    pos_offset = np.zeros(3)
  # VOXELS AS OBJ POINTCLOUD
  # NOTE: Geoms have been pre-allocated with kMaxGeom
  geom_idx = scene.ngeom
  mat = np.empty(9)
  for i, point in enumerate(point_cloud):
    mj.mju_quat2Mat(mat, point[1])
    mj.mjv_initGeom(
      scene.geoms[geom_idx],
      type=mj.mjtGeom.mjGEOM_BOX,
      size=[voxel_size] * 3,
      pos=point[0] + pos_offset, mat=mat,
      rgba=random_rgba(),
    )
    geom_idx += 1
  scene.ngeom += len(point_cloud)

def draw_arrow(scene: mj.MjvScene, from_, to, radius=0.03, rgba=[0.2, 0.2, 0.6, 1]):
  scene.geoms[scene.ngeom].category = mj.mjtCatBit.mjCAT_STATIC
  mj.mjv_initGeom(
    geom=scene.geoms[scene.ngeom],
    type=mj.mjtGeom.mjGEOM_ARROW,
    size=np.zeros(3),
    pos=np.zeros(3),
    mat=np.zeros(9),
    rgba=np.asarray(rgba).astype(np.float32),
  )
  mj.mjv_connector(
    geom=scene.geoms[scene.ngeom],
    type=mj.mjtGeom.mjGEOM_ARROW,
    width=radius,
    from_=from_,
    to=to,
  )
  scene.ngeom += 1

## MUJOCO --
##
def mj_reset_to_home(model: mj.MjModel, data: mj.MjData, home_name: str = "home"):
  key_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, home_name)
  if key_id >= 0:
    mj.mj_resetDataKeyframe(model, data, key_id)

def mj_geom_body(model: mj.MjModel, data: mj.MjData, geom_id: int) : # -> mj.MjDataBodyViews
  return data.body(model.geom_bodyid[geom_id])

def mj_body_tree_body_names(base_body_spec: mj.MjsBody) -> list[str]:
  names = [base_body_spec.name]
  for child_body in base_body_spec.bodies:
    names += mj_body_tree_body_names(child_body)
  return names

def mj_body_geom_ids(model: mj.MjModel, body_name: str) -> list[int]:
  #print(body_name)
  body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
  return [i for i, b_id in enumerate(model.geom_bodyid) if b_id == body_id]

def mj_body_tree_geom_ids(model: mj.MjModel, base_body_spec: mj.MjsBody) -> list[int]:
  res = mj_body_geom_ids(model, base_body_spec.name)
  for child_body_spec in base_body_spec.bodies:
    res += mj_body_tree_geom_ids(model, child_body_spec)
  return res

def mj_add_mocap_body(model_spec: mj.MjSpec, target_body: mj.MjsBody, mocap_name: str,
                      mocap_geom_type: mj.mjtGeom = mj.mjtGeom.mjGEOM_BOX,
                      mocap_size: Optional[np.ndarray] = None):
  if mocap_size is None:
      mocap_size = ([0.05, 0.05, 0.05])
  mocap = model_spec.worldbody.add_body(name=mocap_name, mocap=True,
                                       pos=target_body.pos, quat=target_body.quat)
  mocap.add_geom(type=mocap_geom_type, size=mocap_size, rgba=[0, 1, 0, 0.2],
                 contype=0, conaffinity=0)
  model_spec.add_equality(name="eq1", objtype=mj.mjtObj.mjOBJ_BODY, type=mj.mjtEq.mjEQ_WELD,
                         name1=mocap_name, name2=target_body.name)

def mj_get_mocap_id(model: mj.MjModel, mocap_body_name: str):
  if False:
    return model.body(mocap_body_name).mocapid[0]
  else:
    mocap_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, mocap_body_name)
    return model.body_mocapid[mocap_body_id] if mocap_body_id >= 0 else -1

def mj_move_mocap(model: mj.MjModel, data: mj.MjData, mocap_body_name: str,
                  pos: np.ndarray, quat: Optional[np.ndarray] = None):
  mocap_id = mj_get_mocap_id(model, mocap_body_name)
  data.mocap_pos[mocap_id][:3] = pos
  if quat:
    data.mocap_quat[mocap_id][:4] = quat

def mj_set_body_tree_collision_enabled(base_body_spec: mj.MjsBody, enabled: bool = True):
  for geom_spec in base_body_spec.geoms:
    geom_spec.contype = enabled
    geom_spec.conaffinity = enabled
  for child_body in base_body_spec.bodies:
    mj_set_body_tree_collision_enabled(child_body, enabled)

def mj_check_body_tree_overlapping(model: mj.MjModel, data: mj.MjData, base_body_spec: mj.MjsBody,
                                   threshold: float = 0.01) -> tuple[bool, float]:
  fromto = np.zeros(6)
  geom_ids = mj_body_tree_geom_ids(model, base_body_spec)
  other_geom_ids = np.setdiff1d(np.arange(model.ngeom), geom_ids).tolist()
  for geom_id in geom_ids:
    for other_geom_id in other_geom_ids:
      dist = mj.mj_geomDistance(
        model,
        data,
        geom_id,
        other_geom_id,
        threshold,
        fromto,
      )
      # [fromto] != zeros -> overlapping/collision
      if fromto.any():
        return True, dist
  # No collision: [mj_geomDistance()] always return [threshold]
  return False, threshold

def mj_print_body_contacts(model: mj.MjModel, data: mj.MjData, base_body_spec: mj.MjsBody):
  base_body = data.body(base_body_spec.name)
  geom_ids = mj_body_tree_geom_ids(model, base_body_spec)
  for i in range(data.ncon):
    # Print contacts involving with [body]'s geoms
    contact = data.contact[i]
    if contact.geom1 in geom_ids or contact.geom2 in geom_ids:
      normal = contact.frame[:3]
      dist = contact.dist
      fromto = np.empty((6,), dtype=np.float64)
      fromto[3:] = contact.pos - 0.5 * dist * normal
      fromto[:3] = contact.pos + 0.5 * dist * normal
      print(f"Body at contact #{i}", base_body.xpos, base_body.xquat)
      print(
        "dist:", contact.dist, "\n",
        "fromto:", fromto, "\n",
        "geom1:", mj_geom_body(model, data, contact.geom1).name, "\n",
        "geom2:", mj_geom_body(model, data, contact.geom2).name, "\n",
      )

def mj_save_model_spec(model_spec: mj.MjSpec, path: Optional[str] = None):
  # NOTE:
  # mj_saveLastXML() only works upon model that was loaded with MjModel.[from_xml() or from_xml_string()]
  # mj_saveModel() only writes to MJCB file
  with open(path if path else f"{os.path.splitext(os.path.basename(__file__))[0]}.xml", "w") as f:
    f.writelines(model_spec.to_xml())
    print("Model saved to xml:", path)

def mj_normal_to_quat(normal: Union[np.ndarray, list[float]]) -> np.ndarray :
  mj.mju_normalize3(np.array(normal))

  # Reference direction (world z-axis)
  z_ref = np.array([0, 0, 1])

  # Compute rotation axis (cross product)
  axis = np.cross(z_ref, normal)

  # Compute rotation angle, clipped for numerical stability
  angle = np.arccos(np.clip(np.dot(z_ref, normal), -1.0, 1.0))

  # Special case: (0, 0, -1) normal
  if np.allclose(normal, -z_ref):
    return np.array([0, 1, 0, 0])  # 180-degree rotation around x-axis

  # Normalize rotation axis
  if np.linalg.norm(axis) > 1e-6:  # Avoid divide by zero
    mj.mju_normalize3(axis)
  else:
    axis = np.array([1, 0, 0])  # Default axis if normal is aligned with z_ref

  # Convert [axis, angle] -> quat
  quat = np.zeros(4)
  mj.mju_axisAngle2Quat(quat, axis, angle)
  return quat

def mj_get_states(model: mj.MjModel, data: mj.MjData, nsample: int = 1) -> np.ndarray:
  # https://mujoco.readthedocs.io/en/latest/APIreference/APItypes.html#mjtstate
  # Can only be [mjSTATE_FULLPHYSICS] as checked here:
  # https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/rollout.py#L192
  state_type = mj.mjtState.mjSTATE_FULLPHYSICS
  state = np.zeros((mj.mj_stateSize(model, state_type),))
  mj.mj_getState(model, data, state, state_type)
  return np.tile(state, (nsample, 1))

# https://github.com/google-deepmind/mujoco/blob/main/python/rollout.ipynb
def mj_render_many(model: Union[mj.MjModel, list[mj.MjModel]],
                   data: mj.MjData,
                   state: np.ndarray, framerate: float, camera: Union[int, str, mj.MjvCamera] = -1,
                   shape: np.ndarray = (480, 640),
                   transparent: bool=False, light_pos: Optional[np.ndarray] = None):
  nsample = state.shape[0]

  if not isinstance(model, mj.MjModel):
    model = list(model)

  if isinstance(model, list) and len(model) == 1:
    model = model * nsample
  elif isinstance(model, list):
    assert len(model) == nsample
  else:
    model = [model] * nsample

  # Visual options
  vopt = mj.MjvOption()
  vopt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = transparent
  pert = mj.MjvPerturb()  # Empty MjvPerturb object
  catmask = mj.mjtCatBit.mjCAT_DYNAMIC

  # Simulate and render.
  frames = []
  with mj.Renderer(model[0], *shape) as renderer:
    for i in range(state.shape[1]):
      if len(frames) < i * model[0].opt.timestep * framerate:
        for j in range(state.shape[0]):
          mj.mj_setState(model[j], data, state[j, i, :],
                         mj.mjtState.mjSTATE_FULLPHYSICS)
          mj.mj_forward(model[j], data)

          # Use first model to make the scene, add subsequent models
          if j == 0:
            renderer.update_scene(data, camera, scene_option=vopt)
          else:
            mj.mjv_addGeoms(model[j], data, vopt, pert, catmask, renderer.scene)

        # Add light, if requested
        if light_pos is not None:
          light = renderer.scene.lights[renderer.scene.nlight]
          light.ambient = [0, 0, 0]
          light.attenuation = [1, 0, 0]
          light.castshadow = 1
          light.cutoff = 45
          light.diffuse = [0.8, 0.8, 0.8]
          light.dir = [0, 0, -1]
          light.directional = 0
          light.exponent = 10
          light.headlight = 0
          light.specular = [0.3, 0.3, 0.3]
          light.pos = light_pos
          renderer.scene.nlight += 1

        # Render and add the frame.
        pixels = renderer.render()
        frames.append(pixels)
  return frames

## POINTCLOUD --
##
PclPoint = list[np.ndarray]
def load_pointcloud(filepath: str, normal_as_quat: bool = True) -> list[PclPoint]:
  points = []
  with open(filepath, 'r') as file:
    i = 0
    for line in file:
      # First line: header
      if i == 0:
        i += 1
        continue
      data = [float(num) for num in line.strip().split()]
      pos = data[:3]
      normal = data[3:6]
      points.append([pos, mj_normal_to_quat(normal) if normal_as_quat else
                          normal])
  return points

def generate_pointcloud(mesh_path: str, out_txt_path: str, scale: float = 1.0):
  mesh = o3d.io.read_triangle_mesh(mesh_path)
  print(f"Mesh center:{mesh.get_center()}")
  mesh.scale(scale, (0, 0, 0))
  o3d.utility.random.seed(0)
  pcd = mesh.sample_points_poisson_disk(1000)
  out_pcd_path = f"{os.path.splitext(out_txt_path)[0]}.pcd"
  assert o3d.io.write_point_cloud(out_pcd_path, pcd, write_ascii=True, print_progress=True)
  os.rename(out_pcd_path, out_txt_path)

def recenter_pointcloud(pointcloud: list[PclPoint], out_txt_path: Optional[str] = None):
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector([point[0] for point in pointcloud])
  pcd.normals = o3d.utility.Vector3dVector([point[1] for point in pointcloud])
  pcd.translate(translation=[0, 0, 0], relative=False)
  print(f"Center of pcd: {pcd.get_center()}")
  if out_txt_path:
    out_pcd_path = f"{os.path.splitext(out_txt_path)[0]}.pcd"
    assert o3d.io.write_point_cloud(out_pcd_path, pcd, write_ascii=True, print_progress=True)
    os.rename(out_pcd_path, out_txt_path)

def show_pointcloud(pointcloud: list[PclPoint]):
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector([point[0] for point in pointcloud])
  pcd.normals = o3d.utility.Vector3dVector([point[1] for point in pointcloud])
  pcd.colors = o3d.utility.Vector3dVector(len(pointcloud) * [np.random.rand(3)])
  o3d.visualization.draw(pcd)

## GRASPS --
##
GraspPose = list[np.ndarray]
class GraspType(Enum):
  PRE_GRASP = 1
  GRASP = 2
  POST_GRASP = 3

@dataclass(frozen=False)
class LocalGrasp:
  # NOTE: These poses are relative to point cloud originally and to a target object upon recalculation
  pre_pose: GraspPose # Preparation pose before going to [pose]
  pose: GraspPose # Grasp pose whereby gripper fingers engage with a target object
  post_pose: GraspPose # Post pose
  probability: float
  gripper_closed_width: float
  gripper_open_width: float

  def get_pose(self, grasp_type: GraspType) -> Optional[GraspPose]:
    if grasp_type == GraspType.PRE_GRASP:
      return self.pre_pose
    elif grasp_type == GraspType.GRASP:
      return self.pose
    elif grasp_type == GraspType.POST_GRASP:
      return self.post_pose
    return None

  def recalculate(self, target_obj_body, gripper_body, grasp_type: GraspType):
    """
      RECALCULATE TRUE LOCAL GRASPS, ORIGINALLY RELATIVE TO [target_obj_body]'s POINTCLOUD (from grasplocomo),
      -> TO BECOME RELATIVE TO [target_obj_body] ITSELF
    """
    def recalculate_grasp_pose(grasp_pose, target_obj_body, gripper_body):
      # NOTE: Original grasp post is in [target_obj_body]'s frame, so object is transformed in World frame first then comes the gripper
      obj_negpos = np.empty(3)
      obj_negquat = np.empty(4)
      mj.mju_negPose(obj_negpos, obj_negquat, target_obj_body.xpos, target_obj_body.xquat)
      mj.mju_mulPose(grasp_pose[0], grasp_pose[1], obj_negpos, obj_negquat, gripper_body.xpos, gripper_body.xquat)
    if grasp_type == GraspType.PRE_GRASP:
      recalculate_grasp_pose(self.pre_pose, target_obj_body, gripper_body)
    elif grasp_type == GraspType.GRASP:
      recalculate_grasp_pose(self.pose, target_obj_body, gripper_body)
    elif grasp_type == GraspType.POST_GRASP:
      recalculate_grasp_pose(self.post_pose, target_obj_body, gripper_body)

def load_grasps(filepath: str, post_processing: bool, token: str = "|") -> list[LocalGrasp]:
  output_grasps: list[LocalGrasp] = []
  with open(filepath, 'r') as file:
    i = 0
    for line in file:
      # First line: header
      if i == 0:
        i += 1
        continue
      segments = line.strip().split(token)
      pre_grasp = segments[0]
      grasp = segments[1]
      post_grasp = segments[2]
      probability = float(segments[3])
      gripper_closed_width = float(segments[4])
      gripper_open_width = float(segments[5])

      def read_pose(grasp_segment):
        # Read grasp pose
        data = [float(num) for num in grasp_segment.split()]
        pos = np.array(data[:3])
        quat = np.array(data[3:])
        new_pos = pos.copy()
        new_quat = quat.copy()
        if post_processing:
          # GraspLoCoMo gripper_base model has:
          # - Fingers pointing toward -X, thus need to rotate its grasp around Y -90 deg
          # - Center being offset by -0.0465 along X, so need to shift its grasp along Z 0.0465
          delta_quat_Y = np.empty(4)
          mj.mju_axisAngle2Quat(delta_quat_Y, [0, 1, 0], -90)
          delta_pos_X = 0.0465
          mj.mju_mulPose(new_pos, new_quat,
                         pos, quat,
                         np.array([delta_pos_X, 0, 0]), delta_quat_Y)

          # GraspLoco
        return [new_pos, new_quat]
      output_grasps.append(LocalGrasp(pre_pose = read_pose(pre_grasp),
                                      pose = read_pose(grasp),
                                      post_pose=read_pose(post_grasp),
                                      probability=probability,
                                      gripper_closed_width=gripper_closed_width,
                                      gripper_open_width=gripper_open_width))
  return output_grasps

def write_out_grasps(grasp_list: list[LocalGrasp], out_txt_path: str, token: str = "|"):
  if len(grasp_list) == 0:
    print("There are no grasps to log:", out_txt_path)
    return

  def arr_tostring(arr: np.ndarray) -> str:
    return " ".join(map(str, arr.tolist()))
  def pose_tostring(pose):
    return arr_tostring(pose[0]) + " " + arr_tostring(pose[1])

  with open(out_txt_path, "w") as f:
    f.write("Pregrasp pose | Grasp pose | Postgrasp pose | probability | gripper close | gripper open\n")
    for grasp in grasp_list:
      f.write(pose_tostring(grasp.pre_pose) + token)
      f.write(pose_tostring(grasp.pose) + token)
      f.write(pose_tostring(grasp.post_pose) + token)
      f.write(str(grasp.probability) + token)
      f.write(str(grasp.gripper_closed_width) + token)
      f.write(str(grasp.gripper_open_width) + "\n")
