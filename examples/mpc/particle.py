# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import mujoco
import mujoco.viewer
import numpy as np
from dm_control.mujoco.wrapper.mjbindings.enums import mjtObj

# mjpc
from loop_rate_limiters import RateLimiter
from mujoco import mj_name2id

from mink.mjpc import predictive_sampling


# XML
XML = """
  <mujoco model="Particle Control">
  <option timestep="0.01">
    <flag contact="disable"/>
  </option>

  <visual>
    <headlight ambient=".4 .4 .4" diffuse=".8 .8 .8" specular="0.1 0.1 0.1"/>
    <map znear=".01"/>
    <quality shadowsize="2048"/>
    <global elevation="-15"/>
  </visual>

  <asset>
    <texture name="blue_grid" type="2d" builtin="checker" rgb1=".02 .14 .44" rgb2=".27 .55 1" width="300" height="300" mark="edge" markrgb="1 1 1"/>
    <material name="blue_grid" texture="blue_grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>

    <texture name="grey_grid" type="2d" builtin="checker" rgb1=".26 .26 .26" rgb2=".6 .6 .6" width="300" height="300" mark="edge" markrgb="1 1 1"/>
    <material name="grey_grid" texture="grey_grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1=".66 .79 1" rgb2=".9 .91 .93" width="800" height="800"/>

    <material name="self" rgba=".7 .5 .3 1"/>
    <material name="self_default" rgba=".7 .5 .3 1"/>
    <material name="self_highlight" rgba="0 .5 .3 1"/>
    <material name="effector" rgba=".7 .4 .2 1"/>
    <material name="effector_default" rgba=".7 .4 .2 1"/>
    <material name="effector_highlight" rgba="0 .5 .3 1"/>
    <material name="decoration" rgba=".2 .6 .3 1"/>
    <material name="eye" rgba="0 .2 1 1"/>
    <material name="target" rgba=".6 .3 .3 1"/>
    <material name="target_default" rgba=".6 .3 .3 1"/>
    <material name="target_highlight" rgba=".6 .3 .3 .4"/>
    <material name="site" rgba=".5 .5 .5 .3"/>
  </asset>

  <default>
    <joint type="hinge" axis="0 0 1" limited="true" range="-.29 .29" damping="1"/>
    <motor gear=".1" ctrlrange="-1 1" ctrllimited="true"/>
  </default>

  <worldbody>
    <light name="light" pos="0 0 1"/>
    <camera name="fixed" pos="0 0 .75" quat="1 0 0 0"/>
    <geom name="ground" type="plane" pos="0 0 0" size=".3 .3 .1" material="blue_grid"/>
    <geom name="wall_x" type="plane" pos="-.3 0 .02" zaxis="1 0 0"  size=".02 .3 .02" material="decoration"/>
    <geom name="wall_y" type="plane" pos="0 -.3 .02" zaxis="0 1 0"  size=".3 .02 .02" material="decoration"/>
    <geom name="wall_neg_x" type="plane" pos=".3 0 .02" zaxis="-1 0 0"  size=".02 .3 .02" material="decoration"/>
    <geom name="wall_neg_y" type="plane" pos="0 .3 .02" zaxis="0 -1 0"  size=".3 .02 .02" material="decoration"/>

    <body name="pointmass" pos="0 0 .01">
      <camera name="cam0" pos="0 -0.3 0.3" xyaxes="1 0 0 0 0.7 0.7"/>
      <joint name="root_x" type="slide"  pos="0 0 0" axis="1 0 0" />
      <joint name="root_y" type="slide"  pos="0 0 0" axis="0 1 0" />
      <geom name="pointmass" type="sphere" size=".01" material="self" mass=".3"/>
      <site name="tip" pos="0 0 0" size="0.01"/>
    </body>
    
    <body name="goal" mocap="true" pos="0.25 0 0.01" quat="1 0 0 0">
      <geom type="sphere" size=".01" contype="0" conaffinity="0" rgba="0 1 0 .5"/>
    </body>
  </worldbody>

  <sensor>
    <user name="Position" dim="2" user="0 5.0 0.0 10.0" />
    <user name="Velocity" dim="2" user="0 0.1 0.0 1.0" />
    <user name="Control" dim="2" user="0 0.1 0.0 1.0" />
    <framepos name="trace0" objtype="site" objname="tip"/>
    <framepos name="position" objtype="site" objname="tip"/>
    <framelinvel name="velocity" objtype="site" objname="tip"/>
    <framepos name="goal" objtype="body" objname="goal"/>
  </sensor>

  <actuator>
    <motor name="x_motor" joint="root_x" gear="1" ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="y_motor" joint="root_y" gear="1" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
  </mujoco>
"""

# reward
def reward(model: mujoco.MjModel, data: mujoco.MjData) -> float:
  # sensor data
  #pos_sensor_idx = mj_name2id(model, mjtObj.mjOBJ_SENSOR, "position")
  #position = data.sensordata[pos_sensor_idx:pos_sensor_idx + 2]
  #vel_sensor_idx = mj_name2id(model, mjtObj.mjOBJ_SENSOR, "velocity")
  #velocity = data.sensordata[vel_sensor_idx:vel_sensor_idx + 2]

  position = data.qpos
  velocity = data.qvel

  # position
  goal = data.mocap_pos[0, :2]
  pos_error = position - goal
  r0 = -np.dot(pos_error, pos_error)

  # velocity
  r1 = -np.dot(velocity, velocity)

  # effort
  r2 = -np.dot(data.ctrl, data.ctrl)

  return 5.0 * r0 + 0.1 * r1 + 0.1 * r2

if __name__ == "__main__":
    # create simulation model + data
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    DT = 0.025
    model.opt.timestep = DT

    with mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # planner
        horizon = 0.5
        splinestep = 0.1
        planstep = DT
        nimprove = 10
        nsample = 10
        noise_scale = 0.01
        interp = "zero"
        planner = predictive_sampling.Planner(
            model,
            reward,
            horizon,
            splinestep,
            planstep,
            nsample,
            noise_scale,
            nimprove,
            interp=interp,
        )
        # simulate
        mujoco.mj_resetData(model, data)

        # set goal position
        data.mocap_pos[0, :2] = np.array([0.25, 0.0])

        # history
        qpos = [data.qpos]
        qvel = [data.qvel]
        act = [data.act]
        ctrl = []
        rewards = []

        # verbose
        VERBOSE = False

        rate = RateLimiter(frequency=1/DT, warn=False)
        while viewer.is_running():
          ## predictive sampling

          # improve policy
          planner.improve_policy(
              data.qpos, data.qvel, data.act, data.time, data.mocap_pos, data.mocap_quat
          )

          # get action from policy
          data.ctrl = planner.action_from_policy(data.time)

          # reward
          rewards.append(reward(model, data))

          if VERBOSE:
            print("time  : ", data.time)
            print(" qpos  : ", data.qpos)
            print(" qvel  : ", data.qvel)
            print(" act   : ", data.act)
            print(" action: ", data.ctrl)
            print(" reward: ", rewards[-1])

          # step
          mujoco.mj_step(model, data)

          # history
          qpos.append(data.qpos)
          qvel.append(data.qvel)
          act.append(data.act)
          ctrl.append(ctrl)
          # End exec loop

          # Visualize at fixed FPS
          viewer.sync()
          rate.sleep()

        if VERBOSE:
          print("\nfinal qpos: ", qpos[-1])
          print("goal state : ", data.mocap_pos[0, 0:2])
          print("state error: ", np.linalg.norm(qpos[-1][0:2] - data.mocap_pos[0, 0:2]))
