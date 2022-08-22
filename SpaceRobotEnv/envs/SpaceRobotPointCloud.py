import os

import copy
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

from gym.envs.robotics import utils
from gym.envs.robotics import rotations

import open3d as o3d
import mujoco_py
import cv2 as cv

import matplotlib.pyplot as plt


PATH = os.getcwd()
MODEL_XML_PATH = os.path.join(
    PATH, "SpaceRobotEnv", "assets", "spacerobot", "spacerobot_image.xml"
)
DEFAULT_SIZE = 500


class RobotEnv(gym.GoalEnv):
    def __init__(self, model_path, initial_qpos, n_substeps):

        # load model and simulator
        self.model = mujoco_py.load_model_from_path(model_path)
        self.sim = mujoco_py.MjSim(self.model, nsubsteps=n_substeps)

        # render setting
        self.viewer = None
        self._viewers = {}
        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": int(np.round(1.0 / self.dt)),
        }

        # seed
        self.seed()

        # initalization
        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())
        self.goal = self._sample_goal()

        # set action_space and observation_space
        obs = self._get_obs()
        self._set_action_space()
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["desired_goal"].shape, dtype="float32"
                ),
                achieved_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float32"
                ),
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape, dtype="float32"
                ),
                base=spaces.Box(
                    -np.inf, np.inf, shape=obs["base"].shape, dtype="float32"
                ),
                image=spaces.Box(
                    -np.inf, np.inf, shape=obs["image"].shape, dtype="float32"
                ),
                rawimage=spaces.Box(
                    -np.inf, np.inf, shape=obs["rawimage"].shape, dtype="float32"
                ),
                depth=spaces.Box(
                    -np.inf, np.inf, shape=obs["depth"].shape, dtype="float32"
                ),
            )
        )

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy()
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    def _detecte_collision(self):
        self.collision = self.sim.data.ncon
        return self.collision

    def _sensor_torque(self):
        self.sensor_data = self.sim.data.sensordata
        return self.sensor_data

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        old_action = self.sim.data.ctrl.copy() * (1 / 0.5)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)  # do one step simulation here
        self._step_callback()
        obs = self._get_obs()
        done = False
        info = {
            "is_success": self._is_success(obs["achieved_goal"], self.goal),
            "act": action,
            "old_act": old_action,
        }
        g = self.goal
        ag = obs["achieved_goal"]
        reward = self.compute_reward(ag, g, info)
        return obs, reward, done, info

    def reset(self):
        """Attempt to reset the simulator. Since we randomize initial conditions, it
        is possible to get into a state with numerical issues (e.g. due to penetration or
        Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        In this case, we just keep randomizing until we eventually achieve a valid initial
        configuration.
        """
        super(RobotEnv, self).reset()
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()

        self.goal = self._sample_goal()
        obs = self._get_obs()

        return obs

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def render(self, mode="human", width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        # self._render_callback()
        if mode == "rgb_array":
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            datargb, datadepth = self._get_viewer(mode).read_pixels(
                width, height, depth=True
            )
            # original image is upside-down, so flip it
            return datargb[::-1, :, :], datadepth[::-1]
        elif mode == "human":
            self._get_viewer(mode).render()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = mujoco_py.MjViewer(self.sim)
                self._viewer_setup()

            elif mode == "rgb_array":
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
                self._viewer_setup()
                # self.viewer.cam.trackbodyid = 0
                # original version: cam_pos = np.array([0.7, 0.5, 5, 0.5, 0, -60])
                # latest modification
                cam_pos = np.array([1, 0.5, 5, 0.3, 0, -90])
                for i in range(3):
                    self.viewer.cam.lookat[i] = cam_pos[i]
                self.viewer.cam.distance = cam_pos[3]
                self.viewer.cam.elevation = cam_pos[4]
                self.viewer.cam.azimuth = cam_pos[5]
                # self.viewer.cam.trackbodyid = -1

            self._viewers[mode] = self.viewer
        return self.viewer

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it is successful.
        If a reset is unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        return True

    def _get_obs(self):
        """Returns the observation."""
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation."""
        raise NotImplementedError()

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal."""
        raise NotImplementedError()

    def _sample_goal(self):
        """Samples a new goal and returns it."""
        raise NotImplementedError()

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        pass

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass

    def _render_callback(self):
        """A custom callback【自定义回调】 that is called before rendering. Can be used
        to implement custom visualizations.【可实现自定义可视化】
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.【对模拟状态强制附加约束】
        """
        pass


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class SpacerobotEnv(RobotEnv):
    """Superclass for all SpaceRobot environments."""

    def __init__(
        self,
        model_path,
        n_substeps,
        distance_threshold,
        initial_qpos,
        reward_type,
    ):
        """Initializes a new Fetch environment.
        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.n_substeps = n_substeps
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type

        super(SpacerobotEnv, self).__init__(
            model_path=model_path,
            n_substeps=n_substeps,
            initial_qpos=initial_qpos,
        )

    def compute_reward(self, achieved_goal, desired_goal, info):

        action = info["act"]
        old_action = info["old_act"]
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -(d > self.distance_threshold).astype(np.float32)
        elif self.reward_type == "distance":
            return d
        else:
            # dense reward
            return -(
                0.001 * d ** 2
                + np.log10(d ** 2 + 1e-6)
                + 0.01 * np.linalg.norm(action - old_action) ** 2
            )

    def _set_action(self, action):
        """
        :param action: 3*None->6*None
        :return:
        """
        assert action.shape == (6,)
        self.sim.data.ctrl[:] = action * 0.5 / 10
        for _ in range(self.n_substeps):
            self.sim.step()

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_body_xpos("tip_frame")
        grip_velp = self.sim.data.get_body_xvelp("tip_frame") * self.dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        # get raw_image
        width, height = (64, 64)
        image_raw, depth = self.render(mode="rgb_array", width=width, height=height)
        image = (2.0 / 255.0) * image_raw - 1.0

        image_raw_processed  = cv.cvtColor(image_raw, cv.COLOR_BGR2GRAY)
        depth_processed = (1000*(depth)).astype(np.uint16) 
         # Open3d takes the depth image (uint16 milimeters) as inputs
        
        image_raw_o3d = o3d.geometry.Image(image_raw_processed)
        depth_o3d = o3d.geometry.Image(depth_processed)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_raw_o3d, depth_o3d)

        # show rgbd image 
        # plt.subplot(1, 2, 1)
        # plt.title('Space Robot RGBD image')
        # plt.imshow(rgbd_image.color)
        # plt.subplot(1, 2, 2)
        # plt.title('Space Robot depth image')
        # plt.imshow(rgbd_image.depth)
        # plt.show()

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                    rgbd_image,
                    o3d.camera.PinholeCameraIntrinsic(
                        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
        # Flip it, otherwise the pointcloud will be upside down
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # view point cloud
        # o3d.visualization.draw_geometries([pcd])

        # to save point cloud in directory and file
        # point_cloud_dir : str
        # point_cloud_file = os.path.join(point_cloud_dir, "demo_point_cloud2.ply")
        #o3d.io.write_point_cloud(point_cloud_file, pcd)

        # to read point cloud from file
        # pcd = o3d.io.read_point_cloud(point_cloud_file)
        # o3d.visualization.draw_geometries([pcd])


        # get high-resolution image
        width, height = (640, 640)
        clearimage, _ = self.render(mode="rgb_array", width=width, height=height)  

        base_pos = self.sim.data.qpos[:3].copy()
        base_att = self.sim.data.qpos[3:7].copy()
        base_att = rotations.quat2euler(base_att)
        base = np.hstack((base_pos, base_att))

        achieved_goal = grip_pos.copy()

        obs = np.concatenate(
            [
                self.sim.data.qpos[7:13].copy(),
                self.sim.data.qvel[6:12].copy(),
                grip_pos,
                grip_velp,
                self.goal.copy(),
            ]
        )

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
            "base": base.copy(),
            "image": clearimage.copy(), 
            "rawimage": image_raw.copy(),
            "depth": depth.copy(),
            "point_cloud" : pcd.copy()
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id("tip_frame")
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.0
        self.viewer.cam.elevation = -14.0

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        return True

    def _sample_goal(self):

        goal = self.initial_gripper_xpos[:3].copy()
        d = goal_distance(self.sim.data.get_body_xpos("tip_frame").copy(), goal)

        # random target position
        goal[0] = self.initial_gripper_xpos[0] + self.np_random.uniform(-0.45, 0)
        goal[1] = self.initial_gripper_xpos[1] + self.np_random.uniform(-0.3, 0.3)
        goal[2] = self.initial_gripper_xpos[2] + self.np_random.uniform(0.1, 0.3)

        # show target position
        site_id = self.sim.model.site_name2id("target0")
        self.sim.model.site_pos[site_id] = goal
        self.sim.forward()

        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)
        # return d

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_body_xpos("tip_frame").copy()

        """
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.initial_gripper_xpos + self.np_random.uniform(-self.obj_range, self.obj_range, size=3)
        self.sim.forward()
        """

    def render(self, mode="human", width=500, height=500):
        return super(SpacerobotEnv, self).render(mode, width, height)


class SpaceRobotPointCloud(SpacerobotEnv, gym.utils.EzPickle):
    def __init__(self, reward_type="nosparse"):
        initial_qpos = {
            "arm:shoulder_pan_joint": 0.0,
            "arm:shoulder_lift_joint": 0.0,
            "arm:elbow_joint": 0.0,
            "arm:wrist_1_joint": 0.0,
            "arm:wrist_2_joint": 0.0,
            "arm:wrist_3_joint": 0.0,
        }
        SpacerobotEnv.__init__(
            self,
            MODEL_XML_PATH,
            n_substeps=20,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
        )
        gym.utils.EzPickle.__init__(self)
