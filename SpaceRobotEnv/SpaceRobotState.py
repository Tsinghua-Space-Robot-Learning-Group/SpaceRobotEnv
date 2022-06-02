import os

import copy
import numpy as np

import gym
from gym import error, spaces
from gym.utils import seeding

from gym.envs.robotics import utils
from gym.envs.robotics import rotations

import mujoco_py

MODEL_XML_PATH = os.path.join('mujoco_files', 'spacerobot', 'spacerobot_state.xml')
DEFAULT_SIZE = 500


class RobotEnv(gym.GoalEnv):
    # n_actions是actuator的数量（区别RL算法中的n_actions）
    # n_substeps ?
    def __init__(self, model_path, initial_qpos, n_substeps):

        self.model = mujoco_py.load_model_from_path(model_path)
        self.sim = mujoco_py.MjSim(self.model, nsubsteps=n_substeps)
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.seed()

        self._env_setup(initial_qpos=initial_qpos)  # 设置Robot的初始姿态及target（各joint的初始角度）
        self.initial_state = copy.deepcopy(self.sim.get_state())

        self.goal = self._sample_goal()  # 根据环境中是否有待抓取的目标确定任务目标
        obs = self._get_obs()

        self._set_action_space()  # action_space若要严格参照model中对各个joint的控制信号的限制
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

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

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):  # mujoco中没有step函数,是自己定义的
        # 加入old action 但注意 我们在action上加入了一个系数0.5
        old_action = self.sim.data.ctrl.copy() * (1 / 0.5)

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)  # _set_action(action)函数【在SpaceRobot环境中具体定义】,类比mujoco中的do_simulation()函数
        self.sim.step()  # set action之后step【mujoco-py中用于模拟的API】
        self._step_callback()  # 回调函数不一定要
        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, action, old_action, info)
        return obs, reward, done, info

    def reset(self):  # 重置环境，可在每一个episode结束后调用
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues【数值问题】(e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition【初始条件】 (e.g. an object is within the hand).
        # In this case, we just keep randomizing【保持随机】 until we eventually achieve a valid initial
        # configuration.【有效的初始配置】
        super(RobotEnv, self).reset()

        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()  # _reset_sim()【在低一级环境中定义】

        self.goal = self._sample_goal()  # 根据环境中是否有待抓取的目标确定任务目标

        obs = self._get_obs()

        return obs

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        # self._render_callback()
        if mode == 'rgb_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
            self._viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    # Extension methods
    # ----------------------------

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.【重置模拟，并指示reset是否成功】
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        self.sim.set_state(self.initial_state)  # reset时将状态设置为规定的initial_state
        self.sim.forward()
        return True

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        raise NotImplementedError()

    def _sample_goal(self):
        """Samples a new goal and returns it.
        """
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
    """Superclass for all SpaceRobot environments.
    """

    def __init__(
            self, model_path, n_substeps,
            distance_threshold, initial_qpos, reward_type,
    ):
        """Initializes a new Fetch environment.
        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            ? gripper_extra_height (float): additional height above the table when positioning the gripper【定位gripper】
            ? block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            ? target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            ? target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions采样初始目标位置的均匀分布范围
            ? target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        #        self.gripper_extra_height = gripper_extra_height
        #        self.block_gripper = block_gripper
        #        self.has_object = has_object
        #        self.target_in_the_air = target_in_the_air
        #        self.target_offset = target_offset
        self.n_substeps = n_substeps
        #        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type

        super(SpacerobotEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps,  # n_actions=4,
            initial_qpos=initial_qpos)

        # GoalEnv methods

    # ----------------------------

    def compute_reward(self, achieved_goal, desired_goal, action, old_action, info):

        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, desired_goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        elif self.reward_type == 'distance':
            return d
        else:
            # 使用小曹的奖励函数测试
            return - (0.001 * d ** 2 + np.log10(d ** 2 + 1e-6) + 0.01 * np.linalg.norm(action - old_action) ** 2)

    # RobotEnv methods
    # ----------------------------

    def _set_action(self, action):
        # print('action',action)
        """
        :param action: 3*None->6*None
        :return:
        """
        assert action.shape == (6,)  # 可改
        action = action.copy()  # ensure that we don't change the action outside of this scope
        self.sim.data.ctrl[:] = action * 0.5

        for _ in range(self.n_substeps):
            self.sim.step()

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_body_xpos('tip_frame')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_body_xvelp('tip_frame') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        # position = self.sim.data.qpos.flat.copy()
        # velocity = self.sim.data.qvel.flat.copy()

        object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-1:]
        gripper_vel = robot_qvel[-1:] * dt  # change to a scalar if the gripper is made symmetric

        achieved_goal = grip_pos.copy()
        # 观测量加入了goal
        obs = np.concatenate([
            self.sim.data.qpos[7:13].copy(), self.sim.data.qvel[6:12].copy(),
            grip_pos, grip_velp, self.goal.copy()
        ])
        # obs = np.concatenate([
        #     grip_pos,  grip_velp, self.goal.copy()
        # ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _viewer_setup(self):
        #        body_id = self.sim.model.body_name2id('forearm_link')
        body_id = self.sim.model.body_name2id('tip_frame')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)  # self.initial_state在robot_env中定义;环境重置之后，机械臂的位置回到初始自然state
        self.sim.forward()
        return True

    def _sample_goal(self):
        # 鉴于机械臂的初始位置是0,需要按照初始情况分别设置target的x,y,z坐标
        goal = self.initial_gripper_xpos[:3].copy()
        d = goal_distance(self.sim.data.get_body_xpos('tip_frame').copy(), goal)

        # goal[0] = self.initial_gripper_xpos[0] - 0.37 #self.np_random.uniform(-0.45, 0) # 目标为移动到随机位置
        # goal[1] = self.initial_gripper_xpos[1] + 0.23 #self.np_random.uniform(-0.3, 0.3)
        # goal[2] = self.initial_gripper_xpos[2] + 0.22#self.np_random.uniform(0.1, 0.3)

        goal[0] = self.initial_gripper_xpos[0] + np.random.uniform(-0.4,
                                                                   0)  # self.np_random.uniform(-0.45, 0) # 目标为移动到随机位置
        goal[1] = self.initial_gripper_xpos[1] + np.random.uniform(-0.3, 0.3)  # self.np_random.uniform(-0.3, 0.3)
        goal[2] = self.initial_gripper_xpos[2] + np.random.uniform(0, 0.3)  # self.np_random.uniform(0.1, 0.3)

        d = goal_distance(self.sim.data.get_body_xpos('tip_frame').copy(), goal)
        # print('AD',d)
        # print('goal',goal)

        # 显示target的位置
        site_id = self.sim.model.site_name2id('target0')  # 设置target的位置
        self.sim.model.site_pos[site_id] = goal
        self.sim.forward()

        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)
        # return d

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():  # 机械臂的初始joint状态设置
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_body_xpos('tip_frame').copy()

        '''
        site_id = self.sim.model.site_name2id('target0') # 设置target的位置
        self.sim.model.site_pos[site_id] = self.initial_gripper_xpos + self.np_random.uniform(-self.obj_range, self.obj_range, size=3) # 设置object在所属body中的local坐标

        self.sim.forward()
        '''

    def render(self, mode='human', width=500, height=500):
        return super(SpacerobotEnv, self).render(mode, width, height)



class SpaceRobotState(SpacerobotEnv, gym.utils.EzPickle):
    def __init__(self, reward_type='nosparse'):
        initial_qpos = {
            'arm:shoulder_pan_joint': 0.0,
            'arm:shoulder_lift_joint': 0.0,
            'arm:elbow_joint': 0.0,
            'arm:wrist_1_joint': 0.0,
            'arm:wrist_2_joint': 0.0,
            'arm:wrist_3_joint': 0.0
        }
        SpacerobotEnv.__init__(
            self, MODEL_XML_PATH, n_substeps=20, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        gym.utils.EzPickle.__init__(self)