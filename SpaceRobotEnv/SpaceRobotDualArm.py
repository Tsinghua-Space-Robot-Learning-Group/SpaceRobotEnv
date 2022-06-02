import os

import copy
import numpy as np

import gym
from gym import error, spaces
from gym.utils import seeding

from gym.envs.robotics import utils
from gym.envs.robotics import rotations

import mujoco_py

MODEL_XML_PATH = os.path.join('mujoco_files', 'spacerobot', 'spacerobot_dualarm.xml')
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

        self.goal, self.goal1 = self._sample_goal()  # 根据环境中是否有待抓取的目标确定任务目标,goal=blue
        obs = self._get_obs()

        self._set_action_space()  # action_space若要严格参照model中对各个joint的控制信号的限制
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['desired_goal'].shape, dtype='float32'),
            # desired_goal1=spaces.Box(-np.inf, np.inf, shape=obs['desired_goal1'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            # desired_goal1=spaces.Box(-np.inf, np.inf, shape=obs['desired_goal1'].shape, dtype='float32'),
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

    def step(self, action, t):  # mujoco中没有step函数,是自己定义的
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)  # _set_action(action)函数【在SpaceRobot环境中具体定义】,类比mujoco中的do_simulation()函数
        self._step_callback()  # 回调函数不一定要
        obs = self._get_obs()
        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal) and
                          self._is_success(obs['achieved_goal1'], self.goal1),
        }
        g = np.concatenate([self.goal, self.goal1])
        ag = np.concatenate([obs['achieved_goal'], obs['achieved_goal1']])
        reward = self.compute_reward(ag, g, info)
        cost = self.compute_cost(t)
        # reward = self.compute_reward(obs['achieved_goal'], self.goal, info) + self.compute_reward(obs['achieved_goal1'], self.goal1, info)
        if self.pro_tyep == 'CMDP':

            return obs, reward, cost, done, info
        else:
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

        self.goal, self.goal1 = self._sample_goal()  # reset时重新设置目标
        obs = self._get_obs()

        # TODO: set the position of cube

        body_id = self.sim.model.geom_name2id('cube')
        self.sim.model.geom_pos[body_id] = np.array([0, 0, 6])

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
            datargb, datadepth = self._get_viewer(mode).read_pixels(width, height, depth=True)
            # original image is upside-down, so flip it
            return datargb[::-1, :, :], datadepth[::-1]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)

        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
                self._viewer_setup()
                # cam_pos = np.array([0.5, 0, 5, 0.3, -30, 0])
                # for i in range(3):
                #     self.viewer.cam.lookat[i] = cam_pos[i]
                # self.viewer.cam.distance = cam_pos[3]
                # self.viewer.cam.elevation = cam_pos[4]
                # self.viewer.cam.azimuth = cam_pos[5]
            elif mode == 'rgb_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
                self._viewer_setup()
                # self.viewer.cam.trackbodyid = 0
                # 最新改动
                cam_pos = np.array([0.5, 0, 5, 0.3, -30, 0])
                for i in range(3):
                    self.viewer.cam.lookat[i] = cam_pos[i]
                self.viewer.cam.distance = cam_pos[3]
                self.viewer.cam.elevation = cam_pos[4]
                self.viewer.cam.azimuth = cam_pos[5]
                # self.viewer.cam.trackbodyid = -1

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
            distance_threshold, initial_qpos, reward_type, pro_type, c_coeff,
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
            pro_type ('MDP' or 'CMDP'):  the problem setting whether contains cost or not
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
        self.c_coeff = c_coeff
        self.pro_type = pro_type


        super(SpacerobotEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps,  # n_actions=4,
            initial_qpos=initial_qpos)

        # GoalEnv methods

    # ----------------------------

    def compute_reward(self, achieved_goal, desired_goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, desired_goal)
        d0 = goal_distance(achieved_goal[:3], desired_goal[:3])
        d1 = goal_distance(achieved_goal[3:], desired_goal[3:])

        reward = {
            'sparse': -(d > self.distance_threshold).astype(np.float32),
            'd0': d0,
            'd1': d1,
            'dense': - (0.001 * d ** 2 + np.log10(d ** 2 + 1e-6))
        }

        return reward

    def compute_cost(self, t):
        # get the initial base attitude
        post_base_att = self.sim.data.get_body_xquat('chasersat').copy()

        # get the initial base attitude
        post_base_pos = self.sim.data.get_body_xpos('chasersat').copy()

        """ cost function is continue
        """
        cost = self.c_coeff * t * (goal_distance(post_base_att, self.initial_base_att) + \
                                   goal_distance(post_base_pos, self.initial_base_pos))

        return cost

    # RobotEnv methods
    # ----------------------------

    def _set_action(self, action):
        """
        output action (velocity)
        :param action: angle velocity of joints
        :return: angle velocity of joints
        """
        # print('action',action)
        assert action.shape == (12,)  # 可改
        action = action.copy()  # ensure that we don't change the action outside of this scope
        self.sim.data.ctrl[:] = action * 0.2
        for _ in range(self.n_substeps):
            self.sim.step()

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_body_xpos('tip_frame')
        grip_pos1 = self.sim.data.get_body_xpos('tip_frame1')
        '''
        # get the rotation angle of the target
        grip_rot = self.sim.data.get_body_xquat('tip_frame')
        grip_rot = rotations.quat2euler(grip_rot)
        grip_rot1 = self.sim.data.get_body_xquat('tip_frame1')
        grip_rot1 = rotations.quat2euler(grip_rot1)     
        '''
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_body_xvelp('tip_frame') * dt
        grip_velp1 = self.sim.data.get_body_xvelp('tip_frame1') * dt
        '''
        achieved_goal = np.concatenate([grip_pos.copy(),grip_rot.copy()])
        achieved_goal1 = np.concatenate([grip_pos1.copy(),grip_rot1.copy()]) 
        '''
        # 观测量加goal
        obs = np.concatenate([
            self.sim.data.qpos[:19].copy(), self.sim.data.qvel[:18].copy(), grip_pos, grip_pos1,
            grip_velp, grip_velp1, self.goal.copy(), self.goal1.copy(),
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': grip_pos.copy(),
            'achieved_goal1': grip_pos1.copy(),
            'desired_goal': self.goal.copy(),
            'desired_goal1': self.goal1.copy(),
        }

    def _viewer_setup(self):
        #        body_id = self.sim.model.body_name2id('forearm_link')
        body_id = self.sim.model.body_name2id('wrist_3_link')
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
        goal_pos = self.sim.data.get_body_xpos('tip_frame').copy()  # data initializing
        goal_pos1 = self.sim.data.get_body_xpos('tip_frame1').copy()
        # goal_rot = np.array([0,0,0],dtype=np.float)
        # goal_rot1 = np.array([0,0,0],dtype=np.float)

        # goal[0] = self.initial_gripper_xpos[0] + np.random.uniform(-0.4,0)  # self.np_random.uniform(-0.45, 0) # 目标为移动到随机位置
        # goal[1] = self.initial_gripper_xpos[1] + np.random.uniform(-0.3, 0.3)  # self.np_random.uniform(-0.3, 0.3)
        # goal[2] = self.initial_gripper_xpos[2] + np.random.uniform(0, 0.3)  # self.np_random.uniform(0.1, 0.3)
        goal_pos[0] = self.sim.data.get_body_xpos('tip_frame')[0] + np.random.uniform(-0.40, 0.0)  # 目标为移动到随机位置
        goal_pos[1] = self.sim.data.get_body_xpos('tip_frame')[1] + np.random.uniform(-0.25, 0.30)  # watch out y-
        goal_pos[2] = self.sim.data.get_body_xpos('tip_frame')[2] + np.random.uniform(0.0, 0.30)
        '''
        goal_rot[0] = np.random.uniform(-1.67, 1.67)
        goal_rot[1] = np.random.uniform(-1.67, 1.67)
        goal_rot[2] = np.random.uniform(-1.67, 1.67)
        '''
        goal_pos1[0] = self.sim.data.get_body_xpos('tip_frame1')[0] + np.random.uniform(-0.40, 0.0)  # 目标为移动到随机位置
        goal_pos1[1] = self.sim.data.get_body_xpos('tip_frame1')[1] + np.random.uniform(-0.30, 0.25)  # watch out y+
        goal_pos1[2] = self.sim.data.get_body_xpos('tip_frame1')[2] + np.random.uniform(0.0, 0.30)
        '''
        goal_rot1[0] = np.random.uniform(-1.67, 1.67)
        goal_rot1[1] = np.random.uniform(-1.67, 1.67)
        goal_rot1[2] = np.random.uniform(-1.67, 1.67)     
        '''
        '''
        goal = np.concatenate((goal_pos, goal_rot)) #一维度的数据不影响
        goal1 = np.concatenate((goal_pos1, goal_rot1))
        '''
        goal = goal_pos
        goal1 = goal_pos1
        # 显示target的位置
        site_id = self.sim.model.site_name2id('target0')  # 设置target的位置
        self.sim.model.site_pos[site_id] = goal_pos
        # self.sim.model.site_quat[site_id] = rotations.euler2quat(goal_rot)
        site_id1 = self.sim.model.site_name2id('target1')  # 设置target的位置
        self.sim.model.site_pos[site_id1] = goal_pos1
        # self.sim.model.site_quat[site_id1] = rotations.euler2quat(goal_rot1)
        self.sim.forward()

        return goal.copy(), goal1.copy()

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
        self.initial_gripper_xpos1 = self.sim.data.get_body_xpos('tip_frame1').copy()

        # get the initial base attitude
        self.initial_base_att = self.sim.data.get_body_xquat('chasersat').copy()

        # get the initial base attitude
        self.initial_base_pos = self.sim.data.get_body_xpos('chasersat').copy()

    def render(self, mode='human', width=500, height=500):
        return super(SpacerobotEnv, self).render(mode, width, height)



class SpaceRobotDualArm(SpacerobotEnv, gym.utils.EzPickle):
    def __init__(self, reward_type='sparse',pro_type = 'MDP'):
        initial_qpos = {
            'arm:shoulder_pan_joint': 0.0,
            'arm:shoulder_lift_joint': 0.0,
            'arm:elbow_joint': 0.0,
            'arm:wrist_1_joint': 0.0,
            'arm:wrist_2_joint': 0.0,
            'arm:wrist_3_joint': 0.0,

            'arm:shoulder_pan_joint1': 0.0,
            'arm:shoulder_lift_joint1': 0.0,
            'arm:elbow_joint1': 0.0,
            'arm:wrist_1_joint1': 0.0,
            'arm:wrist_2_joint1': 0.0,
            'arm:wrist_3_joint1': 0.0
        }
        SpacerobotEnv.__init__(
            self, MODEL_XML_PATH, n_substeps=20, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type,pro_type = pro_type,c_coeff=0.1)
        gym.utils.EzPickle.__init__(self)