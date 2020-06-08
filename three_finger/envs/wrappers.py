import numpy as np
import gym
from gym.spaces import Box

from envs.three_finger_envs import GOAL_THRESH
from pybullet_envs.robot_manipulators import Reacher


class ReacherRobotGoalWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.robot = env.unwrapped.robot

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    @property
    def goal(self):
        return np.array(self.robot.target.pose().xyz()[:2])

    def current_obs_to_goal(self):
        return np.array(self.robot.fingertip.pose().xyz()[:2])


class ThreeFingerRobotObsWrapper(gym.ObservationWrapper):
    """
    gym.ObservationWrapper that removes goal information from state
    """
    def __init__(self, env):
        super().__init__(env)
        obs_hi, obs_low = self.observation_space.high, self.observation_space.low
        self.observation_space = Box(obs_low[:-3], obs_hi[:-3])
        self.goal_space = Box(obs_low[-3:], obs_hi[-3:])

    def observation(self, observation):
        return observation[:-3]


class RewScale(gym.RewardWrapper):
    def __init__(self, env, scale):
        gym.RewardWrapper.__init__(self, env)
        self.scale = scale

    def reward(self, r):
        return r * self.scale


class ThreeFingerRobotHerWrapper(gym.Wrapper):
    """
    gym.Wrapper that computes robot object goal location
    """

    def __init__(self, env, goal_space):
        super().__init__(env)
        self.goal_space = goal_space
        self.robot = env.unwrapped.robot

    @property
    def goal(self):
        return self.env.unwrapped.goal

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def current_obs_to_goal(self):
        """
        Returns current object position for ThreeFingerRobot
        :return: numpy.ndarray with shape (3,) containing current object x, y, theta
        """
        return self.robot.object_pos

    def obs_to_goal(self, obs):
        """
        Processes observation and by mapping it to "goal space"
        :param obs: numpy.ndarray observation to map to goal space
        :return: numpy.ndarray with shape (3,) containing current object x, y, theta
        """
        if obs.size == 18:
            return obs[-6:-3]
        elif obs.size == 15:  # assume observation processed by ThreeFingerRobotObsWrapper
            return obs[-3:]


class HacWrapper(gym.Wrapper):
    def __init__(self, env, num_layers, goal_threshold=GOAL_THRESH):
        """
        Returns a gym environment that supports subgoal sampling and returns
        info about subgoal progress in info_dict. Subgoal rewards are not adjusted.
        :param env: Gym.Env instance to be wrapped.
        :param num_layers: Number of Hac layers that generate subgoals.
        :param goal_threshold: How far from goal is considered success
        """
        super().__init__(env)
        self._subgoals = [None for _ in range(num_layers)]
        self._reached_subgoals = [False for _ in range(num_layers)]
        self.num_layers = num_layers
        self.goal_threshold = goal_threshold

        self._state = None
        self.env_reward, self.reward = 0., 0.
        self.info = {}
        self.dense = self.env.reward_type == 'dense'
        self.goal = self.env.goal

    @property
    def subgoals(self):
        return self._subgoals

    @property
    def reached_subgoals(self):
        return self._reached_subgoals

    @subgoals.setter
    def subgoals(self, x):
        i, x = x
        self._subgoals[i] = x

    @property
    def state(self):
        return self._state

    def calc_done(self, achieved_goal, desired_goal):
        diff = desired_goal - achieved_goal
        return np.all(self.goal_threshold >= np.abs(diff))

    def step(self, action):
        """
        Takes atomic actions in the environment, returns reward wrt subgoal of layer 0
        :param action: Action taken by agent
        :return: observation, reward, done, info
        """
        # update appropriate subgoal
        subgoal = self._subgoals[0]
        o, r, d, i = self.env.step(action)
        self._state = o
        # store raw env reward in case it is used
        self.env_reward = r
        # updates which subgoals have been reached
        self._reached_subgoals = [self.calc_done(o, sg) for sg in self._subgoals]
        self.info.update(i)
        self.info['reached_subgoals'] = self.reached_subgoals
        self.reward = self.compute_reward(self._state, subgoal, self.info)
        done = sum(self._reached_subgoals) > 0  # TODO: Return done if reached any subgoal?

        return self._state.copy(), self.reward, done, self.info

    def compute_reward(self, achieved_goal, desired_goal, info={}):
        r = self.env.compute_reward(achieved_goal, desired_goal, info)
        if self.dense:
            return r
        if not self.calc_done(achieved_goal, desired_goal):
            return -1
        else:
            return 0

    def reset(self):
        self._state = self.env.reset()
        self.goal = self.env.goal  # makes sure goal is reset
        self.env_reward, self.reward = 0., 0.
        self._reached_subgoals = [False for _ in range(self.num_layers)]
        self.info = {}
        return self._state.copy()
