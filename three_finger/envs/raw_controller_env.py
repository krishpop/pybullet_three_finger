
"""
3 Finger Robot Gym Environment.
"""
import os, logging, gym, math, time, cv2
import numpy as np
import pybullet as p
from pybullet_utils import bullet_client

from gym import spaces
from gym import GoalEnv
from gym.utils import seeding
from pybullet_utils import bullet_client

from three_finger.envs.dynamics_calculator import DynamicCalculator
from pybullet_envs.robot_bases import XmlBasedRobot
from pybullet_envs.env_bases import MJCFBaseBulletEnv
from pybullet_envs.scene_abstract import SingleRobotEmptyScene
from three_finger.envs.base_env import URDFBasedRobot, ThreeFingerRobot

logger = logging.getLogger(__name__)
ORI_THRESH = 0.1
DIST_THRESH = 0.01
GOAL_THRESH = (DIST_THRESH, DIST_THRESH, ORI_THRESH)
GOAL_RANGE = (0.15, 0.05, np.pi/8)
REW_MIN_CLIP = -2

DELTA_REPOSE = (0.01, 0.01, 0.1)
DELTA_SLIDE = (0.02, 0.02, 0.02)
DELTA_REPOSE2x = (0.04, 0.04, 0.2)
DELTA_SLIDE2x = (0.04, 0.04, 0.04)

REPOSE_DUR = SLIDE_DUR = 30
FLIP_DUR = 1200
DEBUG = False

class Gripper2DEnv(MJCFBaseBulletEnv):
    def __init__(self, render=False, reward_type='contacts', distance_threshold=DIST_THRESH,
                 orientation_threshold=ORI_THRESH, goal_range=GOAL_RANGE, rew_min_clip=REW_MIN_CLIP):
        self.controller = DynamicCalculator()
        self.robot = ThreeFingerRobot()
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.orientation_threshold = orientation_threshold
        self.rew_min_clip = rew_min_clip
        self.max_episode_steps = 200
        self.current_step = 0
        super().__init__(self.robot, render=render)
        self._cam_dist = 1
        self._render_width = 640
        self._render_height = 480
        self.f1_file_path = os.path.join(os.path.dirname(__file__), "data",  "f1")
        self.f2_file_path = os.path.join(os.path.dirname(__file__), "data", "f2")
        self.f3_file_path = os.path.join(os.path.dirname(__file__), "data", "f3")
        if not os.path.exists(self.f1_file_path): raise ValueError("{} does not exist!".format(self.f1_file_path))
        if not os.path.exists(self.f2_file_path): raise ValueError("{} does not exist!".format(self.f2_file_path))
        if not os.path.exists(self.f3_file_path): raise ValueError("{} does not exist!".format(self.f3_file_path))

    @property
    def goal(self):
        return self.robot.goal

    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=0.0, timestep=0.0165, frame_skip=1)

    def step(self, a):
        if self.done:
            return self.robot.calc_state(), 0, self.done, {}
        self.current_step += 1
        self.done = self.current_step == self.max_episode_steps
        if self.done:
            self.current_step = 0
        assert (not self.scene.multiplayer)
        if len(a.shape) == 2:
            a = a.squeeze()
        self.robot.apply_action(a)
        self.scene.global_step()
        state = self.robot.calc_state()

        reward = self.compute_reward(self.robot.object_pos, self.goal)
        return state, reward, self.done, {}

    def compute_reward(self, achieved_goal, desired_goal, info={}):
        x_dist, y_dist, theta_dist = desired_goal - achieved_goal
        if self.reward_type in ['dense', 'contacts', 'quadratic'] and info.get('reward_type', 'dense') == 'dense':
            # referred from gym/robotics/manipulation
            reward = -(10 * np.linalg.norm([x_dist, y_dist]) + np.abs(theta_dist))
            if self.reward_type == 'quadratic':
                reward = reward ** 2
        else:
            reward = ((np.linalg.norm([x_dist, y_dist]) < self.distance_threshold) and
                      (np.abs(theta_dist) < self.orientation_threshold))
            reward = float(reward)
        if self.reward_type == 'contacts':
            for c in self.robot.get_contact_pts():
                if c is None:
                    reward += self.rew_min_clip*2
        return reward

    def camera_adjust(self):
        pass

    def reset(self):
        """
        Overwrite MJCF reset method for initialize purpose. Hard, medium, easy have different init & goal generation method.
        Easy: init & goal are sampled from f1 with same contact location
        """

        def contact_loc_parser(file_name):
            """
            Given a file name (str), parse its contact locations.
            file name is like -0.10,0.04,0.14.npy
            """
            file_name = file_name.split('.npy')
            if len(file_name) < 2: return None  # invalid files like .DS_Store
            file_name = file_name[0]
            contact_loc = [float(di) for di in file_name.split(",")]
            return contact_loc

        def sample_pose(file_path, mode):
            "Given a file_path, sample a valid object pose under the same contact configuration"
            file_list = os.listdir(file_path)
            while 1:
                file_name = self.np_random.choice(file_list)
                ds = contact_loc_parser(file_name)
                if ds is None: continue
                # init pose
                d1, d2, d3 = ds
                object_poses = np.load(os.path.join(file_path, file_name))
                for _ in range(5):
                    init_pos = object_poses[self.np_random.randint(len(object_poses))]
                    goal_pos = object_poses[self.np_random.randint(len(object_poses))]
                    init_pos += np.array([self.np_random.uniform(-0.005, 0.005),
                                          self.np_random.uniform(-0.005, 0.005),
                                          self.np_random.uniform(-0.025, 0.025)])
                    goal_pos += np.array([self.np_random.uniform(-0.005, 0.005),
                                          self.np_random.uniform(-0.005, 0.005),
                                          self.np_random.uniform(-0.025, 0.025)])
                    q = self.controller.compute_IK(init_pos, d1, d2, d3, mode=mode)
                    if q is not None:
                        if mode == 'f1':
                            if q[4] + q[5] >= np.pi + init_pos[2] - 0.1: break  # f3 link2
                            if q[2] + q[3] - 0.1 <= init_pos[2]: break  # f2 link 2
                        elif mode == 'f2':
                            if (q[4] + q[5] - 0.1 <= (np.pi + init_pos[2])): break  # f3 link 2 collide with object
                            if (q[0] + q[1] >= (np.pi + init_pos[2] - 0.1)): break  # f1 link 2 collide with object
                        elif mode == 'f3':
                            if q[0] + q[1] >= np.pi + init_pos[2]: continue
                            if q[2] + q[3] >= init_pos[2]:         continue
                            if q[4] + q[5] >= np.pi + init_pos[2]: continue
                        return init_pos, goal_pos, q
            return

        if (self.physicsClientId < 0):
            self.ownsPhysicsClient = True

            if self.isRender:
                self._p = bullet_client.BulletClient(connection_mode=p.GUI)
            else:
                self._p = bullet_client.BulletClient()

            self.physicsClientId = self._p._client
            self._p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        if self.scene is None:
            self.scene = self.create_single_player_scene(self._p)
        if not self.scene.multiplayer and self.ownsPhysicsClient:
            self.scene.episode_restart(self._p)

        self.robot.scene = self.scene

        self.frame = 0
        self.done = 0
        self.info = {}
        self.reward = 0
        dump = 0
        self.robot.reset(self._p)

        # --------------init & goal init --------------------
        # pick init and goal pose in f1, first pick a contact config, then sample two poses
        rng_set = [[0, 0], [1, 1], [2, 2]]
        rng_id = self.np_random.choice([0, 1, 2], p=[0, 0, 1])
        rng_init_goal_setting = rng_set[rng_id]
        file_paths, modes = [self.f1_file_path, self.f2_file_path, self.f3_file_path], ['f1', 'f2', 'f3']
        init_file_path, goal_file_path = file_paths[rng_init_goal_setting[0]], file_paths[rng_init_goal_setting[1]]
        init_mode, goal_mode = modes[rng_init_goal_setting[0]], modes[rng_init_goal_setting[1]]
        self.mode = init_mode
        while 1:
            self.info_init_object_pos, self.info_target_pos, _ = self.robot.object_pos, self.robot.target_pos, q = sample_pose(
                init_file_path, init_mode)
            for i in range(len(self.robot.object_pos)):
                self.robot.object_pole_joints[i].set_state(self.robot.object_pos[i], 0)
                self.robot.target_pole_joints[i].set_state(self.robot.target_pos[i], 0)
            for i in range(len(q)):
                self.robot.manipulator_joints[i].set_state(q[i], 0)
            self._p.setJointMotorControlArray(0, jointIndices=[i for i in range(9)],
                                              controlMode=p.VELOCITY_CONTROL, forces=[0] * 9)
            self._p.stepSimulation()
            collision_1 = self._p.getContactPoints(bodyA=0, bodyB=0, linkIndexA=0, linkIndexB=8)
            collision_2 = self._p.getContactPoints(bodyA=0, bodyB=0, linkIndexA=2, linkIndexB=8)
            collision_3 = self._p.getContactPoints(bodyA=0, bodyB=0, linkIndexA=4, linkIndexB=8)
            if len(collision_1) > 0 or len(collision_2) > 0 or len(collision_3) > 0: continue
            d1, d2, d3 = self.robot.get_contact_pts()
            if None not in [d1, d2, d3]: break  # some init are invalid, re-sample pose
        s = self.robot.calc_state()
        self._p.setPhysicsEngineParameter(fixedTimeStep=self.scene.timestep * 4,
                                          numSolverIterations=5,
                                          numSubSteps=4)
        return s


class Gripper2DSamplePoseEnv(MJCFBaseBulletEnv):
    def __init__(self, render=False, reward_type='contacts',
                 distance_threshold=DIST_THRESH,
                 orientation_threshold=ORI_THRESH, goal_range=GOAL_RANGE,
                 rew_min_clip=REW_MIN_CLIP,
                 reset_on_drop=False, steps_from_drop=20, goal_difficulty='easy',
                 drop_penalty=-500, ac_penalty=-10, ac_hist_len=0):
        self.set_goal_difficulty(goal_difficulty)
        self.robot = ThreeFingerRobot()
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.orientation_threshold = orientation_threshold
        self.rew_min_clip = rew_min_clip
        self.max_episode_steps = 200
        self.current_step = 0
        self.reset_on_drop = reset_on_drop
        self.steps_from_drop = steps_from_drop
        self.drop_penalty = drop_penalty / steps_from_drop
        self.ac_penalty = ac_penalty
        super().__init__(self.robot, render=render)
        self.ac_hist_len = ac_hist_len
        if ac_hist_len:
            ac_space_low = self.action_space.low
            ob_space_low = self.observation_space.low
            ob_ac_space = np.concatenate([ob_space_low] + 
                                         [ac_space_low]*ac_hist_steps)
            self.observation_space = spaces.Box(low=ob_ac_space,
                                                high=-ob_ac_space)
        self.info = dict(pos_success=False, ori_success=False, dropped=False,
                         is_success=False)
        self._cam_dist = 1
        self._render_width = 640
        self._render_height = 480

        self.done = False
        self.controller = DynamicCalculator()

        self.f1_file_path = os.path.join(os.path.dirname(__file__), "data",  "f1")
        self.f2_file_path = os.path.join(os.path.dirname(__file__), "data", "f2")
        self.f3_file_path = os.path.join(os.path.dirname(__file__), "data", "f3")
        if not os.path.exists(self.f1_file_path): raise ValueError("{} does not exist!".format(self.f1_file_path))
        if not os.path.exists(self.f2_file_path): raise ValueError("{} does not exist!".format(self.f2_file_path))
        if not os.path.exists(self.f3_file_path): raise ValueError("{} does not exist!".format(self.f3_file_path))

    @property
    def goal(self):
        return self.robot.goal
    
    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=0.0, timestep=0.0165, frame_skip=1)

    def step(self, a):
        if self.current_step == self.max_episode_steps or self.done:
            logger.warn('Gripper2DSamplePoseEnv.step: Reached end of episode, need to call env.reset()')
            state = self.robot.calc_state()
            if self.ac_hist_len:
                ac = self.robot.ac_hist[-self.ac_hist_len:]
                state = np.concatenate([state, ac])
            return state, 0, True, self.info
        self.current_step += 1
        self.done = self.current_step == self.max_episode_steps
        assert (not self.scene.multiplayer)
        if len(a.shape) == 2:
            a = a.squeeze()
        self.robot.apply_action(a)
        self.scene.global_step()
        state = self.robot.calc_state()
        if self.ac_hist_len:
            ac = self.robot.ac_hist[-self.ac_hist_len:]
            state = np.concatenate([state, ac])

        self.done, self.info = self.calc_done(self.robot.object_pos, self.goal)
        reward = self.compute_reward(self.robot.object_pos, self.goal)
        return state, reward, self.done, self.info

    def calc_done(self, achieved_goal, desired_goal):
        x_dist, y_dist, theta_dist = achieved_goal - desired_goal
        info = {}
        done = self.done  # does not change status of done
        info['pos_success'] = np.linalg.norm([x_dist, y_dist]) < self.distance_threshold
        info['ori_success'] = (np.abs(theta_dist) < self.orientation_threshold) 
        info['dropped'] = info['is_success'] = False
        if info['pos_success'] and info['ori_success']:
            done = True
            info['is_success'] = True
            info['dropped'] = False
        else:
            cp = self.robot.get_contact_pts() 
            if cp[0] is None or (cp[1] is None and cp[2] is None):
                if self.drop_step is None:
                    self.drop_step = self.current_step
                elif self.current_step - self.drop_step >= self.steps_from_drop:
                    done = True if self.reset_on_drop else done
                    info['dropped'] = True
                    info['is_success'] = False
            else:
                self.drop_step = None
        return done, info

    def compute_reward(self, achieved_goal, desired_goal, info={}):
        last_ac = self.robot.ac_hist[-1]
        x_dist, y_dist, theta_dist = desired_goal - achieved_goal
        reward = 0
        done, i = self.done, self.info 
        if i.get('dropped', False):
            reward = self.drop_penalty
        elif self.reward_type == 'sparse' or info.get('reward_type') == 'sparse':
            # calculate sparse reward
            if i.get('is_success', False):
                reward = 10
            else:
                reward = -1
        else:
            # calculate dense reward
            # referred from gym/robotics/manipulation
            if i['pos_success']:
                reward += 5
            else:
                reward -= 10 * (np.linalg.norm([x_dist, y_dist]))
            if i['ori_success']:
                reward += 3  # more reward for reaching orientation goal than pos goal
            else:
                reward -=  10 * np.abs(theta_dist)
            reward += self.ac_penalty * np.linalg.norm(last_ac)

        if self.reward_type == 'contacts':
            cp = self.robot.get_contact_pts()
            for c in self.robot.get_contact_pts():
                if c is None:
                    reward += self.rew_min_clip*2

        return reward

    def camera_adjust(self):
        pass

    def set_goal_difficulty(self, goal_difficulty):
        assert goal_difficulty in ['easy', 'med', 'hard'], ('Goal difficulty'
                ' must be easy, med, or hard')
        self._goal_difficulty = goal_difficulty

    def _sample_pose(self, file_path, mode, file_name=None):
        "Given a file_path, sample a valid object pose under a random contact configuration"
        file_list = os.listdir(file_path)
        while 1:
            file_name = file_name or self.np_random.choice(file_list)
            ds = self.contact_loc_parser(file_name)
            if ds is None: continue
            # init pose
            d1, d2, d3 = ds
            object_poses = np.load(os.path.join(file_path, file_name))
            for _ in range(5):
                object_pos = object_poses[self.np_random.randint(len(object_poses))]
                object_pos += np.array([self.np_random.uniform(-0.005, 0.005),
                                        self.np_random.uniform(-0.005, 0.005),
                                        self.np_random.uniform(-0.025, 0.025)])
                q = self.controller.compute_IK(object_pos, d1, d2, d3, mode=mode)
                if self._check_link_object_collision(q, object_pos[2], mode):
                    return object_pos, q, file_name
        return

    def _check_link_object_collision(self, q, theta, mode):
        """
        check whether finger link collide with
        """
        if q is None: return False
        if mode == 'f1':
            if q[0] + q[1] >= np.pi + theta: return False  # f2 link 2
            if q[2] + q[3] <= theta:         return False  # f2 link 2
            if q[4] + q[5] >= np.pi + theta: return False  # f3 link2
        elif mode == "f2":
            if q[0] + q[1] >= np.pi + theta: return False
            if q[2] + q[3] <= theta:         return False
            if q[4] + q[5] <= np.pi + theta: return False
        elif mode == "f3":
            if q[0] + q[1] >= np.pi + theta: return False
            if q[2] + q[3] >= theta:         return False
            if q[4] + q[5] >= np.pi + theta: return False
        return True

    @staticmethod
    def contact_loc_parser(file_name):
        """
        Given a file name (str), parse its contact locations.
        file name is like -0.10,0.04,0.14.npy
        """
        file_name = file_name.split('.npy')
        if len(file_name) < 2: return None  # invalid files like .DS_Store
        file_name = file_name[0]
        contact_loc = [float(di) for di in file_name.split(",")]
        return contact_loc
    
    def reset(self):
        """
        Overwrite MJCF reset method for initialize purpose. Hard, medium, easy have different init & goal generation method.
        Easy: init & goal are sampled from f1 with same contact location
        """
        # reset pybullet/robot params
        if (self.physicsClientId < 0):
            self.ownsPhysicsClient = True

            if self.isRender:
                self._p = bullet_client.BulletClient(connection_mode=p.GUI)
            else:
                self._p = bullet_client.BulletClient()

            self.physicsClientId = self._p._client
            self._p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        if self.scene is None:
            self.scene = self.create_single_player_scene(self._p)
        if not self.scene.multiplayer and self.ownsPhysicsClient:
            self.scene.episode_restart(self._p)

        self.robot.scene = self.scene

        self.frame = 0
        self.done = False
        self.current_step = 0
        self.info = dict(pos_success=False, ori_success=False, dropped=False, is_success=False)
        self.drop_step = None
        self.reward = 0
        dump = 0
        self.robot.reset(self._p)

        # --------------init & goal init --------------------
        # pick init and goal pose in f1, first pick a contact config, then sample two poses
        if self._goal_difficulty != 'hard':
            rng_set = [[0, 0], [1, 1], [2, 2]]
            rng_id = self.np_random.choice([0, 1, 2], p=[0, 0, 1])
        else:
            # rng_set = [[0,0], [1,1], [2,2], [0,1], [0,2]]
            # rng_set = [[0,0], [1,1], [2,2], [0,1], [0,2], [2,0], [1,0]]
            rng_set = [[0,1],[0,2], [0,0], [1,1], [2,2]]    # hard goal
            rng_id = self.np_random.choice([0, 1, 2, 3, 4]) # hard goal

        rng_init_goal_setting = rng_set[rng_id]
        file_paths = [self.f1_file_path, self.f2_file_path, self.f3_file_path]
        modes = ['f1', 'f2', 'f3']
        init_file_path, goal_file_path = (file_paths[rng_init_goal_setting[0]],
                                          file_paths[rng_init_goal_setting[1]])
        init_mode, goal_mode = (modes[rng_init_goal_setting[0]],
                                modes[rng_init_goal_setting[1]])
        self.mode = init_mode
        while 1:
            self.robot.object_pos, q, init_file = self._sample_pose(
                    init_file_path, init_mode)
            self.info_init_object_pos = self.robot.object_pos.copy()
            if self._goal_difficulty != 'easy': init_file = None 
            self.robot.target_pos, _, _ = self._sample_pose(
                    goal_file_path, goal_mode, init_file)
            self.info_target_pos = self.robot.target_pos.copy() 
            for i in range(len(self.robot.object_pos)):
                self.robot.object_pole_joints[i].set_state(self.robot.object_pos[i], 0)
                self.robot.target_pole_joints[i].set_state(self.robot.target_pos[i], 0)
            for i in range(len(q)):
                self.robot.manipulator_joints[i].set_state(q[i], 0)
            self._p.setJointMotorControlArray(0, jointIndices=[i for i in range(9)],
                                              controlMode=p.VELOCITY_CONTROL, forces=[0] * 9)
            self._p.stepSimulation()
            collision_1 = self._p.getContactPoints(bodyA=0, bodyB=0, linkIndexA=0, linkIndexB=8)
            collision_2 = self._p.getContactPoints(bodyA=0, bodyB=0, linkIndexA=2, linkIndexB=8)
            collision_3 = self._p.getContactPoints(bodyA=0, bodyB=0, linkIndexA=4, linkIndexB=8)
            if len(collision_1) > 0 or len(collision_2) > 0 or len(collision_3) > 0: continue
            d1, d2, d3 = self.robot.get_contact_pts()
            if None not in [d1, d2, d3]: break  # some init are invalid, re-sample pose
        s = self.robot.calc_state()
        self._p.setPhysicsEngineParameter(fixedTimeStep=self.scene.timestep * 4,
                                          numSolverIterations=5,
                                          numSubSteps=4)
        return s


class Gripper2DHardSamplePoseEnv(Gripper2DSamplePoseEnv):

    def reset(self):
        """
        Overwrite MJCF reset method for initialize purpose. Hard, medium, easy have different init & goal generation method.
        Hard: init are sampled from f1 and goal is f2 (currently)
        """

        def contact_loc_parser(file_name):
            """
            Given a file name (str), parse its contact locations.
            file name is like -0.10,0.04,0.14.npy
            """
            file_name = file_name.split('.npy')
            if len(file_name) < 2: return None  # invalid files like .DS_Store
            file_name = file_name[0]
            contact_loc = [float(di) for di in file_name.split(",")]
            return contact_loc

        def sample_pose(file_path, mode):
            "Given a file_path, sample a valid object pose under a random contact configuration"
            file_list = os.listdir(file_path)
            while 1:
                file_name = self.np_random.choice(file_list)
                ds = contact_loc_parser(file_name)
                if ds is None: continue
                # init pose
                d1, d2, d3 = ds
                object_poses = np.load(os.path.join(file_path, file_name))
                for _ in range(5):
                    object_pos = object_poses[self.np_random.randint(len(object_poses))]
                    object_pos += np.array([self.np_random.uniform(-0.005, 0.005),
                                            self.np_random.uniform(-0.005, 0.005),
                                            self.np_random.uniform(-0.025, 0.025)])
                    q = self.controller.compute_IK(object_pos, d1, d2, d3, mode=mode)
                    if q is not None:
                        if mode == 'f1':
                            if q[4] + q[5] >= np.pi + object_pos[2] - 0.1: break  # f3 link2
                            if q[2] + q[3] - 0.1 <= object_pos[2]: break  # f2 link 2
                        elif mode == 'f2':
                            if (q[4] + q[5] - 0.1 <= (np.pi + object_pos[2])): break  # f3 link 2 collide with object
                            if (q[0] + q[1] >= (np.pi + object_pos[2] - 0.1)): break  # f1 link 2 collide with object
                        elif mode == 'f3':
                            if q[0] + q[1] >= np.pi + object_pos[2]: continue
                            if q[2] + q[3] >= object_pos[2]:         continue
                            if q[4] + q[5] >= np.pi + object_pos[2]: continue
                        return object_pos, q
            return

        if (self.physicsClientId < 0):
            self.ownsPhysicsClient = True

            if self.isRender:
                self._p = bullet_client.BulletClient(connection_mode=p.GUI)
            else:
                self._p = bullet_client.BulletClient()

            self.physicsClientId = self._p._client
            self._p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        if self.scene is None:
            self.scene = self.create_single_player_scene(self._p)
        if not self.scene.multiplayer and self.ownsPhysicsClient:
            self.scene.episode_restart(self._p)

        self.robot.scene = self.scene

        self.frame = 0
        self.done = 0
        self.reward = 0
        dump = 0
        self.robot.reset(self._p)

        # --------------init & goal init --------------------
        # rng_init_goal_setting = self.np_random.permutation([0,1])
        # rng_set = [[0,0], [1,1], [2,2], [0,1], [0,2]]
        # rng_set = [[0,0], [1,1], [2,2], [0,1], [0,2], [2,0], [1,0]]
        rng_set = [[0,1],[0,2], [0,0], [1,1], [2,2]]    # hard goal
        rng_id = self.np_random.choice([0, 1, 2, 3, 4]) # hard goal
        # rng_set = [[0,0], [1,1], [2,2]]    # medium goal
        # rng_id = self.np_random.choice([0, 1, 2]) # medium goal
        rng_init_goal_setting = rng_set[rng_id]
        file_paths, modes = [self.f1_file_path, self.f2_file_path, self.f3_file_path], ['f1', 'f2', 'f3']
        init_file_path, goal_file_path = file_paths[rng_init_goal_setting[0]], file_paths[rng_init_goal_setting[1]]
        init_mode, goal_mode = modes[rng_init_goal_setting[0]], modes[rng_init_goal_setting[1]]
        self.mode = init_mode
        while 1:
            self.info_init_object_pos, _ = self.robot.object_pos, q = sample_pose(init_file_path, init_mode)
            self.info_target_pos, _      = self.robot.target_pos, _ = sample_pose(goal_file_path, goal_mode)
            for i in range(len(self.robot.object_pos)):
                self.robot.object_pole_joints[i].set_state(self.robot.object_pos[i], 0)
                self.robot.target_pole_joints[i].set_state(self.robot.target_pos[i], 0)
            for i in range(len(q)):
                self.robot.manipulator_joints[i].set_state(q[i], 0)
            self._p.setJointMotorControlArray(0, jointIndices=[i for i in range(9)],
                                              controlMode=p.VELOCITY_CONTROL, forces=[0] * 9)
            self._p.stepSimulation()
            collision_1 = self._p.getContactPoints(bodyA=0, bodyB=0, linkIndexA=0, linkIndexB=8)
            collision_2 = self._p.getContactPoints(bodyA=0, bodyB=0, linkIndexA=2, linkIndexB=8)
            collision_3 = self._p.getContactPoints(bodyA=0, bodyB=0, linkIndexA=4, linkIndexB=8)
            if len(collision_1) > 0 or len(collision_2) > 0 or len(collision_3) > 0: continue
            d1, d2, d3 = self.robot.get_contact_pts()
            if None not in [d1,d2,d3]: break    # some init are invalid, re-sample pose
        s = self.robot.calc_state()
        self._p.setPhysicsEngineParameter(fixedTimeStep=self.scene.timestep * 4,
                                          numSolverIterations=5,
                                          numSubSteps=4)
        return s


class Gripper2DGoalEnv(GoalEnv):
    def __init__(self, render=False, reward_type='sparse', distance_threshold=DIST_THRESH,
                 goal_range=GOAL_RANGE, orientation_threshold=ORI_THRESH, reset_on_drop=False,
                 **env_kwargs):
        self.env = Gripper2DSamplePoseEnv(
                render, reward_type, distance_threshold,
                orientation_threshold=ORI_THRESH, goal_range=GOAL_RANGE,
                reset_on_drop=reset_on_drop, **env_kwargs)
        self.robot = self.env.robot
        goal_low = -np.array(goal_range)
        goal_hi = -goal_low
        goal_space = spaces.Box(goal_low, goal_hi)
        observation_space = self.env.observation_space
        self.observation_space = spaces.Dict({'observation': observation_space,
                                              'achieved_goal': goal_space,
                                              'desired_goal': goal_space})
        self.action_space = self.env.action_space
        self.reset_on_drop = reset_on_drop

    def step(self, action):
        obs, r, d, i = self.env.step(action)
        obs_dict = dict(observation=obs, achieved_goal=self.achieved_goal, desired_goal=self.goal)
        r = self.compute_reward(self.achieved_goal, self.goal, {'reward_type': 'sparse'})
        r = int(r)
        return obs_dict, r, d, i

    def reset(self):
        obs = self.env.reset()
        obs_dict = dict(observation=obs, achieved_goal=self.achieved_goal, desired_goal=self.goal)
        return obs_dict

    def compute_reward(self, achieved_goal, desired_goal, info={}):
        return self.env.compute_reward(achieved_goal, desired_goal, info)

    @property
    def achieved_goal(self):
        return self.robot.object_pos

    @property
    def goal(self):
        return self.env.goal


class Gripper2DPrimContEnv(MJCFBaseBulletEnv):
    def __init__(self, render=False, repose_duration=REPOSE_DUR, slide_duration=SLIDE_DUR,
                 delta_repose=DELTA_REPOSE2x, delta_slide=DELTA_SLIDE2x,
                 reward_type='dense', controller_type='pd', distance_threshold=DIST_THRESH,
                 orientation_threshold=ORI_THRESH, rew_min_clip=REW_MIN_CLIP):
        self.robot = ThreeFingerRobot()
        self.controller = DynamicCalculator()
        self.controller_type = controller_type
        self.repose_duration = repose_duration
        self.slide_duration = slide_duration
        self.reward_type = reward_type
        self.delta_repose = delta_repose
        self.delta_slide = delta_slide
        self.distance_threshold = distance_threshold
        self.orientation_threshold = orientation_threshold
        self.rew_min_clip = rew_min_clip
        self.max_episode_steps = 200
        self.current_step = 0
        self.done = False
        super().__init__(self.robot, render=render)
        # actions are 3D for repose and slide actions
        slide_ac_space = spaces.Box(-np.array(self.delta_slide), np.array(self.delta_slide))
        repose_ac_space = spaces.Box(-np.array(self.delta_repose), np.array(self.delta_repose))
        self.action_space = spaces.Dict({'repose': repose_ac_space,
                                         'slide': slide_ac_space})
        self._cam_dist = 1
        self._render_width = 640
        self._render_height = 480

    @property
    def goal(self):
        return self.robot.goal

    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=0.0, timestep=1./240., frame_skip=1)

    def reset(self):
        self.current_step = 0
        return super().reset()

    def step(self, a):
        assert (not self.scene.multiplayer)
        if self.done:
            return self.robot.calc_state(), 0, self.done, {'ac_success': False}
        a, ac_type = a
        self.current_step += 1
        # a: operational space
        try:
            if ac_type == 'repose':
                ac_success = self._repose(a)
            elif ac_type == 'slide':
                ac_success = self._slide(a)
            else:
                raise AssertionError('ac_type must be repose or slide')
        except Exception as e:
            raise Exception('Error: {}, {} action was {}'.format(e, ac_type, a))

        self.scene.global_step()
        state = self.robot.calc_state()

        info = {'ac_success': ac_success}

        self.done = self.calc_done(self.robot.object_pos, self.robot._target_pos, info)
        reward = self.compute_reward(self.robot.object_pos, self.robot._target_pos, info)
        # TODO: Penalize action with negative reward if ac_success is false?
        if ac_success:
            reward = np.where(reward < self.rew_min_clip, self.rew_min_clip, reward)
        return state, reward, self.done, info

    def calc_done(self, achieved_goal, desired_goal, info):
        x_dist, y_dist, theta_dist = desired_goal - achieved_goal
        return (self.done or self.current_step >= self.max_episode_steps or
                not info.get('ac_success') or
                ((np.linalg.norm([x_dist, y_dist]) < self.distance_threshold) and
                 (np.abs(theta_dist) < self.orientation_threshold)))

    def compute_reward(self, achieved_goal, desired_goal, info={}):
        if info.get('ac_success', None) is False:
            return self.rew_min_clip
        x_dist, y_dist, theta_dist = desired_goal - achieved_goal
        if self.reward_type == 'dense':
            reward = -(10 * np.linalg.norm([x_dist, y_dist]) + np.abs(theta_dist))  # referred from gym/robotics/manipulation
        else:
            reward = ((np.linalg.norm([x_dist, y_dist]) < self.distance_threshold) and
                      (np.abs(theta_dist) < self.orientation_threshold))
            reward = float(reward)
        return reward

    def _repose(self, delta_des, mode='f1'):
        _, _, x, _ = self.robot.get_state()
        x_des = delta_des + x
        self.robot.target_local_pole_joints[0].set_state(x_des[0], 0)
        self.robot.target_local_pole_joints[1].set_state(x_des[1], 0)
        self.robot.target_local_pole_joints[2].set_state(x_des[2], 0)

        for i in range(self.repose_duration):
            d1, d2, d3 = self.robot.get_contact_pts()
            q, q_dot, x, x_dot = self.robot.get_state()
            if (d1 is None) or (d2 is None) or (d3 is None):  # lose contact
                # self.scene.global_step()
                self._p.setJointMotorControlArray(0, jointIndices=[0, 1, 2, 3, 4, 5],
                                                  controlMode=p.VELOCITY_CONTROL, forces=[0] * 6)
                return False
            # Inverse Dynamic controller (operational space)
            tau_repose = self.controller.inverse_dynamic_controller_operational_space(mode=mode,
                                                                                      pose_des=x_des,
                                                                                      pose_cur=x,
                                                                                      pose_cur_dot=x_dot,
                                                                                      q=q,
                                                                                      qdot=q_dot,
                                                                                      d1=d1, d2=d2, d3=d3)
            if tau_repose is None:  # target pose is out of manipulator workspace
                # self.scene.global_step()
                self._p.setJointMotorControlArray(0, jointIndices=[0, 1, 2, 3, 4, 5],
                                                  controlMode=p.VELOCITY_CONTROL, forces=[0] * 6)
                return False
            self.robot.apply_action(tau_repose, clip=False)
            self.scene.global_step()
        return True

    def _slide(self, delta_slide, mode='f1'):
        self._p.setTimeStep(1 / 120.)
        finger, = delta_slide.nonzero()
        assert len(finger) == 1, 'only accepts sliding action on one finger, received action {delta_slide}'
        finger = finger.item()
        if (finger == 0 and mode == 'f1') or \
           (finger == 1 and mode == 'f2') or \
           (finger == 2 and mode == 'f3'):
            move_joints = [finger*2, finger*2 + 1]    # finger 1 need fix other two fingers
            fix_joints = list(set(range(6)) - set(move_joints))
            self._p.setJointMotorControlArray(0, jointIndices=fix_joints,
                                              controlMode=p.VELOCITY_CONTROL, forces=[100]*4)

        q_des, _, x_des, _ = self.robot.get_state()
        d = self.robot.get_contact_pts()[finger]
        if d is None:
            self._p.setJointMotorControlArray(0, jointIndices=[0, 1, 2, 3, 4, 5],
                                              controlMode=p.VELOCITY_CONTROL, forces=[0] * 6)
            return False
        d_des = delta_slide[finger] + d
        ddes = np.linspace(d, d_des, self.slide_duration)

        # set indicator position
        self.robot.target_local_pole_joints[0].set_state(x_des[0], 0)
        self.robot.target_local_pole_joints[1].set_state(x_des[1], 0)
        self.robot.target_local_pole_joints[2].set_state(x_des[2], 0)
        self.robot.target_contact_point_joints[finger].set_state(d_des, 0)

        for i in range(self.slide_duration):
            d1, d2, d3 = self.robot.get_contact_pts()
            q, q_dot, x, x_dot = self.robot.get_state()
            if (d1 is None) or (d2 is None) or (d3 is None):  # lose contact
                return False
            tau_slide = self.controller.slide(mode=mode, finger=finger, pose_des=x_des, pose_cur=x, pose_dot=x_dot,
                                              q_des=q_des, q=q, qdot=q_dot, d1=d1, d2=d2, d3=d3, d_des=ddes[i])
            if tau_slide is None:
                self._p.setJointMotorControlArray(0, jointIndices=[0, 1, 2, 3, 4, 5],
                                                  controlMode=p.VELOCITY_CONTROL, forces=[0] * 6)
                return False
            self.robot.apply_action(tau_slide, clip=False)
            self.scene.global_step()

        if (finger == 0 and mode == 'f1') or \
           (finger == 1 and mode == 'f2') or \
           (finger == 2 and mode == 'f3'):
            self._p.setJointMotorControlArray(0, jointIndices=[0, 1, 2, 3, 4, 5],
                                              controlMode=p.VELOCITY_CONTROL, forces=[0]*6)
        return True

    def camera_adjust(self):
        pass


class SafetyGymWrapper(gym.Wrapper):
    def __init__(self, env, fall_cost=1.):
        super(SafetyGymWrapper, self).__init__(env)
        self._fall_cost = fall_cost
        obs_space = self.env.observation_space['observation']
        goal_space = self.env.observation_space['desired_goal']
        obs_space = spaces.Box(np.concatenate([obs_space.low, goal_space.low]),
                               np.concatenate([obs_space.high, goal_space.high]))
        self.observation_space = obs_space

    def reset(self):
        o = super(SafetyGymWrapper, self).reset()
        return np.concatenate([o['observation'], o['desired_goal']])


    def step(self, action):
        o, r, d, i = super(SafetyGymWrapper, self).step(action)
        x_dist, y_dist, theta_dist = o['desired_goal'] - o['achieved_goal']

        i.update({'cost': float(i.get('dropped', False)) * self._fall_cost,
                  'pos_dist': float(np.linalg.norm([x_dist, y_dist])),
                  'ori_dist': theta_dist})
        return np.concatenate([o['observation'], o['desired_goal']]), r, d, i
