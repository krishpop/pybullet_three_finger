
"""
3 Finger Robot Base Environment.
"""
import os, logging, gym, math, time, cv2
import numpy as np
import pybullet as p
from pybullet_utils import bullet_client

from gym import spaces
from gym import GoalEnv
from gym.utils import seeding
from pybullet_utils import bullet_client

from pybullet_envs.robot_bases import XmlBasedRobot
from pybullet_envs.env_bases import MJCFBaseBulletEnv
from pybullet_envs.scene_abstract import SingleRobotEmptyScene

logger = logging.getLogger(__name__)
DEBUG = False

class URDFBasedRobot(XmlBasedRobot):
    """
    Base class for URDF .xml based robots.
    """

    def __init__(self, model_urdf, robot_name, action_dim, obs_dim, basePosition=[0, 0, 0],
                 baseOrientation=[0, 0, 0, 1], fixed_base=False, self_collision=False):
        XmlBasedRobot.__init__(self, robot_name, action_dim, obs_dim, self_collision)
        self.model_urdf = model_urdf
        self.basePosition = basePosition
        self.baseOrientation = baseOrientation
        self.fixed_base = fixed_base
        self.loaded = False
        self.potential = None

    def reset(self, bullet_client=None):
        self._p = bullet_client
        self.ordered_joints = []
        urdf_path = os.path.join(os.path.dirname(__file__), "assets", self.model_urdf)
        if not self.loaded:
            flags = None if not self.self_collision else p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE
            self.parts, self.jdict, self.ordered_joints, self.robot_body = (
                self.addToScene(self._p,
                                self._p.loadURDF(
                                    urdf_path,
                                    basePosition=self.basePosition,
                                    baseOrientation=self.baseOrientation,
                                    useFixedBase=self.fixed_base,
                                    flags=flags))
            )
            self.loaded = True
        self.robot_specific_reset(self._p)
        # optimization: calc_state() can calculate something in self.* for calc_potential() to use
        s = self.calc_state()
        self.potential = self.calc_potential()
        return s

    def robot_specific_reset(self, bullet_client):
        raise NotImplementedError

    def calc_state(self):
        raise NotImplementedError

    def calc_potential(self):
        return 0


class ThreeFingerRobot(URDFBasedRobot):
    def __init__(self, urdf_name="three_fingers_pole_indicator.urdf", measurement_err=False):
        URDFBasedRobot.__init__(self, urdf_name, "Tinguang's Robot",
                                action_dim=6, obs_dim=18, fixed_base=True, self_collision=True)
        obs_low = np.array([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi,     # robot joint pos
                            -5, -5, -5, -5, -5, -5,                             # robot joint vel
                            -np.inf, -np.inf, -np.pi,                           # object current pos
                            -np.inf, -np.inf, -np.pi])                          # object goal pos
        obs_hi = -obs_low
        self.observation_space = spaces.Box(obs_low, obs_hi)
        self.target_pos = None
        self.measurement_err = measurement_err
        self._pos_err = self.sample_noise()
        self.ac_hist_len = 20
        self.ac_hist = np.zeros((self.ac_hist_len, 6))

    @property
    def goal(self):
        return self.target_pos

    def robot_specific_reset(self, bullet_client):
        # init_file_path = os.path.join(os.path.dirname(__file__), 'bullet','three_fingers_pole_indicator.bullet')
        # bullet_client.restoreState(fileName=init_file_path)
        self.manipulator_joints = [self.jdict['1_base_joint'], self.jdict['1_central_joint'],
                                   self.jdict['2_base_joint'], self.jdict['2_central_joint'],
                                   self.jdict['3_base_joint'], self.jdict['3_central_joint']]
        self.target_pole_joints = [self.jdict['target_base_pole_x'], self.jdict['target_base_pole_y'], self.jdict['target_base_pole_theta']]
        self.target_local_pole_joints = [self.jdict['target_local_base_pole_x'], self.jdict['target_local_base_pole_y'], self.jdict['target_local_base_pole_theta']]
        self.target_contact_point_joints = [self.jdict['target_contact_local_point_f1_joint'], self.jdict['target_contact_local_point_f2_joint'], self.jdict['target_contact_local_point_f3_joint']]
        self.object_pole_joints = [self.jdict['object_base_pole_x'], self.jdict['object_base_pole_y'], self.jdict['object_base_pole_theta']]
        self.proximal_links = [self.parts['1_proximal_link'], self.parts['2_proximal_link'], self.parts['3_proximal_link']]
        self.distal_links = [self.parts['1_distal_link'], self.parts['2_distal_link'], self.parts['3_distal_link']]
        self.target = self.parts['target_pole']
        self.target_local = self.parts['target_local_pole']
        self.target_local_contact_point_links = [self.parts['target_contact_local_point_f1'], self.parts['target_contact_local_point_f2'], self.parts['target_contact_local_point_f3']]
        self.object = self.parts['object_pole']
        self.object_pos = self.target_pos = np.array([0,0,0])

        self._p.setJointMotorControlArray(0, jointIndices=list(range(9)),
                                          controlMode=p.VELOCITY_CONTROL, forces=[100] * 9)
        self._p.stepSimulation()
        self._p.setJointMotorControlArray(0, jointIndices=list(range(9)),
                                          controlMode=p.VELOCITY_CONTROL, forces=[0] * 9)
        self.ac_hist = np.zeros((self.ac_hist_len, 6))

    def apply_action(self, a, clip=True):
        # action: torques on 1_base_joint, 1_central_joint, 2_base_joint..., 3_centeral_joint
        self.ac_hist = np.roll(self.ac_hist, -1, 0)
        self.ac_hist[-1,:] = a  # append current ac to ac_hist
        ac_window = self.ac_hist[~np.all(self.ac_hist == 0, axis=-1)]
        a = ac_window.mean(axis=0)
        for i in range(len(self.manipulator_joints)):
            if clip:
                self.manipulator_joints[i].set_motor_torque(0.05 * float(np.clip(a[i], -1, +1)))
            else:
                self.manipulator_joints[i].set_motor_torque(a[i])

    def calc_state(self):
        self.object_pos = np.array([
            self.object_pole_joints[0].get_position(),      # range (-np.inf, np.inf)
            self.object_pole_joints[1].get_position(),      # range (-np.inf, np,inf)
            self.object_pole_joints[2].get_position()      # range [-np.pi, np.pi]
        ])
        if self.measurement_err:
            self.object_pos += self._pos_err
        return np.concatenate([
            np.array([j.get_position() for j in self.manipulator_joints]).flatten(),  # all positions
            np.array([j.get_velocity() for j in self.manipulator_joints]).flatten(),  # all speeds
            self.object_pos,
            self.target_pos,
        ])

    def calc_potential(self):
        pass

    def get_contact_pts(self):
        """
        Return the contact locations (distance to the center of the object)
        """
        robotID = 0
        C1 = self._p.getContactPoints(bodyA=robotID, bodyB=robotID, linkIndexA=1, linkIndexB=8)
        C2 = self._p.getContactPoints(bodyA=robotID, bodyB=robotID, linkIndexA=3, linkIndexB=8)
        C3 = self._p.getContactPoints(bodyA=robotID, bodyB=robotID, linkIndexA=5, linkIndexB=8)

        ds = [None, None, None]
        mass_center = self._p.getLinkState(robotID, 8)[0]
        for i, ci in enumerate([C1, C2, C3]):
            if len(ci) == 0: continue
            ci = ci[0][5]
            ds[i] = np.sign(ci[0] - mass_center[0]) * np.sqrt((ci[0] - mass_center[0]) ** 2 + (ci[1] - mass_center[1]) ** 2)
        if DEBUG:
            if None in ds: print(ds)
        return np.array(ds)

    def get_state(self):
        """
        Return joint position, vel, object position, vel. Used for low-level controller, not for training networks.
        """
        q = np.array([j.get_position() for j in self.manipulator_joints])
        qdot = np.array([j.get_velocity() for j in self.manipulator_joints])
        x = np.array([j.get_position() for j in self.object_pole_joints])
        if self.measurement_err:
            x += self._pos_err
            self._pos_err = self.sample_noise()
        xdot = np.array([j.get_velocity() for j in self.object_pole_joints])
        return q, qdot, x, xdot

    def sample_noise(self):
        return np.random.uniform(-1,1,size=3)*np.array([.005,.005,.05])

