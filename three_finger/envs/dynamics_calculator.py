"""
Dynamic Calulator class for three fingers manipulator, including dynamic equation and
multiple controller.
"""
import numpy as np

class DynamicCalculator(object):
    def __init__(self):
        """
        Init parameters including link mass, link length, link width ...
        This part should be consistent with urdf file.
        """
        self.l1, self.l2, self.l = 0.15, 0.1, 0.5       # link length and target length
        self.m1, self.m2, self.mass = 0.5, 0.5, 0.5     # link mass and target mass
        self.w_link, self.t_link = 0.02, 0.005          # link width, link thickness
        self.w_target = 0.02                            # target width
        self.arena_w, self.arena_h = 0.6, 0.4           # width and height of arena
        self.finger1_x, self.finger2_x, self.finger3_x = 0, -0.1, 0.1   # x coordinates of three fingers

        # PD controller
        kp, kv = 5,1
        self.Kp_PD = np.array([
        	[kp, 0, 0, 0, 0, 0],
            [0, kp, 0, 0, 0, 0],
            [0, 0, kp, 0, 0, 0],
            [0, 0, 0, kp, 0, 0],
            [0, 0, 0, 0, kp, 0],
            [0, 0, 0, 0, 0, kp]
        ])
        self.Kv_PD = np.array([
        	[kv, 0, 0, 0, 0, 0],
            [0, kv, 0, 0, 0, 0],
            [0, 0, kv, 0, 0, 0],
            [0, 0, 0, kv, 0, 0],
            [0, 0, 0, 0, kv, 0],
            [0, 0, 0, 0, 0, kv]
        ])

        # Inverse Dynamic Controller
        kp, kv = 200, 50
        self.Kp_ID_joint_space = np.array([
        	[kp, 0, 0, 0, 0, 0],
            [0, kp, 0, 0, 0, 0],
            [0, 0, kp, 0, 0, 0],
            [0, 0, 0, kp, 0, 0],
            [0, 0, 0, 0, kp, 0],
            [0, 0, 0, 0, 0, kp]
        ])
        self.Kv_ID_joint_space = np.array([
        	[kv, 0, 0, 0, 0, 0],
            [0, kv, 0, 0, 0, 0],
            [0, 0, kv, 0, 0, 0],
            [0, 0, 0, kv, 0, 0],
            [0, 0, 0, 0, kv, 0],
            [0, 0, 0, 0, 0, kv]
        ])

        kp, kv = 50, 30
        self.Kp_ID_operation_space = np.array([
            [kp, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, kp, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, kp, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, kp, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, kp, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, kp, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, kp, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, kp, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, kp]
        ])
        self.Kv_ID_operation_space = np.array([
            [kv, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, kv, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, kv, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, kv, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, kv, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, kv, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, kv, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, kv, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, kv]
        ])

        kp, kv = 30, 30
        self.Kp_sliding = np.array([
            [kp, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, kp * 5, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, kp, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, kp, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, kp * 5, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, kp, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, kp, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, kp * 5, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, kp]
            ])
        self.Kv_sliding = np.array([
            [kv, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, kv / 2, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, kv, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, kv, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, kv / 2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, kv, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, kv, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, kv / 2, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, kv]
        ])

    def compute_forward_kinematics(self, q):
        def compute_FK(q1, q2):
            # in local frame
            x = self.l1 * np.cos(q1) + self.l2 * np.cos(q1 + q2)
            y = self.l1 * np.sin(q1) + self.l2 * np.sin(q1 + q2)
            return [x, y]

        assert len(q) == 6
        x1, y1 = compute_FK(q[0], q[1])
        x2, y2 = compute_FK(q[2], q[3])
        x3, y3 = compute_FK(q[4], q[5])

        # local frame to world frame
        x1, y1 = -x1, self.arena_h/2. - y1
        x2, y2 = x2 + self.finger2_x, y2 - self.arena_h/2.
        x3, y3 = x3 + self.finger3_x, y3 - self.arena_h/2.

        return [x1, y1], [x2, y2], [x3, y3]


    def compute_dynamic_tau_joint_space(self, q, qdot, qddot):
        """
        Given q, qdot, qddot, compute torque according to dynamic equation,
        igonring gravity term.
        """
        assert len(q) == 6 and len(qdot) == 6 and len(qddot) == 6

        M, V= self._get_mass_matrix(q, qdot)
        tau = np.dot(M, qddot) + V
        return tau

    def compute_pole_tau(
        self, pddot
    ):
        """
        Given pole pose (x,y,z), compute the force needed to
        follow its trajectory
        """
        assert len(pddot) == 3

        M = np.array([
            [self.mass, 0, 0],
            [0, self.mass, 0],
            [0, 0, 1 / 12. * self.mass * (self.w_target ** 2 + self.l ** 2)]
        ])

        tau_app = np.dot(M, pddot)
        return tau_app

    def compute_IK(self, pose, d1, d2, d3, mode='f1'):
        """
        Given pole pose, compute joint configurations.
        flag can be f1, f2, f3, representing f1 on the top alone, f2 / f3 on the bottom alone.
        """
        def compute_IK_one_finger(p, theta, flag=1):
            """
            Return joint confituration given the tip position for a 2-link joint robot.
            p should be in local frame, the orientation is ignored (underactuatued).
            """
            # assert len(p) == 2 and p[1] >= 0

            # Considering the width, need modification
            p = [p[0] + self.w_target / 2.0 * np.sin(theta), p[1] - self.w_target / 2.0 * np.cos(theta)]
            l2 = self.l2 - self.w_target/2.0

            q2 = flag * np.arccos((p[0] ** 2 + p[1] ** 2 - self.l1 ** 2 - l2 ** 2) / (2 * self.l1 * l2))
            t1 = np.arctan(p[1] / (p[0] + 0.0001))
            t2 = np.arctan(
                l2 * np.sin(q2) / (self.l1 + l2 * np.cos(q2)))
            q1 = np.arctan(p[1] / (p[0] + 0.0001)) - np.arctan(
                l2 * np.sin(q2) / (self.l1 + l2 * np.cos(q2)))
            q1 = q1 + np.pi if q1 < 0 else q1
            # assert not (np.isnan(q1) or np.isnan(q2)), "{} is out of the workspace".format(p)
            if np.isnan(q1) or np.isnan(q2):
                return None
            # if flag == -1:
            #     q1 += np.pi
            return [q1, q2]

        assert len(pose) == 3
        x, y, theta = pose

        # Compute the position of contact points in world frame
        c1, c2, c3 = self._get_contact_points(pose, d1, d2, d3, mode)

        # Transform world frame to local frame
        c1 = [-c1[0] + self.finger1_x, self.arena_h / 2.0 - c1[1]]
        c2 = [ c2[0] - self.finger2_x, self.arena_h / 2.0 + c2[1]]
        c3 = [ c3[0] - self.finger3_x, self.arena_h / 2.0 + c3[1]]

        # Compute joint angles for each finger
        q1 = compute_IK_one_finger(c1, pose[2])
        q2 = compute_IK_one_finger(c2, pose[2], -1)
        q3 = compute_IK_one_finger(c3, pose[2])

        # if out of workspace, return none
        if q1 is None or q2 is None or q3 is None:
            return None

        return np.array([q1[0], q1[1], q2[0], q2[1], q3[0], q3[1]])

    def PD_controllor(self, pose_des, pose_cur, q, qdot, d1, d2, d3):
        """
        Given a desired pole pose (x,y,theta), calculate applied torque
        """
        # calculate target joint angles given pole pose
        q_des = self.compute_IK(pose_des, d1, d2, d3)
        # contact force to balance the target
        balance_force = 5 * self._get_grasp_matrix_null_space_forces(d1, d2, d3)
        # compute torques to generate balance force
        J = self._get_jacobian(q, pose_cur[2])
        balance_torque = np.dot(J.T, balance_force)

        v = np.dot(self.Kp_PD, (np.array(q_des) - np.array(q))) - np.dot(self.Kv_PD, qdot)
        safe_value = 0.03
        for i in range(len(v)):
            if np.abs(v[i]) > safe_value:
                v[i] = np.sign(v[i]) * safe_value

        tau = v + balance_torque
        return tau

    def inverse_dynamic_controllor_joint_space(self, pose_des, pose_cur, q, qdot, d1, d2, d3):
        """
        Given a desired pole pose (x,y,theta), calculate applied torque
        """
        # calculate target joint angles given pole pose
        q_des = self.compute_IK(pose_des, d1, d2, d3)
        # contact force to balance the target
        balance_force = 5 * self._get_grasp_matrix_null_space_forces(d1, d2, d3)
        # compute torques to generate balance force
        J = self._get_jacobian(q, pose_cur[2])
        balance_torque = np.dot(J.T, balance_force)

        M, V = self._get_mass_matrix(q, qdot)
        v = np.dot(self.Kp_ID_joint_space, (np.array(q_des) - np.array(q))) - np.dot(self.Kv_ID_joint_space, qdot)

        tau = np.dot(M, v) + V + balance_torque
        return tau

    def inverse_dynamic_controller_operational_space(self, pose_des, pose_cur, pose_cur_dot, q, qdot, d1, d2, d3, mode='f1'):
        # contact force to balance the target
        balance_force = 5 * self._get_grasp_matrix_null_space_forces(d1, d2, d3, mode)
        # compute torques to generate balance force
        J = self._get_jacobian(q, pose_cur[2], mode)
        balance_torque = np.dot(J.T, balance_force)

        # Operation Space
        M, V = self._get_mass_matrix(q, qdot)
        # contract frames
        c1, c2, c3 = self._get_contact_points(pose_cur, d1, d2, d3, mode)
        c1_des, c2_des, c3_des = self._get_contact_points(pose_des, d1, d2, d3, mode)
        if mode == 'f1':
            x_cur = [-c1[1], c1[0], pose_cur[2], c2[1], -c2[0], pose_cur[2], c3[1], -c3[0], pose_cur[2]]
            x_des = [-c1_des[1], c1_des[0], pose_des[2], c2_des[1], -c2_des[0], pose_des[2], c3_des[1], -c3_des[0], pose_des[2]]
        elif mode == 'f2':
            x_cur = [-c1[1], c1[0], pose_cur[2], c2[1], -c2[0], pose_cur[2], -c3[1], c3[0], pose_cur[2]]
            x_des = [-c1_des[1], c1_des[0], pose_des[2], c2_des[1], -c2_des[0], pose_des[2], -c3_des[1], c3_des[0], pose_des[2]]
        elif mode == 'f3':
            x_cur = [-c1[1], c1[0], pose_cur[2], -c2[1], c2[0], pose_cur[2], c3[1], -c3[0], pose_cur[2]]
            x_des = [-c1_des[1], c1_des[0], pose_des[2], -c2_des[1], c2_des[0], pose_des[2], c3_des[1], -c3_des[0], pose_des[2]]
        x_vel_cur = np.dot(J, qdot)

        v = np.dot(self.Kp_ID_operation_space, (np.array(x_des) - np.array(x_cur))) - \
            np.dot(self.Kv_ID_operation_space, x_vel_cur)
        Jdot = self._get_jacobian_derivative(q, qdot, pose_cur[2], pose_cur_dot[2], mode)
        tau = np.dot(M, np.dot(np.linalg.pinv(J), v)) + \
              V - np.dot(M, np.dot(np.linalg.pinv(J), np.dot(Jdot, qdot))) + \
              balance_torque
        return tau

    def inverse_dynamic_controller_operational_space_2_fingers(self, pose_cur, pose_cur_dot, q, qdot, balance_force=[0.1,0,0,0.1,0,0,0,0,0]):
        # d1, d2, d3 only two of them are meaningful
        J = self._get_jacobian(q, pose_cur[2])
        balance_torque = np.dot(J.T, balance_force)   # a very small force apply on 2 contacting fingers, no force on flipping finger

        # Operation Space
        M, V = self._get_mass_matrix(q, qdot)
        # contract frames
        x_vel_cur = np.dot(J, qdot)

        kp = 50
        KP = np.diag([kp]*9)

        v = - np.dot(KP, x_vel_cur)
        Jdot = self._get_jacobian_derivative(q, qdot, pose_cur[2], pose_cur_dot[2])
        tau = np.dot(M, np.dot(np.linalg.pinv(J), v)) + \
              V - np.dot(M, np.dot(np.linalg.pinv(J), np.dot(Jdot, qdot))) + \
              balance_torque
        return tau

    def slide_finger_left_in_two(self, pose_des, pose_cur, pose_cur_dot, q_des, q, qdot, d1, d2, d3, d_des, mode='f1'):
        """
        Slide the left finger among the two fingers on the same side.
        """
        # contact force to balance the target
        balance_force = 1 * self._get_grasp_matrix_null_space_forces(d1, d2, d3, mode)
        # compute torques to generate balance force
        J = self._get_jacobian(q, pose_cur[2], mode)
        balance_torque = np.dot(J.T, balance_force)
        #----------------old version, error in joint space------------------------
        # # PD controller
        # if mode == "f1":
        #     q_des_cal = self.compute_IK(pose_des, d1, d_des, d3, mode)
        #     q_des[2:4] = q_des_cal[2:4]
        # elif mode == "f2":
        #     q_des_cal = self.compute_IK(pose_des, d_des, d2, d3, mode)
        #     q_des[:2] = q_des_cal[:2]
        # v = np.dot(self.Kp_PD, (np.array(q_des) - np.array(q))) - np.dot(self.Kv_PD, qdot)
        # safe_value = 0.02
        # for i in range(len(v)):
        #     if np.abs(v[i]) > safe_value:
        #         v[i] = np.sign(v[i]) * safe_value
        #
        # tau = v + balance_torque
        #-------------------------------------------------------------------------

        # Operation Space
        M, V = self._get_mass_matrix(q, qdot)
        # contract frames
        # c1, c2, c3 = self._get_contact_points(pose_cur, d1, d2, d3, mode)
        # if mode == 'f1':
        #     c1_des, c2_des, c3_des = self._get_contact_points(pose_cur, d1, d_des, d3, mode)  # f1
        #     x_cur = [-c1[1], c1[0], pose_cur[2], c2[1], -c2[0], pose_cur[2], c3[1], -c3[0], pose_cur[2]]
        #     x_des = [-c1_des[1], c1_des[0], pose_cur[2], c2_des[1], -c2_des[0], pose_cur[2], c3_des[1], -c3_des[0],
        #              pose_cur[2]]
        # elif mode == 'f2':
        #     c1_des, c2_des, c3_des = self._get_contact_points(pose_cur, d_des, d2, d3, mode)  # f1
        #     x_cur = [-c1[1], c1[0], pose_cur[2], c2[1], -c2[0], pose_cur[2], -c3[1], c3[0], pose_cur[2]]
        #     x_des = [-c1_des[1], c1_des[0], pose_cur[2], c2_des[1], -c2_des[0], pose_cur[2], -c3_des[1], c3_des[0],
        #              pose_cur[2]]
        if mode == 'f1':
            delta_x = np.array([0, 0, 0, 0, d2 - d_des, 0, 0, 0, 0])
        elif mode == 'f2':
            delta_x = np.array([0, d_des - d1, 0, 0, 0, 0, 0, 0, 0])
        elif mode == 'f3':
            delta_x = np.array([0, 0, 0, 0, d_des - d2, 0, 0, 0, 0])
        x_vel_cur = np.dot(J, qdot)

        v = np.dot(self.Kp_sliding, delta_x) - \
            np.dot(self.Kv_sliding, x_vel_cur)
        Jdot = self._get_jacobian_derivative(q, qdot, pose_cur[2], pose_cur_dot[2], mode)
        tau = np.dot(M, np.dot(np.linalg.pinv(J), v)) + \
              V - np.dot(M, np.dot(np.linalg.pinv(J), np.dot(Jdot, qdot))) + \
              balance_torque
        return tau

    def slide_finger_right_in_two(self, pose_des, pose_cur, pose_cur_dot, q_des, q, qdot, d1, d2, d3, d_des, mode='f1'):
        """
        Slide the right finger among the two fingers on the same side.
        """
        # contact force to balance the target
        balance_force = 1 * self._get_grasp_matrix_null_space_forces(d1, d2, d3, mode)
        # compute torques to generate balance force
        J = self._get_jacobian(q, pose_cur[2], mode)
        balance_torque = np.dot(J.T, balance_force)

        #-------------old version in joint space-----------------
        # # PD controller
        # if mode == 'f1' or mode == 'f2':
        #     q_des_cal = self.compute_IK(pose_des, d1, d2, d_des, mode)
        #     q_des[-2:] = q_des_cal[-2:]
        # elif mode == 'f3':
        #     q_des_cal = self.compute_IK(pose_des, d_des, d2, d3, mode)
        #     q_des[:2] = q_des_cal[:2]
        # v = np.dot(self.Kp_PD, (np.array(q_des) - np.array(q))) - np.dot(self.Kv_PD, qdot)
        # safe_value = 0.1
        # for i in range(len(v)):
        #     if np.abs(v[i]) > safe_value:
        #         v[i] = np.sign(v[i]) * safe_value
        # tau = v + balance_torque
        #------------------------------------------------------

        # Operation Space
        M, V = self._get_mass_matrix(q, qdot)
        # contract frames
        # c1, c2, c3 = self._get_contact_points(pose_cur, d1, d2, d3, mode)
        # if mode == 'f1':
        #     c1_des, c2_des, c3_des = self._get_contact_points(pose_des, d1, d2, d_des, mode)  # f1
        #     x_cur = [-c1[1], c1[0], pose_cur[2], c2[1], -c2[0], pose_cur[2], c3[1], -c3[0], pose_cur[2]]
        #     x_des = [-c1_des[1], c1_des[0], pose_cur[2], c2_des[1], -c2_des[0], pose_cur[2], c3_des[1], -c3_des[0],
        #              pose_cur[2]]
        # elif mode == 'f2':
        #     c1_des, c2_des, c3_des = self._get_contact_points(pose_des, d1, d2, d_des, mode)
        #     x_cur = [-c1[1], c1[0], pose_cur[2], c2[1], -c2[0], pose_cur[2], -c3[1], c3[0], pose_cur[2]]
        #     x_des = [-c1_des[1], c1_des[0], pose_cur[2], c2_des[1], -c2_des[0], pose_cur[2], -c3_des[1], c3_des[0],
        #              pose_cur[2]]
        if mode == 'f1':
            delta_x = np.array([0, 0, 0, 0, 0, 0, 0, d3-d_des, 0])
        elif mode == 'f2':
            delta_x = np.array([0, 0, 0, 0, 0, 0, 0, d_des - d3, 0])
        elif mode == 'f3':
            delta_x = np.array([0, d_des - d1, 0, 0, 0, 0, 0, 0, 0])

        x_vel_cur = np.dot(J, qdot)

        v = np.dot(self.Kp_sliding, (delta_x)) - \
            np.dot(self.Kv_sliding, x_vel_cur)
        Jdot = self._get_jacobian_derivative(q, qdot, pose_cur[2], pose_cur_dot[2], mode)
        tau = np.dot(M, np.dot(np.linalg.pinv(J), v)) + \
              V - np.dot(M, np.dot(np.linalg.pinv(J), np.dot(Jdot, qdot))) + \
              balance_torque
        return tau

    def slide(self, finger, pose_des, pose_cur, pose_dot, q_des, q, qdot, d1, d2, d3, d_des, mode='f1'):
        if (finger == 0 and mode == 'f1') or (finger == 1 and mode == 'f2') or (finger == 2 and mode == 'f3'):
            tau = self.slide_finger_alone(pose_des, q, qdot, d1, d2, d3, d_des, mode)
        elif (finger == 1 and mode == 'f1') or (finger == 0 and mode == 'f2') or (finger == 1 and mode == 'f3'):
            tau = self.slide_finger_left_in_two(pose_des, pose_cur, pose_dot, q_des, q, qdot, d1, d2, d3, d_des, mode)
        else:
            tau = self.slide_finger_right_in_two(pose_des, pose_cur, pose_dot, q_des, q, qdot, d1, d2, d3, d_des, mode)
        return tau

    def slide_finger_alone(self, pose_des, q, qdot, d1, d2, d3, d_des, mode='f1'):
        # # PD controller
        # q_des = self.compute_IK(pose_des, d_des, d2, d3)
        # # contact force to balance the target
        # balance_force = 1 * self._get_grasp_matrix_null_space_forces(d1, d2, d3)
        # # compute torques to generate balance force
        # J = self._get_jacobian(q, pose_cur[2])
        # balance_torque = np.dot(J.T, balance_force)
        #
        # v = np.dot(self.Kp_PD, (np.array(q_des) - np.array(q))) - np.dot(self.Kv_PD, qdot)
        # safe_value = 0.1
        # for i in range(len(v)):
        #     if np.abs(v[i]) > safe_value:
        #         v[i] = np.sign(v[i]) * safe_value
        #
        # tau = v + balance_torque
        # return tau

        if mode == 'f1':
            q_des = self.compute_IK(pose_des, d_des, d2, d3, mode)
        elif mode == 'f2':
            q_des = self.compute_IK(pose_des, d1, d_des, d3, mode)
        elif mode == 'f3':
            q_des = self.compute_IK(pose_des, d1, d2, d_des, mode)

        if q_des is None:   # out of workspace
            return None

        v = np.dot(self.Kp_PD, (np.array(q_des) - np.array(q))) - np.dot(self.Kv_PD, qdot)
        safe_value = 0.1
        for i in range(len(v)):
            if np.abs(v[i]) > safe_value:
                v[i] = np.sign(v[i]) * safe_value
        return v

    def _get_grasp_matrix_null_space_forces(self, d1, d2, d3, mode='f1'):
        # return forces in grasp matrix null space
        if mode == 'f1':
            null_space = np.array([d3-d2, 0, 0, d3-d1, 0, 0, d1-d2, 0, 0])
        elif mode == 'f2':
            null_space = np.array([d3-d2, 0, 0, d3-d1, 0, 0, d2-d1, 0, 0])
        elif mode == 'f3':
            null_space = np.array([d3-d2, 0, 0, d1-d3, 0, 0, d1-d2, 0, 0])
        return null_space

    def _get_mass_matrix(self, q, qdot):
        """
        Return the mass matrix and Coriolis term
        """
        assert len(q) == 6 and len(qdot) == 6

        Iz1 = 1 / 12.0 * self.m1 * (self.l1 ** 2 + self.w_link ** 2)
        Iz2 = 1 / 12.0 * self.m2 * (self.l2 ** 2 + self.w_link ** 2)
        alpha = Iz1 + Iz2 + self.m1 * self.l1 ** 2 / 4. + self.m2 * (self.l1 ** 2 + self.l2 ** 2 / 4.)
        beta = self.m2 * self.l1 * self.l2 / 2.
        delta = Iz2 + self.m2 * self.l2 ** 2 / 4.

        M = np.array([
            [alpha + 2 * beta * np.cos(q[1]), delta + beta * np.cos(q[1]), 0, 0, 0, 0],
            [delta + beta * np.cos(q[1]), delta, 0, 0, 0, 0],
            [0, 0, alpha + 2 * beta * np.cos(q[3]), delta + beta * np.cos(q[3]), 0, 0],
            [0, 0, delta + beta * np.cos(q[3]), delta, 0, 0],
            [0, 0, 0, 0, alpha + 2 * beta * np.cos(q[5]), delta + beta * np.cos(q[5])],
            [0, 0, 0, 0, delta + beta * np.cos(q[5]), delta],
        ])

        V = np.dot(np.array([
            [-beta * np.sin(q[1]) * qdot[1], -beta * np.sin(q[1]) * (qdot[0] + qdot[1]), 0, 0, 0, 0],
            [beta * np.sin(q[1]) * qdot[0], 0, 0, 0, 0, 0],
            [0, 0, -beta * np.sin(q[3]) * qdot[3], -beta * np.sin(q[3]) * (qdot[2] + qdot[3]), 0, 0],
            [0, 0, beta * np.sin(q[3]) * qdot[2], 0, 0, 0],
            [0, 0, 0, 0, -beta * np.sin(q[5]) * qdot[5], -beta * np.sin(q[5]) * (qdot[4] + qdot[5])],
            [0, 0, 0, 0, beta * np.sin(q[5]) * qdot[4], 0],
        ]), qdot)

        return M, V

    def _get_hand_matrix(self, theta, d1, d2, d3, mode='f1'):
        """
        Given orientation of pole, return hand matrix.
        """
        if mode == 'f1':
            G = np.array([
                [np.sin(theta), -np.cos(theta), -d1],
                [np.cos(theta), np.sin(theta), -self.w_link / 2.],
                [0, 0, 0],
                [-np.sin(theta), np.cos(theta), d2],
                [-np.cos(theta), -np.sin(theta), -self.w_link / 2.],
                [0, 0, 0],
                [-np.sin(theta), np.cos(theta), d3],
                [-np.cos(theta), -np.sin(theta), -self.w_link / 2.],
                [0, 0, 0],
            ]).T
        elif mode == 'f2':
            G = np.array([
                [np.sin(theta), -np.cos(theta), -d1],
                [np.cos(theta), np.sin(theta), -self.w_link / 2.],
                [0, 0, 0],
                [-np.sin(theta), np.cos(theta), d2],
                [-np.cos(theta), -np.sin(theta), -self.w_link / 2.],
                [0, 0, 0],
                [np.sin(theta), -np.cos(theta), -d3],
                [np.cos(theta), np.sin(theta), -self.w_link / 2.],
                [0, 0, 0],
            ]).T
        elif mode == 'f3':
            G = np.array([
                [np.sin(theta), -np.cos(theta), -d1],
                [np.cos(theta), np.sin(theta), -self.w_link / 2.],
                [0, 0, 0],
                [np.sin(theta), -np.cos(theta), -d2],
                [np.cos(theta), np.sin(theta), -self.w_link / 2.],
                [0, 0, 0],
                [-np.sin(theta), np.cos(theta), d3],
                [-np.cos(theta), -np.sin(theta), -self.w_link / 2.],
                [0, 0, 0],
            ]).T
        return G

    def _get_jacobian(self, q, theta, mode="f1"):
        assert len(q) == 6

        q11, q12, q21, q22, q31, q32 = q[0], q[1], q[2], q[3], q[4], q[5]
        J = np.array([
            [self.l1 * np.cos(q11 - theta) + (self.l2 - self.w_link / 2.0) * np.cos(q11 + q12 - theta),
             (self.l2 - self.w_link / 2.0) * np.cos(q11 + q12 - theta), 0,
             0, 0, 0],
            [self.l1 * np.sin(q11 - theta) + (self.l2 - self.w_link / 2.0) * np.sin(q11 + q12 - theta),
             (self.l2 - self.w_link / 2.0) * np.sin(q11 + q12 - theta), 0,
             0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, self.l1 * np.cos(q21 - theta) + (self.l2 - self.w_link / 2.0) * np.cos(q21 + q22 - theta),
             (self.l2 - self.w_link / 2.0) * np.cos(q21 + q22 - theta), 0, 0],
            [0, 0, self.l1 * np.sin(q21 - theta) + (self.l2 - self.w_link / 2.0) * np.sin(q21 + q22 - theta),
             (self.l2 - self.w_link / 2.0) * np.sin(q21 + q22 - theta), 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, self.l1 * np.cos(q31 - theta) + (self.l2 - self.w_link / 2.0) * np.cos(q31 + q32 - theta),
             (self.l2 - self.w_link / 2.0) * np.cos(q31 + q32 - theta)],
            [0, 0, 0, 0, self.l1 * np.sin(q31 - theta) + (self.l2 - self.w_link / 2.0) * np.sin(q31 + q32 - theta),
             (self.l2 - self.w_link / 2.0) * np.sin(q31 + q32 - theta)],
            [0, 0, 0, 0, 0, 0]
        ])

        if mode == "f2":
            J[-3] *= -1
            J[-2] *= -1
        if mode == "f3":
            J[3] *= -1
            J[4] *= -1
        return J

    def _get_jacobian_derivative(self, q, qdot, theta, theta_dot, mode="f1"):
        assert len(q) == 6 and len(qdot) == 6

        q11, q12, q21, q22, q31, q32 = q[0], q[1], q[2], q[3], q[4], q[5]
        qdot11, qdot12, qdot21, qdot22, qdot31, qdot32 = qdot[0], qdot[1], qdot[2], qdot[3], qdot[4], qdot[5]
        Jdot = np.array([
            [-self.l1 * np.sin(q11 - theta) * (qdot11 - theta_dot) - (self.l2 - self.w_link / 2.0) * np.sin(q11 + q12 - theta) * (
                    qdot11 + qdot12 - theta_dot),
             -(self.l2 - self.w_link / 2.0) * np.sin(q11 + q12 - theta) * (qdot11 + qdot12 - theta_dot), 0, 0, 0, 0],
            [self.l1 * np.cos(q11 - theta) * (qdot11 - theta_dot) + (self.l2 - self.w_link / 2.0) * np.cos(q11 + q12 - theta) * (
                    qdot11 + qdot12 - theta_dot),
             (self.l2 - self.w_link / 2.0) * np.cos(q11 + q12 - theta) * (qdot11 + qdot12 - theta_dot), 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, -self.l1 * np.sin(q21 - theta) * (qdot21 - theta_dot) - (self.l2 - self.w_link / 2.0) * np.sin(q21 + q22 - theta) * (
                    qdot21 + qdot22 - theta_dot),
             -(self.l2 - self.w_link / 2.0) * np.sin(q21 + q22 - theta) * (qdot21 + qdot22 - theta_dot), 0, 0],
            [0, 0, self.l1 * np.cos(q21 - theta) * (qdot21 - theta_dot) + (self.l2 - self.w_link / 2.0) * np.cos(q21 + q22 - theta) * (
                    qdot21 + qdot22 - theta_dot),
             (self.l2 - self.w_link / 2.0) * np.cos(q21 + q22 - theta) * (qdot21 + qdot22 - theta_dot), 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0,
             -self.l1 * np.sin(q31 - theta) * (qdot31 - theta_dot) - (self.l2 - self.w_link / 2.0) * np.sin(q31 + q32 - theta) * (
                     qdot31 + qdot32 - theta_dot),
             -(self.l2 - self.w_link / 2.0) * np.sin(q31 + q32 - theta) * (qdot31 + qdot32 - theta_dot)],
            [0, 0, 0, 0,
             self.l1 * np.cos(q31 - theta) * (qdot31 - theta_dot) + (self.l2 - self.w_link / 2.0) * np.cos(q31 + q32 - theta) * (
                     qdot31 + qdot32 - theta_dot),
             (self.l2 - self.w_link / 2.0) * np.cos(q31 + q32 - theta) * (qdot31 + qdot32 - theta_dot)],
            [0, 0, 0, 0, 0, 0]
        ])

        if mode == "f2":
            Jdot[-3] *= -1
            Jdot[-2] *= -1
        if mode == "f3":
            Jdot[3] *= -1
            Jdot[4] *= -1
        return Jdot

    def _get_contact_points(self, pose, d1, d2, d3, mode='f1'):
        """
        Given pole pose, compute contact point locations
        """
        assert len(pose) == 3
        x, y, theta = pose
        if mode == 'f1':
            c1 = [x - self.w_target / 2.0 * np.sin(theta) + d1 * np.cos(theta),
                  y + self.w_target / 2.0 * np.cos(theta) + d1 * np.sin(theta)]
            c2 = [x + self.w_target / 2.0 * np.sin(theta) + d2 * np.cos(theta),
                  y - self.w_target / 2.0 * np.cos(theta) + d2 * np.sin(theta)]
            c3 = [x + self.w_target / 2.0 * np.sin(theta) + d3 * np.cos(theta),
                  y - self.w_target / 2.0 * np.cos(theta) + d3 * np.sin(theta)]

        elif mode == 'f2':
            c1 = [x - self.w_target / 2.0 * np.sin(theta) + d1 * np.cos(theta),
                  y + self.w_target / 2.0 * np.cos(theta) + d1 * np.sin(theta)]
            c2 = [x + self.w_target / 2.0 * np.sin(theta) + d2 * np.cos(theta),
                  y - self.w_target / 2.0 * np.cos(theta) + d2 * np.sin(theta)]
            c3 = [x - self.w_target / 2.0 * np.sin(theta) + d3 * np.cos(theta),
                  y + self.w_target / 2.0 * np.cos(theta) + d3 * np.sin(theta)]

        elif mode == 'f3':
            c1 = [x - self.w_target / 2.0 * np.sin(theta) + d1 * np.cos(theta),
                  y + self.w_target / 2.0 * np.cos(theta) + d1 * np.sin(theta)]
            c2 = [x - self.w_target / 2.0 * np.sin(theta) + d2 * np.cos(theta),
                  y + self.w_target / 2.0 * np.cos(theta) + d2 * np.sin(theta)]
            c3 = [x + self.w_target / 2.0 * np.sin(theta) + d3 * np.cos(theta),
                  y - self.w_target / 2.0 * np.cos(theta) + d3 * np.sin(theta)]
        else:
            raise ValueError
        return np.array([c1, c2, c3])

