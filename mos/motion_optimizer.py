import torch
import numpy as np
import articulate as art
from qpsolvers import solve_qp
from .tools import *
from .smpl_light import SMPLight, SAMPLight
from .tools import mask_jacobian


def normalize_angle(q):
    r"""
    Normalize radians into [-pi, pi). (np/torch, batch)

    :param q: A tensor (np/torch) of angles in radians.
    :return: The normalized tensor where each angle is in [-pi, pi).
    """
    mod = q % (2 * np.pi)
    mod[mod >= np.pi] -= 2 * np.pi
    return mod

class MotionOptimizer:
    def __init__(self, fps=60, eps=1e-2, bio_axis=True):
        # configuration维度 3 position + 72 euler angle
        self.qdot_size = 75
        self.qdot = np.zeros(self.qdot_size)
        self.reset_states()
        self.bio_axis = bio_axis
        if bio_axis:
            self.body_model = SAMPLight()
        else:
            self.body_model = SMPLight()
        self.dt = 1/fps
        self.w_qp = 0.6
        self.eps = eps
        self.kf = KalmanFilterVelocity(fps=fps, dim=72)
        self.qdot_qp = np.zeros(self.qdot_size)

    def reset_states(self):
        self.last_x = []
        self.q = None
        self.qdot = np.zeros(self.qdot_size)
        self.qdot_qp = np.zeros(self.qdot_size)

    def optimize_frame(self, pose, jvel, trans):
        if self.bio_axis:
            pose = self.body_model.from_smpl(pose)

        q_ref = pose_to_q(pose, trans)[0]
        v_ref = jvel.flatten(0).numpy()

        # # 速度卡尔曼滤波
        # self.kf.predict()
        # self.kf.update(v_ref)
        # v_ref = self.kf.get_state()

        if self.q is None:
            self.q = q_ref
            return pose, trans + jvel[0]

        q_delta = q_ref - self.q
        q_delta[3:] = normalize_angle(q_delta[3:])
        q_dot_ref = q_delta / self.dt

        # print(pose)
        Js = np.array(self.body_model.calc_point_Jacobian(R=pose))
        # 屏蔽无生理自由度的关节
        Js = mask_jacobian(Js)
        # print(Js)

        # minimize   ||A1 * q_dot - b1||^2     for A1, b1 in zip(As1, bs1)
        As1, bs1 = [np.zeros((0, self.qdot_size))], [np.empty(0)]

        A_, b_ = None, None

        # joint position controller (using joint velocity to determine target joint position)
        if True:
            A = Js
            b = v_ref
            As1.append(A)
            bs1.append(b)

        # # joint rotation controller (using joint velocity to determine target joint position)
        # if True:
        #     A = np.hstack((np.zeros((self.qdot_size - 3, 3)), np.eye((self.qdot_size - 3))))
        #     b = q_dot_ref[3:]
        #     As1.append(A * 2)  # 72 * 75
        #     bs1.append(b * 2)  # 72

        # t8 = time.time()
        As1, bs1 = np.vstack(As1), np.concatenate(bs1)
        # print(As1.shape)
        # print(bs1.shape)
        # print(np.dot(As1.T, As1))
        # 注意 这里不用手动乘1/2 算法包内部会自动乘
        P_ = art.math.block_diagonal_matrix_np([np.dot(As1.T, As1)])
        q_ = -np.dot(As1.T, bs1)
        # 正则项 防止非正定
        P_ += np.eye(P_.shape[0]) * self.eps


        # print(np.linalg.eigvalsh(P_))
        # 拼接多个优化问题
        # P_ = np.concatenate((art.math.block_diagonal_matrix_np([np.dot(As1.T, As1)]), xxx, xxx))
        # q_ = np.concatenate((-np.dot(As1.T, bs1)))

        # fast solvers are less accurate/robust, and may fail
        init = self.last_x if len(self.last_x) == len(q_) else None
        # print(P_)

        x = solve_qp(P_, q_, solver='quadprog', initvals=init)

        if x is None or np.linalg.norm(x) > 10000:
            x = solve_qp(P_, q_, solver='cvxopt', initvals=init)
            # t9 = time.time()
        qdot_qp = x[:self.qdot_size]

        qdot = self.qdot_qp * self.w_qp + q_dot_ref * (1 - self.w_qp)
        # qdot = q_dot_qp
        q_optim = self.q + qdot * self.dt
        q_optim[3:] = normalize_angle(q_optim[ 3:])

        # # 处理欧拉角跳变
        # singularity = np.any(np.abs(q_optim[3:] - self.q[3:]) > np.pi/2)
        # if singularity:
        #     print('sos')
        #     q_optim = self.q + q_dot_ref * self.dt
        #     qdot = qdot

        self.qdot = qdot
        self.qdot_qp = qdot_qp
        self.q = q_optim
        self.last_x = x

        # except ValueError:
        #     # raise ValueError
        #     self.err_count += 1
        #     print(f"matrix P is not positive definite")
        #     q = self.q + q_dot_ref * self.dt
        #     self.q = q
        #     self.qdot = q_dot_ref
        #     self.last_x = []


        pose_opt, tran_opt = q_to_pose(q_optim)

        pose_opt = torch.from_numpy(pose_opt).float()
        tran_opt = torch.from_numpy(tran_opt).float().view(-1)

        if self.bio_axis:
            pose_opt = self.body_model.to_smpl(pose_opt)

        return pose_opt, tran_opt
