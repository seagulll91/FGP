__all__ = ['pose_to_q', 'q_to_pose', 'normalize_and_concat', 'KalmanFilterVelocity', 'Body']


import enum
import torch
import numpy as np
from articulate.math import rotation_matrix_to_euler_angle_np, euler_angle_to_rotation_matrix_np, rotation_matrix_to_r6d, \
    normalize_angle, r6d_to_rotation_matrix, rotation_matrix_to_axis_angle, axis_angle_to_rotation_matrix

# 在Jacobian中屏蔽无生理自由度的旋转
# 注意: 如果没对姿态使用constrain_joint_dot操作, 关节18, 19的数据要删除
constrain_joints = np.array([18, 19, 4, 5], dtype=np.int_)
constrain_axis = np.array([2, 2, 2, 2], dtype=np.int_)
# constrain_joints = np.array([18, 18, 18, 19, 19, 19], dtype=np.int_)
# constrain_axis = np.array([0, 1, 2, 0, 1, 2], dtype=np.int_)
constrain_channels_euler = (constrain_joints*3 + constrain_axis).tolist()
constrain_channels_q = (3 + constrain_joints*3 + constrain_axis).tolist()

def constrain_joint_dofs2(poses):
    # 肘关节与膝关节
    joints_origin = poses[:, [4, 5, 18, 19]]
    joints = rotation_matrix_to_r6d(joints_origin).reshape(-1, 4, 6)
    # z轴不旋转 -> x轴方向没有y分量
    joints[:, :, :3] *= torch.FloatTensor([[[1, 0, 1]]])
    joints = r6d_to_rotation_matrix(joints).reshape(-1, 4, 3, 3)
    poses[:, [4, 5, 18, 19]] = joints
    # 减少的旋转分量转移到髋/肩关节
    # delta_R = joints_origin.matmul(joints.transpose(-2, -1))
    delta_R = joints_origin.matmul(joints.transpose(-2, -1))
    poses[:, [1, 2, 16, 17]] = poses[:, [1, 2, 16, 17]].matmul(delta_R)

    return poses

def constrain_joint_dofs(poses):
    # 肘关节与膝关节
    joints_origin = poses[:, [4, 5, 18, 19]]
    joints = rotation_matrix_to_r6d(joints_origin).reshape(-1, 4, 6)
    # z轴不旋转 -> x轴方向没有y分量
    joints[:, :, :3] *= torch.FloatTensor([[[1, 0, 1]]])
    joints = r6d_to_rotation_matrix(joints).reshape(-1, 4, 3, 3)
    poses[:, [4, 5, 18, 19]] = joints
    # 减少的旋转分量转移到髋/肩关节
    # delta_R = joints_origin.matmul(joints.transpose(-2, -1))
    delta_R = joints_origin.matmul(joints.transpose(-2, -1))
    half_delta_R = rotation_matrix_to_axis_angle(delta_R)/2
    half_delta_R = axis_angle_to_rotation_matrix(half_delta_R).reshape(-1, 4, 3, 3)

    #print(half_delta_R)
    #print(delta_R.matmul(half_delta_R.transpose(-2, -1)))
    poses[:, [1, 2, 16, 17]] = poses[:, [1, 2, 16, 17]].matmul(half_delta_R)

    return poses


def pose_to_q(poses, trans, dof_constrains=True):
    r"""
    Convert smpl poses and translations to robot configuration q. (numpy, batch)

    :param poses: Array that can reshape to [n, 24, 3, 3].
    :param trans: Array that can reshape to [n, 3].
    :return: Ndarray in shape [n, 75] (3 root position + 72 joint rotation).
    """
    poses = poses.reshape(-1, 24, 3, 3)
    if dof_constrains:
        poses = constrain_joint_dofs(poses)
    poses = np.array(poses)
    trans = np.array(trans).reshape(-1, 3)
    euler_angle = rotation_matrix_to_euler_angle_np(poses, 'XYZ').reshape(-1, 72)
    # if dof_constrains:
    #     print(euler_angle[:, constrain_channels_euler]*180/np.pi)
    #     euler_angle[:, constrain_channels_euler] *= 0
    qs = np.concatenate((trans, euler_angle), axis=1)
    qs[:, 3:] = normalize_angle(qs[:, 3:])
    return qs

def q_to_pose(qs):
    r"""
    Convert robot configuration q to smpl poses and translations. (numpy, batch)

    :param qs: Ndarray that can reshape to [n, 75] (3 root position + 72 joint rotation).
    :return: Poses ndarray in shape [n, 24, 3, 3] and translation ndarray in shape [n, 3].
    """
    qs = qs.reshape(-1, 75)
    trans, euler_poses = qs[:, :3], qs[:, 3:]
    poses = euler_angle_to_rotation_matrix_np(euler_poses, 'XYZ').reshape(-1, 24, 3, 3)
    return poses, trans

def mask_jacobian(J):
    # J: 72 x 75
    J[:, constrain_channels_q] *= 0
    return J


def normalize_and_concat(glb_acc, glb_rot):
    glb_acc = glb_acc.view(-1, 6, 3)
    glb_rot = glb_rot.view(-1, 6, 3, 3)
    acc = torch.cat((glb_acc[:, :5] - glb_acc[:, 5:], glb_acc[:, 5:]), dim=1).bmm(glb_rot[:, -1])
    ori = torch.cat((glb_rot[:, 5:].transpose(2, 3).matmul(glb_rot[:, :5]), glb_rot[:, 5:]), dim=1)
    data = torch.cat((acc.flatten(1), ori.flatten(1)), dim=1)
    return data

class KalmanFilterVelocity:
    def __init__(self, fps, process_var=0.01, meas_var=1.0, dim=72):
        self.dt = 1 / fps  # 时间间隔
        # 状态向量：速度三分量
        self.x = np.zeros((dim, 1))  # 初始估计状态

        # 状态协方差矩阵P 初始化为单位矩阵
        self.P = np.eye(dim)

        # 状态转移矩阵F (假设速度是稳态，不变)
        self.F = np.eye(dim)

        # 观测矩阵H 直接观测速度
        self.H = np.eye(dim)

        # 过程噪声协方差Q
        self.Q = process_var * np.eye(dim)

        # 观测噪声协方差R
        self.R = meas_var * np.eye(dim)

        self.dim = dim

    def predict(self):
        # 预测状态
        self.x = self.F @ self.x
        # 预测协方差
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        # 计算卡尔曼增益
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # 更新估计
        y = z.reshape(self.dim,1) - self.H @ self.x  # 观测残差
        self.x = self.x + K @ y

        # 更新协方差矩阵
        I = np.eye(self.dim)
        self.P = (I - K @ self.H) @ self.P

    def get_state(self):
        return self.x.flatten()


class Body(enum.Enum):
    r"""
    Prefix L = left; Prefix R = right.
    """
    ROOT = 2
    PELVIS = 2
    SPINE = 2
    LHIP = 5
    RHIP = 17
    SPINE1 = 29
    LKNEE = 8
    RKNEE = 20
    SPINE2 = 32
    LANKLE = 11
    RANKLE = 23
    SPINE3 = 35
    LFOOT = 14
    RFOOT = 26
    NECK = 68
    LCLAVICLE = 38
    RCLAVICLE = 53
    HEAD = 71
    LSHOULDER = 41
    RSHOULDER = 56
    LELBOW = 44
    RELBOW = 59
    LWRIST = 47
    RWRIST = 62
    LHAND = 50
    RHAND = 65
