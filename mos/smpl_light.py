import time
import torch
import numpy as np
import math
from articulate.math.angular import euler_angle_to_rotation_matrix


# 没smpl模型文件也能运行！用于纯姿态的IK&FK

# SMPL骨架
EDGES = {1:[[0, 1], [0, 2], [0, 3]],
         2:[[1, 4], [2, 5], [3, 6]],
         3:[[4, 7], [5, 8], [6, 9]],
         4:[[7, 10], [8, 11], [9, 12], [9, 13], [9, 14]],
         5:[[12, 15], [13, 16], [14, 17]],
         6:[[16, 18], [17, 19]],
         7:[[18, 20], [19, 21]],
         8:[[20, 22], [21, 23]]}

# local关节位置
JOINT_LOCAL = 0.001 * torch.FloatTensor([[   0.0000,    0.0000,    0.0000],
        [  58.5813,  -82.2800,  -17.6641],
        [ -60.3097,  -90.5133,  -13.5425],
        [   4.4394,  124.4036,  -38.3852],
        [  43.4514, -386.4695,    8.0370],
        [ -43.2566, -383.6879,   -4.8430],
        [   4.4884,  137.9564,   26.8203],
        [ -14.7903, -426.8745,  -37.4280],
        [  19.0555, -420.0456,  -34.5617],
        [  -2.2646,   56.0324,    2.8550],
        [  41.0544,  -60.2859,  122.0424],
        [ -34.8399,  -62.1055,  130.3233],
        [ -13.3902,  211.6355,  -33.4676],
        [  71.7025,  113.9997,  -18.8982],
        [ -82.9537,  112.4724,  -23.7074],
        [  10.1132,   88.9373,   50.4099],
        [ 122.9214,   45.2051,  -19.0460],
        [-113.2283,   46.8532,   -8.4721],
        [ 255.3319,  -15.6490,  -22.9465],
        [-260.1275,  -14.3692,  -31.2687],
        [ 265.7092,   12.6981,   -7.3747],
        [-269.1084,    6.7937,   -6.0268],
        [  86.6905,  -10.6360,  -15.5943],
        [ -88.7537,   -8.6516,  -10.1071]])

Z_90_p = torch.FloatTensor([[0, -1,  0],
                            [1,  0,  0],
                            [0,  0,  1]])

Z_90_n = Z_90_p.t()


# for k, v in EDGES.items():


class SMPLight:
    def __init__(self):
        # 父节点-子节点映射, 用于IK
        self.pc_mapping = []
        # 分层次的父节点-子节点映射, 用于FK
        self.layered_pc_mapping = {}
        #
        self.propagation_matrix = torch.zeros(24, 24)
        for k, v in EDGES.items():
            self.pc_mapping += v
            v = torch.LongTensor(v)
            p_id = np.array(v[:, 0]).tolist()
            c_id = np.array(v[:, 1]).tolist()
            self.layered_pc_mapping.update({k:[p_id, c_id]})
            self.propagation_matrix[c_id, p_id] += 1
            self.propagation_matrix[c_id] += self.propagation_matrix[p_id]
        self.pc_mapping = torch.LongTensor(self.pc_mapping)
        self.pc_mapping = [np.array(self.pc_mapping[:, 0]).tolist(), np.array(self.pc_mapping[:, 1]).tolist()]
        self.joint_local = JOINT_LOCAL

        # Jacobian相关
        self.propagation_matrix = (self.propagation_matrix == 1)
        self.propagation_link = [[]]
        for i in range(1, 24):
            self.propagation_link.append(np.array(torch.nonzero(self.propagation_matrix[i], as_tuple=True)[0]).tolist())
        self.joint_end_idx, self.joint_through_idx = [], []
        for i in range(1, 24):
            self.joint_end_idx += [i for _ in range(len(self.propagation_link[i]))]
            # 速度传递途径的关节点
            self.joint_through_idx += self.propagation_link[i]
        # print(self.propagation_link)

    @torch.no_grad()
    def forward_kinematics(self, R, trans=None, calc_joint=False):
        R = R.clone().detach()
        if calc_joint:
            return self._forward_kinematics_with_joint(R, trans)

        for _, mapping in self.layered_pc_mapping.items():
            p_idx, c_idx = mapping[0], mapping[1]
            R[..., c_idx, :, :] = R[..., p_idx, :, :].matmul(R[..., c_idx, :, :])
        return R

    @torch.no_grad()
    def _forward_kinematics_with_joint(self, R, trans):
        R = R.clone().detach()
        # positions n x 24 x 3
        positions = torch.zeros_like(R[..., -1]) + self.joint_local.to(R.device)

        if trans is not None:
            positions[..., 0, :] += trans

        # n x 24 x 3 x 4
        Rk = torch.cat([R, positions.unsqueeze(-1)], dim=-1)
        padding = torch.zeros_like(Rk[..., [-1], :])
        padding[..., -1] += 1

        # 构建传递矩阵: [[R, pos],
        #              [0,  1]]
        # n x 24 x 4 x 4
        Rk = torch.cat([Rk, padding], dim=-2)

        # 前向运动学
        for _, mapping in self.layered_pc_mapping.items():
            p_idx, c_idx = mapping[0], mapping[1]
            Rk[..., c_idx, :, :] = Rk[..., p_idx, :, :].matmul(Rk[..., c_idx, :, :])

        # 获取global的R与pos
        # n x 24 x 3 x 4
        Rk = Rk[..., :-1, :]
        # n x 24 x 3 x 3
        R = Rk[..., :, :-1]
        # n x 24 x 3
        joint = Rk[..., :, -1]

        return R, joint

    @torch.no_grad()
    def inverse_kinematics(self, R):
        R = R.clone().detach()
        p_idx, c_idx = self.pc_mapping[0], self.pc_mapping[1]
        R[..., c_idx, :, :] = R[..., p_idx, :, :].transpose(-2, -1).matmul(R[..., c_idx, :, :])
        return R

    @torch.no_grad()
    def calc_point_Jacobian(self, R):
        # s_dot = J x q_dot
        # J_ij = ds/d_theta

        # i=0
        # j=1
        # # 先实现joint j 对 joint i 的 Jacobian
        # r = (global_Joint[j] - global_Joint[i])
        # # 和单位旋转轴向量叉乘, 得到的是模长为旋转半径, 方向为切线方向的向量r, 正好就是ds/d_theta
        # # ds = d theta * r
        # ds_theta_1 = torch.cross(global_R[j, 0], r).unsqueeze(0)
        # ds_theta_2 = torch.cross(global_R[j, 1], r).unsqueeze(0)
        # ds_theta_3 = torch.cross(global_R[j, 2], r).unsqueeze(0)
        # J = torch.cat([ds_theta_1, ds_theta_2, ds_theta_3])
        # i关节对j关节的影响 三轴角速度引起的三轴线速度
        J = torch.zeros(24, 24, 3, 3).to(R.device)
        J_root_pos = torch.eye(3).unsqueeze(0).repeat(24, 1, 1).reshape(72, 3).to(R.device)

        global_R, global_Joint = self._forward_kinematics_with_joint(R, trans=None)


        r = global_Joint[self.joint_end_idx] - global_Joint[self.joint_through_idx]
        # print(global_R[joint_end_idx, 0].shape, r.shape)
        ds_theta_1 = torch.cross(global_R[self.joint_end_idx, 0], r).unsqueeze(-1)
        ds_theta_2 = torch.cross(global_R[self.joint_end_idx, 1], r).unsqueeze(-1)
        ds_theta_3 = torch.cross(global_R[self.joint_end_idx, 2], r).unsqueeze(-1)
        J_through = torch.cat([ds_theta_1, ds_theta_2, ds_theta_3], dim=-1)
        # print(ds_theta_3.shape)
        J[self.joint_end_idx, self.joint_through_idx] += J_through

        J = J.transpose(1, 2).reshape(72, 72)
        J = torch.cat([J_root_pos, J], dim=-1)

        return J

class SAMPLight(SMPLight):
    def __init__(self):
        super().__init__()
        # 左右手 左右膝盖重映射为axis align
        PART_A2W = torch.FloatTensor([[7.75,  -3.24,  6.71],
                                      [7.75,   3.24, -6.71],
                                      [-5.69, -1.23, -0.33],
                                      [-5.69,  1.23,  0.33]]) * torch.pi / 180
        PART_A2W = euler_angle_to_rotation_matrix(q=PART_A2W)
        PART_IDX= [18, 19, 4, 5]

        self.A2W = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(24, 1, 1)
        self.A2W[PART_IDX] = PART_A2W
        self.W2A = self.A2W.clone().transpose(-2, -1)
        # import pdb
        self.joint_local = self.A2W.matmul(self.joint_local.unsqueeze(-1)).squeeze(-1)
        # pdb.set_trace()

    @torch.no_grad()
    def from_smpl(self, R):
        self.W2A = self.W2A.to(R.device).view_as(R)
        sampl_R_glb = self.forward_kinematics(R).matmul(self.W2A)
        return self.inverse_kinematics(sampl_R_glb)

    @torch.no_grad()
    def to_smpl(self, R):
        self.A2W = self.A2W.to(R.device).view_as(R)
        sampl_R_glb = self.forward_kinematics(R).matmul(self.A2W)
        return self.inverse_kinematics(sampl_R_glb)


class SMPLPose:
    body_model = SMPLight()
    t_pose = torch.eye(3).unsqueeze(0).repeat(24, 1, 1)
    n_pose = t_pose.clone()
    n_pose[17], n_pose[16] = Z_90_p, Z_90_n

    t_pose_ori, t_pose_joint = body_model.forward_kinematics(t_pose, calc_joint=True)
    n_pose_ori, n_pose_joint = body_model.forward_kinematics(n_pose, calc_joint=True)


def DHMatrix(d, a, alpha, theta):
    st = math.sin(theta)
    ct = math.cos(theta)
    sa = math.sin(alpha)
    ca = math.cos(alpha)
    A = torch.FloatTensor([[ct, -st*ca, st*sa,  a*ct],
                           [st,  ct*ca, -ct*sa, a*st],
                           [0,   sa,    ca,     d],
                           [0,   0,     0,      1]])
    return  A

# 升级到BioSMPL 对关节朝向进行重新定义 与生理关节旋转轴对齐 方便进行自由度限制
# 1. 实现SMPL <-> SAMPL
# 2. SAMPL的Jacobian计算


# import articulate as art
# from config import paths

# 计算local joint

# body_model = art.ParametricModel(paths.smpl_file)
# p = torch.eye(3).unsqueeze(0).repeat(24, 1, 1).unsqueeze(0)
# body_shape = torch.zeros(10)
# init_trans = torch.zeros(3)
# # 输入24个关节旋转+体型参数+位移信息, 输出24个关节的旋转+蒙皮点加速度+运动速度
# grot, joint = body_model.forward_kinematics(p, body_shape, init_trans, calc_mesh=False)
# joint = joint.squeeze(0)
#
# sl = SMPLight()

# p_idx, c_idx = sl.pc_mapping[0], sl.pc_mapping[1]
#
# # print(joint)
# joint[c_idx] = joint[c_idx] - joint[p_idx]
# local_joint_position = joint
# #单位转为mm
# # print(local_joint_position*1000)

# # 测试
# sl = SMPLight()
# body_model = art.ParametricModel(paths.smpl_file)
# p = torch.eye(3).unsqueeze(0).repeat(24, 1, 1).unsqueeze(0).repeat(1000, 1, 1, 1)
# body_shape = torch.zeros(10)
# init_trans = torch.zeros(3).unsqueeze(0).repeat(1000,1)
# # 输入24个关节旋转+体型参数+位移信息, 输出24个关节的旋转+蒙皮点加速度+运动速度
# t1 = time.time()
# grot, joint = body_model.forward_kinematics(p, body_shape, init_trans, calc_mesh=False)
# t2 = time.time()
# print(joint, t2-t1)
#
# # p = p.cuda()
# # init_trans = init_trans.cuda()
#
# t1 = time.time()
# grot, joint = sl.forward_kinematics(R=p, trans=init_trans, calc_joint=True)
# t2 = time.time()
# print(joint, t2-t1)

# Jacobian正确性测试
# def euler_angle_to_rotation_matrix(q: torch.Tensor, seq='XYZ'):
#     r"""
#     Turn euler angles into rotation matrices. (torch, batch)
#
#     :param q: Euler angle tensor that can reshape to [batch_size, 3].
#     :param seq: 3 characters belonging to the set {'X', 'Y', 'Z'} for intrinsic
#                 rotations, or {'x', 'y', 'z'} for extrinsic rotations (radians).
#                 See scipy for details.
#     :return: Rotation matrix tensor of shape [batch_size, 3, 3].
#     """
#     from scipy.spatial.transform import Rotation
#     rot = Rotation.from_euler(seq, q.clone().detach().cpu().view(-1, 3).numpy())
#     ret = torch.from_numpy(rot.as_matrix()).float().to(q.device)
#     return ret

# def rotation_matrix_to_euler_angle_np(r, seq='XYZ'):
#     r"""
#     Turn rotation matrices into euler angles. (numpy, batch)
#
#     :param r: Rotation matrix (np/torch) that can reshape to [batch_size, 3, 3].
#     :param seq: 3 characters belonging to the set {'X', 'Y', 'Z'} for intrinsic
#                 rotations, or {'x', 'y', 'z'} for extrinsic rotations (radians).
#                 See scipy for details.
#     :return: Euler angle ndarray of shape [batch_size, 3].
#     """
#     from scipy.spatial.transform import Rotation
#     return Rotation.from_matrix(np.array(r).reshape(-1, 3, 3)).as_euler(seq)

# sl = SMPLight()
# tpose = SMPLPose.t_pose
# delta_theta = 1*np.pi/180
# R = euler_angle_to_rotation_matrix(torch.FloatTensor([delta_theta for _ in range(3)]))
# tpose_delta = tpose.clone()
# tpose_delta[[0]] = tpose_delta[[0]].matmul(R)
# tpose_delta[[3]] = tpose_delta[[3]].matmul(R)
#
# q_delta = torch.zeros(75)
# q_delta[3:6] += delta_theta
# q_delta[12:15] += delta_theta
#
# ds_ik = sl.forward_kinematics(tpose_delta, calc_joint=True)[1] - sl.forward_kinematics(tpose, calc_joint=True)[1]
#
# J = sl.calc_point_Jacobian(R=SMPLPose.t_pose)
# ds_jacobian = J.matmul(q_delta)
#
# # 对比真实值和基于jacobian计算得到的结果
# print(ds_ik.reshape(-1, 6))
# print(ds_jacobian.reshape(-1, 6))
#
# import time
# sl = SMPLight()
# pose = SMPLPose.t_pose
# t1=time.time()
# for i in range(5000):
#     J = sl.calc_point_Jacobian(R=pose)
#     # rotation_matrix_to_euler_angle_np(np.array(pose))
#     # x = sl._forward_kinematics_with_joint(R=pose, trans=None)
# t2=time.time()
# print(5000/(t2-t1))

def euler_angle_to_rotation_matrix(q: torch.Tensor, seq='XYZ'):
    r"""
    Turn euler angles into rotation matrices. (torch, batch)

    :param q: Euler angle tensor that can reshape to [batch_size, 3].
    :param seq: 3 characters belonging to the set {'X', 'Y', 'Z'} for intrinsic
                rotations, or {'x', 'y', 'z'} for extrinsic rotations (radians).
                See scipy for details.
    :return: Rotation matrix tensor of shape [batch_size, 3, 3].
    """
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_euler(seq, q.clone().detach().cpu().view(-1, 3).numpy())
    ret = torch.from_numpy(rot.as_matrix()).float().to(q.device)
    return ret