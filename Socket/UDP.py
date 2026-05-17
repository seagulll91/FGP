import time

import torch
import socket
import threading
import os

import Aplus.tools.smpl_light

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"]=""
from pygame.time import Clock
from socket import *
from Math import *
from Utils import config
from Preprocess.preprocess import preprocess
from Driver.driver import *
import numpy as np
import torch
from articulate.math import r6d_to_rotation_matrix, rotation_matrix_to_axis_angle, normalize_tensor, axis_angle_to_rotation_matrix
import articulate as art
import math
from config import paths
from config import demo_mode
from Aplus.tools.smpl_light import SMPLPose
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from scipy.optimize import minimize

# 使用[pip install numpy-quaternion] 安装
# 而不是[pip install quaternion]
import quaternion

from articulate.math import rotation_matrix_to_euler_angle, euler_angle_to_rotation_matrix, quaternion_to_rotation_matrix

running = False
start_recording = False
index_joint = [3, 6, 9, 13, 14, 16, 17, 18, 19, 20, 21]
index_pose = [0, 3, 6, 9, 13, 14, 16, 17, 18, 19]

def get_tpose_joint_position(pose=None):

    # # 通过smpl模型计算
    body_model = art.ParametricModel(paths.smpl_file)
    #
    # shape = torch.tensor([-0.53062872, -0.03227116, 1.76599983, 1.0983942, 0.69472912, -0.15725264, 0.52576423,
    #                       -0.13045748, 0.24933542, -1.15570345])
    # tran = torch.tensor([[0, 0, 0]])
    #
    # if pose is None:
    #     pose = torch.eye(3).squeeze(0).repeat(24, 1, 1)
    # pose = pose.reshape(-1, 24, 3, 3)
    #
    # grot, joint, vert = body_model.forward_kinematics(pose, shape, tran, calc_mesh=True)

    # 直接使用计算结果
    joint = torch.tensor([[[0.0000, 0.0000, 0.0000],
      [0.0613, -0.0875, -0.0187],
      [-0.0632, -0.0960, -0.0142],
      [0.0044, 0.1332, -0.0386],
      [0.1023, -0.4714, -0.0113],
      [-0.1053, -0.4786, -0.0196],
      [0.0100, 0.2852, -0.0073],
      [0.0864, -0.8966, -0.0513],
      [-0.0853, -0.8977, -0.0568],
      [0.0073, 0.3474, -0.0029],
      [0.1263, -0.9579, 0.0741],
      [-0.1207, -0.9598, 0.0780],
      [-0.0061, 0.5716, -0.0404],
      [0.0794, 0.4669, -0.0244],
      [-0.0761, 0.4651, -0.0291],
      [0.0033, 0.6653, 0.0132],
      [0.2028, 0.5111, -0.0442],
      [-0.1904, 0.5104, -0.0376],
      [0.4604, 0.4961, -0.0665],
      [-0.4515, 0.4975, -0.0699],
      [0.7266, 0.5084, -0.0739],
      [-0.7215, 0.5040, -0.0766],
      [0.8147, 0.4976, -0.0897],
      [-0.8121, 0.4952, -0.0867]]])

    return joint.reshape(24, 3)

#只发送一组数据
def send_data_to_unity(prediction_list,rotation_type,part,fps,tran_list=None):

    #输出提示信息
    n_frame = prediction_list.size(0)
    print(f'***Basic Info***')
    print(f"    Visual_part : {part}")
    print(f"    Frames : {n_frame}\n")

    #建立连接
    server_for_unity = socket(AF_INET, SOCK_STREAM)
    server_for_unity.bind(('127.0.0.1', 8888))
    server_for_unity.listen(1)
    print('Server start. Waiting for unity3d to connect.')

    # 运行unity程序
    run_mode1()
    # run_mode_girl()
    # run_mode_micai()
    # run_mode_up()

    conn, addr = server_for_unity.accept()
    
    i = 0
    skip = 1
    
    running = True
    clock = Clock()
    is_recording = False

    while running:
        # calibration
        clock.tick(fps)
        # print('\r', f'fps: {clock.get_fps()}')
        
        pose = preprocess(prediction_list[i],rotation_type)

        # part of data freeze
        # pose.view(24, 3)[config.joint_set.part[part]] *= 0

        i = (i + skip) % n_frame


        if tran_list is not None:
            tran = tran_list[i]
        else:
            tran = torch.zeros(3)
        # send pose
        s = ','.join(['%g' % v for v in pose]) + '#' + \
            ','.join(['%g' % v for v in tran]) + '$'
        conn.send(s.encode('utf8'))  # I use unity3d to read pose and translation for visualization here
        print(f'\r {i}|{n_frame}', end='')
        # print('\r', '(recording)' if is_recording else '',
        #       '\rOutput FPS:', clock.get_fps(), end='')

    print('Finish.')


#同时发送预测和gt两组数据
def send_datas_to_unity(prediction_list,gt_list,rotation_type,part,fps):
    #输出提示信息
    n_frame_pr = prediction_list.size(0)
    n_frame_gt = gt_list.size(0)

    if n_frame_pr!=n_frame_gt:
        raise Exception('The frames for PR and GT is not equal!')
    
    print(f'***Basic Info***')
    print(f"    Visual_part : {part}")
    print(f"    pr Frames : {n_frame_pr}  (the red model)")
    print(f"    gt Frames : {n_frame_gt}  (the green model)\n")

    #建立连接
    server_for_unity = socket(AF_INET, SOCK_STREAM)
    server_for_unity.bind(('127.0.0.1', 8888))
    server_for_unity.listen(1)
    print('Server start. Waiting for unity3d to connect.')
    
    # 运行unity程序
    run_mode2()

    conn, addr = server_for_unity.accept()

    i = 0
    skip = 1
    
    running = True
    clock = Clock()
    is_recording = False

    while running:
        # calibration
        clock.tick(fps)
        #pose1
        pose = preprocess(prediction_list[i],rotation_type)
        # part of data freeze
        pose.view(24, 3)[config.joint_set.part[part]] *= 0
        
        #pose2
        pose2 = preprocess(gt_list[i],rotation_type)  
        # part of data freeze
        pose2.view(24, 3)[config.joint_set.part[part]] *= 0

        i = (i + skip) % n_frame_pr

        tran = torch.zeros(3)
        # send pose
        s = ','.join(['%g' % v for v in pose]) + '#' + \
            ','.join(['%g' % v for v in tran]) + '$' + \
            ','.join(['%g' % v for v in pose2]) + '#' + \
            ','.join(['%g' % v for v in tran]) + '@'
        conn.send(s.encode('utf8'))  # I use unity3d to read pose and translation for visualization here

        # print('\r', '(recording)' if is_recording else '',
        #       '\tOutput FPS:', clock.get_fps(), end='')

        print(f'\r {i}|{n_frame_pr}', end='')

    print('Finish.')


def bulid_rot(theta, rotation_axis):
    # 基于罗德里格斯公式(Rodrigues' Rotation Formula)
    theta = theta * torch.pi / 180
    I = torch.eye(3)
    kx, ky, kz = tuple(rotation_axis)
    K = torch.FloatTensor([[0, -kz, ky],
                           [kz, 0, -kx],
                           [-ky, kx, 0]])
    s = math.sin(theta)
    c = math.cos(theta)

    # Rodrigues' Rotation Formula
    rot = I + s*K + (1-c)*(K.matmul(K))

    return rot.unsqueeze(0)


def elbow_angle_process(angle):
    angle = angle * np.pi / 180
    angle = torch.tensor([np.cos(angle[0]), -np.sin(angle[0]), np.cos(angle[1]), -np.sin(angle[1])])
    return angle


@torch.no_grad()
def ego_drift_regularization(rot, imu_num=6, ego_yaw_idx=-1):
    rot = rot.reshape(imu_num, 3, 3)
    rot_ego = rot[ego_yaw_idx]

    rot_ego_euler = rotation_matrix_to_euler_angle(rot_ego, seq='YZX').squeeze(0)
    # heading_ref_euler[:, [1, 2]] *= 0
    rot_ego_euler[0] *= 0
    rot_ego = euler_angle_to_rotation_matrix(rot_ego_euler, seq='YZX')

    rot[ego_yaw_idx] = rot_ego
    return rot

def rotation_diversity(rot):
    """
    计算一段序列中rotation的丰富度
    :param rot: batch x seq_len x imu_num x 3 x 3
    :return:
    """
    n_batch, seq_len, imu_num = rot.shape[0], rot.shape[1], rot.shape[2]
    rot = rot.reshape(-1, 3, 3)
    euler_angle = rotation_matrix_to_euler_angle(rot).reshape(n_batch, seq_len, imu_num, 3) * 180 / np.pi
    # 离散化的角度
    dis_angle = torch.div(euler_angle, 15, rounding_mode='floor').long() + torch.LongTensor([12, 6, 12]).reshape(1,
                                                                                                                 1,
                                                                                                                 1,
                                                                                                                 3).to(
        euler_angle.device)
    # 离散空间索引
    dis_angle_idx = torch.clip(dis_angle[:, :, :, [0]], 0, 23) + torch.clip(dis_angle[:, :, :, [1]], 0, 11) * 24 + \
                    torch.clip(dis_angle[:, :, :, [2]], 0, 23) * 12 * 24

    angle_space = torch.zeros(n_batch, seq_len, imu_num, 24 * 12 * 24, dtype=torch.uint8).to(euler_angle.device)
    angle_space.scatter_add_(3, dis_angle_idx, torch.ones_like(angle_space, dtype=torch.uint8))
    angle_space_sum = angle_space.sum(dim=1)
    angle_space_mask = (angle_space_sum > 0).reshape(n_batch, imu_num, -1)
    diversity = angle_space_mask.sum(dim=-1)
    return diversity.cpu()

class DataProcessServer_Upper():
    def __init__(self, rotation_type, part, config, keep_hidden=True, run_unity_package=True, mode=demo_mode.UPPER, track_trans=False):
        if run_unity_package:
            # run_mode1()
            # run_mode_girl()
            # run_mode_micai()
            run_mode_up()
        server_for_unity = socket(AF_INET, SOCK_STREAM)
        server_for_unity.bind(('127.0.0.1', 8888))
        server_for_unity.listen(1)
        print('Server start. Waiting for unity3d to connect.')
        self.conn, self.addr = server_for_unity.accept()
        self.rotation_type = rotation_type
        self.part = part
        self.keep_hidden = keep_hidden
        self.track_trans = track_trans
        self.trans = torch.FloatTensor([[0,0,0]])
        if isinstance(config, list):
            self.config_clothes, self.config_pant = tuple(config)
            self.config = self.config_clothes
        else:
            self.config = config

        self.mode = mode
        if mode == demo_mode.UPPER:
            self.imu_num = 4
        elif mode == demo_mode.FULL:
            self.imu_num = 8

        self.property()

    def get_raw_device_2_bone_rot(self):
        root_2_smpl = self.config_clothes['root_2_smpl'].unsqueeze(0)
        left_2_smpl = self.config_clothes['root_2_left'].transpose(1, 2).matmul(root_2_smpl)
        right_2_smpl = self.config_clothes['root_2_right'].transpose(1, 2).matmul(root_2_smpl)
        back_2_smpl = self.config_clothes['root_2_back'].transpose(1, 2).matmul(root_2_smpl)

        raw_device_2_bone = [left_2_smpl, right_2_smpl, back_2_smpl, root_2_smpl]

        if self.mode == demo_mode.FULL:
            left_lowerleg_2_smpl = self.config_pant['left_lowerleg_2_smpl']
            right_lowerleg_2_smpl = self.config_pant['right_lowerleg_2_smpl']
            left_pelvis_2_smpl = self.config_pant['left_pelvis_2_smpl']
            right_pelvis_2_smpl = self.config_pant['right_pelvis_2_smpl']

            raw_device_2_bone += [left_lowerleg_2_smpl, right_lowerleg_2_smpl, left_pelvis_2_smpl, right_pelvis_2_smpl]
        return torch.cat(raw_device_2_bone, dim=0)

    def globals_bias_estimate(self, tpose_oris, threshold_deg=5):
        R_root = tpose_oris[3]
        if self.mode == demo_mode.UPPER:
            root_2_left = self.config['root_2_left']
            root_2_right = self.config['root_2_right']
            root_2_back = self.config['root_2_back']
            root_2_root = torch.eye(n=3).unsqueeze(dim=0)

            root_2_leafs = torch.cat([root_2_left, root_2_right, root_2_back, root_2_root], dim=0)

            I_bias = tpose_oris.matmul(R_root.matmul(root_2_leafs).transpose(1, 2))

        elif self.mode == demo_mode.FULL:
            root_2_smpl = self.config_clothes['root_2_smpl'].unsqueeze(0)
            left_2_smpl = self.config_clothes['root_2_left'].transpose(1, 2).matmul(root_2_smpl)
            right_2_smpl = self.config_clothes['root_2_right'].transpose(1, 2).matmul(root_2_smpl)
            back_2_smpl = self.config_clothes['root_2_back'].transpose(1, 2).matmul(root_2_smpl)

            left_lowerleg_2_smpl = self.config_pant['left_lowerleg_2_smpl']
            right_lowerleg_2_smpl = self.config_pant['right_lowerleg_2_smpl']
            left_pelvis_2_smpl = self.config_pant['left_pelvis_2_smpl']
            right_pelvis_2_smpl = self.config_pant['right_pelvis_2_smpl']

            # t = [left_2_smpl, right_2_smpl, back_2_smpl, root_2_smpl,
            #                            left_lowerleg_2_smpl, right_lowerleg_2_smpl, left_pelvis_2_smpl, right_pelvis_2_smpl]
            # for v in t:
            #     print(v.shape)

            device_2_smpl = torch.cat([left_2_smpl, right_2_smpl, back_2_smpl, root_2_smpl,
                                       left_lowerleg_2_smpl, right_lowerleg_2_smpl, left_pelvis_2_smpl, right_pelvis_2_smpl], dim=0)

            I_2_smpl = R_root.matmul(root_2_smpl)

            I_bias = tpose_oris.matmul(device_2_smpl).matmul(I_2_smpl.transpose(-2, -1))


        # 仅保留z轴部分旋转[new]
        imu_num = I_bias.shape[0]
        z_mask = torch.tensor([[[1, 1, 0], [1, 1, 0], [0, 0, 1]]]).repeat(imu_num, 1, 1)
        # x, y轴向世界坐标x-y平面投影, z轴向世界坐标z轴投影
        I_bias = I_bias * z_mask
        I_bias = normalize_tensor(x=I_bias.reshape(-1, 3)).view_as(I_bias)

        I_bias_axis = rotation_matrix_to_axis_angle(r=I_bias)

        # 只保留绝对值大于threshold的部分
        threshold = threshold_deg * np.pi / 180
        mask_fix = torch.gt(torch.abs(I_bias_axis), threshold)
        I_bias_axis = I_bias_axis - mask_fix * torch.sgn(I_bias_axis) * threshold
        I_bias = axis_angle_to_rotation_matrix(I_bias_axis)

        print(I_bias)

        # I_bias = torch.eye(3).unsqueeze(0).repeat(4,1,1)
        return I_bias

    def normalize_and_concat(self, smpl_acc, smpl_ori):
        imu_num = self.imu_num
        acc_scale = 30
        smpl_acc = smpl_acc.view(-1, imu_num, 3)
        smpl_ori = smpl_ori.view(-1, imu_num, 3, 3)

        # print(smpl_ori[:100, 0])
        # acc: [n, 4, 3]
        # acc = torch.cat((smpl_acc[:, :(imu_num-1)] - smpl_acc[:, (imu_num-1):], smpl_acc[:, (imu_num-1):]), dim=1).bmm(smpl_ori[:, -1]) / acc_scale
        acc = smpl_acc / acc_scale
        # 不转换为相对加速度 变换至root坐标系
        # acc = smpl_acc.bmm(smpl_ori[:, -1]) / acc_scale

        # ori = torch.cat((smpl_ori[:, (imu_num-1):].transpose(2, 3).matmul(smpl_ori[:, :(imu_num-1)]), smpl_ori[:, (imu_num-1):]), dim=1)
        ori = smpl_ori

        data = torch.cat((acc.flatten(1), ori.flatten(1)), dim=1)
        return data

    def update_data(self, data):

        pose = preprocess(data, self.rotation_type)

        # part of data freeze
        pose.view(24, 3)[config.joint_set.part[self.part]] *= 0

        tran = self.get_trans()
        # send pose
        s = ','.join(['%g' % v for v in pose]) + '#' + \
            ','.join(['%g' % v for v in tran]) + '$'
        self.conn.send(s.encode('utf8'))  # I use unity3d to read pose and translation for visualization here

    def property(self):
        """
        自定义属性
        :return:无返回值
        """
        # 用于calibration的数据
        self.init_oris = None
        self.tpose_accs = None

        self.I_2_Ibias = None

        # 串口版
        self.root_2_smpl = self.config['root_2_smpl']


        self.smpl2imu = None
        self.device2bone = None
        self.acc_offsets = None

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        self.h_1 = np.zeros(shape=[2, 1, 256]).astype(np.float32)
        self.c_1 = np.zeros(shape=[2, 1, 256]).astype(np.float32)
        self.h_2 = np.zeros(shape=[2, 1, 256]).astype(np.float32)
        self.c_2 = np.zeros(shape=[2, 1, 256]).astype(np.float32)

        # 模型输入
        self.acc_cat_oris = []

    def set_calibrate_data(self, input: list):
        """
        设置标定数据
        :param input: [48] -> [4 x 3 (acc) + 4 x 9(oris)]
        :return:无返回值
        """
        # print(input[self.imu_num*3:])
        self.tpose_accs = torch.tensor(input[0:self.imu_num*3]).view(self.imu_num, 3)
        self.init_oris = torch.tensor(input[self.imu_num*3:]).view(self.imu_num, 3, 3)
        # self.tpose_elbow_angle = torch.tensor(input[48:50])

        # 串口版
        # self.I_2_Ibias = self.globals_bias_estimate(tpose_oris=self.init_oris, threshold_deg=5).transpose(1, 2)
        self.I_2_Ibias = torch.eye(3).reshape(-1, 3, 3).repeat(self.imu_num, 1, 1)

        # 修正全局坐标系误差
        self.init_oris = self.I_2_Ibias.matmul(self.init_oris)
        # self.smpl2imu = self.init_oris[3].matmul(self.root_2_smpl).view(3, 3).t()

        back_2_smpl = self.config['root_2_back'].transpose(1, 2).matmul(self.root_2_smpl)
        imu2smpl = self.init_oris[2].matmul(back_2_smpl).view(3, 3)
        # imu2smpl = self.init_oris[2].matmul(self.root_2_smpl).view(3, 3)
        column_y = torch.FloatTensor([[0, 0, 1]])
        column_z = normalize_tensor(
            imu2smpl[:, [2]].t() - (column_y * imu2smpl[:, [2]].t()).sum(dim=1, keepdim=True) * column_y)
        column_x = column_y.cross(column_z, dim=1)
        imu2smpl = torch.stack((column_x, column_y, column_z), dim=-1).squeeze(0)
        self.smpl2imu = imu2smpl.t()

        self.device2bone = self.smpl2imu.matmul(self.init_oris).transpose(1, 2).matmul(torch.eye(3)).cuda()
        self.acc_offsets = self.smpl2imu.matmul(self.init_oris).matmul(self.tpose_accs.unsqueeze(-1)).cuda()
        self.I_2_Ibias = self.I_2_Ibias.cuda()
        self.smpl2imu = self.smpl2imu.cuda()

        print(self.device2bone)

    @torch.no_grad()
    def calibrate(self, input: list):
        """
        用于标定校准处理
        :param input: 传感器数据序列[48] -> [4 x 3 (acc) + 4 x 9(oris)]
        :return: 标定校准处理后的数据/原始传感器数据
        """
        input = torch.tensor(input).cuda()
        accs, oris = input[0:self.imu_num*3].view(-1, self.imu_num, 3), input[self.imu_num*3:].view(-1, self.imu_num, 3, 3)
        oris = self.I_2_Ibias.matmul(oris)
        accs = (self.smpl2imu.matmul(oris).matmul(accs.view(-1, self.imu_num, 3, 1)) - self.acc_offsets).view(-1, self.imu_num, 3)

        oris = self.smpl2imu.matmul(oris).matmul(self.device2bone)

        input_imu = self.normalize_and_concat(accs, oris).view(-1)

        # elbow_angle = elbow_angle - self.tpose_elbow_angle
        # elbow_angle = elbow_angle_process(elbow_angle)
        # input = torch.cat([input_imu, elbow_angle], dim=0)

        input = input_imu

        return np.array(input.cpu())

    def operator(self, input: np.ndarray):
        """
        在标定校准后，进行预处理
        :param input: 标定校准处理后的数据/原始传感器数据
        :return: 无返回数据
        """
        max_length = 200
        input = torch.tensor(input).reshape(-1)
        # acc, rot, angle = input[:12], input[12:48], input[48:]
        # rot_r6d = rotation_matrix_to_r6d(rot.reshape(4, 3, 3))
        # rot_r6d = rot_r6d.reshape(-1)
        # input = torch.cat([acc, rot_r6d], dim=-1)

        self.acc_cat_oris.append(input)
        if len(self.acc_cat_oris) > max_length:
            self.acc_cat_oris = self.acc_cat_oris[-max_length:]

    def to_predict_data(self):
        """
        onnx.run的参数
        :return: output_names, input_feed, run_options
        """

        # 转r6d
        acc_cat_oris = self.acc_cat_oris[-1]
        acc_cat_oris = np.array(acc_cat_oris.unsqueeze(0))

        # 不转r6d
        # acc_cat_oris = self.acc_cat_oris[-1]
        # acc_cat_oris = np.array(acc_cat_oris.unsqueeze(0))


        input_feed = {'imu_data': acc_cat_oris,
                      'h_1': self.h_1, 'c_1': self.c_1,
                      'h_2': self.h_2, 'c_2': self.c_2}
        if self.track_trans:
            input_feed.update({'h_t': self.h_t, 'c_t': self.c_t})

        return input_feed

    def predict_result(self, result, root_fix=False):
        """
        返回的结果
        :return: [24, 3] 24个关节的轴角
        """
        if self.track_trans:
            result, result_2 = result[:-3], result[-3:]
            d_trans, self.h_t, self.c_t = result_2
            self.trans += torch.tensor(d_trans)
        if self.keep_hidden == False:
            result, _, _, _, _ = result
            result = result[-1]
        else:
            result, self.h_1, self.c_1, self.h_2, self.c_2 = result

        result = torch.tensor(result).reshape(24, 3)

        # root_oris = self.acc_cat_oris[-1][-9:].reshape(-1, 3, 3)
        # root_oris = rotation_matrix_to_axis_angle(root_oris)
        # result[0] = root_oris.view_as(result[0])

        return result

    def get_trans(self):
        self.trans = self.trans * 0.99
        return self.trans.view(-1) * torch.FloatTensor([1, -1, 1]) / 30

# 这个版本使用基于关节异常动态判别的方式触发calibration
class DataProcessServer_6IMU_new():
    def __init__(self, rotation_type, part, keep_hidden=True, run_unity_package=True, mode=demo_mode.UPPER,
                 track_trans=True, calibration_session=None):
        # import calibration_animation.main as caan
        # caan.main()

        if run_unity_package:
            run_mode1()
            # run_mode_girl()
            # run_mode_micai()
            # run_mode_up()
        server_for_unity = socket(AF_INET, SOCK_STREAM)
        server_for_unity.bind(('127.0.0.1', 8888))
        server_for_unity.listen(1)
        print('Server start. Waiting for unity3d to connect.')
        self.conn, self.addr = server_for_unity.accept()
        self.rotation_type = rotation_type
        self.part = part
        self.keep_hidden = keep_hidden
        self.track_trans = track_trans

        self.imu_num = 6
        self.clock = Clock()

        self.mode = mode
        self.calibration_session = calibration_session
        self.property()
        self.trans_property()
        self.udp_client = socket(AF_INET, SOCK_DGRAM)  # 创建socket对象，走udp通道
    def property(self):
        """
        自定义属性
        :return:无返回值
        """
        # 用于calibration的数据
        self.init_oris = None
        self.tpose_accs = None

        self.global_shift = torch.eye(3).reshape(-1, 3, 3).repeat(self.imu_num, 1, 1).cuda()
        self.local_shift = torch.eye(3).reshape(-1, 3, 3).repeat(self.imu_num, 1, 1).cuda()

        # 串口版
        # self.root_2_smpl = bulid_rot(theta=180, rotation_axis=[0,0,1]).matmul(bulid_rot(theta=180, rotation_axis=[1,0,0]))
        self.root_2_smpl = bulid_rot(theta=180, rotation_axis=[0, 1, 0])


        self.smpl2imu = None
        self.device2bone = None
        self.acc_offsets = None

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        n_layer = 1
        self.h_1 = np.zeros(shape=[n_layer, 1, 128]).astype(np.float32)
        self.h_2 = np.zeros(shape=[n_layer, 1, 128]).astype(np.float32)
        self.c_1 = np.zeros(shape=[n_layer, 1, 128]).astype(np.float32)
        self.c_2 = np.zeros(shape=[n_layer, 1, 128]).astype(np.float32)

        # 模型输入
        self.acc_cat_oris = []
        self.rot_buffer = []

    def trans_property(self):
        # 位移相关
        self.trans = torch.FloatTensor([0, 0, 0])

        self.body_model = art.ParametricModel(paths.smpl_file)
        p = torch.eye(3).unsqueeze(0).repeat(24, 1, 1).unsqueeze(0)
        self.body_shape = torch.zeros(10)
        self.init_trans = torch.zeros(3)
        # 输入24个关节旋转+体型参数+位移信息, 输出24个关节的旋转+蒙皮点加速度+运动速度
        grot, joint = self.body_model.forward_kinematics(p, self.body_shape, self.init_trans, calc_mesh=False)
        joint = joint[0]

        self.root_height = joint[0][1] - min(joint[10][1], joint[11][1])
        self.root_height_init = joint[0][1] - min(joint[10][1], joint[11][1])
        self.last_joint_pos = joint

        self.max_buffer_len = 1000
        self.G = torch.FloatTensor([0, -9.7, 0])
        self.vel = torch.FloatTensor([0, 0, 0])
        self.root_acc = torch.FloatTensor([0, 0, 0])
        self.last_root_acc = torch.FloatTensor([0, 0, 0])
        self.p_contact = torch.ones(1)
        # 可作为接触点的关节
        self.contactable_joint = [7, 8, 4, 5, 0, 20, 21, 18, 19]
        # 接触点关节对应的力作用点
        self.force_acting_point = [1, 2, 1, 2, 3, 16, 17, 16, 17]

        self.support_order_mask = torch.FloatTensor([0, 0, 1, 1, 2, 3, 3, 4, 4])

        # 重心关节
        self.barycentre_joint = 3
        # 总质量
        self.mass = 1

        self.support_joint_idx = 8

        self.floating_prob = 0

        self.d_trans_fk_last = torch.FloatTensor([0, 0, 0])

        self.vote_buffer = VoteBuffer(n_item=24, buffer_len=10)

    def get_device_2_bone(self):
        root_2_smpl = self.root_2_smpl
        left_arm_2_smpl = bulid_rot(theta=90, rotation_axis=[1, 0, 0]).matmul(bulid_rot(theta=180, rotation_axis=[0, 1, 0]))
        right_arm_2_smpl = bulid_rot(theta=90, rotation_axis=[1, 0, 0]).matmul(bulid_rot(theta=180, rotation_axis=[0, 1, 0]))
        left_leg_smpl = bulid_rot(theta=-90, rotation_axis=[0, 0, 1]).matmul(bulid_rot(theta=-90, rotation_axis=[0, 1, 0]))
        right_leg_smpl = bulid_rot(theta=90, rotation_axis=[0, 0, 1]).matmul(bulid_rot(theta=90, rotation_axis=[0, 1, 0]))
        head_2_smpl = bulid_rot(theta=0, rotation_axis=[1, 0, 0])

        device_2_bone = [left_arm_2_smpl, right_arm_2_smpl, left_leg_smpl, right_leg_smpl, head_2_smpl, root_2_smpl]

        if self.mode == demo_mode.FULL:
            left_lowerleg_2_smpl = self.config_pant['left_lowerleg_2_smpl']
            right_lowerleg_2_smpl = self.config_pant['right_lowerleg_2_smpl']
            left_pelvis_2_smpl = self.config_pant['left_pelvis_2_smpl']
            right_pelvis_2_smpl = self.config_pant['right_pelvis_2_smpl']

            device_2_bone += [left_lowerleg_2_smpl, right_lowerleg_2_smpl, left_pelvis_2_smpl, right_pelvis_2_smpl]
        return torch.cat(device_2_bone, dim=0)

    def normalize_and_concat(self, smpl_acc, smpl_ori):
        imu_num = self.imu_num
        acc_scale = 30
        smpl_acc = smpl_acc.view(-1, imu_num, 3)
        smpl_ori = smpl_ori.view(-1, imu_num, 3, 3)

        # print(smpl_ori[:100, 0])
        # acc: [n, 4, 3]
        # acc = torch.cat((smpl_acc[:, :(imu_num-1)] - smpl_acc[:, (imu_num-1):], smpl_acc[:, (imu_num-1):]), dim=1).bmm(smpl_ori[:, -1]) / acc_scale
        acc = smpl_acc / acc_scale
        # 不转换为相对加速度 变换至root坐标系
        # acc = smpl_acc.bmm(smpl_ori[:, -1]) / acc_scale

        # ori = torch.cat((smpl_ori[:, (imu_num-1):].transpose(2, 3).matmul(smpl_ori[:, :(imu_num-1)]), smpl_ori[:, (imu_num-1):]), dim=1)
        ori = smpl_ori

        data = torch.cat((acc.flatten(1), ori.flatten(1)), dim=1)
        return data

    def update_data(self, data):
        self.clock.tick()
        # print('\r', f'fps: {self.clock.get_fps()}', end='')
        # self.frame_num += 1
        # self.frame_num = self.frame_num % 2
        # if self.frame_num == 0:
        #     return

        pose = preprocess(data, self.rotation_type)

        # part of data freeze
        # pose.view(24, 3)[config.joint_set.part[self.part]] *= 0

        tran = self.get_trans()
        # send pose
        s = ','.join(['%g' % v for v in pose]) + '#' + \
            ','.join(['%g' % v for v in tran]) + '$'
        self.conn.send(s.encode('utf8'))  # I use unity3d to read pose and translation for visualization here

    def device2bone_init(self):
        self.device2bone = self.get_device_2_bone().cuda()

    def set_calibrate_data(self, input: list):
        """
        设置标定数据
        :param input: [48] -> [4 x 3 (acc) + 4 x 9(oris)]
        :return:无返回值
        """
        g = 9.80665
        self.init_oris = torch.tensor(input[self.imu_num*3:]).view(self.imu_num, 3, 3)
        self.tpose_acc = torch.tensor(input[:self.imu_num*3]).view(self.imu_num, 3, 1)
        # print(self.tpose_acc)

        self.acc_offsets = torch.FloatTensor([[0, g, 0]]).repeat(self.imu_num, 1).cuda()
        # self.acc_offsets = self.init_oris.matmul(self.tpose_acc).cuda()
        print(self.acc_offsets)
        # self.tpose_elbow_angle = torch.tensor(input[48:50])

        # 修正全局坐标系误差
        # self.init_oris = self.I_2_Ibias.matmul(self.init_oris)
        # self.smpl2imu = self.init_oris[-1].matmul(self.root_2_smpl).view(3, 3).t().cuda()

        # Build a heading-invariant calibration frame so front-facing and
        # back-facing T-poses map to the same canonical orientation.
        column_y = torch.FloatTensor([[0, 0, 1]])
        column_z = torch.FloatTensor([[0, 1, 0]])
        column_x = column_y.cross(column_z, dim=1)
        imu2smpl = torch.stack((column_x, column_y, column_z), dim=-1).squeeze(0)
        self.smpl2imu = imu2smpl.t().cuda()

        # print(self.smpl2imu)

        if self.device2bone is None:
            self.device2bone = self.smpl2imu.matmul(self.init_oris.cuda()).transpose(1, 2).matmul(torch.eye(3).cuda())

    def anime_update(self, rotation_diversity, trigger):
        send_str = ''
        for rd in rotation_diversity:
            send_str += str(int(rd)) + ' '
        for i, t in enumerate(trigger):
            if t:
                send_str += str(i) + ' '
        self.udp_client.sendto(send_str.encode('utf-8'), ("127.0.0.1", 23338))

    @torch.no_grad()
    def auto_calibrate(self, time_gap=1, times=1):
        import time
        for _ in range(times):
            if len(self.rot_buffer) < 10:
                continue
            time.sleep(time_gap)
            frame_nums = min(len(self.rot_buffer), 512)
            # 降采样到128
            acc_cat_oris = torch.stack(self.rot_buffer[-frame_nums:]).reshape(frame_nums, -1)[::2]
            # 目前版本不使用加速度了, 全置0
            # acc_cat_oris[:, :self.imu_num * 3] *= 0
            oris = acc_cat_oris[:, self.imu_num * 3:].reshape(1, -1, self.imu_num, 3, 3)
            # 旋转丰富度
            diversity = rotation_diversity(oris).reshape(-1)

            diversity_threshold = torch.Tensor([50, 50, 20, 20, 20, 20]) * 1
            trigger_s2 = diversity > diversity_threshold
            # print(diversity)

            trigger = trigger_s2

            self.anime_update(diversity, trigger)

            if len(self.rot_buffer) < 256:
                continue

            keep_mask = ~trigger

            # 根节点不更新
            keep_mask[-1] = True
            skip_count = torch.sum(keep_mask).item()
            if skip_count < 3:
                self.global_shift = r6d_to_rotation_matrix(rotation_matrix_to_r6d(self.global_shift))
                self.local_shift = r6d_to_rotation_matrix(rotation_matrix_to_r6d(self.local_shift))
                oris = oris.reshape(1, -1, self.imu_num*(3 * 3))
                feed = {'imu_rot': np.array(oris)}
                global_shift, local_shift = self.calibration_session.run(output_names=None, input_feed=feed)
                global_shift = torch.FloatTensor(global_shift).reshape(-1, 6)
                global_shift = r6d_to_rotation_matrix(global_shift).transpose(-2,-1).cuda()

                # global_shift = r6d_to_rotation_matrix(torch.tensor(global_shift).reshape(-1, 6))
                # global_shift = drift_rot_2_heading_ref(global_shift).squeeze(0).transpose(-2, -1).cuda()


                local_shift = r6d_to_rotation_matrix(torch.FloatTensor(local_shift).reshape(-1, 6)).transpose(-2, -1).cuda()
                self.rot_buffer = self.rot_buffer[-1:]
                global_shift[keep_mask, :, :] = torch.eye(3).cuda().unsqueeze(0).repeat(skip_count, 1, 1)
                local_shift[keep_mask, :, :] = torch.eye(3).cuda().unsqueeze(0).repeat(skip_count, 1, 1)

                # angle_evaluator = PerJointRotationErrorEvaluator()
                # global_fix = angle_evaluator(global_shift.unsqueeze(0),
                #                              torch.eye(3).cuda().unsqueeze(0).repeat(self.imu_num, 1, 1).unsqueeze(0),
                #                              joint_num=self.imu_num)
                # local_fix = angle_evaluator(local_shift.unsqueeze(0),
                #                             torch.eye(3).cuda().unsqueeze(0).repeat(self.imu_num, 1, 1).unsqueeze(0),
                #                             joint_num=self.imu_num)

                # print('global矫正量', global_fix)
                # print('local矫正量', local_fix)
                self.global_shift = global_shift.matmul(self.global_shift)
                self.local_shift = self.local_shift.matmul(local_shift)
                print('自动校准')
            else:
                continue


    @torch.no_grad()
    def calibrate(self, input: list):
        """
        用于标定校准处理
        :param input: 传感器数据序列[48] -> [4 x 3 (acc) + 4 x 9(oris)]
        :return: 标定校准处理后的数据/原始传感器数据
        """
        input = torch.tensor(input).cuda()
        accs, oris = input[0:self.imu_num * 3].view(-1, self.imu_num, 3), input[self.imu_num * 3:].view(-1,self.imu_num, 3,3)
        # print(oris[-1])
        # acc转到全局坐标 Z轴加速度是反的 补正
        # acc转到全局坐标
        accs = oris.matmul(accs.view(-1, self.imu_num, 3, 1))

        # accs[:, :, 2] *= -1
        # accs =accs.view(-1, self.imu_num, 3).matmul(oris) - self.acc_offsets
        # print(float(accs[0][0][2]))
        # oris转到smpl->bone, 然后补正
        oris = self.global_shift.matmul(self.smpl2imu).matmul(oris).matmul(self.device2bone).matmul(self.local_shift)
        # acc转到smpl 并补正
        accs = self.global_shift.matmul(self.smpl2imu).matmul(accs).view(-1, self.imu_num, 3)

        accs = accs - self.acc_offsets

        self.root_acc = 0.7*accs[0, -1].cpu() + 0.3*self.root_acc

        # print('\r', accs[0, 0], end='')
        input_imu = self.normalize_and_concat(accs, oris).view(-1)

        # elbow_angle = elbow_angle - self.tpose_elbow_angle
        # elbow_angle = elbow_angle_process(elbow_angle)
        # input = torch.cat([input_imu, elbow_angle], dim=0)

        input = input_imu

        return np.array(input.cpu())

    def operator(self, input: np.ndarray):
        """
        在标定校准后，进行预处理
        :param input: 标定校准处理后的数据/原始传感器数据
        :return: 无返回数据
        """
        max_length = 1000
        input = torch.tensor(input).reshape(-1)
        # acc, rot, angle = input[:12], input[12:48], input[48:]
        # rot_r6d = rotation_matrix_to_r6d(rot.reshape(4, 3, 3))
        # rot_r6d = rot_r6d.reshape(-1)
        # input = torch.cat([acc, rot_r6d], dim=-1)

        self.acc_cat_oris.append(input)
        self.rot_buffer.append(input)
        if len(self.acc_cat_oris) > max_length:
            self.acc_cat_oris = self.acc_cat_oris[-max_length:]
        if len(self.rot_buffer) > max_length:
            self.rot_buffer = self.rot_buffer[-max_length:]

    def to_predict_data(self):
        """
        onnx.run的参数
        :return: output_names, input_feed, run_options
        """

        # # 转r6d
        # acc_cat_oris = torch.tensor(self.acc_cat_oris[-1])
        # # print('\r', acc_cat_oris, end='')
        # acc, rot= acc_cat_oris[:self.imu_num*3], acc_cat_oris[self.imu_num*3:]
        # # print(rot.reshape(8, 3, 3))
        # rot_r6d = rotation_matrix_to_r6d(rot.reshape(self.imu_num, 3, 3))
        # rot_r6d = rot_r6d.reshape(-1)
        # acc_cat_oris = torch.cat([acc, rot_r6d], dim=-1)
        # acc_cat_oris = np.array(acc_cat_oris.unsqueeze(0))

        # 不转r6d
        acc_cat_oris = self.acc_cat_oris[-1]
        acc_cat_oris = np.array(acc_cat_oris.unsqueeze(0))


        input_feed = {'imu_data': acc_cat_oris,
                      'h_1': self.h_1,
                      'h_2': self.h_2}

        return input_feed

    def predict_result(self, result):
        """
        返回的结果
        :return: [24, 3] 24个关节的轴角
        """
        pose, vel, self.h_1, self.c_1, self.h_2, self.c_2, = result

        result = torch.FloatTensor(pose).reshape(24, 3)
        result[[7, 8]] *= 0

        p = axis_angle_to_rotation_matrix(result)
        joint_rot, _ = self.body_model.forward_kinematics(p.unsqueeze(0), self.body_shape, self.init_trans,
                                                                  calc_mesh=False)
        # joint_rot = joint_rot[0]
        # if len(self.rot_buffer) > 0:
        joint_rot[0, [18, 19, 4, 5, 15]] = self.rot_buffer[-1][self.imu_num * 3:].reshape(self.imu_num, 3, 3)[:5]
        pose = self.body_model.inverse_kinematics_R(joint_rot).squeeze(0)
        pose[[7, 8, 10, 11, 20, 21, 22, 23]] = torch.eye(3).unsqueeze(0).repeat(8, 1, 1)

        # IMU映射后重新计算joint_pos
        _, joint_pos = self.body_model.forward_kinematics(pose.unsqueeze(0), self.body_shape, self.init_trans,
                                                                  calc_mesh=False)

        pose = rotation_matrix_to_axis_angle(pose)
        result = pose

        joint_pos = joint_pos[0]
        # vel = vel[0]

        # 这个不准
        self.p_contact = ((self.root_acc[1].clamp(min=self.G[1], max=0) - self.G[1]) / 1).clamp(min=0.0001, max=0.9999)


        # 受外力估计
        external_force = self.mass * (self.root_acc - self.G * self.p_contact).reshape(-1, 3)
        support_direction = joint_pos[self.force_acting_point] - joint_pos[self.contactable_joint]
        # print(external_force)
        # print(support_direction)

        support_angles = torch.abs(
            self.compute_angle(external_force.repeat(len(self.force_acting_point), 1), support_direction))
        support_angles[support_angles > 35] = 180 * 5
        support_angles.reshape(-1)
        support_angles += self.support_order_mask * 180
        # print(support_angles)
        min_ang, min_idx = torch.min(support_angles, dim=-1)
        support_joint_idx = self.contactable_joint[min_idx]

        # print('支撑点:', support_joint_idx, '总外力:', external_force)
        if self.p_contact < 0.99:
            print('支撑概率:', self.p_contact)

        d_trans_fk = self.last_joint_pos[support_joint_idx] - joint_pos[support_joint_idx]

        self.vote_buffer.vote(support_joint_idx, weight=35-(min_ang-self.support_order_mask[min_idx]*180))
        if self.support_joint_idx != self.vote_buffer.get_max_vote():
            print('支撑点:', self.support_joint_idx)
            # print(self.vote_buffer.buffer)
        self.support_joint_idx = self.vote_buffer.get_max_vote()


        self.d_trans_fk_last = d_trans_fk

        self.last_joint_pos = joint_pos

        self.vel = self.p_contact * d_trans_fk * 60 + (1 - self.p_contact) * self.root_acc / 60
        self.trans += self.vel / 60

        lowest_position, _ = torch.min(joint_pos[:, 1], dim=-1)
        # support_position = joint_pos[self.support_joint_idx, 1]
        root_height_fk = joint_pos[0][1] - lowest_position - self.root_height_init

        self.trans[1] = max(root_height_fk, self.p_contact * root_height_fk + (1 - self.p_contact) * self.trans[1])

        # print('========================')
        # track_joint = {'左肩': 16, '右肩': 17, '脊柱-1': 3, '脊柱-2': 6, '脊柱-3': 9, '左肘': 18, '右肘': 19, '腰部': 0,
        #                '左胯': 1, '右胯': 2, '左膝': 4, '右膝': 5}
        # for jn, ji in track_joint.items():
        #     print(jn, ':', np.array(result[ji]).tolist())

        return result

    def predict_result2(self, result):
        """
        返回的结果
        :return: [24, 3] 24个关节的轴角
        """
        pose, vel, self.h_1, self.c_1, self.h_2, self.c_2, = result

        # print(pose)

        vel = torch.FloatTensor(vel).reshape(-1, 3) * 10

        contact_left = 1 - (torch.norm(vel[10], 2) - 0.04).clamp(min=0, max=0.2) / 0.2
        contact_right = 1 - (torch.norm(vel[11], 2) - 0.04).clamp(min=0, max=0.2) / 0.2

        vel_left = torch.norm(vel[10], 2)
        vel_right = torch.norm(vel[11], 2)

        print(vel_left, vel_right)

        result = torch.FloatTensor(pose).reshape(24, 3)
        result[[7, 8]] *= 0

        p = axis_angle_to_rotation_matrix(result)
        joint_rot, joint_pos = self.body_model.forward_kinematics(p.unsqueeze(0), self.body_shape, self.init_trans,
                                                                  calc_mesh=False)

        # imu_raw_rot = self.rot_buffer[-1].reshape(12, 3, 3)
        # joint_rot = joint_rot[0]
        # joint_rot[[18, 19, 16, 17]] = imu_raw_rot[:4]
        # fix_pose = self.body_model.inverse_kinematics_R(joint_rot.unsqueeze(0)).squeeze(0)
        # result = rotation_matrix_to_axis_angle(fix_pose)

        joint_pos = joint_pos[0]
        # print(joint_pos, self.last_joint_pos)
        # fk位移
        if vel_left < vel_right:
            d_trans_fk = self.last_joint_pos[10] - joint_pos[10]
        else:
            d_trans_fk = self.last_joint_pos[11] - joint_pos[11]
        self.last_joint_pos = joint_pos

        # nn位移
        d_trans_nn = torch.FloatTensor(vel).reshape(-1, 3)[0] / 60

        d_root_height_nn = d_trans_nn[1]

        # nn & fk高度
        root_height_fk = joint_pos[0][1] - min(joint_pos[10][1], joint_pos[11][1]) - self.root_height_init
        s = max(contact_left, contact_right)

        # s = (torch.abs(self.root_acc[1] - self.G[1]) / 8.0).clamp(min=0.0001, max=0.9999)
        # s = ((self.root_acc[1].clamp(min=self.G[1], max=0) - self.G[1]) / 5).clamp(min=0.0001, max=0.9999)

        if s < 0.1:
            self.floating_prob = min(max(self.floating_prob + 0.3, 0), 1)
            d_trans = d_trans_nn
        else:
            self.floating_prob = min(max(self.floating_prob - 0.3, 0), 1)
            d_trans = self.floating_prob * d_trans_nn + (1 - self.floating_prob) * d_trans_fk

        self.root_height = (1 - self.floating_prob) * root_height_fk + self.floating_prob * (
                self.root_height + d_root_height_nn)

        self.root_height = max(root_height_fk, self.root_height)

        # if self.track_trans:
        self.trans[[0, 2]] += d_trans[[0, 2]]
        self.trans[1] = self.root_height

        # print('========================')
        # track_joint = {'左肩': 16, '右肩': 17, '脊柱-1': 3, '脊柱-2': 6, '脊柱-3': 9, '左肘': 18, '右肘': 19, '腰部': 0,
        #                '左胯': 1, '右胯': 2, '左膝': 4, '右膝': 5}
        # for jn, ji in track_joint.items():
        #     print(jn, ':', np.array(result[ji]).tolist())

        return result

    def get_trans(self):
        return self.trans.view(-1)

    def compute_angle(self, vectors1, vectors2):
        """
        计算两个批次三维向量的夹角
        :param vectors1: 第一个批次的三维向量，形状为 (batch_size, 3)
        :param vectors2: 第二个批次的三维向量，形状为 (batch_size, 3)
        :return: 夹角的度数，形状为 (batch_size,)
        """

        # 确保输入是浮点数类型
        vectors1 = vectors1.float()
        vectors2 = vectors2.float()

        # 计算两个向量的点积
        dot_product = torch.sum(vectors1 * vectors2, dim=-1)

        # 计算向量的模
        norm_vectors1 = torch.norm(vectors1, dim=-1)
        norm_vectors2 = torch.norm(vectors2, dim=-1)

        # 计算余弦值
        cos_angles = dot_product / (norm_vectors1 * norm_vectors2)

        # 为了避免数值超出 [-1, 1] 范围，使用 clamp 均分
        cos_angles = torch.clamp(cos_angles, -1.0, 1.0)

        # 计算夹角，转换为弧度
        angles_rad = torch.acos(cos_angles)

        # 转换为度
        angles_deg = angles_rad * (180.0 / torch.pi)

        return angles_deg

class DataProcessServer_FullBody():
    def __init__(self, rotation_type, part, config, keep_hidden=True, run_unity_package=True, mode=demo_mode.UPPER,
                 track_trans=False, calibration_session=None, physics_optim=False, cali_pose='T', beta=None, imu_num = 10, unity_mode = 'single'):

        self.run_unity_package = run_unity_package
        # PIP的物理优化模块
        if physics_optim:
            from mos.motion_optimizer import MotionOptimizer
            self.predict_result = self.predict_result_optim
            self.motion_optimizer = MotionOptimizer(fps=30)
        self.beta_len = None
        self.rotation_type = rotation_type
        self.part = part
        self.keep_hidden = keep_hidden
        self.track_trans = track_trans
        self.trans = torch.FloatTensor([0, 0, 0])
        self.R_wrist_vp_last = np.array([0, 0, 0], dtype=np.float32)
        self.L_wrist_vp_last = np.array([0, 0, 0], dtype=np.float32)
        # self.Ground_2_root = torch.FloatTensor([0, 0, 0])
        self.invalid_r_counter = 0
        self.invalid_l_counter = 0

        self.imu_num = imu_num
        self.clock = Clock()
        if beta is not None:
            self.beta = beta
        self.mode = mode
        if isinstance(config, list):
            self.config_clothes, self.config_pant = tuple(config)
            self.config = self.config_clothes
        else:
            self.config = config
            self.config_clothes = config

        self.calibration_session = calibration_session
        self.property()
        self.udp_client = socket(AF_INET, SOCK_DGRAM)  # 创建socket对象，走udp通道
        from Aplus.tools.smpl_light import SMPLight
        # if self.beta is not None:
        self.R_upper = 0
        self.R_lower = 0
        self.L_upper = 0
        self.L_lower = 0

        self.body_model = SMPLight()
        # self.body_model = SMPLight_with_bodyshape()
        # self.body_model.set_beta(beta=self.beta)

        # else:
        #     self.body_model = SMPLight()
        # self.body_model = SMPLight()

        p = torch.eye(3).unsqueeze(0).repeat(24, 1, 1).unsqueeze(0)
        grot, joint = self.body_model.forward_kinematics(p, calc_joint=True)
        joint = joint[0]

        if cali_pose == 'N':
            self.cali_pose = SMPLPose.n_pose_ori
        else:
            self.cali_pose = SMPLPose.t_pose_ori
        self.root_height = joint[0][1] - min(joint[10][1], joint[11][1])
        self.root_height_init = joint[0][1] - min(joint[10][1], joint[11][1])
        self.last_joint_pos = joint
        self.frame_num = 0
        self.floating_prob = 0
        self.max_buffer_len = 1000
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.unity_mode = unity_mode
        self.global_tran_origin = None

    def unity_start(self, run_unity_package):

        server_for_unity = socket(AF_INET, SOCK_STREAM)
        server_for_unity.bind(('127.0.0.1', 8888))
        server_for_unity.listen(1)
        print('Server start.')
        if run_unity_package:
            print('内置Unity启动中')
            if self.unity_mode == 'single_global':
                run_mode3()
            elif self.unity_mode == 'twins':
                run_mode2()
            elif self.unity_mode == 'single':
                run_mode1()
            elif self.unity_mode == 'single_local':
                run_mode4()
            # run_mode_vision()
            # run_mode_vision()
            # run_mode_girl()
            # run_mode_micai()
            # run_mode_up()
        else:
            print('手动启动Unity以开始动作捕捉')
        self.conn, self.addr = server_for_unity.accept()
    def property(self):
        """
        自定义属性
        :return:无返回值
        """
        # 用于calibration的数据
        self.init_oris = None
        self.tpose_accs = None

        self.global_shift = torch.eye(3).reshape(-1, 3, 3).repeat(self.imu_num, 1, 1)
        self.local_shift = torch.eye(3).reshape(-1, 3, 3).repeat(self.imu_num, 1, 1)

        self.global_shift0 = torch.eye(3).reshape(-1, 3, 3).repeat(self.imu_num, 1, 1)
        self.local_shift0 = torch.eye(3).reshape(-1, 3, 3).repeat(self.imu_num, 1, 1)


        # 串口版
        # self.root_2_smpl = bulid_rot(theta=180, rotation_axis=[0,0,1]).matmul(bulid_rot(theta=180, rotation_axis=[1,0,0]))
        self.root_2_smpl = bulid_rot(theta=180, rotation_axis=[0, 1, 0])
        self.device2bone = None

        self.smpl2imu = None
        self.acc_offsets = None

        self.h_1 = np.zeros(shape=[2, 1, 256]).astype(np.float32)
        self.c_1 = np.zeros(shape=[2, 1, 256]).astype(np.float32)
        self.h_2 = np.zeros(shape=[2, 1, 256]).astype(np.float32)
        self.c_2 = np.zeros(shape=[2, 1, 256]).astype(np.float32)
        self.h_3 = np.zeros(shape=[2, 1, 256]).astype(np.float32)
        self.c_3 = np.zeros(shape=[2, 1, 256]).astype(np.float32)

        self.diversity_threshold = torch.FloatTensor([50, 50, 30, 30, 30, 30, 30, 30, 30, 30])
        # self.diversity_threshold = torch.FloatTensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        # 模型输入
        self.acc_cat_oris = []
        self.rot_buffer = []

    def get_device_2_bone(self):
        root_2_smpl = self.config_clothes['root_2_smpl'].unsqueeze(0)
        left_2_smpl = self.config_clothes['root_2_left'].transpose(1, 2).matmul(root_2_smpl)
        right_2_smpl = self.config_clothes['root_2_right'].transpose(1, 2).matmul(root_2_smpl)
        back_2_smpl = self.config_clothes['root_2_back'].transpose(1, 2).matmul(root_2_smpl)

        device_2_bone = [left_2_smpl, right_2_smpl, back_2_smpl, root_2_smpl]

        if self.mode == demo_mode.FULL:
            left_lowerleg_2_smpl = self.config_pant['left_lowerleg_2_smpl']
            right_lowerleg_2_smpl = self.config_pant['right_lowerleg_2_smpl']
            left_pelvis_2_smpl = self.config_pant['left_pelvis_2_smpl']
            right_pelvis_2_smpl = self.config_pant['right_pelvis_2_smpl']

            device_2_bone += [left_lowerleg_2_smpl, right_lowerleg_2_smpl, left_pelvis_2_smpl, right_pelvis_2_smpl]
        return torch.cat(device_2_bone, dim=0)

    def normalize_and_concat(self, smpl_acc, smpl_ori):
        imu_num = self.imu_num

        smpl_acc = smpl_acc.view(-1, imu_num, 3)
        smpl_ori = smpl_ori.view(-1, imu_num, 3, 3)

        # print(smpl_ori[:100, 0])
        # acc: [n, 4, 3]
        # acc = torch.cat((smpl_acc[:, :(imu_num-1)] - smpl_acc[:, (imu_num-1):], smpl_acc[:, (imu_num-1):]), dim=1).bmm(smpl_ori[:, -1]) / acc_scale
        acc = smpl_acc
        # 不转换为相对加速度 变换至root坐标系

        ori = smpl_ori

        data = torch.cat((acc.flatten(1), ori.flatten(1)), dim=1)
        return data

    def update_data(self, data):

        pose = preprocess(data, self.rotation_type)

        tran = self.get_trans()
        # if trans is not None:
        #     # tran = trans
        #     tran_y = tran[1]
        #     trans[1] = tran_y
        # send pose
        # if self.unity_mode == 'single':
        s = ','.join(['%g' % v for v in pose]) + '#' + \
            ','.join(['%g' % v for v in tran]) + '$'
        # elif self.unity_mode == 'twins':
        #     s = (
        #     ','.join(['%g' % v for v in pose]) + '#' +
        #     ','.join(['%g' % v for v in tran]) + '$' +
        #     ','.join(['%g' % v for v in pose]) + '#' +
        #     ','.join(['%g' % v for v in tran]) + '@'
        #     )
        self.conn.send(s.encode('utf8'))  # I use unity3d to read pose and translation for visualization here
    
    def update_data_global(self, data,tran,reset_origin=False):

        pose = preprocess(data, self.rotation_type).numpy()
        pose = pose.reshape(24,3)
        # pose[ :, 0] *= -1
        # pose[ :, 2] *= -1
        pose = pose.reshape(-1)


        shape = torch.zeros(10).numpy()

        s = ','.join(['%g' % v for v in pose]) + '#' + \
            ','.join(['%g' % v for v in tran]) + '$' + \
            ','.join(['%g' % v for v in shape]) + '@'
        

        self.conn.send(s.encode('utf8'))  # I use unity3d to read pose and translation for visualization here

    def update_data_twins(self, data1,tran1,data2,tran2):

        pose1 = preprocess(data1, self.rotation_type).numpy()
        pose2 = preprocess(data2, self.rotation_type).numpy()

        # tran = self.get_trans()
        s = (
        ','.join(['%g' % v for v in pose1]) + '#' +
        ','.join(['%g' % v for v in tran1]) + '$' +
        ','.join(['%g' % v for v in pose2]) + '#' +
        ','.join(['%g' % v for v in tran2]) + '@'
        )
        self.conn.send(s.encode('utf8')) 


    def apply_fk(self, rotations, positions, parents):
        """简化FK: 输入每个局部旋转（axis-angle），根据parent关系迭代得到每个节点的全局位置"""
        global_rot = [R.from_rotvec(rotations[0])]
        global_pos = [positions[0]]

        for i in range(1, len(rotations)):
            parent = parents[i]
            r = R.from_rotvec(rotations[i])
            rel_pos = positions[i] - positions[parent]
            global_rot.append(global_rot[parent] * r)
            global_pos.append(global_pos[parent] + global_rot[parent].apply(rel_pos))

        return np.array(global_pos)

    def ik_optimize_hand(self, joint_pos, joint_rot, vp_left_wrist, vp_right_wrist):
        # 索引定义（SMPL标准）
        LEFT_IDS = [17, 19, 21]  # 左肩 左肘 左腕
        RIGHT_IDS = [16, 18, 20]  # 右肩 右肘 右腕

        # 对应局部 parent 索引
        PARENT_L = [-1, 0, 1]
        PARENT_R = [-1, 0, 1]

        def optimize_chain(j_ids, vp_target, parent_list):
            # 提取局部位置
            global_pos = joint_pos[j_ids]
            local_pos = np.zeros_like(global_pos)
            for i, p in enumerate(parent_list):
                if p == -1:
                    local_pos[i] = np.zeros(3)
                else:
                    local_pos[i] = global_pos[i] - global_pos[p]

            # 初始旋转向量
            init_rots = [R.from_matrix(joint_rot[j]).as_rotvec() for j in j_ids]

            def fk_loss(rotvecs):
                rotvecs = rotvecs.reshape(-1, 3)
                fk_pos = self.apply_fk(rotvecs, local_pos, parent_list)
                return (fk_pos[-1] - vp_target)

            result = least_squares(fk_loss, np.concatenate(init_rots), method='trf')
            opt_rots = result.x.reshape(-1, 3)
            final_pos = self.apply_fk(opt_rots, local_pos, parent_list)
            return [R.from_rotvec(r).as_matrix() for r in opt_rots], final_pos

        # 左右手优化
        left_rot, left_pos = optimize_chain(LEFT_IDS, vp_left_wrist, PARENT_L)
        right_rot, right_pos = optimize_chain(RIGHT_IDS, vp_right_wrist, PARENT_R)

        # 更新全局 joint_rot（原地替换）
        for idx, mat in zip(LEFT_IDS, left_rot):
            joint_rot[idx] = torch.from_numpy(mat).to(joint_rot.device).type_as(joint_rot)

        for idx, mat in zip(RIGHT_IDS, right_rot):
            joint_rot[idx] = torch.from_numpy(mat).to(joint_rot.device).type_as(joint_rot)

        # 拼接新位置（6个点：左肩、左肘、左腕，右肩、右肘、右腕）
        new_positions = np.vstack([left_pos, right_pos])  # shape: (6, 3)

        return joint_rot, new_positions

    def fk_wrist(self, R_upper, R_lower, shoulder_pos, L_upper, L_lower):
        elbow_pos = shoulder_pos + R_upper @ np.array([0, 0, L_upper])
        wrist_pos = elbow_pos + R_upper @ R_lower @ np.array([0, 0, L_lower])
        return elbow_pos, wrist_pos

    def fk_wrist(self, R_upper, R_lower, shoulder_pos, L_upper, L_lower):
        elbow_pos = shoulder_pos + R_upper @ np.array([0, 0, L_upper])
        wrist_pos = elbow_pos + R_upper @ R_lower @ np.array([0, 0, L_lower])
        return elbow_pos, wrist_pos

    def optimize_arm(self,
            shoulder_pos, wrist_pos_vp,
            R_upper_prior, R_lower_prior,
            L_upper, L_lower,
            enable_elbow_prior=False, elbow_prior=None,
            λ_pos=3000.0, λ_R1=0.1, λ_R2=0.01, λ_elbow=1.0
    ):
        """
        优化手臂旋转，使 wrist 对齐 VP 提供的位置。

        参数：
            shoulder_pos: (3,) 肩膀位置
            wrist_pos_vp: (3,) VP 提供的手腕目标位置
            R_upper_prior: (3,3) 初始大臂旋转（可来自IMU）
            R_lower_prior: (3,3) 初始小臂旋转
            L_upper, L_lower: float，骨骼长度
            enable_elbow_prior: 是否启用肘部先验
            elbow_prior: (3,) 可选肘部参考位置
            λ_pos, λ_R1, λ_R2, λ_elbow: 各项损失权重

        返回：
            R_upper_opt, R_lower_opt: 优化后的旋转
            elbow_pos, wrist_pos: 对应关节位置
            joints_global: [shoulder, elbow, wrist]
        """
        # rotvec 初始化
        rotvec_upper_init = R.from_matrix(R_upper_prior).as_rotvec()
        rotvec_lower_init = R.from_matrix(R_lower_prior).as_rotvec()
        x0 = np.concatenate([rotvec_upper_init, rotvec_lower_init])

        def loss_fn(x):
            R_upper = R.from_rotvec(x[:3]).as_matrix()
            R_lower = R.from_rotvec(x[3:]).as_matrix()

            elbow_pred, wrist_pred = self.fk_wrist(R_upper, R_lower, shoulder_pos, L_upper, L_lower)

            loss = 0.0
            # 手腕位置误差
            loss += λ_pos * np.linalg.norm(wrist_pred - wrist_pos_vp) ** 2
            # 大臂先验偏离惩罚
            loss += λ_R1 * np.linalg.norm(R.from_matrix(R_upper.T @ R_upper_prior).as_rotvec()) ** 2
            # 小臂先验偏离惩罚
            loss += λ_R2 * np.linalg.norm(R.from_matrix(R_lower.T @ R_lower_prior).as_rotvec()) ** 2
            # 肘部先验（可选）
            if enable_elbow_prior and elbow_prior is not None:
                loss += λ_elbow * np.linalg.norm(elbow_pred - elbow_prior) ** 2

            return loss

        res = minimize(loss_fn, x0, method='BFGS')

        R_upper_opt = R.from_rotvec(res.x[:3]).as_matrix()
        R_lower_opt = R.from_rotvec(res.x[3:]).as_matrix()
        elbow_opt, wrist_opt = self.fk_wrist(R_upper_opt, R_lower_opt, shoulder_pos, L_upper, L_lower)

        joints_global = np.stack([shoulder_pos, elbow_opt, wrist_opt], axis=0)

        return R_upper_opt, R_lower_opt, elbow_opt, wrist_opt, joints_global

    def compute_elbow(self, shoulder_pos, wrist_pos, L_upper, L_lower, elbow_hint=None):
        v = wrist_pos - shoulder_pos
        D = np.linalg.norm(v)
        if D > L_upper + L_lower:
            # 超出IK可达范围，直接伸直
            return shoulder_pos + L_upper * (v / D)

        v = v / D
        if elbow_hint is None:
            elbow_hint = np.array([0, 1, 0])  # 默认向上
        n = np.cross(v, elbow_hint)
        if np.linalg.norm(n) < 1e-5:
            elbow_hint = np.array([1, 0, 0])
            n = np.cross(v, elbow_hint)
        n = n / np.linalg.norm(n)
        u = np.cross(n, v)

        a = (L_upper ** 2 - L_lower ** 2 + D ** 2) / (2 * D)
        h = np.sqrt(max(L_upper ** 2 - a ** 2, 0.0))

        P = shoulder_pos + a * v
        elbow_pos = P + h * u
        return elbow_pos

    def solve_elbow_direct(self, shoulder_pos, wrist_pos, elbow_pos, L_upper, L_lower):
        # 确保长度为 float（避免是 torch.Tensor）
        L_upper = float(L_upper)
        L_lower = float(L_lower)

        elbow_pos = elbow_pos
        lower_dir = wrist_pos - elbow_pos.numpy()
        lower_dir = lower_dir / (np.linalg.norm(lower_dir) + 1e-8)
        wrist_proj = elbow_pos + lower_dir * L_lower
        return elbow_pos, wrist_proj

    def detect_vp_hand_failure_state(self, R_wrist, L_wrist, head_pos):
        """
        检测 Vision Pro 双手是否失效或单手失效状态。

        Returns:
            str: 'both_invalid', 'right_invalid', 'left_invalid', 'valid'
        """

        if not hasattr(self, "R_wrist_hist"):
            self.R_wrist_hist = [R_wrist.copy()]
            self.L_wrist_hist = [L_wrist.copy()]
            self.state = "valid"
            return "valid"

        self.R_wrist_hist.append(R_wrist.copy())
        self.L_wrist_hist.append(L_wrist.copy())
        if len(self.R_wrist_hist) > 10:
            self.R_wrist_hist.pop(0)
            self.L_wrist_hist.pop(0)

        # === 参数配置 ===
        jump_thresh = 0.05  # 超过这个阈值认为是突变（恢复）
        near_head_thresh = 0.1  # 靠近头部的距离阈值
        still_thresh = 0.005  # 静止时最大运动距离

        def is_near_head(pos):
            return np.linalg.norm(pos - head_pos) < near_head_thresh

        def is_static(pos_hist):
            dists = [np.linalg.norm(pos_hist[i + 1] - pos_hist[i]) for i in range(len(pos_hist) - 1)]
            return all(d < still_thresh for d in dists)

        def is_following_head(wrist_hist):
            head_motions = [np.linalg.norm(head_pos - wrist_hist[i]) for i in range(len(wrist_hist))]
            return all(d < near_head_thresh for d in head_motions)

        right_near = is_near_head(R_wrist)
        left_near = is_near_head(L_wrist)
        right_static = is_static(self.R_wrist_hist)
        left_static = is_static(self.L_wrist_hist)

        # === 判断逻辑 ===
        if right_near and left_near and right_static and left_static:
            state = "both_invalid"
        elif right_near and is_following_head(self.R_wrist_hist):
            state = "right_invalid"
        elif left_near and is_following_head(self.L_wrist_hist):
            state = "left_invalid"
        else:
            state = "valid"

        self.state = state
        return state

    def detect_hand_failure(self, p_l, p_r, p_head,
                            R_l, R_r, R_head,
                            last_l, last_r,
                            dist_thresh=0.05, move_thresh=0.005,
                            rot_target_deg=90.0, rot_tol=5.0,
                            hand_dist_thresh=0.03):

        def rotation_angle(R1, R2):
            R_diff = R2.T @ R1
            cos_theta = np.clip((np.trace(R_diff) - 1) / 2.0, -1.0, 1.0)
            return np.degrees(np.arccos(cos_theta))

        # 距离计算
        dist_lh = np.linalg.norm(p_l - p_head)
        dist_rh = np.linalg.norm(p_r - p_head)
        # print("dist_lh", dist_lh, "dist_rh", dist_rh)
        move_l = np.linalg.norm(p_l - last_l)
        move_r = np.linalg.norm(p_r - last_r)
        dist_lr = np.linalg.norm(p_l - p_r)

        # 旋转差
        rot_l = rotation_angle(R_l, R_head)
        rot_r = rotation_angle(R_r, R_head)

        # 状态初始化
        left_invalid = False
        right_invalid = False
        both_invalid = False
        # LL = abs(rot_l - rot_target_deg)
        # RR = abs(rot_r - rot_target_deg)
        # print("********", LL, RR)
        # ============ 单手失效 ============
        if dist_lh < dist_thresh and abs(rot_l - rot_target_deg) < rot_tol:
            left_invalid = True
            print("左手", left_invalid)
        if dist_rh < dist_thresh and abs(rot_r - rot_target_deg) < rot_tol:
            right_invalid = True
            print("右手", right_invalid)
        # ============ 双手失效 ============
        if dist_lr < hand_dist_thresh:
            left_invalid = True
            right_invalid = True
            both_invalid = True

        return {
            "left_invalid": left_invalid,
            "right_invalid": right_invalid,
            "both_invalid": both_invalid,
            "rot_l": rot_l,
            "rot_r": rot_r,
            "dist_lh": dist_lh,
            "dist_rh": dist_rh,
            "dist_lr": dist_lr
        }

    def _normalize(self, v, eps=1e-8):
        n = np.linalg.norm(v)
        return v / (n + eps)


    def device2bone_init(self):
        self.device2bone = self.get_device_2_bone()

    def set_calibrate_data(self, input_sensor: list):
        """
        设置标定数据
        :param input: [48] -> [4 x 3 (acc) + 4 x 9(oris)]
        :return:无返回值
        """
        # print(input[self.imu_num*3:])
        self.init_oris = torch.tensor(input_sensor[self.imu_num * 3:]).view(self.imu_num, 3, 3)
        # print(self.init_oris)
        # for i in range(self.imu_num):
        #     err = False
        #     if torch.isnan(self.init_oris[i]).any():
        #         err = True
        #         print(f'传感器 {i+1} 数据获取失败!')
        #         print(self.init_oris[i])
        #     if err:
        #         input('请重启设备后重试!')
        self.acc_offsets = torch.FloatTensor([[0, 0, 9.8]]).repeat(self.imu_num, 1).unsqueeze(-1)


        back_2_smpl = self.config_clothes['root_2_back'].transpose(1, 2).matmul(self.root_2_smpl)
        imu2smpl = self.init_oris[-1].matmul(back_2_smpl).view(3, 3)

        column_y = torch.FloatTensor([[0, 0, 1]])
        column_z = normalize_tensor(
            imu2smpl[:, [2]].t() - (column_y * imu2smpl[:, [2]].t()).sum(dim=1, keepdim=True) * column_y)
        column_x = column_y.cross(column_z, dim=1)
        imu2smpl = torch.stack((column_x, column_y, column_z), dim=-1).squeeze(0)
        self.smpl2imu = imu2smpl.t()

        cali_pose_ori = self.cali_pose[[18, 16, 19, 17, 12, 4, 5, 1, 2, 0]]
        device2bone = self.smpl2imu.matmul(self.init_oris).transpose(1, 2).matmul(cali_pose_ori)

        if self.device2bone is None:
            self.device2bone = device2bone
        else:
            self.device2bone = device2bone

        print(self.smpl2imu)

        self.unity_start(run_unity_package=self.run_unity_package)
    def set_calibrate_data_8(self, input_sensor: list):
        """
        设置标定数据
        :param input: [48] -> [4 x 3 (acc) + 4 x 9(oris)]
        :return:无返回值
        """
        """
        设置标定数据
        :param input: [48] -> [4 x 3 (acc) + 4 x 9(oris)]
        :return:无返回值
        """
        # print(input[self.imu_num*3:])
        self.tpose_oris = torch.tensor(input_sensor[self.imu_num * 3:]).view(self.imu_num, 3, 3)
        # self.acc_offsets = self.tpose_oris.matmul(torch.tensor(input[0:self.imu_num * 3]).view(self.imu_num, 3, 1))
        self.acc_offsets = torch.FloatTensor([[0, 0, 9.8]]).repeat(self.imu_num, 1).unsqueeze(-1)

        # self.acc_offsets = acc_offset.view(self.imu_num, 3)
        print(self.acc_offsets)

        back_2_smpl = self.config_clothes['root_2_back'].transpose(1, 2).matmul(self.root_2_smpl)
        imu2smpl = self.tpose_oris[2].matmul(back_2_smpl).view(3, 3)
        # imu2smpl = self.tpose_oris[2].matmul(self.root_2_smpl).view(3, 3)
        column_y = torch.FloatTensor([[0, 0, 1]])
        column_z = normalize_tensor(
            imu2smpl[:, [2]].t() - (column_y * imu2smpl[:, [2]].t()).sum(dim=1, keepdim=True) * column_y)
        column_x = column_y.cross(column_z, dim=1)
        imu2smpl = torch.stack((column_x, column_y, column_z), dim=-1).squeeze(0)
        self.smpl2imu = imu2smpl.t()

        # T-Pose
        cali_pose_ori = SMPLPose.t_pose_ori[[18, 17, 12, 0, 4, 5, 0, 0]]
        # N-Pose
        # cali_pose_ori = SMPLPose.n_pose_ori[[18, 17, 12, 0, 4, 5, 0, 0]]

        device2bone = self.smpl2imu.matmul(self.tpose_oris).transpose(1, 2).matmul(cali_pose_ori)

        if self.device2bone is None:
            self.device2bone = device2bone
        else:
            self.device2bone[2] = device2bone[2]

        print(self.smpl2imu)
        self.unity_start(run_unity_package=self.run_unity_package)

    def set_head_calibrate_data(self, T_vp_world, joint_rot, joint_pos):
        """
        T_vp_world: (4,4) Vision Pro 当前帧头部在世界坐标中的位姿
        pose_smpl_axis_angle: (24, 3) 当前 SMPL 姿态（每个关节的 axis-angle 旋转）

        使用 FK + 指定骨骼链 [0,3,6,9,12,15] 累乘构建 SMPL 头部位姿
        """
        import numpy as np
        import torch

        # 转换为旋转矩阵 (24, 3, 3)
        joint_rot = axis_angle_to_rotation_matrix(torch.tensor(joint_rot))  # (24, 3, 3)

        # 通过 FK 得到每个关节的全局旋转和位置
        # joint_rot, joint_pos = self.body_model.forward_kinematics(R_smpl, calc_joint=True)

        # 构建 0 → 15 的 T-chain
        joint_chain = [0, 3, 6, 9, 12, 15]
        T_chain = np.eye(4)
        for idx in joint_chain:
            T = np.eye(4)
            T[:3, :3] = joint_rot[idx].detach().cpu().numpy()
            T[:3, 3] = joint_pos[idx].detach().cpu().numpy()  # 累积世界位移（简化）
            T_chain = T @ T_chain  # 左乘叠加

        T_head_world = T_chain  # 得到 SMPL head 的全局位姿

        # 计算偏移：T_offset = T_head⁻¹ * T_vp
        T_offset = np.linalg.inv(T_head_world) @ T_vp_world
        self.T_offset = T_offset
        return T_offset


    def calibrate_head_offset(self, T_vp_world, joint_rot, joint_pos):
        """
        计算头显与SMPL第15号关节之间的相对变换T_offset。
        通常在静止帧（如T-pose）调用一次。

        参数：
            T_vp_world: np.ndarray (4,4)，Vision Pro 头部的世界位姿
            joint_rot: torch.Tensor (24, 3, 3)，SMPL每个关节的全局旋转矩阵
            joint_pos: torch.Tensor (24, 3)，SMPL每个关节的全局位移

        返回：
            T_offset: np.ndarray (4,4)，SMPL头部坐标系 → VP坐标系
        """
        joint_rot = axis_angle_to_rotation_matrix(torch.tensor(joint_rot))  # (24, 3, 3)

        foot_y = joint_pos[[7, 8], 1]
        joint_pos[:, 1] -= torch.min(foot_y)
        T_head = np.eye(4)
        T_head[:3, :3] = joint_rot[15].detach().cpu().numpy()
        T_head[:3, 3] = joint_pos[15].detach().cpu().numpy()

        T_offset = np.linalg.inv(T_head) @ T_vp_world

        return T_offset

    def get_smpl_head_world(self, joint_rot, joint_pos, chain=[0, 3, 6, 9, 12, 15]):
        """
        沿SMPL骨骼链累乘当前帧从root到head的变换，得到头部在世界坐标下的T。

        参数：
            joint_rot: torch.Tensor (24, 3, 3)，每个关节的全局旋转
            joint_pos: torch.Tensor (24, 3)，每个关节的全局位置
            chain: list[int]，root → head 的关节索引路径

        返回：
            T_head_world: np.ndarray (4,4)，SMPL头部的世界位姿
        """
        joint_rot = axis_angle_to_rotation_matrix(torch.tensor(joint_rot))  # (24, 3, 3)

        T = np.eye(4)
        for idx in chain:
            Ti = np.eye(4)
            Ti[:3, :3] = joint_rot[idx].detach().cpu().numpy()
            Ti[:3, 3] = joint_pos[idx].detach().cpu().numpy()
            T = T @ Ti
        return T

    def anime_update(self, rotation_diversity, trigger):
        send_str = ''
        for rd in rotation_diversity:
            send_str += str(int(rd)) + ' '
        for i, t in enumerate(trigger):
            if t:
                send_str += str(i) + ' '
        self.udp_client.sendto(send_str.encode('utf-8'), ("127.0.0.1", 23338))

    @torch.no_grad()
    def auto_calibrate(self):
        if len(self.acc_cat_oris) < 256:
            # print(len(self.acc_cat_oris))
            return
        frame_nums = 256

        acc_cat_oris = torch.stack(self.acc_cat_oris[-frame_nums:]).reshape(frame_nums, -1)
        # 目前版本不使用加速度了, 全置0
        oris = acc_cat_oris[:, 3*self.imu_num:].reshape(1, -1, self.imu_num, 3, 3)
        # 旋转丰富度
        diversity = rotation_diversity(oris.clone()).reshape(-1)

        trigger = diversity > self.diversity_threshold


        keep_mask = ~trigger

        # 根节点不更新
        # keep_mask[-1] = True
        skip_count = torch.sum(keep_mask).item()
        if skip_count < 8:
            # 矩阵正交化:
            self.global_shift = r6d_to_rotation_matrix(rotation_matrix_to_r6d(self.global_shift))
            self.local_shift = r6d_to_rotation_matrix(rotation_matrix_to_r6d(self.local_shift))
            acc_cat_oris = acc_cat_oris.reshape(1, -1, self.imu_num * (3 + 3 * 3))
            # print(oris[0,0])
            feed = {'imu_rot': np.array(acc_cat_oris)}
            global_shift, local_shift = self.calibration_session.run(output_names=None, input_feed=feed)

            global_shift = r6d_to_rotation_matrix(torch.FloatTensor(global_shift).reshape(-1, 6))
            global_shift = ego_drift_regularization(global_shift, imu_num=self.imu_num).squeeze(0).transpose(-2,
                                                                                                            -1)

            local_shift = r6d_to_rotation_matrix(torch.FloatTensor(local_shift).reshape(-1, 6)).transpose(-2, -1)

            # print(global_shift)
            # print(local_shift)

            global_shift[keep_mask, :, :] = torch.eye(3).unsqueeze(0).repeat(skip_count, 1, 1)
            local_shift[keep_mask, :, :] = torch.eye(3).unsqueeze(0).repeat(skip_count, 1, 1)

            self.global_shift = global_shift.matmul(self.global_shift)
            self.local_shift = self.local_shift.matmul(local_shift)

            # 不浪费数据

            self.acc_cat_oris = self.acc_cat_oris[-1:]

            print('自动校准')

            return self.global_shift, self.local_shift
            # print(self.global_shift)
            # print(self.local_shift)


    @torch.no_grad()
    def calibrate(self, input: list,acc_scale = 1):
        """
        用于标定校准处理
        :param input: 传感器数据序列[48] -> [4 x 3 (acc) + 4 x 9(oris)]
        :return: 标定校准处理后的数据/原始传感器数据
        """
        input = torch.FloatTensor(input)


        accs, oris = input[0:self.imu_num * 3].view(-1, self.imu_num, 3), input[self.imu_num * 3:].view(-1,
                                                                                                        self.imu_num, 3,
                                                                                                        3)
        # acc转到全局坐标 Z轴加速度是反的 补正
        # acc转到全局坐标
        accs = oris.matmul(accs.view(-1, self.imu_num, 3, 1))
        self.acc_offsets = torch.FloatTensor([[0, 0, 9.8]]).repeat(self.imu_num, 1).unsqueeze(-1)

        accs = accs - self.acc_offsets
        # oris转到smpl->bone, 然后补正
        oris = self.global_shift0.matmul(self.smpl2imu).matmul(oris).matmul(self.device2bone).matmul(self.local_shift0)
        # acc转到smpl 并补正
        accs = self.global_shift0.matmul(self.smpl2imu).matmul(accs).view(-1, self.imu_num, 3)/acc_scale

        # print('\r', accs[0, 0], end='')
        input_imu = self.normalize_and_concat(accs, oris).view(-1)

        input = input_imu

        return np.array(input.cpu())
    
    @torch.no_grad()
    def calibrate_auto(self, input: list,acc_scale = 1):
        """
        用于标定校准处理
        :param input: 传感器数据序列[48] -> [4 x 3 (acc) + 4 x 9(oris)]
        :return: 标定校准处理后的数据/原始传感器数据
        """
        input = torch.FloatTensor(input)


        accs, oris = input[0:self.imu_num * 3].view(-1, self.imu_num, 3), input[self.imu_num * 3:].view(-1,
                                                                                                        self.imu_num, 3,
                                                                                                        3)
        # acc转到全局坐标 Z轴加速度是反的 补正
        # acc转到全局坐标
        accs = oris.matmul(accs.view(-1, self.imu_num, 3, 1))
        self.acc_offsets = torch.FloatTensor([[0, 0, 9.8]]).repeat(self.imu_num, 1).unsqueeze(-1)

        accs = accs - self.acc_offsets
        # oris转到smpl->bone, 然后补正
        oris = self.global_shift.matmul(self.smpl2imu).matmul(oris).matmul(self.device2bone).matmul(self.local_shift)
        # acc转到smpl 并补正
        accs = self.global_shift.matmul(self.smpl2imu).matmul(accs).view(-1, self.imu_num, 3)/acc_scale

        # print('\r', accs[0, 0], end='')
        input_imu = self.normalize_and_concat(accs, oris).view(-1)

        input = input_imu

        return np.array(input.cpu())

    def operator(self, input: np.ndarray):
        """
        在标定校准后，进行预处理
        :param input: 标定校准处理后的数据/原始传感器数据
        :return: 无返回数据
        """
        input = torch.FloatTensor(input).reshape(-1)

        self.acc_cat_oris.append(input)
        # self.rot_buffer.append(input[self.imu_num * 3:].reshape(self.imu_num, 3, 3))
        if len(self.acc_cat_oris) > self.max_buffer_len:
            self.acc_cat_oris = self.acc_cat_oris[-self.max_buffer_len:]
        # if len(self.rot_buffer) > self.max_buffer_len:
        #     self.rot_buffer = self.rot_buffer[-self.max_buffer_len:]

    def to_predict_data(self):
        """
        onnx.run的参数
        :return: output_names, input_feed, run_options
        """

        # 不转r6d
        acc_cat_oris = self.acc_cat_oris[-1]
        acc_cat_oris = np.array(acc_cat_oris.unsqueeze(0))

        input_feed = {'imu_data': acc_cat_oris,
                      'h_1': self.h_1, 'c_1': self.c_1,
                      'h_2': self.h_2, 'c_2': self.c_2,
                      'h_3': self.h_3, 'c_3': self.c_3}

        return input_feed

    def predict_result(self, result):
        """
        返回的结果
        :return: [24, 3] 24个关节的轴角
        """
        pose, joint, vel, self.h_1, self.c_1, self.h_2, self.c_2, self.h_3, self.c_3 = result
        vel = torch.tensor(vel).reshape(-1, 3)
        contact_left = 1 - (torch.norm(vel[10], 2) - 0.02).clamp(min=0, max=0.2) / 0.2
        contact_right = 1 - (torch.norm(vel[11], 2) - 0.02).clamp(min=0, max=0.2) / 0.2

        vel_left = torch.norm(vel[10], 2)
        vel_right = torch.norm(vel[11], 2)

        result = torch.tensor(pose).reshape(24, 3)
        result[[7, 8]] *= 0

        p = axis_angle_to_rotation_matrix(result)
        joint_rot, joint_pos = self.body_model.forward_kinematics(p, calc_joint=True)
        # cali_rot_loose = joint_rot.reshape(24, 3, 3)[[12, 0]]
        # self.rot_buffer[-1][3] = joint_rot[0]


        # fk位移
        if vel_left < vel_right:
            d_trans_fk = self.last_joint_pos[10] - joint_pos[10]
        else:
            d_trans_fk = self.last_joint_pos[11] - joint_pos[11]
        self.last_joint_pos = joint_pos

        # nn位移
        d_trans_nn = torch.Tensor(vel).reshape(-1, 3)[0] / 30
        d_root_height_nn = d_trans_nn[1]

        # nn & fk高度
        root_height_fk = joint_pos[0][1] - min(joint_pos[10][1], joint_pos[11][1]) - self.root_height_init
        s = max(contact_left, contact_right)
        if s < 0.1:
            self.floating_prob = min(max(self.floating_prob + 0.3, 0), 1)
            d_trans = d_trans_nn
        else:
            self.floating_prob = min(max(self.floating_prob - 0.3, 0), 1)
            # self.root_height = root_height_fk
            # d_trans = d_trans_fk
            d_trans = self.floating_prob * d_trans_nn + (1 - self.floating_prob) * d_trans_fk
        self.root_height = (1 - self.floating_prob) * root_height_fk + self.floating_prob * (
                self.root_height + d_root_height_nn)
        self.root_height = max(root_height_fk, self.root_height)

        if self.track_trans:
            self.trans[[0, 2]] += d_trans[[0, 2]]
        self.trans[1] = self.root_height

        return result


    def compute_root_from_vp(self, vp_head_pos, R_head, T_head2neck_ref, T_neck2root_ref):
        neck_pos = vp_head_pos + R_head @ T_head2neck_ref
        root_pos = neck_pos + R_head @ T_neck2root_ref
        return root_pos

    def predict_VP_result_optim(self, result, VP_head_rot):
        """
        VP_head_rot 坐标系z轴朝上
        返回的结果
        :return: [24, 3] 24个关节的轴角
        """
        R_T = torch.tensor([
            [-1., 0., 0.],
            [0., 0., 1.],
            [0., 1., 0.]
        ], dtype=torch.float64)

        R_head = torch.tensor([
            [-1., 0., 0.],
            [0., 0., 1.],
            [0., 1., 0.]
        ], dtype=torch.float64)

        # R_align = np.array([
        #     [1, 0, 0],
        #     [0, 0, 1],
        #     [0, -1, 0]
        # ])
        VP_head_rot[:3, :3] = R_T @ VP_head_rot[:3, :3] @ R_head.T

        # VP_head_rot[:3, :3] = R_align @ VP_head_rot[:3, :3]  # 旋转对齐
        # print(VP_head_rot[:3, :3])
        pose, joint, vel, self.h_1, self.c_1, self.h_2, self.c_2, self.h_3, self.c_3 = result
        vel = torch.tensor(vel).reshape(-1, 3)
        p = axis_angle_to_rotation_matrix(torch.FloatTensor(pose))
        # p[12] = torch.FloatTensor(VP_head_rot[:3, :3])
        joint_rot, joint_pos = self.body_model.forward_kinematics(p, calc_joint=True)
        joint_rot[15] = torch.tensor(VP_head_rot[:3, :3])
        p = self.body_model.inverse_kinematics(joint_rot)
        # joint_rot, joint_pos = self.body_model.forward_kinematics(joint_rot, calc_joint=True)

        y_coords = joint_pos[:, 1]  # 提取第2列（y轴）
        max_idx = torch.argmax(y_coords).item()
        min_idx = torch.argmin(y_coords).item()
        max_y = y_coords[max_idx].item()
        min_y = y_coords[min_idx].item()
        height = max_y - min_y
        # print(f"最大Y值点索引: {max_idx}，Y = {max_y:.3f}, 最小Y值点索引: {min_idx}，Y = {min_y:.3f}, 估算身高: {height:.3f} 米")

        pose, optim_trans = self.motion_optimizer.optimize_frame(pose=p, jvel=vel, trans=self.trans)
        pose = rotation_matrix_to_axis_angle(pose)

        contact_left = 1 - (torch.norm(vel[10], 2) - 0.02).clamp(min=0, max=0.2) / 0.2
        contact_right = 1 - (torch.norm(vel[11], 2) - 0.02).clamp(min=0, max=0.2) / 0.2

        vel_left = torch.norm(vel[10], 2)
        vel_right = torch.norm(vel[11], 2)

        result = pose.reshape(24, 3)
        result[[7, 8]] *= 0


        # self.rot_buffer[-1][3] = joint_rot[0]


        # fk位移
        if vel_left < vel_right:
            d_trans_fk = self.last_joint_pos[10] - joint_pos[10]
        else:
            d_trans_fk = self.last_joint_pos[11] - joint_pos[11]
        # print('左脚:', self.last_joint_pos[10], joint_pos[10], '右脚：', self.last_joint_pos[11], joint_pos[11], '位移：', d_trans_fk)
        self.last_joint_pos = joint_pos

        # nn位移
        # print(optim_trans)
        d_trans_nn = optim_trans - self.trans
        d_root_height_nn = d_trans_nn[1]

        # nn & fk高度
        root_height_fk = joint_pos[0][1] - min(joint_pos[10][1], joint_pos[11][1]) - self.root_height_init
        s = max(contact_left, contact_right)
        if s < 0.1:
            self.floating_prob = min(max(self.floating_prob + 0.3, 0), 1)
            d_trans = d_trans_nn
        else:
            self.floating_prob = min(max(self.floating_prob - 0.3, 0), 1)
            # self.root_height = root_height_fk
            # d_trans = d_trans_fk
            d_trans = self.floating_prob * d_trans_nn + (1 - self.floating_prob) * d_trans_fk
        self.root_height = (1 - self.floating_prob) * root_height_fk + self.floating_prob * (
                self.root_height + d_root_height_nn)
        self.root_height = max(root_height_fk, self.root_height)

        self.trans[[0, 2]] += d_trans[[0, 2]]
        self.trans[1] = self.root_height
        return result, joint_pos, joint_rot

    def predict_result_optim(self, result):
        """
        返回的结果
        :return: [24, 3] 24个关节的轴角
        """
        pose, joint, vel, self.h_1, self.c_1, self.h_2, self.c_2, self.h_3, self.c_3 = result
        vel = torch.tensor(vel).reshape(-1, 3)

        p = axis_angle_to_rotation_matrix(torch.FloatTensor(pose))
        joint_rot, joint_pos = self.body_model.forward_kinematics(p, calc_joint=True)

        pose, optim_trans = self.motion_optimizer.optimize_frame(pose=p, jvel=vel, trans=self.trans)
        pose = rotation_matrix_to_axis_angle(pose)

        contact_left = 1 - (torch.norm(vel[10], 2) - 0.02).clamp(min=0, max=0.2) / 0.2
        contact_right = 1 - (torch.norm(vel[11], 2) - 0.05).clamp(min=0, max=0.2) / 0.2

        vel_left = torch.norm(vel[10], 2)
        vel_right = torch.norm(vel[11], 2)

        result = pose.reshape(24, 3)
        result[[7, 8]] *= 0

        # fk位移
        if vel_left < vel_right:
            d_trans_fk = self.last_joint_pos[10] - joint_pos[10]
        else:
            d_trans_fk = self.last_joint_pos[11] - joint_pos[11]
        # print('左脚:', self.last_joint_pos[10], joint_pos[10], '右脚：', self.last_joint_pos[11], joint_pos[11], '位移：', d_trans_fk)
        self.last_joint_pos = joint_pos

        # nn位移
        # print(optim_trans)
        d_trans_nn = optim_trans - self.trans
        d_root_height_nn = d_trans_nn[1]

        # nn & fk高度
        root_height_fk = joint_pos[0][1] - min(joint_pos[10][1], joint_pos[11][1]) - self.root_height_init
        s = max(contact_left, contact_right)
        if s < 0.1:
            self.floating_prob = min(max(self.floating_prob + 0.3, 0), 1)
            d_trans = d_trans_nn
        else:
            self.floating_prob = min(max(self.floating_prob - 0.3, 0), 1)
            # self.root_height = root_height_fk
            # d_trans = d_trans_fk
            d_trans = self.floating_prob * d_trans_nn + (1 - self.floating_prob) * d_trans_fk
        self.root_height = (1 - self.floating_prob) * root_height_fk + self.floating_prob * (
                self.root_height + d_root_height_nn)
        self.root_height = max(root_height_fk, self.root_height)

        self.trans[[0, 2]] += d_trans[[0, 2]]
        self.trans[1] = self.root_height
        return result, joint_pos
    def predict_result_optim_0911test(self, result):
        """
        返回的结果
        :return: [24, 3] 24个关节的轴角
        """
        pose, joint, vel, self.h_1, self.c_1, self.h_2, self.c_2, self.h_3, self.c_3 = result
        vel = torch.tensor(vel).reshape(-1, 3)

        p = axis_angle_to_rotation_matrix(torch.FloatTensor(pose))

        self.physics_models.update_state(pose=p, vel=vel, stationary=None)
        _optim_pose, _trans = self.physics_models.get_state()
        result = rotation_matrix_to_axis_angle(_optim_pose[0])
        self.trans = _trans

        return result, _trans

    def predict_result_physics_optim(self, result):
        """
        返回的结果
        :return: [24, 3] 24个关节的轴角
        """
        pose, joint, vel, self.h_1, self.c_1, self.h_2, self.c_2, self.h_3, self.c_3 = result
        vel = torch.tensor(vel).reshape(-1, 3)
        joint = joint.reshape(-1, 3)
        # h_left, h_right = joint[10, 1], joint[11, 1]
       

        contact_left = 1 - (torch.norm(vel[10], 2) - 0.02).clamp(min=0, max=0.15) / 0.15
        contact_right = 1 - (torch.norm(vel[11], 2) - 0.02).clamp(min=0, max=0.15) / 0.15
        contact = torch.stack([contact_left, contact_right]).reshape(-1).clamp(min=0.001, max=0.999)
        # contact = (contact+contact_fix)/2
        # print(contact)

        result = torch.tensor(pose).reshape(24, 3)
        result[[7, 8]] *= 0

        pose = axis_angle_to_rotation_matrix(result)

        # joint_velocity = vel.view(24, 3).matmul(pose[0].transpose(-2, -1))
        joint_velocity = vel.view(24, 3)


        pose, trans = self.dynamics_optimizer.optimize_frame(pose, joint_velocity*1, contact, acc=None)

        joint_rot, joint_pos = self.body_model.forward_kinematics(pose, calc_joint=True)

        pose = rotation_matrix_to_axis_angle(pose)

        # self.trans = trans.unsqueeze(0)
        # ===================================================================

        joint_pos = joint_pos[0]
        # print(joint_pos, self.last_joint_pos)
        # fk位移
        if contact_left > contact_right:
            d_trans_fk = self.last_joint_pos[10] - joint_pos[10]
        else:
            d_trans_fk = self.last_joint_pos[11] - joint_pos[11]
        self.last_joint_pos = joint_pos

        # nn位移
        d_trans_nn = torch.Tensor(vel).reshape(-1, 3)[0] / 30

        d_root_height_nn = d_trans_nn[1]

        # nn & fk高度
        root_height_fk = joint_pos[0][1] - min(joint_pos[10][1], joint_pos[11][1]) - self.root_height_init
        s = max(contact_left, contact_right)
        if s < 0.1:
            self.floating_prob = min(max(self.floating_prob + 0.5, 0), 1)
            d_trans = d_trans_nn
        else:
            self.floating_prob = min(max(self.floating_prob - 0.1, 0), 1)
            # self.root_height = root_height_fk
            # d_trans = d_trans_fk
            d_trans = self.floating_prob * d_trans_nn + (1 - self.floating_prob) * d_trans_fk
        self.root_height = (1 - self.floating_prob) * root_height_fk + self.floating_prob * (
                self.root_height + d_root_height_nn)
        self.root_height = max(root_height_fk, self.root_height)

        if self.track_trans:
            # print(d_trans)
            # print(self.trans)
            self.trans[[0, 2]] += d_trans[[0, 2]]
            self.trans[1] = self.root_height

        return pose

    def get_trans(self):
        if self.track_trans is False:
            trans = self.trans.view(-1)
            trans[[0, 2]] *= 0
            return trans
        return self.trans.view(-1)



class DataProcessServer_8IMU_V3():
    def __init__(self, rotation_type, part, config, keep_hidden=True, run_unity_package=True, mode=demo_mode.UPPER,
                 use_trans=True, calibration_session=None,unity_mode='single'):
        # import calibration_animation.main as caan
        # caan.main()
        self.run_unity_package = run_unity_package
        # if run_unity_package:
        #     os.system('start Unity_exe/mode1/mode1.exe')

        # server_for_unity = socket(AF_INET, SOCK_STREAM)
        # server_for_unity.bind(('127.0.0.1', 8888))
        # server_for_unity.listen(1)
        # print('Server start. Waiting for unity3d to connect.')
        # self.conn, self.addr = server_for_unity.accept()
        self.rotation_type = rotation_type
        self.part = part
        self.keep_hidden = keep_hidden
        self.use_trans = use_trans
        self.trans = torch.FloatTensor([0, 0, 0])
        self.imu_num = 8
        self.clock = Clock()
        self.unity_mode = unity_mode
        self.mode = mode
        if isinstance(config, list):
            self.config_clothes, self.config_pant = tuple(config)
            self.config = self.config_clothes
        else:
            self.config = config
            self.config_clothes = config

        self.calibration_session = calibration_session
        self.property()
        self.udp_client = socket(AF_INET, SOCK_DGRAM)  # 创建socket对象，走udp通道


        from Aplus.tools.smpl_light import SMPLight
        # self.body_model = art.ParametricModel(paths.smpl_file)
        self.body_model = SMPLight()

        p = torch.eye(3).unsqueeze(0).repeat(24, 1, 1).unsqueeze(0)
        self.body_shape = torch.zeros(10)
        self.init_trans = torch.zeros(3)
        # 输入24个关节旋转+体型参数+位移信息, 输出24个关节的旋转+蒙皮点加速度+运动速度
        grot, joint = self.body_model.forward_kinematics(p, calc_joint=True)
        joint = joint[0]

        self.root_height = joint[0][1] - min(joint[10][1], joint[11][1])
        self.root_height_init = joint[0][1] - min(joint[10][1], joint[11][1])
        self.last_joint_pos = joint
        self.frame_num = 0
        self.floating_prob = 0
        self.max_buffer_len = 1000
        self.global_tran_origin = None
        self.unity_start(run_unity_package=self.run_unity_package)

    def unity_start(self, run_unity_package):

        server_for_unity = socket(AF_INET, SOCK_STREAM)
        server_for_unity.bind(('127.0.0.1', 8888))
        server_for_unity.listen(1)
        print('Server start.')
        if run_unity_package:
            print('内置Unity启动中')
            if self.unity_mode == 'single_global':
                run_mode3()
            elif self.unity_mode == 'twins':
                run_mode2()
            elif self.unity_mode == 'single':
                run_mode1()
            # run_mode_vision()
            # run_mode_vision()
            # run_mode_girl()
            # run_mode_micai()
            # run_mode_up()
        else:
            print('手动启动Unity以开始动作捕捉')
        self.conn, self.addr = server_for_unity.accept()

    def property(self):
        """
        自定义属性
        :return:无返回值
        """
        # 用于calibration的数据
        self.tpose_oris = None
        self.tpose_accs = None

        self.global_shift = torch.eye(3).reshape(-1, 3, 3).repeat(self.imu_num, 1, 1)
        self.local_shift = torch.eye(3).reshape(-1, 3, 3).repeat(self.imu_num, 1, 1)

        # 串口版
        # self.root_2_smpl = bulid_rot(theta=180, rotation_axis=[0,0,1]).matmul(bulid_rot(theta=180, rotation_axis=[1,0,0]))
        self.root_2_smpl = bulid_rot(theta=180, rotation_axis=[0, 1, 0])
        self.device2bone = None

        self.smpl2imu = None
        self.acc_offsets = None

        self.h_1 = np.zeros(shape=[2, 1, 256]).astype(np.float32)
        self.c_1 = np.zeros(shape=[2, 1, 256]).astype(np.float32)
        self.h_2 = np.zeros(shape=[2, 1, 256]).astype(np.float32)
        self.c_2 = np.zeros(shape=[2, 1, 256]).astype(np.float32)
        self.h_3 = np.zeros(shape=[2, 1, 256]).astype(np.float32)
        self.c_3 = np.zeros(shape=[2, 1, 256]).astype(np.float32)

        # 模型输入
        self.acc_cat_oris = []
        self.diversity_threshold = torch.Tensor([50, 50, 20, 20, 20, 20, 20, 20]) * 1

    def get_device_2_bone(self):
        root_2_smpl = self.config_clothes['root_2_smpl'].unsqueeze(0)
        left_2_smpl = self.config_clothes['root_2_left'].transpose(1, 2).matmul(root_2_smpl)
        right_2_smpl = self.config_clothes['root_2_right'].transpose(1, 2).matmul(root_2_smpl)
        back_2_smpl = self.config_clothes['root_2_back'].transpose(1, 2).matmul(root_2_smpl)

        device_2_bone = [left_2_smpl, right_2_smpl, back_2_smpl, root_2_smpl]

        if self.mode == demo_mode.FULL:
            left_lowerleg_2_smpl = self.config_pant['left_lowerleg_2_smpl']
            right_lowerleg_2_smpl = self.config_pant['right_lowerleg_2_smpl']
            left_pelvis_2_smpl = self.config_pant['left_pelvis_2_smpl']
            right_pelvis_2_smpl = self.config_pant['right_pelvis_2_smpl']

            device_2_bone += [left_lowerleg_2_smpl, right_lowerleg_2_smpl, left_pelvis_2_smpl, right_pelvis_2_smpl]
        return torch.cat(device_2_bone, dim=0)

    def normalize_and_concat(self, smpl_acc, smpl_ori):
        imu_num = self.imu_num
        smpl_acc = smpl_acc.view(-1, imu_num, 3)
        smpl_ori = smpl_ori.view(-1, imu_num, 3, 3)

        # print(smpl_ori[:100, 0])
        # acc: [n, 4, 3]
        # acc = torch.cat((smpl_acc[:, :(imu_num-1)] - smpl_acc[:, (imu_num-1):], smpl_acc[:, (imu_num-1):]), dim=1).bmm(smpl_ori[:, -1]) / acc_scale
        acc = smpl_acc 
        # 不转换为相对加速度 变换至root坐标系
        # acc = smpl_acc.bmm(smpl_ori[:, -1]) / acc_scale

        # ori = torch.cat((smpl_ori[:, (imu_num-1):].transpose(2, 3).matmul(smpl_ori[:, :(imu_num-1)]), smpl_ori[:, (imu_num-1):]), dim=1)
        ori = smpl_ori

        data = torch.cat((acc.flatten(1), ori.flatten(1)), dim=1)
        return data

    def update_data(self, data):
        pose = data.view(-1)
        tran = self.get_trans()
        # send pose
        s = ','.join(['%g' % v for v in pose]) + '#' + \
            ','.join(['%g' % v for v in tran]) + '$'
        self.conn.send(s.encode('utf8'))  # I use unity3d to read pose and translation for visualization here

    def update_data_global(self, data,tran,reset_origin=False):

        pose = preprocess(data, self.rotation_type).numpy()
        pose = pose.reshape(24,3)
        pose[ :, 0] *= -1
        pose[ :, 2] *= -1
        pose = pose.reshape(-1)

        # tran = torch.as_tensor(tran, dtype=torch.float32).view(-1).cpu()

        if reset_origin or self.global_tran_origin is None:
            self.global_tran_origin = tran.clone()


        shape = torch.zeros(10).numpy()

        s = ','.join(['%g' % v for v in pose]) + '#' + \
            ','.join(['%g' % v for v in tran]) + '$' + \
            ','.join(['%g' % v for v in shape]) + '@'

        self.conn.send(s.encode('utf8'))  # I use unity3d to read pose and translation for visualization here
    
    def update_data_twins(self, data1,data2):

        pose1 = preprocess(data1, self.rotation_type)
        pose2 = preprocess(data2, self.rotation_type)

        tran = self.get_trans()
        # if trans is not None:
        #     # tran = trans
        #     tran_y = tran[1]
        #     trans[1] = tran_y
        # send pose
        
        s = (
        ','.join(['%g' % v for v in pose1]) + '#' +
        ','.join(['%g' % v for v in tran]) + '$' +
        ','.join(['%g' % v for v in pose2]) + '#' +
        ','.join(['%g' % v for v in tran]) + '@'
        )
        self.conn.send(s.encode('utf8')) 

    def device2bone_init(self):
        self.device2bone = self.get_device_2_bone()

    def set_calibrate_data(self, input: list):
        """
        设置标定数据
        :param input: [48] -> [4 x 3 (acc) + 4 x 9(oris)]
        :return:无返回值
        """
        # print(input[self.imu_num*3:])
        self.tpose_oris = torch.tensor(input[self.imu_num * 3:]).view(self.imu_num, 3, 3)
        # self.acc_offsets = self.tpose_oris.matmul(torch.tensor(input[0:self.imu_num * 3]).view(self.imu_num, 3, 1))
        self.acc_offsets = torch.FloatTensor([[0, 0, 9.8]]).repeat(self.imu_num, 1).unsqueeze(-1)

        # self.acc_offsets = acc_offset.view(self.imu_num, 3)
        print(self.acc_offsets)

        back_2_smpl = self.config_clothes['root_2_back'].transpose(1, 2).matmul(self.root_2_smpl)
        imu2smpl = self.tpose_oris[2].matmul(back_2_smpl).view(3, 3)
        # imu2smpl = self.tpose_oris[2].matmul(self.root_2_smpl).view(3, 3)
        column_y = torch.FloatTensor([[0, 0, 1]])
        column_z = normalize_tensor(
            imu2smpl[:, [2]].t() - (column_y * imu2smpl[:, [2]].t()).sum(dim=1, keepdim=True) * column_y)
        column_x = column_y.cross(column_z, dim=1)
        imu2smpl = torch.stack((column_x, column_y, column_z), dim=-1).squeeze(0)
        self.smpl2imu = imu2smpl.t()

        # T-Pose
        cali_pose_ori = SMPLPose.t_pose_ori[[18, 17, 12, 0, 4, 5, 0, 0]]
        # N-Pose
        # cali_pose_ori = SMPLPose.n_pose_ori[[18, 17, 12, 0, 4, 5, 0, 0]]

        device2bone = self.smpl2imu.matmul(self.tpose_oris).transpose(1, 2).matmul(cali_pose_ori)

        if self.device2bone is None:
            self.device2bone = device2bone
        else:
            self.device2bone[2] = device2bone[2]

        print(self.smpl2imu)

    def anime_update(self, rotation_diversity, trigger):
        send_str = ''
        for rd in rotation_diversity:
            send_str += str(int(rd)) + ' '
        for i, t in enumerate(trigger):
            if t:
                send_str += str(i) + ' '
        self.udp_client.sendto(send_str.encode('utf-8'), ("127.0.0.1", 23338))

    @torch.no_grad()
    def auto_calibrate(self):
        if len(self.acc_cat_oris) < 256:
            return
        # frame_nums = min(512, len(self.clean_rot_buffer))
        frame_nums = 256
        # 降采样到128
        acc_cat_rot = torch.stack(self.acc_cat_oris[-frame_nums:]).reshape(frame_nums, -1)
        oris = acc_cat_rot[:, self.imu_num*3:]
        # 目前版本不使用加速度了, 全置0
        oris = oris.reshape(1, -1, self.imu_num, 3, 3)
        # oris_ego = oris.clone()
        # leafs = [0, 1, 2, 4, 5]
        # oris_ego[:, :, leafs] = oris_ego[:, :, [-1]].transpose(-2, -1).matmul(oris_ego[:, :, leafs])
        # 旋转丰富度
        diversity = rotation_diversity(oris.cuda()).reshape(-1)
        # print(diversity)


        trigger_s2 = diversity > self.diversity_threshold
        # print(diversity)

        trigger = trigger_s2

        keep_mask = ~trigger

        # 根节点不更新
        # keep_mask[-1] = True
        skip_count = torch.sum(keep_mask).item()
        if skip_count < 8:
            # 矩阵正交化:
            self.global_shift = r6d_to_rotation_matrix(rotation_matrix_to_r6d(self.global_shift))
            self.local_shift = r6d_to_rotation_matrix(rotation_matrix_to_r6d(self.local_shift))
            acc_cat_rot = acc_cat_rot.reshape(1, -1, self.imu_num * (3 * 3 + 3))
            feed = {'imu_rot': np.array(acc_cat_rot)}
            global_shift, local_shift = self.calibration_session.run(output_names=None, input_feed=feed)

            global_shift = r6d_to_rotation_matrix(torch.tensor(global_shift).reshape(-1, 6))
            global_shift = ego_drift_regularization(global_shift, imu_num=self.imu_num).squeeze(0).transpose(-2,
                                                                                                            -1)

            local_shift = r6d_to_rotation_matrix(torch.tensor(local_shift).reshape(-1, 6)).transpose(-2, -1)

            global_shift[keep_mask, :, :] = torch.eye(3).unsqueeze(0).repeat(skip_count, 1, 1)
            local_shift[keep_mask, :, :] = torch.eye(3).unsqueeze(0).repeat(skip_count, 1, 1)

            self.global_shift = global_shift.matmul(self.global_shift)
            self.local_shift = self.local_shift.matmul(local_shift)

            # 不浪费数据
            self.acc_cat_oris = self.acc_cat_oris[-1:]

            # print('自动校准')


    @torch.no_grad()
    def auto_calibrate_new(self, times=1):
        if len(self.clean_rot_buffer) < 256:
            return

        frame_nums = min(512, len(self.clean_rot_buffer))
        # 降采样到128
        oris = torch.stack(self.clean_rot_buffer[-frame_nums:]).reshape(frame_nums, -1).cuda()
        # 目前版本不使用加速度了, 全置0
        oris = oris.reshape(1, -1, self.imu_num, 3, 3)
        # 旋转丰富度
        diversity = rotation_diversity(oris).reshape(-1)

        diversity_threshold = torch.Tensor([50, 50, 20, 20, 20, 20, 20, 20]) * 1
        trigger_s2 = diversity > diversity_threshold
        # print(diversity)

        trigger = trigger_s2

        keep_mask = ~trigger

        # 根节点不更新
        # keep_mask[-1] = True
        skip_count = torch.sum(keep_mask).item()
        delta_global = torch.eye(3).unsqueeze(0).repeat(self.imu_num, 1, 1).to(self.global_shift.device)
        delta_local  = torch.eye(3).unsqueeze(0).repeat(self.imu_num, 1, 1).to(self.local_shift.device)
        if skip_count < 8:
            for _ in range(times):
                oris = oris.reshape(1, -1, self.imu_num * (3 * 3))
                feed = {'imu_rot': np.array(oris.cpu())}
                global_shift, local_shift = self.calibration_session.run(output_names=None, input_feed=feed)

                global_shift = r6d_to_rotation_matrix(torch.tensor(global_shift).reshape(-1, 6)).cuda()
                global_shift = ego_drift_regularization(global_shift, imu_num=self.imu_num).squeeze(0).transpose(-2,
                                                                                                                -1).cuda()

                local_shift = r6d_to_rotation_matrix(torch.tensor(local_shift).reshape(-1, 6)).transpose(-2, -1).cuda()

                global_shift[keep_mask, :, :] = torch.eye(3).cuda().unsqueeze(0).repeat(skip_count, 1, 1)
                local_shift[keep_mask, :, :] = torch.eye(3).cuda().unsqueeze(0).repeat(skip_count, 1, 1)

                # 循环结束条件
                from articulate.math.angular import rotation_matrix_2_angle
                angle_glob_max = rotation_matrix_2_angle(global_shift).max() * 180 / torch.pi
                angle_loca_max = rotation_matrix_2_angle(local_shift).max() * 180 / torch.pi

                print(angle_glob_max, angle_loca_max)

                if max(angle_glob_max, angle_loca_max) < 1.5:
                    break

                oris = oris.reshape(-1, self.imu_num, 3, 3)
                oris = global_shift.matmul(oris).matmul(local_shift)
                delta_global = global_shift.matmul(delta_global)
                delta_local = delta_local.matmul(local_shift)




            # 矩阵正交化:
            self.global_shift = r6d_to_rotation_matrix(rotation_matrix_to_r6d(self.global_shift))
            self.local_shift = r6d_to_rotation_matrix(rotation_matrix_to_r6d(self.local_shift))
            self.global_shift = delta_global.matmul(self.global_shift)
            self.local_shift = self.local_shift.matmul(delta_local)

            # 不浪费数据
            # oris = oris.reshape(1, -1, self.imu_num, 3, 3).cuda()
            # oris = global_shift.matmul(oris).matmul(local_shift).reshape(-1, self.imu_num * 3 * 3).cpu()
            self.clean_rot_buffer = []
            # for i in range(oris.shape[0]):
            #     self.clean_rot_buffer.append(oris[i])
            print('自动校准')


    @torch.no_grad()
    def calibrate(self, input: list,acc_scale = 1):
        """
        用于标定校准处理
        :param input: 传感器数据序列[48] -> [4 x 3 (acc) + 4 x 9(oris)]
        :return: 标定校准处理后的数据/原始传感器数据
        """
        input = torch.FloatTensor(input)
        accs, oris = input[0:self.imu_num * 3].view(-1, self.imu_num, 3), input[self.imu_num * 3:].view(-1,
                                                                                                        self.imu_num, 3,
                                                                                                        3)
        # acc转到全局坐标 Z轴加速度是反的 补正
        # acc转到全局坐标
        accs = oris.matmul(accs.view(-1, self.imu_num, 3, 1))

        accs = accs - self.acc_offsets
        # oris转到smpl->bone, 然后补正
        oris = self.global_shift.matmul(self.smpl2imu).matmul(oris).matmul(self.device2bone).matmul(self.local_shift)
        # acc转到smpl 并补正
        accs = self.global_shift.matmul(self.smpl2imu).matmul(accs).view(-1, self.imu_num, 3)/acc_scale

        # print('\r', torch.norm(accs, dim=-1, p=2).view(-1), end='')
        input_imu = self.normalize_and_concat(accs, oris).view(-1)

        input = input_imu

        return np.array(input)

    def operator(self, input: np.ndarray):
        """
        在标定校准后，进行预处理
        :param input: 标定校准处理后的数据/原始传感器数据
        :return: 无返回数据
        """
        input = torch.FloatTensor(input).reshape(-1)

        self.acc_cat_oris.append(input)
        if len(self.acc_cat_oris) > self.max_buffer_len:
            self.acc_cat_oris = self.acc_cat_oris[-self.max_buffer_len:]

    def to_predict_data(self):
        """
        onnx.run的参数
        :return: output_names, input_feed, run_options
        """

        # 不转r6d
        acc_cat_oris = self.acc_cat_oris[-1]
        acc_cat_oris = np.array(acc_cat_oris.unsqueeze(0))

        input_feed = {'imu_data': acc_cat_oris,
                      'h_1': self.h_1, 'c_1': self.c_1,
                      'h_2': self.h_2, 'c_2': self.c_2,
                      'h_3': self.h_3, 'c_3': self.c_3}

        return input_feed

    def predict_result(self, result):
        """
        返回的结果
        :return: [24, 3] 24个关节的轴角
        """
        pose, joint, vel, self.h_1, self.c_1, self.h_2, self.c_2, self.h_3, self.c_3 = result
        vel = torch.tensor(vel).reshape(-1, 3)
        joint = joint.reshape(-1, 3)
        h_left, h_right = joint[10, 1], joint[11, 1]
        # contact_fix = 1 - (
        #     (torch.tensor(([h_left, h_right]) - min(h_left, h_right)) / 0.15).clamp(min=0.001, max=0.999))

        contact_left = 1 - (torch.norm(vel[10], 2) - 0.02).clamp(min=0, max=0.15) / 0.15
        contact_right = 1 - (torch.norm(vel[11], 2) - 0.02).clamp(min=0, max=0.15) / 0.15

        result = torch.tensor(pose).reshape(24, 3)
        result[[7, 8]] *= 0

        p = axis_angle_to_rotation_matrix(result)
        # print('========================')
        # from smpl2bio import smpl2bio
        # print(smpl2bio(p))
        joint_rot, joint_pos = self.body_model.forward_kinematics(p, calc_joint=True)

        if contact_left > contact_right:
            d_trans_fk = self.last_joint_pos[10] - joint_pos[10]
        else:
            d_trans_fk = self.last_joint_pos[11] - joint_pos[11]
        self.last_joint_pos = joint_pos

        # nn位移
        d_trans_nn = torch.Tensor(vel).reshape(-1, 3)[0] / 30
        d_root_height_nn = d_trans_nn[1]

        # nn & fk高度
        root_height_fk = joint_pos[0][1] - min(joint_pos[10][1], joint_pos[11][1]) - self.root_height_init
        s = max(contact_left, contact_right)
        if s < 0.1:
            self.floating_prob = min(max(self.floating_prob + 0.5, 0), 1)
            d_trans = d_trans_nn
        else:
            self.floating_prob = min(max(self.floating_prob - 0.1, 0), 1)
            # self.root_height = root_height_fk
            # d_trans = d_trans_fk
            d_trans = self.floating_prob * d_trans_nn + (1 - self.floating_prob) * d_trans_fk
        self.root_height = (1 - self.floating_prob) * root_height_fk + self.floating_prob * (
                self.root_height + d_root_height_nn)
        self.root_height = max(root_height_fk, self.root_height)

        if self.use_trans:
            self.trans[[0, 2]] += d_trans[[0, 2]]
        self.trans[1] = self.root_height

        # print('========================')
        # track_joint = {'左肩': 16, '右肩': 17, '脊柱-1': 3, '脊柱-2': 6, '脊柱-3': 9, '左肘': 18, '右肘': 19, '腰部': 0,
        #                '左胯': 1, '右胯': 2, '左膝': 4, '右膝': 5}
        # for jn, ji in track_joint.items():
        #     print(jn, ':', np.array(result[ji]).tolist())

        return result


    def get_trans(self):
        return self.trans.view(-1)



class VoteBuffer():
    def __init__(self, n_item, buffer_len):
        self.i = 0
        self.buffer_len = buffer_len
        self.buffer = torch.FloatTensor(n_item, buffer_len)
    def vote(self, item_id, weight=None):
        self.buffer[:, self.i] *= 0
        score = 1
        if weight is not None:
            score = weight
        self.buffer[item_id, self.i] += score
        self.i += 1
        self.i = self.i % self.buffer_len

    def get_max_vote(self):
        vote_sum = torch.sum(self.buffer, dim=-1, keepdim=False)
        return torch.argmax(vote_sum, dim=-1)

class DataProcessServer_NOITOM():
    def __init__(self, rotation_type, part, keep_hidden=True, run_unity_package=True, mode=demo_mode.UPPER,
                 track_trans=True, calibration_session=None, run_server=True):
        # import calibration_animation.main as caan
        # caan.main()

        if run_unity_package:
            run_mode1()
            # run_mode_girl()
            # run_mode_micai()
            # run_mode_up()
        if run_server:
            server_for_unity = socket(AF_INET, SOCK_STREAM)
            server_for_unity.bind(('127.0.0.1', 8888))
            server_for_unity.listen(1)
            print('Server start. Waiting for unity3d to connect.')
            self.conn, self.addr = server_for_unity.accept()
        self.rotation_type = rotation_type
        self.part = part
        self.keep_hidden = keep_hidden
        self.track_trans = track_trans

        self.imu_num = 6
        self.clock = Clock()

        self.mode = mode
        self.calibration_session = calibration_session
        self.property()
        self.trans_property()
        self.udp_client = socket(AF_INET, SOCK_DGRAM)  # 创建socket对象，走udp通道
    def property(self):
        """
        自定义属性
        :return:无返回值
        """
        # 用于calibration的数据
        self.init_oris = None
        self.tpose_accs = None

        self.global_shift = torch.eye(3).reshape(-1, 3, 3).repeat(self.imu_num, 1, 1).cuda()
        self.local_shift = torch.eye(3).reshape(-1, 3, 3).repeat(self.imu_num, 1, 1).cuda()

        # 串口版
        # self.root_2_smpl = bulid_rot(theta=180, rotation_axis=[0,0,1]).matmul(bulid_rot(theta=180, rotation_axis=[1,0,0]))
        self.root_2_smpl = bulid_rot(theta=180, rotation_axis=[0, 1, 0])


        self.smpl2imu = None
        self.device2bone = None
        self.acc_offsets = None

        n_layer = 2
        self.h_1 = np.zeros(shape=[n_layer, 1, 256]).astype(np.float32)
        self.h_2 = np.zeros(shape=[n_layer, 1, 256]).astype(np.float32)
        self.h_3 = np.zeros(shape=[n_layer, 1, 256]).astype(np.float32)
        self.c_1 = np.zeros(shape=[n_layer, 1, 256]).astype(np.float32)
        self.c_2 = np.zeros(shape=[n_layer, 1, 256]).astype(np.float32)
        self.c_3 = np.zeros(shape=[n_layer, 1, 256]).astype(np.float32)

        # 模型输入
        self.acc_cat_oris = []

    def set_init_state(self, init_states):
        import copy
        init_states = copy.copy(init_states)
        print('加载初始状态')
        for i in range(len(init_states)):
            init_states[i] = np.array(init_states[i].detach()).astype(np.float32)
        self.h_1, self.c_1, self.h_2, self.c_2, self.h_3, self.c_3 = tuple(init_states)


    def trans_property(self):
        # 位移相关
        self.trans = torch.FloatTensor([0, 0, 0])

        self.body_model = Aplus.tools.smpl_light.SMPLight()
        p = torch.eye(3).unsqueeze(0).repeat(24, 1, 1)
        # 输入24个关节旋转+体型参数+位移信息, 输出24个关节的旋转+蒙皮点加速度+运动速度
        grot, joint = self.body_model.forward_kinematics(p, calc_joint=True)

        self.root_height = joint[0][1] - min(joint[10][1], joint[11][1])
        self.root_height_init = joint[0][1] - min(joint[10][1], joint[11][1])
        self.last_joint_pos = joint

        self.max_buffer_len = 1000
        self.vel = torch.FloatTensor([0, 0, 0])
        self.p_contact = torch.ones(1)

        self.floating_prob = 0

        self.d_trans_fk_last = torch.FloatTensor([0, 0, 0])

    def get_device_2_bone(self):
        root_2_smpl = self.root_2_smpl
        left_arm_2_smpl = bulid_rot(theta=90, rotation_axis=[1, 0, 0]).matmul(bulid_rot(theta=180, rotation_axis=[0, 1, 0]))
        right_arm_2_smpl = bulid_rot(theta=90, rotation_axis=[1, 0, 0]).matmul(bulid_rot(theta=180, rotation_axis=[0, 1, 0]))
        left_leg_smpl = bulid_rot(theta=-90, rotation_axis=[0, 0, 1]).matmul(bulid_rot(theta=-90, rotation_axis=[0, 1, 0]))
        right_leg_smpl = bulid_rot(theta=90, rotation_axis=[0, 0, 1]).matmul(bulid_rot(theta=90, rotation_axis=[0, 1, 0]))
        head_2_smpl = bulid_rot(theta=0, rotation_axis=[1, 0, 0])

        device_2_bone = [left_arm_2_smpl, right_arm_2_smpl, left_leg_smpl, right_leg_smpl, head_2_smpl, root_2_smpl]

        if self.mode == demo_mode.FULL:
            left_lowerleg_2_smpl = self.config_pant['left_lowerleg_2_smpl']
            right_lowerleg_2_smpl = self.config_pant['right_lowerleg_2_smpl']
            left_pelvis_2_smpl = self.config_pant['left_pelvis_2_smpl']
            right_pelvis_2_smpl = self.config_pant['right_pelvis_2_smpl']

            device_2_bone += [left_lowerleg_2_smpl, right_lowerleg_2_smpl, left_pelvis_2_smpl, right_pelvis_2_smpl]
        return torch.cat(device_2_bone, dim=0)

    def normalize_and_concat(self, smpl_acc, smpl_ori):
        imu_num = self.imu_num
        acc_scale = 30
        smpl_acc = smpl_acc.view(-1, imu_num, 3)
        smpl_ori = smpl_ori.view(-1, imu_num, 3, 3)

        # print(smpl_ori[:100, 0])
        # acc: [n, 4, 3]
        # acc = torch.cat((smpl_acc[:, :(imu_num-1)] - smpl_acc[:, (imu_num-1):], smpl_acc[:, (imu_num-1):]), dim=1).bmm(smpl_ori[:, -1]) / acc_scale
        acc = smpl_acc / acc_scale
        # 不转换为相对加速度 变换至root坐标系
        # acc = smpl_acc.bmm(smpl_ori[:, -1]) / acc_scale

        # ori = torch.cat((smpl_ori[:, (imu_num-1):].transpose(2, 3).matmul(smpl_ori[:, :(imu_num-1)]), smpl_ori[:, (imu_num-1):]), dim=1)
        ori = smpl_ori

        data = torch.cat((acc.flatten(1), ori.flatten(1)), dim=1)
        return data

    def update_data(self, data):
        self.clock.tick()
        # print('\r', f'fps: {self.clock.get_fps()}', end='')
        # self.frame_num += 1
        # self.frame_num = self.frame_num % 2
        # if self.frame_num == 0:
        #     return

        pose = preprocess(data, self.rotation_type)

        # part of data freeze
        # pose.view(24, 3)[config.joint_set.part[self.part]] *= 0

        tran = self.get_trans()
        # send pose
        s = ','.join(['%g' % v for v in pose]) + '#' + \
            ','.join(['%g' % v for v in tran]) + '$'
        self.conn.send(s.encode('utf8'))  # I use unity3d to read pose and translation for visualization here

    def device2bone_init(self):
        self.device2bone = self.get_device_2_bone().cuda()

    def set_calibrate_data(self, input: list):
        """
        设置标定数据
        :param input: [48] -> [4 x 3 (acc) + 4 x 9(oris)]
        :return:无返回值
        """
        g = 9.8
        self.init_oris = torch.tensor(input[self.imu_num*3:]).view(self.imu_num, 3, 3)
        self.tpose_acc = torch.tensor(input[:self.imu_num*3]).view(self.imu_num, 3, 1)
        # print(self.tpose_acc)

        self.acc_offsets = torch.FloatTensor([[0, g, 0]]).repeat(self.imu_num, 1).cuda()
        # self.acc_offsets = self.init_oris.matmul(self.tpose_acc).cuda()
        print(self.acc_offsets)
        # self.tpose_elbow_angle = torch.tensor(input[48:50])

        # 修正全局坐标系误差
        # self.init_oris = self.I_2_Ibias.matmul(self.init_oris)
        # self.smpl2imu = self.init_oris[-1].matmul(self.root_2_smpl).view(3, 3).t().cuda()

        imu2smpl = self.init_oris[-1].matmul(self.root_2_smpl).view(3, 3)
        column_y = torch.FloatTensor([[0, 0, 1]])
        column_z = normalize_tensor(imu2smpl[:, [2]].t() - (column_y * imu2smpl[:, [2]].t()).sum(dim=1, keepdim=True) * column_y)
        column_x = column_y.cross(column_z, dim=1)
        imu2smpl = torch.stack((column_x, column_y, column_z), dim=-1).squeeze(0)
        self.smpl2imu = imu2smpl.t().cuda()

        # print(self.smpl2imu)

        if self.device2bone is None:
            self.device2bone = self.smpl2imu.matmul(self.init_oris.cuda()).transpose(1, 2).matmul(torch.eye(3).cuda())

    def anime_update(self, rotation_diversity, trigger):
        send_str = ''
        for rd in rotation_diversity:
            send_str += str(int(rd)) + ' '
        for i, t in enumerate(trigger):
            if t:
                send_str += str(i) + ' '
        self.udp_client.sendto(send_str.encode('utf-8'), ("127.0.0.1", 23338))

    @torch.no_grad()
    def auto_calibrate(self, time_gap=1, times=1):
        import time
        for _ in range(times):
            if len(self.acc_cat_oris) < 256:
                continue
            time.sleep(time_gap)
            frame_nums = 256
            # 降采样到128
            acc_cat_oris = torch.stack(self.acc_cat_oris[-frame_nums:]).reshape(frame_nums, -1)[::2]
            # 目前版本不使用加速度了, 全置0
            # acc_cat_oris[:, :self.imu_num * 3] *= 0
            oris = acc_cat_oris[:, self.imu_num * 3:].reshape(1, -1, self.imu_num, 3, 3)
            # 旋转丰富度
            diversity = rotation_diversity(oris).reshape(-1)

            diversity_threshold = torch.Tensor([30, 45, 30, 30, 20, 15]) * 1
            trigger_s2 = diversity > diversity_threshold
            # print(diversity)

            trigger = trigger_s2

            self.anime_update(diversity, trigger)

            keep_mask = ~trigger

            # 根节点不更新
            keep_mask[-1] = True
            skip_count = torch.sum(keep_mask).item()
            if skip_count < 5:
                self.global_shift = r6d_to_rotation_matrix(rotation_matrix_to_r6d(self.global_shift))
                self.local_shift = r6d_to_rotation_matrix(rotation_matrix_to_r6d(self.local_shift))
                acc_cat_oris = acc_cat_oris.reshape(1, -1, self.imu_num*(3 * 3 + 3))
                feed = {'imu_rot': np.array(acc_cat_oris)}
                global_shift, local_shift = self.calibration_session.run(output_names=None, input_feed=feed)
                # global_shift = torch.FloatTensor(global_shift).reshape(-1, 6)
                # global_shift = r6d_to_rotation_matrix(global_shift).transpose(-2,-1).cuda()

                global_shift = r6d_to_rotation_matrix(torch.tensor(global_shift).reshape(-1, 6))
                global_shift = ego_drift_regularization(global_shift).squeeze(0).transpose(-2, -1).cuda()


                local_shift = r6d_to_rotation_matrix(torch.FloatTensor(local_shift).reshape(-1, 6)).transpose(-2, -1).cuda()
                self.acc_cat_oris = self.acc_cat_oris[-1:]
                global_shift[keep_mask, :, :] = torch.eye(3).cuda().unsqueeze(0).repeat(skip_count, 1, 1)
                local_shift[keep_mask, :, :] = torch.eye(3).cuda().unsqueeze(0).repeat(skip_count, 1, 1)

                # angle_evaluator = PerJointRotationErrorEvaluator()
                # global_fix = angle_evaluator(global_shift.unsqueeze(0),
                #                              torch.eye(3).cuda().unsqueeze(0).repeat(self.imu_num, 1, 1).unsqueeze(0),
                #                              joint_num=self.imu_num)
                # local_fix = angle_evaluator(local_shift.unsqueeze(0),
                #                             torch.eye(3).cuda().unsqueeze(0).repeat(self.imu_num, 1, 1).unsqueeze(0),
                #                             joint_num=self.imu_num)

                # print('global矫正量', global_fix)
                # print('local矫正量', local_fix)
                self.global_shift = global_shift.matmul(self.global_shift)
                self.local_shift = self.local_shift.matmul(local_shift)
                print('自动校准')
            else:
                continue


    @torch.no_grad()
    def calibrate(self, input: list):
        """
        用于标定校准处理
        :param input: 传感器数据序列[48] -> [4 x 3 (acc) + 4 x 9(oris)]
        :return: 标定校准处理后的数据/原始传感器数据
        """
        input = torch.tensor(input).cuda()
        accs, oris = input[0:self.imu_num * 3].view(-1, self.imu_num, 3), input[self.imu_num * 3:].view(-1,self.imu_num, 3,3)
        # print(oris[-1])
        # acc转到全局坐标
        accs = oris.matmul(accs.view(-1, self.imu_num, 3, 1))

        # oris转到smpl->bone, 然后补正
        oris = self.global_shift.matmul(self.smpl2imu).matmul(oris).matmul(self.device2bone).matmul(self.local_shift)
        # acc转到smpl 并补正
        accs = self.global_shift.matmul(self.smpl2imu).matmul(accs).view(-1, self.imu_num, 3)

        accs = accs - self.acc_offsets

        # print('\r', accs[0, 0], end='')
        input_imu = self.normalize_and_concat(accs, oris).view(-1)

        input = input_imu

        return np.array(input.cpu())

    def operator(self, input: np.ndarray):
        """
        在标定校准后，进行预处理
        :param input: 标定校准处理后的数据/原始传感器数据
        :return: 无返回数据
        """
        max_length = 1000
        input = torch.tensor(input).reshape(-1)
        # acc, rot, angle = input[:12], input[12:48], input[48:]
        # rot_r6d = rotation_matrix_to_r6d(rot.reshape(4, 3, 3))
        # rot_r6d = rot_r6d.reshape(-1)
        # input = torch.cat([acc, rot_r6d], dim=-1)

        self.acc_cat_oris.append(input)
        # self.rot_buffer.append(input)
        if len(self.acc_cat_oris) > max_length:
            self.acc_cat_oris = self.acc_cat_oris[-max_length:]
        # if len(self.rot_buffer) > max_length:
        #     self.rot_buffer = self.rot_buffer[-max_length:]

    def to_predict_data(self):
        """
        onnx.run的参数
        :return: output_names, input_feed, run_options
        """

        # # 转r6d
        # acc_cat_oris = torch.tensor(self.acc_cat_oris[-1])
        # # print('\r', acc_cat_oris, end='')
        # acc, rot= acc_cat_oris[:self.imu_num*3], acc_cat_oris[self.imu_num*3:]
        # # print(rot.reshape(8, 3, 3))
        # rot_r6d = rotation_matrix_to_r6d(rot.reshape(self.imu_num, 3, 3))
        # rot_r6d = rot_r6d.reshape(-1)
        # acc_cat_oris = torch.cat([acc, rot_r6d], dim=-1)
        # acc_cat_oris = np.array(acc_cat_oris.unsqueeze(0))

        # 不转r6d
        acc_cat_oris = self.acc_cat_oris[-1]
        acc_cat_oris = np.array(acc_cat_oris.unsqueeze(0))


        input_feed = {'imu_data': acc_cat_oris,
                      'h_1': self.h_1,
                      'c_1': self.c_1,
                      'h_2': self.h_2,
                      'c_2': self.c_2,
                      'h_3': self.h_3,
                      'c_3': self.c_3
                      }

        return input_feed

    def predict_result(self, result):
        """
        返回的结果
        :return: [24, 3] 24个关节的轴角
        """
        pose, _, vel, self.h_1, self.c_1, self.h_2, self.c_2, self.h_3, self.c_3,= result

        vel = torch.FloatTensor(vel).reshape(-1, 3)

        contact_left = 1 - (torch.norm(vel[10], 2) - 0.04).clamp(min=0, max=0.2) / 0.2
        contact_right = 1 - (torch.norm(vel[11], 2) - 0.04).clamp(min=0, max=0.2) / 0.2

        vel_left = torch.norm(vel[10], 2)
        vel_right = torch.norm(vel[11], 2)

        # print(vel_left, vel_right)

        result = torch.FloatTensor(pose).reshape(24, 3)
        result[[7, 8]] *= 0

        p = axis_angle_to_rotation_matrix(result)
        joint_rot, joint_pos = self.body_model.forward_kinematics(p, calc_joint=True)

        imu_raw_rot = self.acc_cat_oris[-1][self.imu_num*3:].reshape(6, 3, 3)
        joint_rot[[18, 19, 4, 5, 15]] = imu_raw_rot[:5]
        fix_pose = self.body_model.inverse_kinematics(joint_rot)
        result = rotation_matrix_to_axis_angle(fix_pose)
        result[[20,21,22,23,7,8,10,11]] *= 0

        # print(joint_pos, self.last_joint_pos)
        # fk位移
        if vel_left < vel_right:
            d_trans_fk = self.last_joint_pos[10] - joint_pos[10]
        else:
            d_trans_fk = self.last_joint_pos[11] - joint_pos[11]
        self.last_joint_pos = joint_pos

        # nn位移
        d_trans_nn = torch.FloatTensor(vel).reshape(-1, 3)[0] / 60

        d_root_height_nn = d_trans_nn[1]

        # nn & fk高度
        root_height_fk = joint_pos[0][1] - min(joint_pos[10][1], joint_pos[11][1]) - self.root_height_init
        s = max(contact_left, contact_right)

        # s = (torch.abs(self.root_acc[1] - self.G[1]) / 8.0).clamp(min=0.0001, max=0.9999)
        # s = ((self.root_acc[1].clamp(min=self.G[1], max=0) - self.G[1]) / 5).clamp(min=0.0001, max=0.9999)

        if s < 0.1:
            self.floating_prob = min(max(self.floating_prob + 0.3, 0), 1)
            d_trans = d_trans_nn
        else:
            self.floating_prob = min(max(self.floating_prob - 0.3, 0), 1)
            d_trans = self.floating_prob * d_trans_nn + (1 - self.floating_prob) * d_trans_fk

        self.root_height = (1 - self.floating_prob) * root_height_fk + self.floating_prob * (
                self.root_height + d_root_height_nn)

        self.root_height = max(root_height_fk, self.root_height)

        if self.track_trans:
            self.trans[[0, 2]] += d_trans[[0, 2]]
        self.trans[1] = self.root_height

        # print('========================')
        # track_joint = {'左肩': 16, '右肩': 17, '脊柱-1': 3, '脊柱-2': 6, '脊柱-3': 9, '左肘': 18, '右肘': 19, '腰部': 0,
        #                '左胯': 1, '右胯': 2, '左膝': 4, '右膝': 5}
        # for jn, ji in track_joint.items():
        #     print(jn, ':', np.array(result[ji]).tolist())

        return result

    def get_trans(self):
        return self.trans.view(-1) + torch.FloatTensor([0, 0.06, 0])

    def compute_angle(self, vectors1, vectors2):
        """
        计算两个批次三维向量的夹角
        :param vectors1: 第一个批次的三维向量，形状为 (batch_size, 3)
        :param vectors2: 第二个批次的三维向量，形状为 (batch_size, 3)
        :return: 夹角的度数，形状为 (batch_size,)
        """

        # 确保输入是浮点数类型
        vectors1 = vectors1.float()
        vectors2 = vectors2.float()

        # 计算两个向量的点积
        dot_product = torch.sum(vectors1 * vectors2, dim=-1)

        # 计算向量的模
        norm_vectors1 = torch.norm(vectors1, dim=-1)
        norm_vectors2 = torch.norm(vectors2, dim=-1)

        # 计算余弦值
        cos_angles = dot_product / (norm_vectors1 * norm_vectors2)

        # 为了避免数值超出 [-1, 1] 范围，使用 clamp 均分
        cos_angles = torch.clamp(cos_angles, -1.0, 1.0)

        # 计算夹角，转换为弧度
        angles_rad = torch.acos(cos_angles)

        # 转换为度
        angles_deg = angles_rad * (180.0 / torch.pi)

        return angles_deg

class DataProcessServer_NOITOM4():
    def __init__(self, rotation_type, part, keep_hidden=True, run_unity_package=True, mode=demo_mode.UPPER,
                 track_trans=True, calibration_session=None, run_server=True):
        # import calibration_animation.main as caan
        # caan.main()

        if run_unity_package:
            run_mode1()
            # run_mode_girl()
            # run_mode_micai()
            # run_mode_up()
        if run_server:
            server_for_unity = socket(AF_INET, SOCK_STREAM)
            server_for_unity.bind(('127.0.0.1', 8888))
            server_for_unity.listen(1)
            print('Server start. Waiting for unity3d to connect.')
            self.conn, self.addr = server_for_unity.accept()
        self.rotation_type = rotation_type
        self.part = part
        self.keep_hidden = keep_hidden
        self.track_trans = track_trans

        self.imu_num = 6
        self.clock = Clock()

        self.mode = mode
        self.calibration_session = calibration_session
        self.property()
        self.trans_property()
        self.udp_client = socket(AF_INET, SOCK_DGRAM)  # 创建socket对象，走udp通道
    def property(self):
        """
        自定义属性
        :return:无返回值
        """
        # 用于calibration的数据
        self.init_oris = None
        self.tpose_accs = None

        self.global_shift = torch.eye(3).reshape(-1, 3, 3).repeat(self.imu_num, 1, 1).cuda()
        self.local_shift = torch.eye(3).reshape(-1, 3, 3).repeat(self.imu_num, 1, 1).cuda()

        # 串口版
        # self.root_2_smpl = bulid_rot(theta=180, rotation_axis=[0,0,1]).matmul(bulid_rot(theta=180, rotation_axis=[1,0,0]))
        self.root_2_smpl = bulid_rot(theta=180, rotation_axis=[0, 1, 0])


        self.smpl2imu = None
        self.device2bone = None
        self.acc_offsets = None

        n_layer = 2
        self.h_1 = np.zeros(shape=[n_layer, 1, 256]).astype(np.float32)
        self.h_2 = np.zeros(shape=[n_layer, 1, 256]).astype(np.float32)
        self.h_3 = np.zeros(shape=[n_layer, 1, 256]).astype(np.float32)
        self.c_1 = np.zeros(shape=[n_layer, 1, 256]).astype(np.float32)
        self.c_2 = np.zeros(shape=[n_layer, 1, 256]).astype(np.float32)
        self.c_3 = np.zeros(shape=[n_layer, 1, 256]).astype(np.float32)

        # 模型输入
        self.acc_cat_oris = []

    def set_init_state(self, init_states):
        import copy
        init_states = copy.copy(init_states)
        print('加载初始状态')
        for i in range(len(init_states)):
            init_states[i] = np.array(init_states[i].detach()).astype(np.float32)
        self.h_1, self.c_1, self.h_2, self.c_2, self.h_3, self.c_3 = tuple(init_states)


    def trans_property(self):
        # 位移相关
        self.trans = torch.FloatTensor([0, 0, 0])

        self.body_model = Aplus.tools.smpl_light.SMPLight()
        p = torch.eye(3).unsqueeze(0).repeat(24, 1, 1)
        # 输入24个关节旋转+体型参数+位移信息, 输出24个关节的旋转+蒙皮点加速度+运动速度
        grot, joint = self.body_model.forward_kinematics(p, calc_joint=True)

        self.root_height = joint[0][1] - min(joint[10][1], joint[11][1])
        self.root_height_init = joint[0][1] - min(joint[10][1], joint[11][1])
        self.last_joint_pos = joint

        self.max_buffer_len = 1000
        self.vel = torch.FloatTensor([0, 0, 0])
        self.p_contact = torch.ones(1)

        self.floating_prob = 0

        self.d_trans_fk_last = torch.FloatTensor([0, 0, 0])

    def get_device_2_bone(self):
        root_2_smpl = self.root_2_smpl
        left_arm_2_smpl = bulid_rot(theta=90, rotation_axis=[1, 0, 0]).matmul(bulid_rot(theta=180, rotation_axis=[0, 1, 0]))
        right_arm_2_smpl = bulid_rot(theta=90, rotation_axis=[1, 0, 0]).matmul(bulid_rot(theta=180, rotation_axis=[0, 1, 0]))
        left_leg_smpl = bulid_rot(theta=-90, rotation_axis=[0, 0, 1]).matmul(bulid_rot(theta=-90, rotation_axis=[0, 1, 0]))
        right_leg_smpl = bulid_rot(theta=90, rotation_axis=[0, 0, 1]).matmul(bulid_rot(theta=90, rotation_axis=[0, 1, 0]))
        head_2_smpl = bulid_rot(theta=0, rotation_axis=[1, 0, 0])

        device_2_bone = [left_arm_2_smpl, right_arm_2_smpl, left_leg_smpl, right_leg_smpl, head_2_smpl, root_2_smpl]

        if self.mode == demo_mode.FULL:
            left_lowerleg_2_smpl = self.config_pant['left_lowerleg_2_smpl']
            right_lowerleg_2_smpl = self.config_pant['right_lowerleg_2_smpl']
            left_pelvis_2_smpl = self.config_pant['left_pelvis_2_smpl']
            right_pelvis_2_smpl = self.config_pant['right_pelvis_2_smpl']

            device_2_bone += [left_lowerleg_2_smpl, right_lowerleg_2_smpl, left_pelvis_2_smpl, right_pelvis_2_smpl]
        return torch.cat(device_2_bone, dim=0)

    def normalize_and_concat(self, smpl_acc, smpl_ori):
        imu_num = self.imu_num
        acc_scale = 30
        smpl_acc = smpl_acc.view(-1, imu_num, 3)
        smpl_ori = smpl_ori.view(-1, imu_num, 3, 3)

        # print(smpl_ori[:100, 0])
        # acc: [n, 4, 3]
        # acc = torch.cat((smpl_acc[:, :(imu_num-1)] - smpl_acc[:, (imu_num-1):], smpl_acc[:, (imu_num-1):]), dim=1).bmm(smpl_ori[:, -1]) / acc_scale
        acc = smpl_acc / acc_scale
        # 不转换为相对加速度 变换至root坐标系
        # acc = smpl_acc.bmm(smpl_ori[:, -1]) / acc_scale

        # ori = torch.cat((smpl_ori[:, (imu_num-1):].transpose(2, 3).matmul(smpl_ori[:, :(imu_num-1)]), smpl_ori[:, (imu_num-1):]), dim=1)
        ori = smpl_ori

        data = torch.cat((acc.flatten(1), ori.flatten(1)), dim=1)
        return data

    def update_data(self, data):
        self.clock.tick()
        # print('\r', f'fps: {self.clock.get_fps()}', end='')
        # self.frame_num += 1
        # self.frame_num = self.frame_num % 2
        # if self.frame_num == 0:
        #     return

        pose = preprocess(data, self.rotation_type)

        # part of data freeze
        # pose.view(24, 3)[config.joint_set.part[self.part]] *= 0

        tran = self.get_trans()
        # send pose
        s = ','.join(['%g' % v for v in pose]) + '#' + \
            ','.join(['%g' % v for v in tran]) + '$'
        self.conn.send(s.encode('utf8'))  # I use unity3d to read pose and translation for visualization here

    def device2bone_init(self):
        self.device2bone = self.get_device_2_bone().cuda()

    def set_calibrate_data(self, input: list):
        """
        设置标定数据
        :param input: [48] -> [4 x 3 (acc) + 4 x 9(oris)]
        :return:无返回值
        """
        g = 9.8
        self.init_oris = torch.tensor(input[self.imu_num*3:]).view(self.imu_num, 3, 3)
        self.tpose_acc = torch.tensor(input[:self.imu_num*3]).view(self.imu_num, 3, 1)
        # print(self.tpose_acc)

        self.acc_offsets = torch.FloatTensor([[0, g, 0]]).repeat(self.imu_num, 1).cuda()
        # self.acc_offsets = self.init_oris.matmul(self.tpose_acc).cuda()
        print(self.acc_offsets)
        # self.tpose_elbow_angle = torch.tensor(input[48:50])

        # 修正全局坐标系误差
        # self.init_oris = self.I_2_Ibias.matmul(self.init_oris)
        # self.smpl2imu = self.init_oris[-1].matmul(self.root_2_smpl).view(3, 3).t().cuda()

        imu2smpl = self.init_oris[-1].matmul(self.root_2_smpl).view(3, 3)
        column_y = torch.FloatTensor([[0, 0, 1]])
        column_z = normalize_tensor(imu2smpl[:, [2]].t() - (column_y * imu2smpl[:, [2]].t()).sum(dim=1, keepdim=True) * column_y)
        column_x = column_y.cross(column_z, dim=1)
        imu2smpl = torch.stack((column_x, column_y, column_z), dim=-1).squeeze(0)
        self.smpl2imu = imu2smpl.t().cuda()

        # print(self.smpl2imu)

        if self.device2bone is None:
            self.device2bone = self.smpl2imu.matmul(self.init_oris.cuda()).transpose(1, 2).matmul(torch.eye(3).cuda())

    def anime_update(self, rotation_diversity, trigger):
        send_str = ''
        for rd in rotation_diversity:
            send_str += str(int(rd)) + ' '
        for i, t in enumerate(trigger):
            if t:
                send_str += str(i) + ' '
        self.udp_client.sendto(send_str.encode('utf-8'), ("127.0.0.1", 23338))

    @torch.no_grad()
    def auto_calibrate(self, time_gap=1, times=1):
        import time
        for _ in range(times):
            if len(self.acc_cat_oris) < 256:
                continue
            time.sleep(time_gap)
            frame_nums = 256
            # 降采样到128
            acc_cat_oris = torch.stack(self.acc_cat_oris[-frame_nums:]).reshape(frame_nums, -1)[::2]
            # 目前版本不使用加速度了, 全置0
            # acc_cat_oris[:, :self.imu_num * 3] *= 0
            oris = acc_cat_oris[:, self.imu_num * 3:].reshape(1, -1, self.imu_num, 3, 3)
            # 旋转丰富度
            diversity = rotation_diversity(oris).reshape(-1)

            diversity_threshold = torch.Tensor([30, 45, 30, 30, 20, 15]) * 1
            trigger_s2 = diversity > diversity_threshold
            # print(diversity)

            trigger = trigger_s2

            self.anime_update(diversity, trigger)

            keep_mask = ~trigger

            # 根节点不更新
            keep_mask[-1] = True
            skip_count = torch.sum(keep_mask).item()
            if skip_count < 5:
                self.global_shift = r6d_to_rotation_matrix(rotation_matrix_to_r6d(self.global_shift))
                self.local_shift = r6d_to_rotation_matrix(rotation_matrix_to_r6d(self.local_shift))
                acc_cat_oris = acc_cat_oris.reshape(1, -1, self.imu_num*(3 * 3 + 3))
                feed = {'imu_rot': np.array(acc_cat_oris)}
                global_shift, local_shift = self.calibration_session.run(output_names=None, input_feed=feed)
                # global_shift = torch.FloatTensor(global_shift).reshape(-1, 6)
                # global_shift = r6d_to_rotation_matrix(global_shift).transpose(-2,-1).cuda()

                global_shift = r6d_to_rotation_matrix(torch.tensor(global_shift).reshape(-1, 6))
                global_shift = ego_drift_regularization(global_shift).squeeze(0).transpose(-2, -1).cuda()


                local_shift = r6d_to_rotation_matrix(torch.FloatTensor(local_shift).reshape(-1, 6)).transpose(-2, -1).cuda()
                self.acc_cat_oris = self.acc_cat_oris[-1:]
                global_shift[keep_mask, :, :] = torch.eye(3).cuda().unsqueeze(0).repeat(skip_count, 1, 1)
                local_shift[keep_mask, :, :] = torch.eye(3).cuda().unsqueeze(0).repeat(skip_count, 1, 1)

                # angle_evaluator = PerJointRotationErrorEvaluator()
                # global_fix = angle_evaluator(global_shift.unsqueeze(0),
                #                              torch.eye(3).cuda().unsqueeze(0).repeat(self.imu_num, 1, 1).unsqueeze(0),
                #                              joint_num=self.imu_num)
                # local_fix = angle_evaluator(local_shift.unsqueeze(0),
                #                             torch.eye(3).cuda().unsqueeze(0).repeat(self.imu_num, 1, 1).unsqueeze(0),
                #                             joint_num=self.imu_num)

                # print('global矫正量', global_fix)
                # print('local矫正量', local_fix)
                self.global_shift = global_shift.matmul(self.global_shift)
                self.local_shift = self.local_shift.matmul(local_shift)
                print('自动校准')
            else:
                continue


    @torch.no_grad()
    def calibrate(self, input: list):
        """
        用于标定校准处理
        :param input: 传感器数据序列[48] -> [4 x 3 (acc) + 4 x 9(oris)]
        :return: 标定校准处理后的数据/原始传感器数据
        """
        input = torch.tensor(input).cuda()
        accs, oris = input[0:self.imu_num * 3].view(-1, self.imu_num, 3), input[self.imu_num * 3:].view(-1,self.imu_num, 3,3)
        # print(oris[-1])
        # acc转到全局坐标
        accs = oris.matmul(accs.view(-1, self.imu_num, 3, 1))

        # oris转到smpl->bone, 然后补正
        oris = self.global_shift.matmul(self.smpl2imu).matmul(oris).matmul(self.device2bone).matmul(self.local_shift)
        # acc转到smpl 并补正
        accs = self.global_shift.matmul(self.smpl2imu).matmul(accs).view(-1, self.imu_num, 3)

        accs = accs - self.acc_offsets

        # print('\r', accs[0, 0], end='')
        input_imu = self.normalize_and_concat(accs, oris).view(-1)

        input = input_imu

        return np.array(input.cpu())

    def operator(self, input: np.ndarray):
        """
        在标定校准后，进行预处理
        :param input: 标定校准处理后的数据/原始传感器数据
        :return: 无返回数据
        """
        max_length = 1000
        input = torch.tensor(input).reshape(-1)
        acc, rot, angle = input[:12], input[12:48], input[48:]
        rot_r6d = rotation_matrix_to_r6d(rot.reshape(4, 3, 3))
        rot_r6d = rot_r6d.reshape(-1)
        input = torch.cat([acc, rot_r6d], dim=-1)

        self.acc_cat_oris.append(input)
        # self.rot_buffer.append(input)
        if len(self.acc_cat_oris) > max_length:
            self.acc_cat_oris = self.acc_cat_oris[-max_length:]
        # if len(self.rot_buffer) > max_length:
        #     self.rot_buffer = self.rot_buffer[-max_length:]

    def to_predict_data(self):
        """
        onnx.run的参数
        :return: output_names, input_feed, run_options
        """

        # # 转r6d
        # acc_cat_oris = torch.tensor(self.acc_cat_oris[-1])
        # # print('\r', acc_cat_oris, end='')
        # acc, rot= acc_cat_oris[:self.imu_num*3], acc_cat_oris[self.imu_num*3:]
        # # print(rot.reshape(8, 3, 3))
        # rot_r6d = rotation_matrix_to_r6d(rot.reshape(self.imu_num, 3, 3))
        # rot_r6d = rot_r6d.reshape(-1)
        # acc_cat_oris = torch.cat([acc, rot_r6d], dim=-1)
        # acc_cat_oris = np.array(acc_cat_oris.unsqueeze(0))

        # 不转r6d
        acc_cat_oris = self.acc_cat_oris[-1]
        acc_cat_oris = np.array(acc_cat_oris.unsqueeze(0))


        input_feed = {'imu_data': acc_cat_oris,
                      'h_1': self.h_1,
                      'c_1': self.c_1,
                      'h_2': self.h_2,
                      'c_2': self.c_2,
                      'h_3': self.h_3,
                      'c_3': self.c_3
                      }

        return input_feed

    def predict_result(self, result):
        """
        返回的结果
        :return: [24, 3] 24个关节的轴角
        """
        pose, self.h_1, self.c_1, self.h_2, self.c_2, self.h_3, self.c_3 = result

        # vel = torch.FloatTensor(vel).reshape(-1, 3)

        # contact_left = 1 - (torch.norm(vel[10], 2) - 0.04).clamp(min=0, max=0.2) / 0.2
        # contact_right = 1 - (torch.norm(vel[11], 2) - 0.04).clamp(min=0, max=0.2) / 0.2

        # vel_left = torch.norm(vel[10], 2)
        # vel_right = torch.norm(vel[11], 2)

        # print(vel_left, vel_right)

        result = torch.FloatTensor(pose).reshape(24, 3)
        result[[7, 8]] *= 0

        p = axis_angle_to_rotation_matrix(result)
        joint_rot, joint_pos = self.body_model.forward_kinematics(p, calc_joint=True)

        # imu_raw_rot = self.acc_cat_oris[-1][self.imu_num*3:].reshape(6, 3, 3)
        # joint_rot[[18, 19, 4, 5, 15]] = imu_raw_rot[:5]
        fix_pose = self.body_model.inverse_kinematics(joint_rot)
        result = rotation_matrix_to_axis_angle(fix_pose)
        result[[20,21,22,23,7,8,10,11]] *= 0

        # print(joint_pos, self.last_joint_pos)
        # fk位移
        # if vel_left < vel_right:
        #     d_trans_fk = self.last_joint_pos[10] - joint_pos[10]
        # else:
        #     d_trans_fk = self.last_joint_pos[11] - joint_pos[11]
        # self.last_joint_pos = joint_pos

        # # nn位移
        # d_trans_nn = torch.FloatTensor(vel).reshape(-1, 3)[0] / 60

        # d_root_height_nn = d_trans_nn[1]

        # # nn & fk高度
        # root_height_fk = joint_pos[0][1] - min(joint_pos[10][1], joint_pos[11][1]) - self.root_height_init
        # s = max(contact_left, contact_right)

        # s = (torch.abs(self.root_acc[1] - self.G[1]) / 8.0).clamp(min=0.0001, max=0.9999)
        # s = ((self.root_acc[1].clamp(min=self.G[1], max=0) - self.G[1]) / 5).clamp(min=0.0001, max=0.9999)

        # if s < 0.1:
        #     self.floating_prob = min(max(self.floating_prob + 0.3, 0), 1)
        #     d_trans = d_trans_nn
        # else:
        #     self.floating_prob = min(max(self.floating_prob - 0.3, 0), 1)
        #     d_trans = self.floating_prob * d_trans_nn + (1 - self.floating_prob) * d_trans_fk

        # self.root_height = (1 - self.floating_prob) * root_height_fk + self.floating_prob * (
        #         self.root_height + d_root_height_nn)

        # self.root_height = max(root_height_fk, self.root_height)

        # if self.track_trans:
        #     self.trans[[0, 2]] += d_trans[[0, 2]]
        # self.trans[1] = self.root_height

        # print('========================')
        # track_joint = {'左肩': 16, '右肩': 17, '脊柱-1': 3, '脊柱-2': 6, '脊柱-3': 9, '左肘': 18, '右肘': 19, '腰部': 0,
        #                '左胯': 1, '右胯': 2, '左膝': 4, '右膝': 5}
        # for jn, ji in track_joint.items():
        #     print(jn, ':', np.array(result[ji]).tolist())

        return result

    def get_trans(self):
        return self.trans.view(-1) + torch.FloatTensor([0, 0.06, 0])

    def compute_angle(self, vectors1, vectors2):
        """
        计算两个批次三维向量的夹角
        :param vectors1: 第一个批次的三维向量，形状为 (batch_size, 3)
        :param vectors2: 第二个批次的三维向量，形状为 (batch_size, 3)
        :return: 夹角的度数，形状为 (batch_size,)
        """

        # 确保输入是浮点数类型
        vectors1 = vectors1.float()
        vectors2 = vectors2.float()

        # 计算两个向量的点积
        dot_product = torch.sum(vectors1 * vectors2, dim=-1)

        # 计算向量的模
        norm_vectors1 = torch.norm(vectors1, dim=-1)
        norm_vectors2 = torch.norm(vectors2, dim=-1)

        # 计算余弦值
        cos_angles = dot_product / (norm_vectors1 * norm_vectors2)

        # 为了避免数值超出 [-1, 1] 范围，使用 clamp 均分
        cos_angles = torch.clamp(cos_angles, -1.0, 1.0)

        # 计算夹角，转换为弧度
        angles_rad = torch.acos(cos_angles)

        # 转换为度
        angles_deg = angles_rad * (180.0 / torch.pi)

        return angles_deg
