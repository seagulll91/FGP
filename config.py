import torch
import numpy as np
import quaternion
import math
from articulate.math import quaternion_to_rotation_matrix

imu_num = 6



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

amass_data = ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh', 'Transitions_mocap', 'SSM_synced', 'CMU',
              'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'BMLmovi', 'EKUT', 'TCD_handMocap', 'ACCAD',
              'BioMotionLab_NTroje', 'BMLhandball', 'MPI_Limits', 'DFaust67']

class joint_set:
    joint_name_list = ["pelvis", "l_hip", "r_hip", "spine1", "l_knee", "r_knee", "spine2", "l_ankle", "r_ankle",
                       "spine3", "l_toe", "r_toe", "neck", "l_collar", "r_collar", "head", "l_shoulder", "r_shoulder",
                       "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_palm", "r_palm"]
    # 设置要预测的关节
    # index_pose = torch.tensor([3, 6, 9, 12, 16, 17, 18, 19])
    # index_pose = torch.tensor([0, 3, 6, 9, 13, 14, 16, 17, 18, 19])
    # index_joint = torch.tensor([3, 6, 9, 13, 14, 16, 17, 18, 19, 20, 21])
    index_pose = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19])
    index_joint = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 16, 17, 18, 19, 20, 21])

    joint_num = len(index_pose)
    endpoint_num = len(index_joint)

    internal_joint = torch.tensor([0, 3, 6, 9, 13, 14])
    external_joint = torch.tensor([16, 17, 18, 19])

    index_l_elbow = 20
    index_r_elbow = 21

class garment_imu_set:
    # 起点/终点
    imu_axis = {'left' :{'axis_1':[916, 4795],  'axis_2':[4796, 4828], 'order':'zx'},
                'right':{'axis_1':[8504, 2109], 'axis_2':[8529, 8369], 'order':'zx'},
                'back' :{'axis_1':[7916, 6021], 'axis_2':[6022, 1226], 'order':'xy'},
                'root' :{'axis_1':[7972, 6000], 'axis_2':[1220, 5999], 'order':'xy'}}
class paths:
    imu4_dir = 'E:\DATA\processed_data\\'  # 采集的数据的路径

    raw_amass_dir = 'E:/DATA/AMASS'      # raw AMASS dataset path (raw_amass_dir/ACCAD/ACCAD/s001/*.npz)
    amass_dir = 'E:\DATA\processed_AMASS'         # output path for the synthetic AMASS dataset

    raw_dipimu_dir = 'data/dataset_raw/DIP_IMU'   # raw DIP-IMU dataset path (raw_dipimu_dir/s_01/*.pkl)
    dipimu_dir = 'E:\DATA\DIP_IMU6'      # output path for the preprocessed DIP-IMU dataset
    lip_dir = 'F:\宽松全身'

    # DIP recalculates the SMPL poses for TotalCapture dataset. You should acquire the pose data from the DIP authors.
    raw_totalcapture_dip_dir = 'data/dataset_raw/TotalCapture/DIP_recalculate'  # contain ground-truth SMPL pose (*.pkl)
    raw_totalcapture_official_dir = 'data/dataset_raw/TotalCapture/official'    # contain official gt (S1/acting1/gt_skel_gbl_pos.txt)
    totalcapture_dir = 'data/dataset_work/TotalCapture'          # output path for the preprocessed TotalCapture dataset

    example_dir = 'data/example'                    # example IMU measurements
    # smpl_file = 'models/SMPL_male.pkl'              # official SMPL model path
    smpl_file = 'E:\DATA\smpl\smpl/SMPL_MALE.pkl'  # official SMPL model path

acc_scale = 30
vel_scale = 3

class device_config:
    # 深蓝宽松
    new_1 = {'imu_order': [1, 2, 0, 3],
             'root_2_smpl': torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).matmul(torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])).float(),
             'root_2_left': bulid_rot(theta=-90, rotation_axis=[0, 1, 0]).matmul(bulid_rot(theta=-90, rotation_axis=[0, 0, 1])),
             'root_2_right': bulid_rot(theta=-90, rotation_axis=[0, 1, 0]).matmul(bulid_rot(theta=90, rotation_axis=[0, 0, 1])),
             'root_2_back': bulid_rot(theta=180, rotation_axis=[1, 0, 0]),
             'mac': 'C0D63C4A1ECC'}
    # 迷彩款
    new_2 = {'imu_order': [2, 1, 0, 3],
             'root_2_smpl': torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).matmul(torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])).float(),
             'root_2_left': bulid_rot(theta=-90, rotation_axis=[0, 1, 0]).matmul(bulid_rot(theta=-90, rotation_axis=[0, 0, 1])),
             'root_2_right': bulid_rot(theta=-90, rotation_axis=[0, 1, 0]).matmul(bulid_rot(theta=90, rotation_axis=[0, 0, 1])),
             'root_2_back': torch.eye(n=3).unsqueeze(dim=0),
             'mac': 'xx'}



    # 夹克 数据顺序：背部 右手 左手 腰部 模型输入顺序要求为 左手 右手 背部 腰部
    jacket = {'imu_order': [2, 1, 0, 3],
             'root_2_smpl': torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).matmul(torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])).float(),
             'root_2_left': bulid_rot(theta=-90, rotation_axis=[0, 1, 0]).matmul(bulid_rot(theta=-90, rotation_axis=[0, 0, 1])),
             'root_2_right':bulid_rot(theta=-90, rotation_axis=[0, 1, 0]).matmul(bulid_rot(theta=90, rotation_axis=[0, 0, 1])),
             'root_2_back': torch.eye(n=3).unsqueeze(dim=0),
             'mac': '?'}

    # 灰色裤子 数据顺序：右腰 右腿 左腿 左腰 模型输入顺序要求为 左腿 右腿 左腰 右腰
    pant = {'imu_order': [2, 1, 3, 0],
             'left_pelvis_2_smpl': bulid_rot(theta=-90, rotation_axis=[1, 0, 0]).matmul(bulid_rot(theta=-90, rotation_axis=[0, 0, 1])),
             'right_pelvis_2_smpl': bulid_rot(theta=-90, rotation_axis=[0, 1, 0]).matmul(bulid_rot(theta=180, rotation_axis=[0, 0, 1])),
             'left_lowerleg_2_smpl': bulid_rot(theta=90, rotation_axis=[0, 0, 1]),
             'right_lowerleg_2_smpl': bulid_rot(theta=90, rotation_axis=[0, 0, 1]),
             'mac': 'C0D63C4A1ECE'}

    pant_micai = {'imu_order': [1, 2, 3, 0],
            'left_pelvis_2_smpl': bulid_rot(theta=-90, rotation_axis=[1, 0, 0]).matmul(
                bulid_rot(theta=-90, rotation_axis=[0, 0, 1])),
            'right_pelvis_2_smpl': bulid_rot(theta=-90, rotation_axis=[0, 1, 0]).matmul(
                bulid_rot(theta=180, rotation_axis=[0, 0, 1])),
            'left_lowerleg_2_smpl': bulid_rot(theta=90, rotation_axis=[0, 0, 1]),
            'right_lowerleg_2_smpl': bulid_rot(theta=90, rotation_axis=[0, 0, 1]),
            'mac': 'C0D63C4A1ECE'}

    jacket_17 = {'imu_order': [2, 1, 0, 3],
              'root_2_smpl': torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).matmul(
                  torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])).float(),
              'root_2_left': bulid_rot(theta=-90, rotation_axis=[0, 1, 0]).matmul(
                  bulid_rot(theta=-90, rotation_axis=[0, 0, 1])),
              'root_2_right': bulid_rot(theta=-90, rotation_axis=[0, 1, 0]).matmul(
                  bulid_rot(theta=90, rotation_axis=[0, 0, 1])),
              'root_2_back': torch.eye(n=3).unsqueeze(dim=0),
              'mac': 'C0D63C4C97A6'}

    micai_jacket = {'imu_order': [1, 2, 0, 3],
                 'root_2_smpl': bulid_rot(theta=90, rotation_axis=[0, 0, 1]).reshape(3,3),
                 'root_2_right': bulid_rot(theta=-90, rotation_axis=[0, 1, 0]).matmul(bulid_rot(theta=180, rotation_axis=[0, 1, 0])).matmul(
                     bulid_rot(theta=-90, rotation_axis=[0, 0, 1])),
                 'root_2_left': bulid_rot(theta=-90, rotation_axis=[0, 1, 0]).matmul(bulid_rot(theta=180, rotation_axis=[0, 1, 0])).matmul(
                     bulid_rot(theta=90, rotation_axis=[0, 0, 1])),
                 'root_2_back': bulid_rot(theta=180, rotation_axis=[0, 1, 0]),
                 'mac': 'C0D63C4C97A6'}

    jacket_208 = {'imu_order': [3, 0, 2, 1],  # 208交付
                  'root_2_smpl': torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).matmul(
                      torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])).float(),
                  'root_2_left': bulid_rot(theta=-90, rotation_axis=[0, 1, 0]).matmul(
                      bulid_rot(theta=-90, rotation_axis=[0, 0, 1])),
                  'root_2_right': bulid_rot(theta=-90, rotation_axis=[0, 1, 0]).matmul(
                      bulid_rot(theta=90, rotation_axis=[0, 0, 1])),
                  'root_2_back': torch.eye(n=3).unsqueeze(dim=0),
                  'mac': 'C0D63C4A1ECC'}
    
    # 模型输入顺序：
    # 左手，右手，左大臂，右大臂，背部（删掉的root）
    # 左脚，右脚，左大腿，右大腿，腰
    #
    # 硬件IMU顺序：
    # 右大臂，右手，腰，背，左大臂，左手
    # 右大腿，右脚，腰，左大腿，左脚
    #
    jacket_6IMU = {'imu_order': [5,1,4,0,3],  # 6+5
                  'root_2_smpl': torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).matmul(
                      torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])).float(),
                  'root_2_left': bulid_rot(theta=-90, rotation_axis=[0, 1, 0]).matmul(
                      bulid_rot(theta=-90, rotation_axis=[0, 0, 1])),
                  'root_2_right': bulid_rot(theta=-90, rotation_axis=[0, 1, 0]).matmul(
                      bulid_rot(theta=90, rotation_axis=[0, 0, 1])),
                  'root_2_back': torch.eye(n=3).unsqueeze(dim=0),
                  'mac': 'C0D63C4A1ECC'}
    pants_5IMU = {'imu_order': [4,1,3,0,2],  # 6+5
                   'root_2_smpl': torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).matmul(
                       torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])).float(),
                   'root_2_left': bulid_rot(theta=-90, rotation_axis=[0, 1, 0]).matmul(
                       bulid_rot(theta=-90, rotation_axis=[0, 0, 1])),
                   'root_2_right': bulid_rot(theta=-90, rotation_axis=[0, 1, 0]).matmul(
                       bulid_rot(theta=90, rotation_axis=[0, 0, 1])),
                   'root_2_back': torch.eye(n=3).unsqueeze(dim=0),
                   'mac': 'C0D63C4A1ECC'}

    pant_208 = {
        'imu_order': [3, 1, 2, 0],  # 208交付
        # 'imu_order': [1, 3, 0, 2],  # 东北交付
        'left_pelvis_2_smpl': bulid_rot(theta=-90, rotation_axis=[0, 1, 0]).matmul(
            bulid_rot(theta=180, rotation_axis=[1, 0, 0])),
        'right_pelvis_2_smpl': bulid_rot(theta=180, rotation_axis=[0, 0, 1]).matmul(
            bulid_rot(theta=90, rotation_axis=[0, 1, 0])),
        'left_lowerleg_2_smpl': bulid_rot(theta=90, rotation_axis=[0, 0, 1]),
        'right_lowerleg_2_smpl': bulid_rot(theta=90, rotation_axis=[0, 0, 1]),
        'mac': 'C0D63C4A1ECE'}

class demo_mode:
    UPPER = 0
    FULL = 1

# def bulid_rot_old(theta, rotation_axis):
#     w = np.cos(theta * np.pi / 360)
#     s = np.sin(theta * np.pi / 360)
#     x = s * rotation_axis[0]
#     y = s * rotation_axis[1]
#     z = s * rotation_axis[2]
#
#     q = quaternion.from_float_array([w, x, y, z])
#     q = torch.Tensor([q.w, q.x, q.y, q.z]).float()
#     rot = quaternion_to_rotation_matrix(q)
#
#     return rot



class joint_set:
    joint_name_list = ["pelvis", "l_hip", "r_hip", "spine1", "l_knee", "r_knee", "spine2", "l_ankle", "r_ankle",
                       "spine3", "l_toe", "r_toe", "neck", "l_collar", "r_collar", "head", "l_shoulder", "r_shoulder",
                       "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_palm", "r_palm"]
    
    leaf = [12, 20, 21]
    full = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    reduced = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
    ignored = [0, 20, 21, 22, 23]

    joint_list=[i for i in range(len(joint_name_list))]
    
    lower_body = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    upper_body=list(set(joint_list)-set(lower_body))

    only_lower_body=upper_body
    only_upper_body=lower_body
    
    lower_body_parent = [None, 0, 0, 1, 2, 3, 4, 5, 6]

    left_hand=[13,16,18,20,22]
    only_left_hand=list(set(joint_list) - set(left_hand))

    right_hand=[14,17,19,21,23]
    only_right_hand=list(set(joint_list) - set(right_hand))

    hands=[13,16,18,20,22,14,17,19,21,23]
    only_hands=list(set(joint_list) - set(hands))

    head=[12,15]
    only_head=list(set(joint_list)-set(head))

    spine=[0,3,6,9]
    only_spine=list(set(joint_list)-set(spine))
    
    left_leg=[2,5,8,11]
    only_left_leg=list(set(joint_list)-set(left_leg))

    right_leg=[1,4,7,10]
    only_right_leg=list(set(joint_list)-set(right_leg))

    part={
        'body':[],
        'upper_body':only_upper_body,
        'lower_body':only_lower_body,
        'head':only_head,
        'spine':only_spine,
        'left_hand':only_left_hand,
        'right_hand':only_right_hand,
        'left_leg':only_left_leg,
        'right_leg':only_right_leg,
        'hands':only_hands
    }

    n_leaf = len(leaf)
    n_full = len(full)
    n_reduced = len(reduced)
    n_ignored = len(ignored)


class old_joint_set:
    leaf = [7, 8, 12, 20, 21]
    full = list(range(1, 24))
    reduced = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
    ignored = [0, 7, 8, 10, 11, 20, 21, 22, 23]

    lower_body = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    lower_body_parent = [None, 0, 0, 1, 2, 3, 4, 5, 6]

    n_leaf = len(leaf)
    n_full = len(full)
    n_reduced = len(reduced)
    n_ignored = len(ignored)


acc_scale = 30
vel_scale = 3
