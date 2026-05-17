import torch
import numpy as np
from Math import *

def load_data(path):

    #pt/pth文件
    if path[-3:]=='.pt' or path[-4:]=='.pth':
        prediction_list=torch.load(path,map_location='cpu')
    
    #npy文件
    elif path[:-3]=='npy':
        prediction_list = np.load(path)
        prediction_list=torch.tensor(prediction_list)
    
    else:
        raise Exception('Unknown file fomart, only .pt/.pth/.npy is avaliable')

    return prediction_list

def preprocess(pose,rotation_type):

    if rotation_type=='AXIS_ANGLE':
        pose=pose.reshape(-1)

    elif rotation_type=='DCM':
        pose = rotation_matrix_to_axis_angle(pose.view(-1)).view(-1)

    #3 Untested types,May be wrong
    elif rotation_type=='QUATERNION':
        pose = quaternion_to_axis_angle(pose.view(-1)).view(-1)

    elif rotation_type=='EULER_ANGLE':
        pose = euler_angle_to_rotation_matrix(pose.view(-1))
        pose = rotation_matrix_to_axis_angle(pose.view(-1)).view(-1)

    elif rotation_type=='R6D':
        pose = r6d_to_rotation_matrix(pose.view(-1))
        pose = rotation_matrix_to_axis_angle(pose.view(-1)).view(-1)

    else:
        raise Exception('unknown rotation representation')
     
    return pose
