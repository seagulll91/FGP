import time
import numpy as np
from datetime import datetime

def read_txt(txt_path):
    """
    读取保存为txt的数据, 数据第一列为时间戳
    Args:
        txt_path: txt文件路径

    Returns:
        time_sec: 以秒记的时间戳, numpy格式
        data: 数据, numpy格式
    """
    f = open(txt_path, 'r', encoding='ascii')
    data_buffer = []
    time = []
    for line in f.readlines():
        data = line.split(',')
        # 记录数据与时间戳
        time.append(time_stamp2float(data[0]))
        data_buffer.append(data[1:])
    data = np.array(data_buffer, dtype='float')
    time_sec = np.array(time)
    return time_sec, data
def get_time_stamp():
    return datetime.now().strftime('%Y-%m-%d-%H:%M:%S.%f')
def time_stamp2float(s):
    """
    将time.time()获取的时间戳字符串转换为秒
    Args:
        s: 时间戳字符串
    Returns:

    """
    t = s.split("-")[3].split(":")
    return float(t[0]) * 3600 + float(t[1]) * 60 + float(t[2])

def fps_fix(time:np.ndarray, data:np.ndarray, fps_fix:int):
    """
    通过时间戳信息稳定数据帧率
    Args:
        time: 时间戳(单位秒)
        data: 数据 n x feature_dim, numpy格式
        fps_fix: 目标fps

    Returns:

    """

    record_time = time[-1] - time[0]
    # 稳定帧率的时间戳
    n_frames_fixed = int(record_time * fps_fix)
    time_fix = np.linspace(start=time[0], stop=time[-1], num=n_frames_fixed).tolist()
    processed_data = []

    # 基于稳定帧率的时间戳进行插值
    for i in range(data.shape[1]):
        dim_i = data[:, i]
        dim_i = np.interp(time_fix, time, dim_i)[:, np.newaxis]
        processed_data.append(dim_i)
    processed_data = np.concatenate(processed_data, axis=-1)
    return processed_data

def fps_change(data, fps_origin, fps_target):
    """
    对帧率固定的数据进行fps变换
    Args:
        data: 数据 n x feature_dim, numpy格式
        fps_origin: 原始fps
        fps_target: 目标fps

    Returns:

    """
    record_time = len(data)
    # 稳定帧率的时间戳
    n_frames_fixed = int(record_time * fps_target / fps_origin)
    time_origin = range(record_time)
    time_fix = np.linspace(start=0, stop=record_time-1, num=n_frames_fixed).tolist()
    processed_data = []

    # 基于稳定帧率的时间戳进行插值
    for i in range(data.shape[1]):
        dim_i = data[:, i]
        dim_i = np.interp(time_fix, time_origin, dim_i)[:, np.newaxis]
        processed_data.append(dim_i)
    processed_data = np.concatenate(processed_data, axis=-1)
    return processed_data


