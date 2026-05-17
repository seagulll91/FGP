import asyncio
from bleak import BleakScanner, BleakClient, BleakError
import ctypes
import datetime
import time
import numpy as np
import threading
from Aplus.tools.pd_controll import FpsController
# 预设的两个设备地址，需要根据实际情况修改
# DEVICE_ADDRESSES = ["D5:87:74:70:AF:82", "CB:6F:67:6F:07:A8"]  ##[第一套6+5的衣服, 裤子]
# DEVICE_ADDRESSES = ["CE:2D:7E:3A:A6:35", "E3:6A:16:93:4F:45"]  ##[chenshi6+5的衣服, 裤子]
# DEVICE_ADDRESSES = ["FD:1F:AD:F8:92:D7", "EC:5C:24:8D:A1:D4"]  ##[第二套6+5的衣服, 裤子]
# DEVICE_ADDRESSES = ["FB:90:BA:BD:E2:93", "F6:C8:9C:8A:AA:E8"]
DEVICE_ADDRESSES = ["DB:08:BA:3F:70:41", "E6:D6:B2:A2:77:61"]  ##[防晒服6+5的衣服2号, 裤子2号]
# DEVICE_ADDRESSES = ["C3:4C:F4:84:D5:67", "CA:92:9F:7B:96:DD"]  ##[防晒服6+5的衣服, 裤子]
# DEVICE_ADDRESSES = ["C3:4C:F4:84:D5:67", "D2:72:6F:CC:8A:9B"]  ##[防晒服6+5的衣服, 裤子]



import asyncio
from bleak import BleakScanner, BleakClient, BleakError
import ctypes
import datetime
import time
import numpy as np
import threading
from Aplus.tools.pd_controll import FpsController
# 预设的两个设备地址，需要根据实际情况修改
# DEVICE_ADDRESSES = ["D5:87:74:70:AF:82", "CB:6F:67:6F:07:A8"]  ##[第一套6+5的衣服, 裤子]
# DEVICE_ADDRESSES = ["CE:2D:7E:3A:A6:35", "E3:6A:16:93:4F:45"]  ##[chenshi6+5的衣服, 裤子]
# DEVICE_ADDRESSES = ["FD:1F:AD:F8:92:D7", "EC:5C:24:8D:A1:D4"]  ##[第二套6+5的衣服, 裤子]
# DEVICE_ADDRESSES = ["FB:90:BA:BD:E2:93", "F6:C8:9C:8A:AA:E8"]
DEVICE_ADDRESSES = ["DB:08:BA:3F:70:41", "E6:D6:B2:A2:77:61"]  ##[防晒服6+5的衣服2号, 裤子2号]
# DEVICE_ADDRESSES = ["C3:4C:F4:84:D5:67", "CA:92:9F:7B:96:DD"]  ##[防晒服6+5的衣服, 裤子]
# DEVICE_ADDRESSES = ["C3:4C:F4:84:D5:67", "D2:72:6F:CC:8A:9B"]  ##[防晒服6+5的衣服, 裤子]






# 全局数据缓冲区
global data_up_buffer 
global data_down_buffer 
data_up_buffer = []
data_down_buffer = []

global up_packets_cnt
global up_get_time
global down_get_time
global down_packets_cnt
up_packets_cnt = 0
up_get_time = time.time()
down_get_time = time.time()
down_packets_cnt = 0

# # 锁保护
# up_buffer_lock = threading.Lock()
# down_buffer_lock = threading.Lock()

# 定义C结构体的对应Python结构体
class ImuData(ctypes.Structure):
    _pack_ = 1  # 设置对齐方式为单字节对齐
    _fields_ = [
        ("packetCnt", ctypes.c_uint16),
        ("sampleTimeFine", ctypes.c_uint32),
        ("acceleration", ctypes.c_float * 3),
        ("quaternion", ctypes.c_float * 4)
    ]

class OneAxisData(ctypes.Structure):
    _pack_ = 1  # 设置对齐方式为单字节对齐
    _fields_ = [
        ("angle", ctypes.c_float)
    ]

class TofData(ctypes.Structure):
    _pack_ = 1  # 设置对齐方式为单字节对齐
    _fields_ = [
        ("dis", ctypes.c_uint16 * 16)
    ]

class SensorDatas(ctypes.Structure):
    _pack_ = 1  # 设置对齐方式为单字节对齐
    _fields_ = [
        ("timestamp", ctypes.c_uint32),
        ("imuData", ImuData * 2)

    ]

class Left_SensorDatas(ctypes.Structure):
    _pack_ = 1  # 设置对齐方式为单字节对齐
    _fields_ = [
        ("timestamp", ctypes.c_uint32),
        ("imuData", ImuData * 4),
    ]

class MyDatas(ctypes.Structure):
    _pack_ = 1  # 设置对齐方式为单字节对齐
    _fields_ = [
        ("header", ctypes.c_ubyte * 2),
        ("len", ctypes.c_uint16),
        ("sensors0", Left_SensorDatas),
        ("sensors1", SensorDatas),
        ("chksum", ctypes.c_ubyte)
    ]


# 目标设备名称
TARGET_NAME = "SmartWear"

# 最大重试次数
MAX_RETRIES = 3


async def handle_up_data(data):
    global data_up_buffer
    global up_get_time
    global up_packets_cnt
    # 每次通知只处理当前这一包数据，不能在回调里自旋，否则会卡住后续 BLE 通知。
    starttime = time.time()
    try:
        md = MyDatas.from_buffer_copy(data)
        if md.header[0] != 0xFF or md.header[1] != 0xFE:
            return
        
        # current_time = time.time()
        # if current_time - up_get_time >=2:
        #     up_get_time = time.time()
        #     packets_cnt = md.sensors0.imuData[0].packetCnt
        #     FPS = (packets_cnt - up_packets_cnt)/2
        #     up_packets_cnt = packets_cnt
        #     # print('已接收衣服数据boot:', len(data_up_buffer))
        # print('\r', f'当前上衣传输帧率(帧/秒):{fc.get_fps():.2f}')
        imu_data_array = []

        for i in range(0, 4):
            # 第一条sensor的0,2条传感器有数据
            imu_data = md.sensors0.imuData[i]

            acc_data = [imu_data.acceleration[0], imu_data.acceleration[1], imu_data.acceleration[2]]
            quat_data = [imu_data.quaternion[3], imu_data.quaternion[0], imu_data.quaternion[1],
                        imu_data.quaternion[2]]
            imu_data_array.append(acc_data + quat_data)

        for imu_data in md.sensors1.imuData:
            acc_data = [imu_data.acceleration[0], imu_data.acceleration[1], imu_data.acceleration[2]]
            quat_data = [imu_data.quaternion[3], imu_data.quaternion[0], imu_data.quaternion[1],
                        imu_data.quaternion[2]]

            imu_data_array.append(acc_data + quat_data)

        data = np.array(imu_data_array).reshape(6, 7)

        # 过滤nan值
        if np.any(np.isnan(data)):
            print('nan')
            return

        # 检测无效四元数
        q = data[:, 3:]
        # 计算每个向量的平方和
        squared_sums = np.sum(q ** 2, axis=-1)
        # 判断是否有平方和为0的向量
        if np.any(squared_sums < 1e-3):
            # print(q)
            return

        data_up_buffer.append(data)
        data_up_buffer = data_up_buffer[-128:]
        endtime = time.time()
        # print('up', endtime - starttime)
    except Exception as e:
        print(f"上衣数据处理错误: {e}")

async def handle_down_data(data):
    global data_down_buffer
    global down_get_time
    global down_packets_cnt
    # 每次通知只处理当前这一包数据，不能在回调里自旋，否则会卡住后续 BLE 通知。
    starttime = time.time()
    try:
        md = MyDatas.from_buffer_copy(data)
        if md.header[0] != 0xFF or md.header[1] != 0xFE:
            return
        
        # current_time = time.time()
        # if current_time - down_get_time >=2:
        #     down_get_time = time.time()
        #     packets_cnt = md.sensors0.imuData[0].packetCnt
        #     FPS = (packets_cnt - down_packets_cnt)/2
        #     down_packets_cnt = packets_cnt
        #     print('已接收裤子数据boot:', len(data_down_buffer))
        # print('\r', f'当前裤子传输帧率(帧/秒):{fc.get_fps():.2f}')

        imu_data_array = []

        for i in [0, 1, 2]:
            # 第一条sensor的0,2条传感器有数据
            imu_data = md.sensors0.imuData[i]

            acc_data = [imu_data.acceleration[0], imu_data.acceleration[1], imu_data.acceleration[2]]
            quat_data = [imu_data.quaternion[3], imu_data.quaternion[0], imu_data.quaternion[1],
                        imu_data.quaternion[2]]
            imu_data_array.append(acc_data + quat_data)

        for imu_data in md.sensors1.imuData:
            acc_data = [imu_data.acceleration[0], imu_data.acceleration[1], imu_data.acceleration[2]]
            quat_data = [imu_data.quaternion[3], imu_data.quaternion[0], imu_data.quaternion[1],
                        imu_data.quaternion[2]]

            imu_data_array.append(acc_data + quat_data)

        data = np.array(imu_data_array).reshape(5, 7)

        # 过滤nan值
        if np.any(np.isnan(data)):
            return

        # 检测无效四元数
        q = data[:, 3:]
        # 计算每个向量的平方和
        squared_sums = np.sum(q ** 2, axis=-1)
        # 判断是否有平方和为0的向量
        if np.any(squared_sums < 1e-3):
            # print(q)
            return

        data_down_buffer.append(data)
        data_down_buffer = data_down_buffer[-128:]
        # print(data_down_buffer[-1])
        endtime = time.time()
        # print('down', endtime - starttime)
            
    except Exception as e:
        print(f"裤子数据处理错误: {e}")

# def get_up_data():
#     global data_up_buffer
#     with up_buffer_lock:
#         return data_up_buffer
#
#
# def get_down_data():
#     global data_down_buffer
#     with down_buffer_lock:
#         return data_down_buffer


async def notification_handler(sender, data, device_address):
    ###异步notify通知处理函数###
    try:
        # 数据入队处理
        if device_address == DEVICE_ADDRESSES[0]:
            await handle_up_data(data)
        else:
            await handle_down_data(data)
    except Exception as e:
        print(f"通知处理错误: {e}")


async def connect_and_explore(device_address):
    ###设备搜索、开启通知###
    retries = 0
    target_uuid = "adaf0101-c332-42a8-93bd-25e905756cb8"  #指定IMU数据uuid
    while retries < MAX_RETRIES:
        try:
            async with BleakClient(device_address, timeout=30.0) as client:
                print('\r',f"Connected to {device_address}")
                
                # 启动通知
                await client.start_notify(
                    target_uuid,
                    lambda s, d: asyncio.create_task(
                        notification_handler(s, d, device_address)  # 正确传递三个参数
                    )
                )

                print('\r',f"start_notify from {device_address}")
                
                # 保持连接
                while True:
                    await asyncio.sleep(3600)  # 长时间运行

        except (TimeoutError, BleakError) as e:
            print('\r',f"Connection attempt {retries+1} failed: {e}")
            retries += 1
        except Exception as e:
            print('\r',f"Unexpected error: {e}")
            retries += 1
    else:
        print('\r',f"Failed to connect to {device_address}")


# 异步程序封装
async def run_async_ble():
    print('启动异步接收')
    devices = [connect_and_explore(addr) for addr in DEVICE_ADDRESSES]
    await asyncio.gather(*devices)


        
async def bleak_receive():
    ###主协程###
    # 启动设备连接
    devices = [connect_and_explore(addr) for addr in DEVICE_ADDRESSES]
    await asyncio.gather(*devices)

if __name__ == "__main__":
    try:
        asyncio.run(bleak_receive())
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
