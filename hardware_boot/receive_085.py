import time
import serial
import serial.tools.list_ports as list_ports
import ctypes
import datetime
from Aplus.tools.pd_controll import FpsController
import numpy as np

# DEFAULT_BAUD = 921600
DEFAULT_BAUD = 115200

MAX_MTI3_NUM = 2
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
        ("imuData", ImuData * MAX_MTI3_NUM),
        # ("angleData", OneAxisData),
        # ("tofData", TofData)
    ]

class Left_SensorDatas(ctypes.Structure):
    _pack_ = 1  # 设置对齐方式为单字节对齐
    _fields_ = [
        ("timestamp", ctypes.c_uint32),
        ("imuData", ImuData * 4),
        # ("angleData", OneAxisData),
        # ("tofData", TofData)
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

class DongleReceiver:
    """
    085低成本衣物接收方案，需要import ‘dongle_decode’ 中的类
    """

    def __init__(self, port):
        isFirst = True
        pos = 0
        portList = []
        devCnt = 0
        for item in list_ports.comports():
            if item.vid == 0x2FE3 and item.pid == 0x0004 and item.hwid.endswith('2'):
                if devCnt == 0:
                    print("Port Lists:")
                devCnt = devCnt + 1
                print(f'[{pos}]:{item.device} {item.hwid}')
                portList.append(item.device)
                pos = pos + 1
        if devCnt == 0:
            print("Not Serial Fouled!")
            return
        if devCnt > 1:
            sel = port
        else:
            sel = 0
        if sel >= pos:
            print("Error Sel!")
            return

        self.port = portList[sel]
        self.ser = serial.Serial(self.port, DEFAULT_BAUD, timeout=2)
        self.fc = FpsController(50)
        print('Connect to Uart Done!')
        time.sleep(0.5)

        if self.ser is None:
            raise CmdException("Failed to connect uart")

    def FlushSer(self):
        self.ser.flushOutput()
        self.ser.flushInput()

    def AutoReceive(self, data_buffer):

        if self.ser is None:
            return
        cnt = 0
        prefix = '[Notify]:'
        while True:
            self.fc.sleep()
            rawline = self.ser.readline()
            rawline = rawline.decode(encoding="utf-8", errors="replace").strip()
            if rawline.startswith(prefix):
                rawline = rawline[len(prefix):]
                self.DecodeData(rawline, data_buffer)
                # print(self.fc.get_fps())


    def DecodeData(self, data, data_buffer):
        # global data_buffer_angle
        # current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        # binary_data = None
        # try:
        #     binary_data = bytes.fromhex(data)
        # except ValueError:
        #     pass  # 忽略错误，继续执行
        # 将二进制数据解析为MyDatas结构体
        try:
            binary_data = bytes.fromhex(data)
            md = MyDatas.from_buffer_copy(binary_data)
        except (ValueError, TypeError) as e:
            print(f"发生错误: {e}")
            return

        # 将二进制数据解析为MyDatas结构体
        # md = MyDatas.from_buffer_copy(binary_data)
        imu_data_array = []

        for i in [0, 2]:
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

        data = np.array(imu_data_array).reshape(4, 7)

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

        data_buffer.append(data)
        data_buffer = data_buffer[-128:]

class CmdException(Exception):
    pass

def main():
    try:
        dongle = DongleReceiver()
        dongle.FlushSer()
        dongle.AutoReceive()

    except Exception as e:
        print(str(e))

if __name__ == '__main__':
    main()