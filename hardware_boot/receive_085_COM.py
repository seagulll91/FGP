import time
import serial
import serial.tools.list_ports as list_ports
import ctypes
import datetime
import threading

from collections import deque  # 全局缓冲区管理

from Aplus.tools.pd_controll import FpsController
import numpy as np

DEFAULT_BAUD = 460800

MAX_MTI3_NUM = 2
cnt_lost = 0
last_packetCnt = 0
begin_time = 0
final_time = 0

# 全局数据缓冲区
global data_up_buffer
global data_down_buffer
data_up_buffer = deque(maxlen=128)
data_down_buffer = deque(maxlen=128)


# 定义C结构体的对应Python结构体
class ImuData(ctypes.Structure):
    _pack_ = 1  # 设置对齐方式为单字节对齐
    _fields_ = [
        ("acceleration", ctypes.c_float * 3),
        ("quaternion", ctypes.c_float * 4)
    ]
class MyDatas(ctypes.Structure):
    _pack_ = 1  # 设置对齐方式为单字节对齐
    _fields_ = [
        ("header", ctypes.c_ubyte * 2),
        ("len", ctypes.c_uint16),
        ("imuData", ImuData * 12),
        ("chksum", ctypes.c_ubyte)
    ]

def find_ch340_ports():
    # """检测所有CH340设备"""
    ch340_list = []
    for port in list_ports.comports():
        # 双重匹配机制提高兼容性
        if ("USB-SERIAL CH340" in port.description) or \
           ("1A86:7523" in port.hwid.upper()):  # VID/PID验证
            ch340_list.append({
                "device": port.device,
                "desc": port.description,
                "hwid": port.hwid
            })
    return ch340_list

class MultiPortManager():
    def __init__(self):
        self.active_ports = {}  # 存储{端口名: serial对象}
        self.frame_size = 341  # 完整帧长度
    def connect_devices(self, max_devices=1):
        found_ports = find_ch340_ports()
        for port_info in found_ports[:max_devices]:
            try:
                ser = serial.Serial(
                    port=port_info["device"],
                    baudrate=DEFAULT_BAUD,
                    timeout=2
                )
                if ser.is_open:
                    self.active_ports[port_info["device"]] = (
                        ser, 
                        deque(maxlen=2048)# 每个端口关联一个deque缓冲区
                    )
                    print(f"成功连接 {port_info['desc']}")
            except serial.SerialException as e:
                print(f"连接失败 {port_info['device']}: {str(e)}")
    
    # def FlushSer(self):
    # self.ser.flushOutput()
    # self.ser.flushInput()
    def close_all(self):
        # """安全关闭所有端口"""
        for name, ser in self.active_ports.items():
            ser.close()

data_buffer_1 = data_up_buffer
data_buffer_2 = data_down_buffer

def start_data_threads(port_manager,data_buffer_1,data_buffer_2):
    threads = []
    for port_name, (ser, buffer) in port_manager.active_ports.items():
        thread = threading.Thread(
            target=data_handler,
            args=(ser, buffer, port_manager.frame_size, data_buffer_1, data_buffer_2),  # 传递独立缓冲区
            name=f"Thread-{port_name}",
            daemon=True
        )
        thread.start()
        threads.append(thread)
    return threads

def data_handler(ser, buffer, frame_size, data_buffer_1,data_buffer_2):
    last_frame_time = time.time()  # 初始化上一次帧时间

    while True:
        # 读取数据到本线程缓冲区


        if ser.in_waiting > 0:
            backlog = ser.in_waiting
            # print(f"串口积压字节数: {backlog}")
            data = ser.read(backlog)
            buffer.extend(data)
            # 只在新数据到达时记录一次时间


            current_time = time.time()
            interval = current_time - last_frame_time
            last_frame_time = current_time
            fps = 1.0 / interval if interval > 0 else float('inf')
            # print(f"[串口到达] 当前间隔: {interval * 1000:.2f} ms, 推测速率: {fps:.2f} fps")
        # 帧检测逻辑
        # while len(buffer) >= frame_size:   ###frame_size=341
        #     header_pos = -1
        #     for i in range(len(buffer)-1):
        #         if buffer[i] == 0xFF and buffer[i+1] == 0xFE:
        #             header_pos = i
        #             break
        #     if header_pos != -1 and len(buffer) >= header_pos + frame_size:
        #         frame = bytes([buffer.popleft() for _ in range(frame_size)])
        #         start = time.perf_counter()
        #         DecodeData(frame, data_buffer_1, data_buffer_2)
        #         print(f"[Decode] decode took {(time.perf_counter() - start) * 1000:.2f} ms")
        #         # ======================= 帧率统计核心 =======================
        #         # current_time = time.time()
        #         # interval = current_time - last_frame_time
        #         # last_frame_time = current_time
        #         #
        #         # fps = 1.0 / interval if interval > 0 else float('inf')
        #         # print(f"[帧率] 当前帧间隔: {interval * 1000:.2f} ms, 速率: {fps:.2f} fps")
        while len(buffer) >= frame_size:
            # 寻找合法帧头
            header_pos = -1
            for i in range(len(buffer) - 1):
                if buffer[i] == 0xFF and buffer[i + 1] == 0xFE:
                    header_pos = i
                    break

            if header_pos == -1:
                # 没找到帧头，清理无效数据避免 buffer 无限增长
                buffer.popleft()
                continue

            if len(buffer) < header_pos + frame_size:
                # 数据还不够一帧，先等待下次再读
                break

            # 移除帧头前的脏数据
            for _ in range(header_pos):
                buffer.popleft()

            # 取出一帧数据
            frame = bytes([buffer.popleft() for _ in range(frame_size)])

            start = time.perf_counter()
            DecodeData(frame, data_buffer_1, data_buffer_2)
            # print(f"[Decode] decode took {(time.perf_counter() - start) * 1000:.2f} ms")
def DecodeData(binary_data , data_buffer_1, data_buffer_2):
    # start_time = time.perf_counter()  # ⏱ 开始计时

    try:
        # 增加CRC校验（示例）
        # if self._crc16(data[:-2]) != int.from_bytes(data[-2:], 'big'):
        #     print("CRC校验失败")
        #     return

        # 将二进制数据解析为MyDatas结构体
        md = MyDatas.from_buffer_copy(binary_data)

        imu_data_array = []
        # 0, 2, 4, 5, 6, 8, 10, 11传感器有数据(4+4)
        # for i in [0, 2, 4, 5, 6, 8, 10, 11]:
        for i in range(0, 12):
            if i == 9:
                continue
            imu_data = md.imuData[i]

            acc_data = [imu_data.acceleration[0], imu_data.acceleration[1], imu_data.acceleration[2]]
            quat_data = [imu_data.quaternion[3], imu_data.quaternion[0], imu_data.quaternion[1],
                         imu_data.quaternion[2]]
            imu_data_array.append(acc_data + quat_data)



        data = np.stack(imu_data_array, axis=0)
      

        # 过滤nan值1
        if np.any(np.isnan(data)):
            return

        # 检测无效四元数
        q = data[:, 3:]
        # 计算每个向量的平方和
        squared_sums = np.sum(q ** 2, axis=-1)
        # 判断是否有平方和为0的向量
        if np.any(squared_sums < 1e-3):
            print('q')
            return


        data_buffer_1.append(data[:6, :])
        data_buffer_2.append(data[6:, :])
        if len(data_buffer_1) > 128:
            data_buffer_1 = data_buffer_1[-128:]
        if len(data_buffer_2) > 128:
            data_buffer_2 = data_buffer_2[-128:]
        # print('***', len(data_buffer_1), len(data_buffer_2))

    except Exception as e:
        print(f"解析异常：{str(e)}")
        return
    # finally:
    #     end_time = time.perf_counter()
    #     duration_ms = (end_time - start_time) * 1000  # 转换为毫秒
    #     print(f"[DecodeData] 处理耗时: {duration_ms:.3f} ms")

def check_connection(ser):
    # """硬件信号检测"""
    try:
        ser.rts = True
        time.sleep(0.01)
        return ser.cts  # 检测CTS硬件信号
    except:
        return False
def auto_reconnect(self):
    # """每30秒检测断线重连"""
    while True:
        time.sleep(30)
        for name in list(self.active_ports.keys()):
            if not check_connection(self.active_ports[name]):
                print(f"检测到断线: {name}")
                self.active_ports[name].close()
                del self.active_ports[name]
                self.connect_devices(max_devices=1)


class CmdException(Exception):
    pass

if __name__ == "__main__":
    manager = MultiPortManager()
    
    # # 设备连接
    # manager.connect_devices(max_devices=1)  # 可连接1-2个设备
    # data_buffer_1 = {}
    # data_buffer_2 = {}
    # # 启动数据处理
    # threads = start_data_threads(manager,data_buffer_1,data_buffer_1)
    
    # # 保活监控
    # watchdog = threading.Thread(
    #     target=auto_reconnect,
    #     daemon=True
    # )
    
    # global final_time
    # final_time = time.time()
    # print("use time", final_time - begin_time)
    # print("frequence", 1000/(final_time - begin_time))
    # 主线程监控
    try:
        ser1 = serial.Serial("COM5", 460800)    # 打开COM2，将波特率配置为115200，其余参数使用默认值
        while True: 
            time.sleep(1)
    except KeyboardInterrupt:
        manager.close_all()
