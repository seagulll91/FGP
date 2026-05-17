import serial
import serial.tools.list_ports
import struct
from collections import namedtuple


MTI_MODULE_NUM = 4
ONE_AXIS_MODULE_NUM = 2
BT_FRAME_HEADER_LEN = 2
QUATERNION_DATA_LEN = 16
SAMPLETIMEFINE_DATA_LEN = 4
ACCELERATION_DATA_LEN = 12
PACKETCOUNTER_DATA_LEN = 2
MtiData = namedtuple('MtiData', ['packet_counter', 'sample_time_fine', 'acceleration', 'quaternion'])
OneAxisData = namedtuple('OneAxisData', ['angle'])
BtFrame = namedtuple('BtFrame', ['header', 'len', 'mti_data', 'one_axis_data', 'check_sum'])


def parse_bt_frame(data):
    index = 0
    # Header
    header = data[index:index + BT_FRAME_HEADER_LEN]
    header = [byte for byte in header]
    index += BT_FRAME_HEADER_LEN

    # Length
    length = struct.unpack('<H', data[index:index + 2])[0]
    index += 2

    # MTI data
    mti_data_list = []
    for _ in range(MTI_MODULE_NUM):
        packet_counter = data[index:index + PACKETCOUNTER_DATA_LEN]
        packet_counter = [byte for byte in packet_counter]
        index += PACKETCOUNTER_DATA_LEN

        sample_time_fine = data[index:index + SAMPLETIMEFINE_DATA_LEN]
        sample_time_fine = [byte for byte in sample_time_fine]
        index += SAMPLETIMEFINE_DATA_LEN

        acceleration = data[index:index + ACCELERATION_DATA_LEN]
        acceleration = struct.unpack(">3f", acceleration)
        acceleration = [round(value, 10) for value in acceleration]
        index += ACCELERATION_DATA_LEN

        quaternion = data[index:index + QUATERNION_DATA_LEN]
        quaternion = struct.unpack(">4f", quaternion)
        quaternion = [round(value, 10)/1000 for value in quaternion]
        index += QUATERNION_DATA_LEN

        mti_data_list.append(MtiData(packet_counter, sample_time_fine, acceleration, quaternion))

    # One-axis data
    one_axis_data_list = []
    for _ in range(ONE_AXIS_MODULE_NUM):
        angle = struct.unpack('<f', data[index:index + 4])[0]
        angle = round(angle, 6)
        index += 4
        one_axis_data_list.append(OneAxisData(angle))

    # Checksum
    check_sum = data[index]

    return BtFrame(header, length, mti_data_list, one_axis_data_list, check_sum)