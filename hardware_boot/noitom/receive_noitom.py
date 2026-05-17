r"""
    Live demo using Noitom Perception Neuron Lab IMUs.
"""

import time
import torch
from hardware_boot.noitom import *


class NoitomIMUSet:
    g = 9.80665
    def __init__(self, udp_port=7070):
        app = MCPApplication()
        settings = MCPSettings()
        settings.set_udp(udp_port)
        settings.set_calc_data()
        app.set_settings(settings)
        app.open()
        time.sleep(0.5)
        sensors = [None for _ in range(6)]
        evts = []
        while len(evts) == 0:
            evts = app.poll_next_event()
            for evt in evts:
                assert evt.event_type == MCPEventType.SensorModulesUpdated
                sensor_module_handle = evt.event_data.sensor_module_data.sensor_module_handle
                sensor_module = MCPSensorModule(sensor_module_handle)
                print(sensor_module.get_id())
                sensors[sensor_module.get_id()-1] = sensor_module

        print('find %d sensors' % len([_ for _ in sensors if _ is not None]))
        self.app = app
        self.sensors = sensors
        self.t = 0

    def get(self):
        evts = self.app.poll_next_event()
        if len(evts) > 0:
            self.t = evts[0].timestamp
        q, a = [], []
        for sensor in self.sensors:
            q.append(sensor.get_posture())
            a.append(sensor.get_accelerated_velocity())

        # assuming g is positive (= 9.8), we need to change left-handed system to right-handed by reversing axis x, y, z
        q = torch.tensor(q)  # rotation is not changed
        a = -torch.tensor(a) / 1000 * self.g                         # acceleration is reversed
        # a = R.bmm(a.unsqueeze(-1)).squeeze(-1) + torch.tensor([0., 0., self.g])   # calculate global free acceleration
        return self.t, q, a

# if __name__ == '__main__':
#
#     imu_set = NoitomIMUSet(udp_port=8080)
#     clock = Clock()
#
#     while True:
#         clock.tick(100)
#         tframe, RIS, aI = imu_set.get()
#         quat1 = [0] * 4
#         quat1[:3] = RIS[0][1:]
#         quat1[3] = RIS[0][0]
#         quat2 = [0] * 4
#         quat2[:3] = RIS[1][1:]
#         quat2[3] = RIS[1][0]
#         rot1_ = R.from_quat(quat1)
#         rot2_ = R.from_quat(quat2)
#         # print('\r',(rot1*rot2.inv()).as_euler('zxy',degrees=True),end='')
#         # print('\rfps: ', clock.get_fps(), tframe, '|', RIS, aI, end='')
#
#         quat1 = torch.FloatTensor(RIS[0])
#         quat2 = torch.FloatTensor(RIS[1])
#         rot1 = quaternion_to_rotation_matrix(quat1)
#         rot2 = quaternion_to_rotation_matrix(quat2)
#         rot = rot1.matmul(rot2.transpose(1, 2))
#         euler_ang = rotation_matrix_to_euler_angle(rot).view(-1) * 180 / np.pi
#         euler_ang = np.array(euler_ang).tolist()
#         print('\r', euler_ang, (rot1_*rot2_.inv()).as_euler('zxy',degrees=True), end='')


