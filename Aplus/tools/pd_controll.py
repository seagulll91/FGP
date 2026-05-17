from pygame.time import Clock
import time
# class FpsController:
#     def __init__(self, set_fps, kp=0.05, td=0.1):
#         self.target_time_gap = 1 / set_fps
#         self.used_time_gap = 1 / set_fps
#         self.kp = kp
#         self.td = td
#         self.d_err = None
#         self.past_time_gap = None
#         self.clock = Clock()
#         self.past_d_err = 0
#
#     def sleep(self):
#         """
#         在while循环中调用这个函数即可, 会自动控制暂停间隔让fps向设定值靠近
#         """
#         time.sleep(self.used_time_gap)
#         self.clock.tick()
#         current_fps = max(self.clock.get_fps(), 1)
#         if self.past_time_gap is None:
#             self.past_time_gap = 1 / current_fps
#         else:
#             # 通过当前time gap与target的差异更新后续使用的time gap
#             current_time_gap = 1 / current_fps
#             d_err = self.target_time_gap - current_time_gap
#             dd_err = d_err - self.past_d_err
#             u = self.kp*(d_err + self.td*dd_err)
#
#             self.past_time_gap = current_time_gap
#             self.past_d_err = d_err
#             # 最终使用的time gap一定大于0且小于target_time_gap
#             self.used_time_gap = min(max(self.used_time_gap + u, 0), self.target_time_gap)
#
#     def get_fps(self):
#         return self.clock.get_fps()

class FpsController:
    def __init__(self, set_fps, kp=0.025, td=0.01):
        self.target_time_gap = 1 / set_fps
        self.used_time_gap = 1 / set_fps
        self.kp = kp
        self.td = td
        self.d_err = None
        self.past_time_gap = None
        self.clock = Clock()
        self.past_d_err = 0
        self.time_past = 0
        self.time_now = 0

    def sleep(self):
        """
        在while循环中调用这个函数即可, 会自动控制暂停间隔让fps向设定值靠近
        """
        time.sleep(self.used_time_gap)
        self.clock.tick()
        # 第一次运行
        if self.past_time_gap is None:
            self.init()
        else:
            # 通过当前time gap与target的差异更新后续使用的time gap
            self.time_past = self.time_now
            self.time_now = time.time()
            current_time_gap = self.time_now - self.time_past
            d_err = self.target_time_gap - current_time_gap
            dd_err = d_err - self.past_d_err
            u = self.kp*(d_err + self.td*dd_err)

            self.past_time_gap = current_time_gap
            self.past_d_err = d_err
            # 最终使用的time gap一定大于0且小于target_time_gap
            self.used_time_gap = min(max(self.used_time_gap + u, 0), self.target_time_gap)

    def init(self):
        self.past_time_gap = self.used_time_gap
        self.time_now = time.time()
        self.time_past = self.time_now - self.used_time_gap
    def get_fps(self):
        return self.clock.get_fps()

