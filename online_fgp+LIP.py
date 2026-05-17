import sys

sys.coinit_flags = 0  # 0 means MTA
import argparse
from collections import deque
from threading import Thread
# import onnxruntime as rt
from Socket.UDP import *
from Utils import config

from hardware_boot import receive_085_COM
from hardware_boot.receive_085_COM import *
from Aplus.tools.pd_controll import FpsController
from Aplus.tools.dataset_maker import get_time_stamp
from datetime import datetime, timedelta
import os
import time
import articulate as art
from Aplus.models import EasyLSTM

# _LIPN_ROOT = os.path.abspath(
#     os.path.join(os.path.dirname(__file__), "..", "baseline", "LIPN")
# )
# if os.path.isdir(_LIPN_ROOT) and _LIPN_ROOT not in sys.path:
#     sys.path.insert(0, _LIPN_ROOT)
from LIP_model import BiPoser  # baseline/LIPN
import numpy as np
import torch
import onnxruntime as rt

# from denoise_model import IMUDenoiseFlowDiT,MaskedIMUAutoEncode,
from canonicalization_model import IMUDenoiseFlowDiT,IMUDenoiseFlowDiT2
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
global BUFFER_SIZE

global CALIBRATION_DONE
global my_server
global i_session
global imu_order_clothes
global imu_order_pant
import logging

logging.basicConfig(level=logging.INFO)

CALIBRATION_DONE = False
BUFFER_SIZE = 500

clock = Clock()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    # Ampere+：矩阵乘用 TF32，常能明显加速 Transformer 类模型
    torch.set_float32_matmul_precision("high")

RUNNING = True

DENOISE_ENABLED = True
DENOISE_WINDOW_SIZE = 30
# 以下三项仅在 DENOISE_EVERY_FRAME=False 时使用（旧版 chunk/stride 调度）
DENOISE_CHUNK_SIZE = 4
DENOISE_STRIDE = 2
DENOISE_STEPS = 1
DENOISE_INTERVAL = 1
DENOISE_EVERY_FRAME = True
# PyTorch 2.x：首次推理会编译较慢，之后通常更快（与部分动态 shape 不兼容时可关）
DENOISE_TORCH_COMPILE = False
DENOISE_OUTPUT_MODE = "causal_ema"
DENOISE_EMA_ALPHA = 0.08
DENOISE_ACC_SCALE = 30.0
# DENOISE_CKPT = r"C:\Users\15482\Desktop\GlobalPose\data\denoised_ckpt\flowdit_ep200_ae30.pth"
DENOISE_CKPT = 'checkpoints/canonicalization/U100hL7.07h_residual.pth'

# LIP (BiPoser) pose model — matches LIPN 10-IMU denoised checkpoint
LIP_IMU_NUM = 10
LIP_ACC_SCALE = 30.0
LIP_NUM_JOINTS = 24
LIP_CHECKPOINT_PATH = os.path.join(
    "checkpoints", "LIP", "LIP_10_real_denoised_7.07_10imu_10.pth"
)
SMPL_MODEL_FILE = "models/SMPL_male.pkl"

UNITY_DOUBLE_VIEW = True
PERF_PRINT_ENABLED = True

# 是否同步运行可视化
ShowRealtimePose = True

# 选择是否从离线 txt 文件读取数据（参考 data_record 保存的格式）
OFFLINE_MODE = True

OFFLINE_PT = False  # 是否使用离线的 .pt 文件数据进行推理

# 离线数据文件路径，默认使用本脚本创建的 sensor_data.txt
# OFFLINE_DIR = 'data/W_20260115_gxp/142222_gxp_skipRope'
OFFLINE_DIR = r'C:\Users\15482\Desktop\ClothData\10IMU\GreenOutfit\G_20260103_zkb\151719_zkb_badminton'
# 离线读取 .pt 文件开关与路径（已移除，当前仅支持 txt）
user_name = 'gxx'
record_id = 1
motion = 'walk'

data_save_root = f'./data_record/20260106_{user_name}'
os.makedirs(data_save_root,exist_ok= True)
# 创建文件夹
output_dir = os.path.join(data_save_root, f'{record_id}_{user_name}')
os.makedirs(output_dir,exist_ok= True)
txt_file_sensor = open(os.path.join(output_dir, "sensor_data.txt"), 'w', encoding='ascii')

def unity_args_setting():
    parser = argparse.ArgumentParser(description='Visual System')

    parser.add_argument('--mode', default='single', type=str, choices=['single', 'twins','single_global']
                        , help='single for a model ,twins for two models at the same time')

    parser.add_argument('--skeleton', default='smpl', type=str, choices=['smpl', 'h36m']
                        , help='The type of skeleton used by your data')

    parser.add_argument('--rotation_type', default='AXIS_ANGLE', type=str,
                        choices=['AXIS_ANGLE', 'DCM', 'QUATERNION', 'R6D', 'EULER_ANGLE']
                        , help='Rotation representations. Quaternions are in wxyz. Euler angles are in local XYZ.')

    parser.add_argument('--part', default='body', type=str
                        ,
                        choices=['body', 'upper_body', 'lower_body', 'head', 'spine', 'left_hand', 'right_hand',
                                 'left_leg',
                                 'right_leg', 'hands']
                        , help='You can choose the part of visualization')

    parser.add_argument('--fps', default=60, type=int
                        , help='The frame rate at which the animation is played')

    args = parser.parse_args()

    return args


args = unity_args_setting()


def data_receive(data_getter, data_buffer, fps=40):
    from Aplus.tools.pd_controll import FpsController
    fc = FpsController(set_fps=fps)
    try:
        while RUNNING:
            fc.sleep()
            data = data_getter()
            data_buffer.extend(data)
    except Exception as e:
        print(str(e))
def compute_gyro_from_R(
    R_prev: torch.Tensor,
    R_cur: torch.Tensor,
    dt: float,
    angle_eps: float = 1e-4,
):
    """
    R_prev, R_cur: [N, 3, 3]
    return w: [N, 3] in rad/s
    """
    R_delta = R_prev.transpose(-1, -2).matmul(R_cur)
    aa = art.math.rotation_matrix_to_axis_angle(R_delta)
    if angle_eps > 0:
        aa = torch.where(aa.abs() < angle_eps, torch.zeros_like(aa), aa)
    w = aa / max(dt, 1e-6)
    w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    # if max(w.view(-1).cpu().numpy())>100:
    #     print(11)
    return w.clamp(-60.0, 60.0)


def project_to_so3(R: torch.Tensor) -> torch.Tensor:
    """
    Project a batch of 3x3 matrices to the nearest SO(3) via SVD.
    """
    orig_shape = R.shape
    R = R.reshape(-1, 3, 3).to(torch.float64)
    U, _, Vh = torch.linalg.svd(R)
    R_proj = U @ Vh
    det = torch.det(R_proj)
    mask = det < 0
    if mask.any():
        U_fix = U.clone()
        U_fix[mask, :, -1] *= -1
        R_proj = U_fix @ Vh
    return R_proj.to(torch.float32).reshape(orig_shape)


def load_flowdit(flow_ckpt_path: str) -> IMUDenoiseFlowDiT:
    model = IMUDenoiseFlowDiT2(
            pretrained_ae=None,
            cin_state=9,
            cin_cond=9,
            cout=9,
            depth=6,
            nhead=4,
            ffn=512,
            dropout=0.1,
            max_T=512,
            t_embed_dim=256,
            enc_d_model=384,
            enc_layers=4,
            enc_nhead=8
        ).to(device)

    state = torch.load(flow_ckpt_path, map_location="cpu")
    state = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


@torch.no_grad()
def flow_sampler_euler(
    flow_model,
    x_noisy_9: torch.Tensor,
    vis_ids_full: torch.Tensor,
    mem_pad_mask: torch.Tensor = None,
    steps: int = 1,
    use_fp16: bool = False,
):
    """
    x_noisy_9: [1, L, K, 9] = [acc3, r6d6]
    return: [1, L, K, 9]
    """
    x = x_noisy_9
    x_cond = x_noisy_9
    B = x.shape[0]

    for i in range(steps):
        t = torch.full((B,), i / max(steps, 1), device=x.device, dtype=x.dtype)
        dt = 1.0 / max(steps, 1)

        if use_fp16 and x.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                v = flow_model(
                    x_t_9=x,
                    x_n_9=x_cond,
                    vis_ids=vis_ids_full,
                    mem_pad_mask=mem_pad_mask,
                    # t=t,
                )
        else:
            v = flow_model(
                x_t_9=x,
                x_n_9=x_cond,
                vis_ids=vis_ids_full,
                mem_pad_mask=mem_pad_mask,
                # t=t,
            )

        x = x + dt * v
        # x  = v

    return x


class RealtimeIMUDenoiser:
    def __init__(
        self,
        ckpt_path: str,
        window_size: int = 30,
        chunk_size: int = 10,
        stride: int = 5,
        steps: int = 1,
        interval: int = 1,
        output_mode: str = "causal_ema",
        ema_alpha: float = 0.7,
        acc_scale: float = 30.0,
        use_fp16: bool = False,
        every_frame: bool = True,
        torch_compile: bool = False,
    ):
        self.ckpt_path = ckpt_path
        self.window_size = int(window_size)
        self.chunk_size = max(1, int(chunk_size))
        self.stride = max(1, int(stride))
        self.steps = int(steps)
        self.interval = max(1, int(interval))
        self.every_frame = bool(every_frame)
        self.output_mode = str(output_mode)
        self.ema_alpha = float(ema_alpha)
        self.acc_scale = float(acc_scale)
        self.use_fp16 = bool(use_fp16)
        self.K_out = 10
        self.model = load_flowdit(ckpt_path)
        if torch_compile and hasattr(torch, "compile") and device.type == "cuda":
            try:
                # default：编译较快；需要极限吞吐可改为 mode="max-autotune"（编译很久）
                self.model = torch.compile(self.model, mode="default")
                print("[IMU Denoise] torch.compile enabled (first frames may be slow)")
            except Exception as e:
                print(f"[IMU Denoise] torch.compile skipped: {e}")
        ws = self.window_size
        k = self.K_out
        self._x_noisy = torch.zeros(1, ws, k, 9, device=device, dtype=torch.float32)
        self._mem_pad_mask = torch.zeros(1, ws * k, dtype=torch.bool, device=device)
        self.vis_ids_full = torch.arange(self.K_out, dtype=torch.long, device=device).unsqueeze(0)
        self.reset()

    def reset(self):
        self.acc_buffer = deque(maxlen=self.window_size)
        self.rot_buffer = deque(maxlen=self.window_size)
        self.frame_count = 0
        self.last_chunk_frame = 0
        self.prev_out_acc = None
        self.prev_out_rot = None
        self.pending_outputs = deque()

    def _apply_causal_smoothing(self, acc: torch.Tensor, rot: torch.Tensor):
        if self.output_mode != "causal_ema" or self.prev_out_acc is None or self.prev_out_rot is None:
            self.prev_out_acc = acc
            self.prev_out_rot = rot
            return acc, rot

        alpha = min(max(self.ema_alpha, 0.0), 1.0)
        smoothed_acc = alpha * self.prev_out_acc + (1.0 - alpha) * acc
        smoothed_rot = alpha * self.prev_out_rot + (1.0 - alpha) * rot
        smoothed_rot = art.math.normalize_rotation_matrix(smoothed_rot)

        self.prev_out_acc = smoothed_acc
        self.prev_out_rot = smoothed_rot
        return smoothed_acc, smoothed_rot

    def _fill_model_input(self, acc_seq: torch.Tensor, rot_seq: torch.Tensor):
        """In-place pack [W,K,9] into self._x_noisy (avoids cat + unsqueeze alloc per step)."""
        ws = self.window_size
        r6d = art.math.rotation_matrix_to_r6d(rot_seq.reshape(-1, 3, 3)).reshape(ws, self.K_out, 6)
        self._x_noisy[0, :, :, :3].copy_(acc_seq / self.acc_scale)
        self._x_noisy[0, :, :, 3:9].copy_(r6d)

    @torch.inference_mode()
    def step(self, acc: torch.Tensor, rot: torch.Tensor):
        """
        acc: [10, 3]
        rot: [10, 3, 3]
        return denoised acc/rot for the latest frame
        """
        acc = acc.detach().to(device=device, dtype=torch.float32)
        rot = art.math.normalize_rotation_matrix(rot.detach().to(device=device, dtype=torch.float32))

        self.acc_buffer.append(acc)
        self.rot_buffer.append(rot)
        self.frame_count += 1

        if not self.every_frame:
            if self.pending_outputs:
                den_acc, den_rot = self.pending_outputs.popleft()
                return self._apply_causal_smoothing(den_acc, den_rot)

        if len(self.acc_buffer) < self.window_size:
            return acc, rot

        if not self.every_frame:
            if self.interval > 1 and self.frame_count % self.interval != 0:
                return acc, rot
            if self.frame_count - self.last_chunk_frame < self.stride:
                return acc, rot

        acc_seq = torch.stack(list(self.acc_buffer), dim=0)
        rot_seq = torch.stack(list(self.rot_buffer), dim=0)
        self._fill_model_input(acc_seq, rot_seq)

        clean_9 = flow_sampler_euler(
            flow_model=self.model,
            x_noisy_9=self._x_noisy,
            vis_ids_full=self.vis_ids_full,
            mem_pad_mask=self._mem_pad_mask,
            steps=self.steps,
            use_fp16=self.use_fp16,
        )[0]

        if self.every_frame:
            last = clean_9[-1]
            den_acc = last[:, :3] * self.acc_scale
            den_rot = art.math.r6d_to_rotation_matrix(last[:, 3:9].reshape(-1, 6)).reshape(
                self.K_out, 3, 3
            )
            return self._apply_causal_smoothing(den_acc, den_rot)

        tail_len = min(self.chunk_size, clean_9.shape[0])
        clean_tail = clean_9[-tail_len:]
        den_acc_tail = clean_tail[:, :, :3] * self.acc_scale
        den_rot_tail = art.math.r6d_to_rotation_matrix(clean_tail[:, :, 3:9].reshape(-1, 6)).reshape(
            tail_len, self.K_out, 3, 3
        )

        self.pending_outputs = deque(
            (den_acc_tail[i], den_rot_tail[i])
            for i in range(max(0, tail_len - self.stride), tail_len)
        )
        self.last_chunk_frame = self.frame_count

        if self.pending_outputs:
            den_acc, den_rot = self.pending_outputs.popleft()
            return self._apply_causal_smoothing(den_acc, den_rot)

        return acc, rot


def pose_mat_to_repr(pose_mat: torch.Tensor, rotation_type: str):
    """
    pose_mat: [24,3,3] torch
    return: 1D numpy array, compatible with preprocess()
    """
    pose_mat = pose_mat.float()

    if rotation_type == "DCM":
        return pose_mat.reshape(-1).cpu().numpy()              # 24*9
    elif rotation_type == "AXIS_ANGLE":
        aa = art.math.rotation_matrix_to_axis_angle(pose_mat)  # [24,3]
        return aa.reshape(-1).cpu()                # 24*3
    elif rotation_type == "QUATERNION":
        q = art.math.rotation_matrix_to_quaternion(pose_mat)   # [24,4] (wxyz)
        return q.reshape(-1).cpu().numpy()                     # 24*4
    elif rotation_type == "R6D":
        r6d = art.math.rotation_matrix_to_r6d(pose_mat)        # [24,6]
        return r6d.reshape(-1).cpu().numpy()                   # 24*6
    else:
        # 默认直接用 DCM（最安全）
        return pose_mat.reshape(-1).cpu().numpy()


def send_pose_to_unity(server, data, duplicate_second_actor: bool = False):
    if not duplicate_second_actor:
        server.update_data(data)
        return

    pose = preprocess(data, server.rotation_type)
    tran = server.get_trans()
    s = (
        ','.join(['%g' % v for v in pose]) + '#' +
        ','.join(['%g' % v for v in tran]) + '$' +
        ','.join(['%g' % v for v in pose]) + '#' +
        ','.join(['%g' % v for v in tran]) + '@'
    )
    server.conn.send(s.encode('utf8'))

def dynamic_calibration(t_gap=1):
    while True:
        time.sleep(1)
        if CALIBRATION_DONE:
            my_server.auto_calibrate()
            # break
        else:
            continue
w_ema = None
EMA = 0.85
USE_DYNAMIC_GYRO_DT = False
GYRO_ANGLE_EPS = 1e-4
imu_denoiser = None
fps_print_count = 0
fps_print_last_time = None
perf_denoise_ms_accum = 0.0
perf_pose_ms_accum = 0.0
FOOT_LOCK_ENABLED = False
FOOT_LOCK_BLEND = 0.75
FOOT_LOCK_VEL_THRESH = 0.045
FOOT_LOCK_HEIGHT_THRESH = 0.06
FOOT_GROUND_MARGIN = 0.01
foot_lock_state = None
lip_pose_predictor = None
smpl_body_model = None


def pack_lip_imu_frame(acc: torch.Tensor, rot: torch.Tensor, imu_num=LIP_IMU_NUM, acc_scale=LIP_ACC_SCALE):
    """Match LIP offline eval: root-relative acc/rot, acc/scale, rot as r6d."""
    acc = acc.float().clamp(-60.0, 60.0)
    rot = art.math.normalize_rotation_matrix(rot.float())
    acc_rooted = torch.cat([acc[: imu_num - 1] - acc[imu_num - 1], acc[imu_num - 1 :]], dim=0)
    acc_rooted = acc_rooted @ rot[imu_num - 1]
    rot_rooted = torch.cat(
        [torch.matmul(rot[imu_num - 1].transpose(0, 1), rot[: imu_num - 1]), rot[imu_num - 1 :]],
        dim=0,
    )
    acc_feat = (acc_rooted / acc_scale).reshape(-1)
    rot_r6d = art.math.rotation_matrix_to_r6d(rot_rooted.reshape(-1, 3, 3)).reshape(imu_num, 6)
    return torch.cat([acc_feat, rot_r6d.reshape(-1)], dim=-1)


class RealtimeLIPPredictor:
    def __init__(self, model: BiPoser, imu_num=LIP_IMU_NUM):
        self.model = model
        self.imu_num = imu_num
        self.input_dim = imu_num * 9
        self.reset()

    def reset(self):
        self.h_s1 = None
        self.c_s1 = None
        self.h_s2 = None
        self.c_s2 = None

    def _init_lstm_state(self, lstm_module):
        h_0 = lstm_module.h_0.repeat(1, 1, 1)
        c_0 = lstm_module.c_0.repeat(1, 1, 1)
        return h_0, c_0

    @torch.inference_mode()
    def step(self, acc: torch.Tensor, rot: torch.Tensor):
        x = pack_lip_imu_frame(acc, rot, imu_num=self.imu_num).view(1, 1, -1).to(device)
        if self.h_s1 is None:
            h_0, c_0 = self._init_lstm_state(self.model.net_s1)
            out_1, self.h_s1, self.c_s1 = self.model.net_s1(x, h_0, c_0)
            h_0, c_0 = self._init_lstm_state(self.model.net_s2)
            out_2, self.h_s2, self.c_s2 = self.model.net_s2(torch.cat([x, out_1], dim=-1), h_0, c_0)
        else:
            out_1, self.h_s1, self.c_s1 = self.model.net_s1(x, self.h_s1, self.c_s1)
            out_2, self.h_s2, self.c_s2 = self.model.net_s2(
                torch.cat([x, out_1], dim=-1), self.h_s2, self.c_s2
            )

        pred_r6d = out_2.squeeze(0).squeeze(0).view(LIP_NUM_JOINTS, 6)
        pose_mat = art.math.r6d_to_rotation_matrix(pred_r6d)
        return pose_mat, torch.zeros(3, device=device)


def build_lip_model(imu_num=LIP_IMU_NUM):
    model_s1 = EasyLSTM(
        n_input=imu_num * 9,
        n_hidden=256,
        n_output=LIP_NUM_JOINTS * 3,
        n_lstm_layer=2,
        bidirectional=False,
        output_type="seq",
        dropout=0.2,
    ).to(device)
    model_s2 = EasyLSTM(
        n_input=imu_num * 9 + LIP_NUM_JOINTS * 3,
        n_hidden=256,
        n_output=LIP_NUM_JOINTS * 6,
        n_lstm_layer=2,
        bidirectional=False,
        output_type="seq",
        dropout=0.2,
    ).to(device)
    return BiPoser(net_s1=model_s1, net_s2=model_s2, export_mode=True).to(device)


def reset_foot_lock_state():
    global foot_lock_state
    foot_lock_state = {
        "prev_global_feet": None,
        "locked_feet": None,
        "initialized": False,
    }


@torch.no_grad()
def stabilize_root_translation(pose_mat: torch.Tensor, tran: torch.Tensor):
    global foot_lock_state

    if foot_lock_state is None:
        reset_foot_lock_state()

    if smpl_body_model is None:
        return tran
    joint_pos = smpl_body_model.forward_kinematics(
        pose_mat.unsqueeze(0), tran=tran.unsqueeze(0)
    )[1][0].detach().cpu()
    foot_idx = torch.tensor([10, 11], dtype=torch.long)
    local_feet = joint_pos[foot_idx]
    tran = tran.detach().cpu().clone()
    global_feet = local_feet + tran.view(1, 3)

    # Keep the supporting foot close to the ground to reduce hovering.
    min_foot_y = float(global_feet[:, 1].min().item())
    if min_foot_y < FOOT_GROUND_MARGIN:
        tran[1] += FOOT_GROUND_MARGIN - min_foot_y
        global_feet[:, 1] += FOOT_GROUND_MARGIN - min_foot_y

    state = foot_lock_state
    if not state["initialized"]:
        state["prev_global_feet"] = global_feet.clone()
        state["locked_feet"] = global_feet.clone()
        state["initialized"] = True
        return tran

    prev_global_feet = state["prev_global_feet"]
    locked_feet = state["locked_feet"]
    foot_speed = torch.norm(global_feet - prev_global_feet, dim=1)
    foot_height = global_feet[:, 1]
    contact_mask = (foot_height < FOOT_LOCK_HEIGHT_THRESH) & (foot_speed < FOOT_LOCK_VEL_THRESH)

    if contact_mask.any():
        drift = global_feet[contact_mask][:, [0, 2]] - locked_feet[contact_mask][:, [0, 2]]
        drift_mean = drift.mean(dim=0)
        tran[[0, 2]] -= FOOT_LOCK_BLEND * drift_mean
        global_feet = local_feet + tran.view(1, 3)
        global_feet[:, 1] = torch.maximum(global_feet[:, 1], torch.full_like(global_feet[:, 1], FOOT_GROUND_MARGIN))

    locked_feet[contact_mask] = global_feet[contact_mask]
    locked_feet[~contact_mask] = global_feet[~contact_mask]
    state["prev_global_feet"] = global_feet.clone()
    state["locked_feet"] = locked_feet
    return tran


def data_transmit(fps=30):
    global CALIBRATION_DONE
    global i_session
    global imu_order_clothes
    global imu_order_pant
    global R_prev_cache, last_time_cache
    global w_ema, imu_denoiser
    global fps_print_count, fps_print_last_time
    global perf_denoise_ms_accum, perf_pose_ms_accum, lip_pose_predictor
    fc = FpsController(set_fps=fps)
    # 如果使用离线文件，先读取全部行，否则等待传感器数据
    if OFFLINE_MODE:
        offline_path = os.path.join(OFFLINE_DIR, "sensor_data.txt")
        print(f"使用离线 txt 数据文件: {offline_path}")
        if not os.path.exists(offline_path):
            raise FileNotFoundError(f"离线数据文件不存在: {offline_path}")
        with open(offline_path, 'r', encoding='ascii') as f:
            offline_lines = f.readlines()
        if len(offline_lines) < 1:
            raise ValueError("离线数据文件为空")
        line_idx = 0
        print('已加载离线数据, 接下来进行校准数据采集, 请输入任意字符开始')
    else:
        # 等待数据接收
        while RUNNING:
            time.sleep(2)
            # print('已接收衣服数据app:', len(receive_bleak_085.data_up_buffer))
            # print('已接收裤子数据app:', len(receive_bleak_085.data_down_buffer))
            if len(receive_085_COM.data_up_buffer) < 2:
                print('\r', '等待接收衣服数据...', end='')
                continue
            if len(receive_085_COM.data_down_buffer) < 2:
                print('\r', '等待接收裤子数据...', end='')
                continue
            else:
                break
        input('数据接收成功, 接下来进行校准数据采集, 请输入任意字符开始')
    # 校准前倒计时3秒
    # for i in range(3):
    #     time.sleep(1)
    #     print(3 - i)
    frame_idx_pt = 0
    tran0 = None
    while RUNNING:
        # time.sleep(1 / fps)
        fc.sleep()
        

        if CALIBRATION_DONE == False:
            # input('接下来请按任意键开始姿态校准')
            print('请保持站立')
            for i in range(5):
                time.sleep(1)
                print(3 - i)

            print('校准数据采集完成!上传中...')
            # 读取数据 重新排序 拼接（仅支持 txt 离线）
            if OFFLINE_MODE:
                # 离线 txt 每行格式: timestamp, accs(N*3), oris(N*9)
                # 取最后60帧平均作为t-pose
                recent_lines = offline_lines[:60]
                vals = []
                for ln in recent_lines:
                    parts = ln.strip().split(',')
                    if len(parts) < 2:
                        continue
                    nums = [float(x) for x in parts[1:] if x != '']
                    vals.append(nums)
                arr = np.array(vals)  # [M, N*12]
                N = 11
                acc_all = arr[:, : N * 3].reshape(-1, N, 3)[:, [0,1,2,3,4,6,7,8,9,10],:]  # [M,N,3]
                oris_all = arr[:, N * 3: N * 3 + N * 9].reshape(-1, N, 3, 3)[:, [0,1,2,3,4,6,7,8,9,10],:]  # [M,N,3,3]
                tpose_acc = torch.FloatTensor(acc_all.mean(axis=0))  # [N,3]
                tpose_oris = torch.FloatTensor(oris_all.mean(axis=0))  # [N,3,3]
                tpose_acc = tpose_acc.reshape(-1)
                tpose_oris = tpose_oris.reshape(-1)
                tpose_data = np.array(torch.cat([tpose_acc, tpose_oris], dim=0)).tolist()
            else:
                tpose_data_clothes = np.array(receive_085_COM.data_up_buffer)[-60:].reshape(-1, 6, 7)[:, imu_order_clothes,
                                     :].mean(axis=0)
                tpose_data_pant = np.array(receive_085_COM.data_down_buffer)[-60:].reshape(-1, 5, 7)[:, imu_order_pant,
                                  :].mean(axis=0)
                tpose_data = np.concatenate([tpose_data_clothes, tpose_data_pant], axis=0)
                # print(tpose_data.shape)
                tpose_acc = torch.FloatTensor(tpose_data[:, 0:3])
                tpose_q = torch.FloatTensor(tpose_data[:, 3:7])
                tpose_oris = quaternion_to_rotation_matrix(tpose_q)

                tpose_acc = tpose_acc.reshape(-1)
                tpose_oris = tpose_oris.reshape(-1)

                tpose_data = np.array(torch.cat([tpose_acc, tpose_oris], dim=0)).tolist()

            # ------------添加calibration数据传输代码-------------
            print('校准数据设置中')
            # my_server.set_calibrate_data(tpose_data)
            print(len(tpose_data))
            my_server.set_calibrate_data(tpose_data)
            torch.save(
                {
                    "tpose_acc": tpose_acc.cpu(),      # [N,3]
                    "tpose_oris": tpose_oris.cpu(),    # [N,3,3]
                },
                os.path.join(output_dir, "tpose_data.pt"),
            )
            torch.save(my_server.smpl2imu, os.path.join(output_dir, "smpl2imu.pt"))
            torch.save(my_server.device2bone, os.path.join(output_dir, "device2bone.pt"))
            print('保存calibration参数')

            # -------------------------------------------------
            CALIBRATION_DONE = True
            record_begin_t = time.time()
            if lip_pose_predictor is not None:
                lip_pose_predictor.reset()
            reset_foot_lock_state()
            R_prev_cache = None
            last_time_cache = None
            w_ema = None
            fps_print_count = 0
            fps_print_last_time = time.time()
            perf_denoise_ms_accum = 0.0
            perf_pose_ms_accum = 0.0
            if imu_denoiser is not None:
                imu_denoiser.reset()
            print("[LIP] LSTM state reset.")
            # 开始上传实时数据
        else:
            if OFFLINE_MODE:
                # 从离线 txt 读取每行数据
                if line_idx >= len(offline_lines):
                    print('\n已到达离线数据末尾，停止运行')
                    return
                parts = offline_lines[line_idx].strip().split(',')
                nums = [float(x) for x in parts[1:] if x != '']
                data = nums  # accs + oris flattened
                line_idx += 1

            else:
                now_t = time.time()
                data_clothes = receive_085_COM.data_up_buffer[-1].reshape(6, 7)[imu_order_clothes]
                data_pant = receive_085_COM.data_down_buffer[-1].reshape(5, 7)[imu_order_pant]
                data = np.concatenate([data_clothes, data_pant], axis=0)
                # for i, row in enumerate(data):
                #     print(f'Sensor {i + 1}: ' + ' | '.join(f'{x:.4f}' for x in row))
                accs = torch.FloatTensor(data[:, 0:3])
                q = torch.FloatTensor(data[:, 3:7])
                oris = quaternion_to_rotation_matrix(q)

                # print(oris[:4])
                # print('++++++++++++++')
                # print(q)
                accs = accs.view(-1)
                oris = oris.view(-1)

                data = np.array(torch.cat([accs, oris], dim=0)).tolist()
                last_time_cache = now_t

            latest_time = get_time_stamp()
            sensor_data = latest_time + ','
            for d in data:
                sensor_data += str(d) + ','
            sensor_data = sensor_data[:-1]
            sensor_data += '\n'

            # txt_file_sensor.write(sensor_data)
            # txt_file_sensor.flush()

            # -------------------------
            # LIP 推理缓存
            # -------------------------
            # R_prev_cache = None
            # last_time_cache = None

            # ------------添加数据实时传输代码-------------
            if ShowRealtimePose:

                # ===========================
                # LIP realtime inference
                # ===========================
                if "R_prev_cache" not in globals():
                    R_prev_cache = None
                    last_time_cache = None



                # 对输入数据执行校准（离线 txt 与在线均需校准）
                
                if OFFLINE_PT:
                    index = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
                    index = [0, 1, 2, 3, 4, 5,6,7,8,9]
                    # 流式读取 acc.pt 和 rot.pt，每次只取一帧
                    if "acc_pt_data" not in globals():
                        acc_pt_data = torch.load(os.path.join(OFFLINE_DIR, "acc_denoised_10.pt"))[:, index]
                        rot_pt_data = torch.load(os.path.join(OFFLINE_DIR, "rot_denoised_10.pt"))[:, index]
                        
                    N_frame = acc_pt_data.shape[0]
                    if frame_idx_pt >= N_frame:
                        print("\n已到达离线.pt数据末尾，停止运行")
                        return
                    acc = acc_pt_data[frame_idx_pt]
                    rot = rot_pt_data[frame_idx_pt]
                    frame_idx_pt += 1
                    print(f'使用离线 pt 数据帧 {frame_idx_pt}/{N_frame}', end='\r')

                    acc_pose = acc.to(device=device, dtype=torch.float32).view(10, 3)
                    rot_pose = art.math.normalize_rotation_matrix(
                        rot.to(device=device, dtype=torch.float32).view(10, 3, 3)
                    )
                else:
                    # 你的 reorder 逻辑保留
                    if len(data)==132:
                        data0 = np.concatenate([data[:5*3], data[6*3:11*3],
                                            data[11*3:11*3+5*9], data[11*3+6*9:]], axis=0)
                    else:
                        data0 = data
                    data = my_server.calibrate(data0, acc_scale = 30)
                    data_auto = my_server.calibrate_auto(data0, acc_scale = 30)

                    acc = torch.tensor(data[:10*3], dtype=torch.float32, device=device).view(10, 3)*30
                    rot = torch.tensor(data[10*3:], dtype=torch.float32, device=device).view(10, 3, 3)
                    # rot = art.math.normalize_rotation_matrix(rot)
                    # if max(acc.view(-1).cpu().numpy())>50:
                    #     print(11)
                    if imu_denoiser is not None:
                        denoise_t0 = time.perf_counter()
                        acc, rot = imu_denoiser.step(acc, rot)
                        perf_denoise_ms_accum += (time.perf_counter() - denoise_t0) * 1000.0
                        if len(imu_denoiser.acc_buffer) < imu_denoiser.window_size:
                            continue

                    acc_pose = acc.view(10, 3)
                    rot_pose = rot.view(10, 3, 3)

                if lip_pose_predictor is None:
                    continue

                pose_t0 = time.perf_counter()
                pose_mat, tran = lip_pose_predictor.step(acc_pose, rot_pose)
                pose_mat = pose_mat.detach().cpu()
                tran = tran.detach().cpu()
                perf_pose_ms_accum += (time.perf_counter() - pose_t0) * 1000.0
                if tran0 is None:
                    tran0 = tran
                # pose_mat = pose_mat.detach().cpu()
                # pose_aa = art.math.rotation_matrix_to_axis_angle(pose_mat)
                # ✅ 把 pose_mat 转成 preprocess能吃的格式
                pose_repr = pose_mat_to_repr(pose_mat, my_server.rotation_type)
                pose_repr[45:48] =torch.tensor([0,0,0])
                pose_repr[36:39] =torch.tensor([0,0,0])
                
 
                # my_server.operator(data_auto)
                # data_feed = my_server.to_predict_data()
                # result = i_session.run(output_names=None, input_feed=data_feed)
                # pose_baselne = my_server.predict_result(result)

                # ✅ 用你原来的 update_data，保持 UDP/Unity 不变
                try:
                    if hasattr(my_server, 'conn') and my_server.conn is not None:
                        if FOOT_LOCK_ENABLED:
                            tran = stabilize_root_translation(pose_mat, tran)
                        if args.mode == 'single_global':
                            # tran[1]+=my_server.root_height
                            tran[2] *= 0.7
                            tran[1]+= 1
                            my_server.update_data_global(pose_repr,tran)
                        elif args.mode == 'single_local':
                            tran = torch.tensor([0,0,0],dtype=pose_repr.dtype)
                            my_server.update_data_twins(pose_repr, tran, pose_repr,tran)
                        # elif args.mode == 'twins':
                        #     tran = torch.tensor([0,0,0],dtype=pose_repr.dtype)
                        #     my_server.update_data_twins(pose_repr, tran, pose_baselne ,tran)
                            
                        fps_print_count += 1
                        now_fps_time = time.time()
                        if fps_print_last_time is None:
                            fps_print_last_time = now_fps_time
                        elapsed = now_fps_time - fps_print_last_time
                        if PERF_PRINT_ENABLED and elapsed >= 1.0:
                            fps_val = fps_print_count / max(elapsed, 1e-6)
                            denoise_ms = perf_denoise_ms_accum / max(fps_print_count, 1)
                            pose_ms = perf_pose_ms_accum / max(fps_print_count, 1)
                            print(
                                f"\rFPS: {fps_val:.2f} | denoise: {denoise_ms:.2f} ms | pose: {pose_ms:.2f} ms",
                                end=''
                            )
                            fps_print_count = 0
                            fps_print_last_time = now_fps_time
                            perf_denoise_ms_accum = 0.0
                            perf_pose_ms_accum = 0.0
                    else:
                        pass
                except Exception as e:
                    print(f"update_data 发生错误: {e}")
            # -----------------------------------------
            
            



if __name__ == "__main__":

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    # args.mode = 'twins'
    args.mode = 'single_global'
    # args.mode = 'single_local'
    # args.mode = 'twins'
    
    
    lip_ckpt = os.path.abspath(LIP_CHECKPOINT_PATH)
    if not os.path.exists(lip_ckpt):
        raise FileNotFoundError(f"LIP checkpoint not found: {lip_ckpt}")

    lip_net = build_lip_model(LIP_IMU_NUM)
    lip_net.restore(lip_ckpt)
    lip_net.eval()
    lip_pose_predictor = RealtimeLIPPredictor(lip_net, imu_num=LIP_IMU_NUM)

    smpl_path = os.path.abspath(SMPL_MODEL_FILE)
    if os.path.exists(smpl_path):
        smpl_body_model = art.ParametricModel(smpl_path, device=device)
    else:
        smpl_body_model = None
        print(f"[LIP] SMPL model not found (foot lock needs it): {smpl_path}")
    print(f"[LIP] loaded checkpoint: {lip_ckpt}")

    device_config_clothes = config.device_config.jacket_6IMU
    imu_order_clothes = device_config_clothes['imu_order']

    device_config_pant = config.device_config.pants_5IMU
    imu_order_pant = device_config_pant['imu_order']
    beta = [1.23633747e+00, -7.79002046e-02, 1.39674025e-01, -6.36826841e-01,
            -1.84661462e-01, 7.05933973e-01, 6.46934647e-02, 1.38463431e-01,
            -6.61722247e-02, 2.84141800e-05]
    my_server = DataProcessServer_FullBody(rotation_type=args.rotation_type, part=args.part,
                                           config=[device_config_clothes, device_config_pant], mode=demo_mode.FULL,
                                           track_trans=True,
                                           calibration_session=None, run_unity_package=True,
                                           physics_optim=False,
                                           cali_pose='T', beta=beta,unity_mode=args.mode)

    if DENOISE_ENABLED:
        if os.path.exists(DENOISE_CKPT):
            imu_denoiser = RealtimeIMUDenoiser(
                ckpt_path=DENOISE_CKPT,
                window_size=DENOISE_WINDOW_SIZE,
                chunk_size=DENOISE_CHUNK_SIZE,
                stride=DENOISE_STRIDE,
                steps=DENOISE_STEPS,
                interval=DENOISE_INTERVAL,
                output_mode=DENOISE_OUTPUT_MODE,
                ema_alpha=DENOISE_EMA_ALPHA,
                acc_scale=DENOISE_ACC_SCALE,
                use_fp16=(device.type == "cuda"),
                every_frame=DENOISE_EVERY_FRAME,
                torch_compile=DENOISE_TORCH_COMPILE,
            )
            print(
                f"[IMU Denoise] enabled, window={DENOISE_WINDOW_SIZE}, "
                f"every_frame={DENOISE_EVERY_FRAME}, torch_compile={DENOISE_TORCH_COMPILE}, "
                f"chunk={DENOISE_CHUNK_SIZE}, stride={DENOISE_STRIDE}, "
                f"steps={DENOISE_STEPS}, interval={DENOISE_INTERVAL}, "
                f"mode={DENOISE_OUTPUT_MODE}, ema={DENOISE_EMA_ALPHA}, ckpt={DENOISE_CKPT}"
            )
        else:
            print(f"[IMU Denoise] checkpoint not found, skip denoise: {DENOISE_CKPT}")
            imu_denoiser = None

    clock = Clock()

    manager = MultiPortManager()
    manager.connect_devices()

    # 线程池
    t_pool = []
    # 连接传感器

    start_data_threads(manager, receive_085_COM.data_up_buffer, receive_085_COM.data_down_buffer)
    t_pool.append(Thread(target=data_transmit, kwargs={'fps': 30}))
    # t_pool.append(Thread(target=dynamic_calibration, kwargs={'t_gap': 2}))
    

    # 依次启动线程
    for t in t_pool:
        t.start()


#
