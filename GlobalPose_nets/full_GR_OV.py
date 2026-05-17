__all__ = ['Full_GR_OV']


import os
import time
import torch
import numpy as np
import articulate as art
import carticulate as cart
from scipy.sparse import csc_array
from scipy.sparse.linalg import lsqr
from qpsolvers import solve_qp


class Scene:
    SupportBlock = 0
    GraspBlock = 1

    def __init__(self, viewer):
        self.viewer = viewer
        self.blocks = []

    def add_block(self, type, position, lifetime=120, render=True):
        exist = False
        for block in self.blocks:
            if block['type'] == type and np.abs(block['position'][1] - position[1]) < 0.05:
                position[1] = block['position'][1]
            if block['type'] == type and np.linalg.norm(block['position'] - position) < 0.1:
                block['lifetime'] = lifetime
                exist = True
        if not exist:
            name = str(time.time_ns()) + '_' + str(np.random.randint(0, 1000))
            self.blocks.append({'name': name, 'type': type, 'position': position, 'lifetime': lifetime})
            self.viewer.instantiate(type, name, position, render=render)

    def update(self, render=True):
        for block in self.blocks:
            block['lifetime'] -= 1
            if block['lifetime'] <= 0:
                self.viewer.destroy(block['name'], render=render)
        self.blocks = [block for block in self.blocks if block['lifetime'] > 0]


class Full_GR_OV(torch.nn.Module):
    dt = 1 / 60
    mu = 0.7           # environment fiction coefficient
    kp_pose = 3600     # kp in pose PD controller
    kd_pose = 60       # kd in pose PD controller
    kp_tran = 3600     # kp in tran PD controller
    kd_tran = 60       # kd in tran PD controller
    alpha_pd = 0.0     # relaxation in stable PD controller
    floor_y = -0.97    # floor height
    beta_velocity = 1
    beta_cjoint = 1
    beta_extforce = 0.4
    beta_torque = 1e-3 / 80
    v_imu = (1961, 5424, 1176, 4662, 411, 3021)
    j_reduce = (1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19)
    j_ignore = (0, 7, 8, 10, 11, 20, 21, 22, 23)
    j_contact = (0, 10, 11, 22, 23)

    class Visualization:
        enable = False
        show_residual_force = False
        show_contact_force = True
        show_block = True
        show_contact = False
        show_stationary = False
        show_torque = False

    def __init__(self):
        from articulate.utils.torch import RNN, RNNWithInit
        super(Full_GR_OV, self).__init__()
        self.plnet = RNNWithInit(input_linear=False,
                                 input_size=84,
                                 output_size=18,
                                 hidden_size=512,
                                 num_rnn_layer=3,
                                 dropout=0.4)
        self.iknet = torch.nn.ModuleDict({
            'net1': RNN(input_linear=False,
                        input_size=63,
                        output_size=72,
                        hidden_size=512,
                        num_rnn_layer=3,
                        dropout=0.4),
            'net2': RNN(input_linear=False,
                        input_size=117,
                        output_size=90,
                        hidden_size=512,
                        num_rnn_layer=3,
                        dropout=0.4)
        })
        self.vrnet = RNNWithInit(input_linear=False,
                                 input_size=243,
                                 output_size=9,
                                 hidden_size=512,
                                 num_rnn_layer=3,
                                 dropout=0.4)

        # for training
        # self.plnet.load_state_dict(torch.load('data/weights/Pose-GR/PL/best_weights.pt'))
        # self.iknet.load_state_dict(torch.load('data/weights/Pose-GR/IK/best_weights.pt'))
        # self.vrnet.load_state_dict(torch.load('data/weights/Tran-OV/VR/best_weights.pt'))

        # for testing
        self.load_state_dict(torch.load('data/weights/Full-GR-OV/full/weights.pt'))

        self.B = np.array([[self.mu, -self.mu, 0,       0       ],
                           [1,       1,        1,       1       ],
                           [0,       0,        self.mu, -self.mu]]) / np.sqrt(1 + self.mu ** 2)  # basis of friction cone
        self.body_model = art.ParametricModel('models/SMPL_male.pkl', vert_mask=self.v_imu)
        self.physics_model = cart.get_dynamic_model('models/SMPL_male.pkl')
        # self.rnn_initialize()  # using T-pose
        self.eval()

        if self.Visualization.enable:
            from articulate.utils.unity import MotionViewer
            MotionViewer.colors = [(1, 1, 1)]
            self.viewer = MotionViewer(1)
            self.viewer.connect()
            self.scene = Scene(self.viewer)
            self.force_lpf = [art.LowPassFilter(0.3) for _ in range(5)]
            self.torque_lpf = art.LowPassFilter(0.3)
            self.tran_offset = torch.zeros(3)
            if self.Visualization.show_torque:
                self.viewer.show_torque(0, [1, 2, 4, 5, 16, 17, 18, 19])

    @torch.no_grad()
    def rnn_initialize(self, init_pose=None, init_vel=None):
        r"""
        Initialize the hidden states of the RNNs.

        :param init_pose: Pose in shape [24, 3, 3]. T-pose by default.
        :param init_vel: Root world-space velocity in shape [3]. Zero by default.
        """
        init_pose = torch.eye(3).expand(1, 24, 3, 3) if init_pose is None else init_pose.cpu().view(1, 24, 3, 3)
        init_vel = torch.zeros(3) if init_vel is None else init_vel.cpu().view(3)
        vRR_V = init_vel[1].view(1).clone()
        init_vel[1] = 0
        _, j, v = self.body_model.forward_kinematics(init_pose, calc_mesh=True)
        pRL, gR = (v[0, :5] - v[0, 5:]).mm(init_pose[0, 0]).ravel(), -init_pose[0, 0, 1]
        x1 = torch.cat((pRL, gR)).to(self.plnet.init_net[0].weight.device)
        h, vRR_H, c = -j[:, :, 1].min().view(1), init_pose[0, 0].t().mm(init_vel.unsqueeze(-1)).squeeze(-1), torch.zeros(5)
        x2 = torch.cat((vRR_V, vRR_H, c)).to(self.vrnet.init_net[0].weight.device)
        self.pl1hc = [_.contiguous() for _ in self.plnet.init_net(x1).view(1, 2, self.plnet.num_layers, self.plnet.hidden_size).permute(1, 2, 0, 3)]
        self.vr1hc = [_.contiguous() for _ in self.vrnet.init_net(x2).view(1, 2, self.vrnet.num_layers, self.vrnet.hidden_size).permute(1, 2, 0, 3)]
        self.ik1hc = None
        self.ik2hc = None
        self.last_cjoint = torch.tensor([0, h + self.floor_y, 0]) + j[0, self.j_contact]
        self.fast_tran = torch.zeros(3)
        self.fast_initialized = False
        self.is_init = False
        self.contact = np.zeros(5, dtype=bool)
        self.contact_counter = np.zeros(5, dtype=int)

    @torch.no_grad()
    def _explain_residual_force(self, contact, contact_Jacobian, contact_position, residual_force):
        if np.any(contact):
            J = contact_Jacobian.reshape(5, 3, 75)[contact, :, :6].reshape(-1, 6)
            B, lb = [], []
            for i in np.where(contact)[0]:
                if i > 3 and contact_position[i, 1] > self.floor_y + 0.15:      # hand grasp
                    B.append(np.eye(3))
                    lb.append(-np.ones(3) * np.inf)
                else:
                    B.append(self.B)
                    lb.append(np.zeros(4))
            JTB = J.T @ art.math.block_diagonal_matrix_np(B)
            P = csc_array(JTB.T @ JTB + self.beta_extforce * np.eye(JTB.shape[1]))
            q = -JTB.T @ residual_force
            force = solve_qp(P, q, lb=np.concatenate(lb), solver='osqp')
            error = np.linalg.norm(JTB @ force - residual_force)
        else:
            B = None
            force = np.zeros(0)
            error = np.linalg.norm(residual_force)
        return force, error, B

    @torch.no_grad()
    def forward_frame(self, a, w, R):
        aRB = a.mm(R[5])
        wRB = w.mm(R[5])
        RRB = R[5].t().matmul(R[:5])
        gR0 = -R[5, 1]

        # PL-s1
        x = torch.cat((aRB.ravel(), wRB.ravel(), RRB.ravel(), gR0))
        x, self.pl1hc = self.plnet.rnn(x.view(1, 1, -1), self.pl1hc)
        x = self.plnet.linear2(x.squeeze())        # pRB, gR
        gR1 = art.math.normalize_tensor(x[15:])
        RRB = art.math.from_to_rotation_matrix(gR0, gR1).matmul(RRB)

        # IK-s1
        x = torch.cat((RRB.ravel(), gR1, x[:15]))
        x, self.ik1hc = self.iknet.net1.rnn(x.view(1, 1, -1), self.ik1hc)
        x = self.iknet.net1.linear2(x.squeeze())   # pRJ, gR
        gR2 = art.math.normalize_tensor(x[69:])
        RRB = art.math.from_to_rotation_matrix(gR1, gR2).matmul(RRB)

        # IK-s2
        x = torch.cat((RRB.ravel(), gR2, x[:69]))
        x, self.ik2hc = self.iknet.net2.rnn(x.view(1, 1, -1), self.ik2hc)
        x = self.iknet.net2.linear2(x.squeeze())   # RRJ

        # get pose estimation
        RRJ = art.math.r6d_to_rotation_matrix(x).cpu()
        glb_pose = torch.eye(3).repeat(1, 24, 1, 1)
        glb_pose[:, self.j_reduce] = RRJ.view(1, 15, 3, 3)
        pose = self.body_model.inverse_kinematics_R(glb_pose).view(24, 3, 3)
        pose[self.j_ignore, ...] = torch.eye(3)
        pRJ = self.body_model.forward_kinematics(pose.unsqueeze(0))[1][0, 1:]
        pose[0] = R[5].mm(art.math.from_to_rotation_matrix(gR2, gR0).squeeze()).cpu()

        # VR-s1
        aRB = a.cpu().mm(pose[0])
        wRB = w.cpu().mm(pose[0])
        x = torch.cat((RRJ.ravel(), pRJ.ravel(), aRB.ravel(), wRB.ravel(), gR2.cpu())).to(gR2.device)
        x, self.vr1hc = self.vrnet.rnn(x.view(1, 1, -1), self.vr1hc)
        x = self.vrnet.linear2(x.squeeze())  # vRR_V, vRR_H, stationary_prob

        # get translation estimation
        vRR_V, vRR_H, stationary_prob = x[0].item(), x[1:4].cpu(), x[4:].sigmoid().cpu()
        vWR = pose[0].mm(vRR_H.unsqueeze(-1)).squeeze(-1)
        vWR[1] = vRR_V
        cjoint = torch.cat((torch.zeros(1, 3), pRJ.mm(pose[0].t())))[self.j_contact, :]
        stationary_weight = (stationary_prob * 5 - 3).clip(0, 1)
        velocity = (stationary_weight.unsqueeze(0).mm(self.last_cjoint - cjoint)[0] / self.dt + self.beta_velocity * vWR) / (self.beta_velocity + stationary_weight.sum())
        self.last_cjoint = cjoint

        # physics optimization
        if not self.is_init:
            self.is_init = True
            self.physics_model.set_state_R(pose.numpy(), np.array([0, self.floor_y - cjoint[:, 1].min().item(), 0]), np.zeros(75))
        else:
            # get current state
            pose_cur, tran_cur, qdot = self.physics_model.get_state_R()
            cjoint_cur = np.vstack([self.physics_model.get_position(j) for j in self.j_contact])
            cvel_cur = np.vstack([self.physics_model.get_linear_velocity(j) for j in self.j_contact])
            cJ_cur = np.vstack([self.physics_model.get_linear_Jacobian(j) for j in self.j_contact])
            cJdot_cur = np.vstack([self.physics_model.get_linear_Jacobian_dot(j) for j in self.j_contact])
            M = self.physics_model.mass_matrix()
            h = self.physics_model.inverse_dynamics(np.zeros(75))
            stationary = stationary_prob.numpy() > 0.7
            qdot = self._sanitize_numpy(qdot, "qdot", clip_val=500.0)
            cjoint_cur = self._sanitize_numpy(cjoint_cur, "cjoint_cur", clip_val=20.0)
            cvel_cur = self._sanitize_numpy(cvel_cur, "cvel_cur", clip_val=50.0)
            cJ_cur = self._sanitize_numpy(cJ_cur, "cJ_cur", clip_val=1e3)
            cJdot_cur = self._sanitize_numpy(cJdot_cur, "cJdot_cur", clip_val=1e3)
            M = self._sanitize_numpy(M, "mass_matrix", clip_val=1e5)
            h = self._sanitize_numpy(h, "inverse_dynamics", clip_val=1e5)
            qdot = self._sanitize_numpy(qdot, "qdot", clip_val=500.0)
            cjoint_cur = self._sanitize_numpy(cjoint_cur, "cjoint_cur", clip_val=20.0)
            cvel_cur = self._sanitize_numpy(cvel_cur, "cvel_cur", clip_val=50.0)
            cJ_cur = self._sanitize_numpy(cJ_cur, "cJ_cur", clip_val=1e3)
            cJdot_cur = self._sanitize_numpy(cJdot_cur, "cJdot_cur", clip_val=1e3)
            M = self._sanitize_numpy(M, "mass_matrix", clip_val=1e5)
            h = self._sanitize_numpy(h, "inverse_dynamics", clip_val=1e5)

            # dual RSPD controller
            pose_dtype = pose.dtype
            qdot_t = torch.from_numpy(qdot[3:]).to(dtype=pose_dtype)
            pose_cur_t = torch.from_numpy(pose_cur).to(dtype=pose_dtype)
            R = art.math.axis_angle_to_rotation_matrix(qdot_t * self.alpha_pd * self.dt)
            delta_pose = art.math.rotation_matrix_to_axis_angle(
                pose_cur_t.bmm(R).transpose(1, 2).bmm(pose)
            ).ravel().numpy()
            thetaddotdes = (self.kp_pose * delta_pose - self.kd_pose * qdot[3:]) / (1 + self.kd_pose * self.alpha_pd * self.dt)
            cjoint = tran_cur + velocity.numpy() * self.dt + cjoint.numpy()
            cjoint = art.math.lerp(cjoint, cjoint_cur, stationary_weight.view(5, 1).numpy())
            delta_tran = cjoint - cjoint_cur - cvel_cur * self.alpha_pd * self.dt
            rddotdes = (self.kp_tran * delta_tran - self.kd_tran * cvel_cur).ravel() / (1 + self.kd_tran * self.alpha_pd * self.dt)

            # unconstrained tracking
            cjoint_cur[0, 1] -= 0.15
            cjoint[0, 1] -= 0.15
            k = np.ones((5, 3)) * self.beta_cjoint
            k[self.contact] *= 10
            A = np.vstack((np.hstack((np.zeros((72, 3)), np.eye(72))), np.sqrt(k.reshape(15, 1)) * cJ_cur, np.sqrt(self.beta_torque) * M))
            b = np.concatenate((thetaddotdes, np.sqrt(k.reshape(15)) * (-cJdot_cur @ qdot + rddotdes), np.sqrt(self.beta_torque) * (-h)))
            qddot = lsqr(csc_array(A), b)[0]
            residual_force = M[:6] @ qddot + h[:6]

            # determine potential contact
            vdist = np.abs(cjoint_cur[np.newaxis, :, 1] - cjoint_cur[:, np.newaxis, 1])
            contact = stationary & (self.contact | (cjoint_cur[:, 1] < self.floor_y + 0.05))
            if np.any(contact):
                contact |= stationary & (vdist[contact].min(axis=0) < 0.05)
            potential_contact = stationary & ~contact
            if contact[0] or potential_contact[0]:  # root joint
                lleg = self.physics_model.get_position(4) - self.physics_model.get_position(1)
                rleg = self.physics_model.get_position(5) - self.physics_model.get_position(2)
                if min(np.arccos(-lleg[1] / np.linalg.norm(lleg)), np.arccos(-rleg[1] / np.linalg.norm(rleg))) < np.pi / 4:
                    contact[0], potential_contact[0] = False, False

            # explain residual force by contacts
            force, err, forceB = self._explain_residual_force(contact, cJ_cur, cjoint_cur, residual_force)
            for i in np.argsort(cjoint_cur[:, 1]):  # add potential contact from lowest to highest
                if err > 400 and potential_contact[i]:
                    contact[i] = True
                    force_new, err_new, forceB_new = self._explain_residual_force(contact, cJ_cur, cjoint_cur, residual_force)
                    self.contact_counter[i] = self.contact_counter[i] + 1 if err_new / err < 0.6 else 0
                    if self.contact_counter[i] >= 5:
                        force, err, forceB = force_new, err_new, forceB_new
                    else:
                        contact[i] = False
                else:
                    self.contact_counter[i] = 0

            # update contact position
            near_ground = cjoint[:, 1] < self.floor_y + 0.15
            # object_y = []
            # for i in np.where(contact & ~near_ground)[0]:
            #     is_added = False
            #     for y in object_y:
            #         if not is_added and abs(sum(y) / len(y) - cjoint[i, 1]) < 0.05:
            #             y.append(cjoint[i, 1])
            #             is_added = True
            #     if not is_added:
            #         object_y.append([cjoint[i, 1]])
            # object_y = np.array([sum(y) / len(y) for y in object_y])
            # for i in np.where(contact)[0]:
            #     cjoint[i, 1] = art.math.lerp(cjoint[i, 1], self.floor_y, 0.1) if near_ground[i] else object_y[np.abs(cjoint[i, 1] - object_y).argmin()]
            for i in np.where(contact & near_ground)[0]:
                cjoint[i, 1] = art.math.lerp(cjoint[i, 1], self.floor_y, 0.1)
            cjoint[cjoint[:, 1] < self.floor_y, 1] = self.floor_y

            # re-optimization and update state
            delta_tran = cjoint - cjoint_cur - cvel_cur * self.alpha_pd * self.dt
            rddotdes = (self.kp_tran * delta_tran - self.kd_tran * cvel_cur).ravel() / (1 + self.kd_tran * self.alpha_pd * self.dt)
            if np.any(contact):
                expected_force_dim = sum(b.shape[1] for b in forceB) if forceB is not None else 0
                if force.shape[0] == expected_force_dim and expected_force_dim > 0:
                    J = cJ_cur.reshape(5, 3, 75)[contact].reshape(-1, 75)
                    B = art.math.block_diagonal_matrix_np(forceB)
                    force = B @ force
                    torque = J.T @ force
                else:
                    # If the QP solver failed to produce a valid force vector, skip
                    # contact torque for this frame instead of crashing.
                    contact[:] = False
                    force = np.zeros(0)
                    torque = np.zeros(75)
            else:
                torque = np.zeros(75)
            A = np.vstack((np.hstack((np.zeros((72, 3)), np.eye(72))), np.sqrt(k.reshape(15, 1)) * cJ_cur, np.sqrt(self.beta_torque * 3) * M))
            b = np.concatenate((thetaddotdes, np.sqrt(k.reshape(15)) * (-cJdot_cur @ qdot + rddotdes), np.sqrt(self.beta_torque * 3) * (-h + torque)))
            qddot = lsqr(csc_array(A), b)[0]
            self.physics_model.update_state(qddot, self.dt)
            self.contact = contact

            # visualization
            if self.Visualization.enable:
                force = iter(force.reshape(-1, 3))
                self.viewer.clear_line(render=False)
                self.viewer.clear_point(render=False)
                self.viewer.update(torch.from_numpy(pose_cur), torch.from_numpy(tran_cur) - self.tran_offset, render=False)
                cjoint -= self.tran_offset.numpy()
                for i in np.where(contact)[0]:
                    if self.Visualization.show_block and not near_ground[i]:
                        if i < 3 or True:   # always show block
                            self.scene.add_block(Scene.SupportBlock, cjoint[i] - [0, 0.2, 0], lifetime=120, render=False)
                        else:
                            self.scene.add_block(Scene.GraspBlock, cjoint[i], lifetime=30, render=False)
                    if self.Visualization.show_contact_force:
                        self.viewer.draw_line(cjoint[i], cjoint[i] + self.force_lpf[i](next(force)) * 0.002, [1, 0, 0], 0.02, render=False)
                    if self.Visualization.show_contact:
                        self.viewer.draw_point(cjoint[i], [0, 1, 0], 0.15, render=False)
                if self.Visualization.show_stationary:
                    for i in np.where(stationary & ~contact)[0]:
                        self.viewer.draw_point(cjoint[i], [0.4, 0.4, 1], 0.15, render=False)
                if self.Visualization.show_block:
                    self.scene.update(render=False)
                if self.Visualization.show_residual_force:
                    self.viewer.draw_line_from_joint(0, 0, residual_force[:3] * 0.001, (0, 0, 1), 0.02, render=False)
                if self.Visualization.show_torque:
                    self.viewer.show_torque(0, [1, 2, 4, 5, 16, 17, 18, 19], render=False)
                    tau = M @ qddot + h - torque
                    tau = self.torque_lpf(tau)
                    self.viewer.update_torque(tau[3:], render=False)
                else:
                    self.viewer.hide_torque(render=False)
                self.viewer.render()

        refined_pose, refined_tran, qdot = self.physics_model.get_state_R()
        return torch.from_numpy(refined_pose), torch.from_numpy(refined_tran)

    def forward(self, x, fast=True):
        # PL-s1
        RRB = [x_[:, 36:81].view(-1, 5, 3, 3) for x_, y_ in x]
        gR0 = [x_[:, 81:] for x_, y_ in x]
        x1 = [(x_, y_[:18]) for x_, y_ in x]
        x1 = self.plnet(x1)       # pRB, gR

        # IK-s1
        pRB = [x_[:, :15] for x_ in x1]
        gR1 = [art.math.normalize_tensor(x_[:, 15:].clone().detach()) for x_ in x1]
        RRB = [art.math.from_to_rotation_matrix(gR0_, gR1_).unsqueeze(1).matmul(RRB_) for gR0_, gR1_, RRB_ in zip(gR0, gR1, RRB)]
        x2 = [torch.cat((RRB_.flatten(1), gR1_, pRB_), dim=1) for RRB_, gR1_, pRB_ in zip(RRB, gR1, pRB)]
        x2 = self.iknet.net1(x2)   # pRJ, gR

        # IK-s2
        pRJ = [x_[:, :69] for x_ in x2]
        gR2 = [art.math.normalize_tensor(x_[:, 69:].clone().detach()) for x_ in x2]
        RRB = [art.math.from_to_rotation_matrix(gR1_, gR2_).unsqueeze(1).matmul(RRB_) for gR1_, gR2_, RRB_ in zip(gR1, gR2, RRB)]
        x3 = [torch.cat((RRB_.flatten(1), gR2_, pRJ_), dim=1) for RRB_, gR2_, pRJ_ in zip(RRB, gR2, pRJ)]
        x3 = self.iknet.net2(x3)   # RRJ

        # VR-s1
        if fast:   # faster approximation
            RRJ = [art.math.r6d_to_rotation_matrix(x3_.detach()).view(-1, 135) for x3_ in x3]
            awRB = [x_[:, :36].view(-1, 12, 3).bmm(art.math.from_to_rotation_matrix(gR2_, gR0_)).view(-1, 36) for gR2_, gR0_, (x_, y_) in zip(gR2, gR0, x)]
            x4 = [(torch.cat((RRJ_, pRJ_.detach(), awRB_, gR2_), dim=1), torch.zeros_like(y_[-9:]) if torch.isnan(y_[-9]) else y_[-9:]) for RRJ_, pRJ_, awRB_, gR2_, (x_, y_) in zip(RRJ, pRJ, awRB, gR2, x)]
        else:
            RRJ, pRJ, aRB_new, wRB_new, pose = [], [], [], [], []
            aRB = [x_[:, :18].view(-1, 6, 3) for x_, y_ in x]
            wRB = [x_[:, 18:36].view(-1, 6, 3) for x_, y_ in x]
            for i in range(len(x3)):
                x3_ = x3[i].clone().detach().cpu()
                RRJ_ = art.math.r6d_to_rotation_matrix(x3_).view(-1, 15, 3, 3)
                glb_pose_ = torch.eye(3).repeat(RRJ_.shape[0], 24, 1, 1)
                glb_pose_[:, self.j_reduce] = RRJ_
                pose_ = self.body_model.inverse_kinematics_R(glb_pose_).view(-1, 24, 3, 3)
                pose_[:, self.j_ignore] = torch.eye(3)
                pRJ_ = self.body_model.forward_kinematics(pose_)[1][:, 1:]
                aRB_ = aRB[i].bmm(art.math.from_to_rotation_matrix(gR2[i], gR0[i]))
                wRB_ = wRB[i].bmm(art.math.from_to_rotation_matrix(gR2[i], gR0[i]))
                RRJ.append(RRJ_)
                pRJ.append(pRJ_)
                aRB_new.append(aRB_)
                wRB_new.append(wRB_)
                pose.append(pose_)
            no_translation = [torch.isnan(y_[-9]).item() for x_, y_ in x]
            x4 = [(torch.cat((RRJ_.flatten(1).to(aRB_.device), pRJ_.flatten(1).to(aRB_.device), aRB_.flatten(1), wRB_.flatten(1), gR2_), dim=1), x_[1][-9:] if not nt_ else torch.zeros_like(x_[1][-9:])) for RRJ_, pRJ_, aRB_, wRB_, gR2_, x_, nt_ in zip(RRJ, pRJ, aRB_new, wRB_new, gR2, x, no_translation)]
        x4 = self.vrnet(x4)   # vRR_V, vRR_H, stationary_prob
        result = [torch.cat((x1_, x2_, x3_, x4_), dim=1) for x1_, x2_, x3_, x4_ in zip(x1, x2, x3, x4)]
        return result



class Full_GR_OV_10(torch.nn.Module):
    dt = 1 / 60
    mu = 0.7           # environment fiction coefficient
    kp_pose = 3600     # kp in pose PD controller
    kd_pose = 60       # kd in pose PD controller
    kp_tran = 3600     # kp in tran PD controller
    kd_tran = 60       # kd in tran PD controller
    alpha_pd = 0.0     # relaxation in stable PD controller
    floor_y = -0.97    # floor height
    beta_velocity = 1
    beta_cjoint = 1
    beta_extforce = 0.4
    beta_torque = 1e-3 / 80
    v_imu = (1961, 5424, 1505, 4917, 1305, 3187, 6585, 850, 4333, 4298)
    j_reduce = (1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19)
    j_ignore = (0, 7, 8, 10, 11, 20, 21, 22, 23)
    j_contact = (0, 10, 11, 22, 23)
    imu_num = len(v_imu)
    leaf_idx = [0, 1, 5, 6, 4]
    root_idx = imu_num - 1

    class Visualization:
        enable = False
        show_residual_force = False
        show_contact_force = True
        show_block = True
        show_contact = False
        show_stationary = False
        show_torque = False

    def __init__(self):
        from articulate.utils.torch import RNN, RNNWithInit
        super(Full_GR_OV_10, self).__init__()
        self.plnet = RNNWithInit(
            input_linear=False,
            input_size=144,
            output_size=18,
            hidden_size=512,
            num_rnn_layer=3,
            dropout=0.4,
        )
        self.iknet = torch.nn.ModuleDict({
            'net1': RNN(
                input_linear=False,
                input_size=99,
                output_size=72,
                hidden_size=512,
                num_rnn_layer=3,
                dropout=0.4,
            ),
            'net2': RNN(
                input_linear=False,
                input_size=153,
                output_size=90,
                hidden_size=512,
                num_rnn_layer=3,
                dropout=0.4,
            ),
        })
        self.vrnet = RNNWithInit(
            input_linear=False,
            input_size=267,
            output_size=9,
            hidden_size=512,
            num_rnn_layer=3,
            dropout=0.4,
        )

        # for testing
        # self.load_state_dict(torch.load('data/weights_amass/Full-GR-OV/full/weights.pt')) # amass trained 10 IMU pose weights
        # self.load_state_dict(torch.load('data/weights_4_amass/Full-GR-OV/full/weights.pt')) # amass trained 4 IMU pose weights
        # self.load_state_dict(torch.load('data/weights_finetuned_with_amass/Full-GR-OV/full/weights.pt'))
        # self.load_state_dict(torch.load('data/weights_finetuned_sc///Full-GR-OV/full/weights_40.pt'))
        # self.load_state_dict(torch.load('data/weights_finetuned_ae30/Full-GR-OV/full//weights_40.pt'))
        # self.load_state_dict(torch.load(r'C:\Users\15482\Desktop\GlobalPose\checkpoints\flowdit_gp_finetune\flowdit_gp_finetune_ep4.pth'))
        # self.load_state_dict(torch.load(r'C:\Users\15482\Desktop\GlobalPose\data\U100hL7.07h_w_amass_finetuned_w_denoised\weights.pt')) # FGP final weights
        # self.load_state_dict(torch.load(r'C:\Users\15482\Desktop\GlobalPose\data\U100hL7.07h_w_amass_denoised_residual\Full-GR-OV\full\weights.pt'))

        self.load_state_dict(torch.load('checkpoints/globalpose/weights.pt'))

        # self.load_state_dict(torch.load('data/weights_finetuned_loose/Full-GR-OV/full//weights_40.pt'))
        # self.load_state_dict(torch.load('data/weights/Pose-GR/full/weights.pt'))
        # self.load_state_dict(torch.load('data/weights_loose_finetuned/weights.pt'))
        self.B = np.array([
            [self.mu, -self.mu, 0,       0      ],
            [1,       1,        1,       1      ],
            [0,       0,        self.mu, -self.mu],
        ]) / np.sqrt(1 + self.mu ** 2)  # basis of friction cone

        self.body_model = art.ParametricModel('models/SMPL_male.pkl', vert_mask=self.v_imu)
        self.physics_model = cart.get_dynamic_model('models/SMPL_male.pkl')
        self.rnn_initialize()  # using T-pose
        self.eval()

        if self.Visualization.enable:
            from articulate.utils.unity import MotionViewer
            MotionViewer.colors = [(1, 1, 1)]
            self.viewer = MotionViewer(1)
            self.viewer.connect()
            self.scene = Scene(self.viewer)
            self.force_lpf = [art.LowPassFilter(0.3) for _ in range(5)]
            self.torque_lpf = art.LowPassFilter(0.3)
            self.tran_offset = torch.zeros(3)
            if self.Visualization.show_torque:
                self.viewer.show_torque(0, [1, 2, 4, 5, 16, 17, 18, 19])

    # ========= 新增：安全 lsqr 封装 =========
    def _safe_lsqr(self, A, b, name="qddot", clip_val=500.0, damp=1e-2, iters=100):
        """
        对 lsqr 解做安全处理：
        - damp 正则避免病态爆炸
        - NaN / inf → 直接清零
        - 元素级裁剪到 [-clip_val, clip_val]
        """
        A = np.asarray(A, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)

        if not np.all(np.isfinite(A)):
            print(f"[safe_lsqr] {name} A non-finite, sanitizing")
            A = np.nan_to_num(A, nan=0.0, posinf=clip_val, neginf=-clip_val)

        if not np.all(np.isfinite(b)):
            print(f"[safe_lsqr] {name} b non-finite, sanitizing")
            b = np.nan_to_num(b, nan=0.0, posinf=clip_val, neginf=-clip_val)

        sol, *_ = lsqr(csc_array(A), b, damp=damp, iter_lim=iters)
        sol = np.asarray(sol, dtype=np.float64)

        if not np.all(np.isfinite(sol)):
            print(f"[safe_lsqr] {name} non-finite, zeroed")
            sol[:] = 0.0
            return sol

        np.clip(sol, -clip_val, clip_val, out=sol)
        return sol

    def _sanitize_torch(self, x, name, clip_val=None):
        if torch.isfinite(x).all():
            return x
        print(f"[warn] {name} non-finite, sanitizing")
        pos = clip_val if clip_val is not None else 0.0
        neg = -clip_val if clip_val is not None else 0.0
        x = torch.nan_to_num(x, nan=0.0, posinf=pos, neginf=neg)
        if clip_val is not None:
            x = x.clamp(-clip_val, clip_val)
        return x

    def _sanitize_numpy(self, x, name, clip_val=None):
        x = np.asarray(x, dtype=np.float64)
        if np.all(np.isfinite(x)):
            return x
        print(f"[warn] {name} non-finite, sanitizing")
        pos = clip_val if clip_val is not None else 0.0
        neg = -clip_val if clip_val is not None else 0.0
        x = np.nan_to_num(x, nan=0.0, posinf=pos, neginf=neg)
        if clip_val is not None:
            np.clip(x, -clip_val, clip_val, out=x)
        return x

    @torch.no_grad()
    def rnn_initialize(self, init_pose=None, init_vel=None):
        r"""
        Initialize the hidden states of the RNNs.

        :param init_pose: Pose in shape [24, 3, 3]. T-pose by default.
        :param init_vel: Root world-space velocity in shape [3]. Zero by default.
        """
        init_pose = torch.eye(3).expand(1, 24, 3, 3) if init_pose is None else init_pose.cpu().view(1, 24, 3, 3)
        init_vel = torch.zeros(3) if init_vel is None else init_vel.cpu().view(3)
        vRR_V = init_vel[1].view(1).clone()
        init_vel[1] = 0
        _, j, v = self.body_model.forward_kinematics(init_pose, calc_mesh=True)

        pRL, gR = (v[0, :self.root_idx] - v[0, self.root_idx:]).mm(init_pose[0, 0]), -init_pose[0, 0, 1]
        pRL = pRL[self.leaf_idx, :].ravel()
        x1 = torch.cat((pRL, gR)).to(self.plnet.init_net[0].weight.device)
        h, vRR_H, c = -j[:, :, 1].min().view(1), init_pose[0, 0].t().mm(init_vel.unsqueeze(-1)).squeeze(-1), torch.zeros(5)
        x2 = torch.cat((vRR_V, vRR_H, c)).to(self.vrnet.init_net[0].weight.device)
        self.pl1hc = [
            _.contiguous()
            for _ in self.plnet
            .init_net(x1)
            .view(1, 2, self.plnet.num_layers, self.plnet.hidden_size)
            .permute(1, 2, 0, 3)
        ]
        self.vr1hc = [
            _.contiguous()
            for _ in self.vrnet
            .init_net(x2)
            .view(1, 2, self.vrnet.num_layers, self.vrnet.hidden_size)
            .permute(1, 2, 0, 3)
        ]
        self.ik1hc = None
        self.ik2hc = None
        self.last_cjoint = torch.tensor([0, h + self.floor_y, 0]) + j[0, self.j_contact]
        self.is_init = False
        self.contact = np.zeros(5, dtype=bool)
        self.contact_counter = np.zeros(5, dtype=int)

    @torch.no_grad()
    def _explain_residual_force(self, contact, contact_Jacobian, contact_position, residual_force):
        if np.any(contact):
            J = contact_Jacobian.reshape(5, 3, 75)[contact, :, :6].reshape(-1, 6)
            B, lb = [], []
            for i in np.where(contact)[0]:
                if i > 3 and contact_position[i, 1] > self.floor_y + 0.15:  # hand grasp
                    B.append(np.eye(3))
                    lb.append(-np.ones(3) * np.inf)
                else:
                    B.append(self.B)
                    lb.append(np.zeros(4))
            JTB = J.T @ art.math.block_diagonal_matrix_np(B)
            P = csc_array(JTB.T @ JTB + self.beta_extforce * np.eye(JTB.shape[1]))
            q = -JTB.T @ residual_force
            force = solve_qp(P, q, lb=np.concatenate(lb), solver='osqp')

            # ✅ QP 失败或数值异常 → 回退
            if (force is None) or (not np.all(np.isfinite(force))):
                force = np.zeros(P.shape[0], dtype=np.float64)

            # 防止极端爆炸
            if np.linalg.norm(force) > 1e6:
                force = np.zeros_like(force)

            error = np.linalg.norm(JTB @ force - residual_force)
        else:
            B = None
            force = np.zeros(0)
            error = np.linalg.norm(residual_force)
        return force, error, B

    def _safe_leg_angle(self, v, y_idx=1, eps=1e-8):
        v = np.asarray(v, dtype=np.float64)
        if not np.all(np.isfinite(v)):
            return np.nan
        n = np.linalg.norm(v)
        if (not np.isfinite(n)) or (n < eps):
            return np.nan
        c = -v[y_idx] / n
        c = np.clip(c, -1.0, 1.0)
        return float(np.arccos(c))

    @torch.no_grad()
    def forward_frame(self, a, w, R):
        """
        a: [imu_num, 3]
        w: [imu_num, 3]
        R: [imu_num, 3, 3]   # each sensor orientation in global/world
        假设 root IMU 在最后一个索引：root_idx = imu_num - 1
        """
        imu_num = self.imu_num
        root_idx = imu_num - 1
        n_nonroot = imu_num - 1

        a = self._sanitize_torch(a.detach().to(torch.float32), "imu_acc", clip_val=100.0)
        w = self._sanitize_torch(w.detach().to(torch.float32), "imu_gyro", clip_val=100.0)
        R = self._sanitize_torch(R.detach().to(torch.float32), "imu_rot", clip_val=10.0)
        R = art.math.normalize_rotation_matrix(R)

        a_dim = imu_num * 3
        w_dim = imu_num * 3
        aw_dim = a_dim + w_dim
        rrb_dim = n_nonroot * 9

        # ---- global -> root frame (root = last) ----
        aRB = a.mm(R[root_idx])                      # [imu_num,3]
        wRB = w.mm(R[root_idx])                      # [imu_num,3]
        RRB = R[root_idx].t().matmul(R[:root_idx])   # [n_nonroot,3,3]
        gR0 = -R[root_idx, 1]                        # [3]

        # PL-s1
        x = torch.cat((aRB.reshape(-1), wRB.reshape(-1), RRB.reshape(-1), gR0), dim=0)  # [144]
        x, self.pl1hc = self.plnet.rnn(x.view(1, 1, -1), self.pl1hc)
        x = self.plnet.linear2(x.squeeze())          # pRB(15), gR(3)

        gR1 = art.math.normalize_tensor(x[15:])
        if not torch.isfinite(gR1).all():
            print("[warn] gR1 non-finite, fallback to gR0")
            gR1 = gR0.clone()
        RRB = art.math.from_to_rotation_matrix(gR0, gR1).matmul(RRB)  # broadcast: [n_nonroot,3,3]

        # IK-s1
        x = torch.cat((RRB.reshape(-1), gR1, x[:15]), dim=0)          # [n_nonroot*9 + 3 + 15] = 99
        x, self.ik1hc = self.iknet.net1.rnn(x.view(1, 1, -1), self.ik1hc)
        x = self.iknet.net1.linear2(x.squeeze())      # pRJ(69), gR(3)

        gR2 = art.math.normalize_tensor(x[69:])
        if not torch.isfinite(gR2).all():
            print("[warn] gR2 non-finite, fallback to gR1")
            gR2 = gR1.clone()
        RRB = art.math.from_to_rotation_matrix(gR1, gR2).matmul(RRB)

        # IK-s2
        x = torch.cat((RRB.reshape(-1), gR2, x[:69]), dim=0)          # [n_nonroot*9 + 3 + 69] = 153
        x, self.ik2hc = self.iknet.net2.rnn(x.view(1, 1, -1), self.ik2hc)
        x = self.iknet.net2.linear2(x.squeeze())      # RRJ r6d (90)

        # ---- get pose estimation ----
        RRJ = art.math.r6d_to_rotation_matrix(x).cpu()                # [15,3,3]
        RRJ = self._sanitize_torch(RRJ, "RRJ", clip_val=10.0)
        RRJ = art.math.normalize_rotation_matrix(RRJ)
        glb_pose = torch.eye(3).repeat(1, 24, 1, 1)
        glb_pose[:, self.j_reduce] = RRJ.view(1, 15, 3, 3)

        pose = self.body_model.inverse_kinematics_R(glb_pose).view(24, 3, 3)
        pose = self._sanitize_torch(pose, "pose_ik", clip_val=10.0)
        pose = art.math.normalize_rotation_matrix(pose)
        pose[self.j_ignore, ...] = torch.eye(3)

        pRJ = self.body_model.forward_kinematics(pose.unsqueeze(0))[1][0, 1:]  # [23,3]
        pRJ = self._sanitize_torch(pRJ, "pRJ", clip_val=10.0)

        # root orientation: put root back to world frame
        R_align = art.math.from_to_rotation_matrix(gR2, gR0)
        if R_align.dim() == 3:
            R_align = R_align.squeeze(0)   # [3,3]
        pose[0] = (R[root_idx] @ R_align).cpu()

        # ---- VR-s1 ----
        aRB2 = a.cpu().mm(pose[0])                   # [imu_num,3]
        wRB2 = w.cpu().mm(pose[0])                   # [imu_num,3]

        x = torch.cat((
            RRJ.reshape(-1),                         # 135
            pRJ.reshape(-1),                         # 69
            aRB2.reshape(-1),                        # imu_num*3 = 30
            wRB2.reshape(-1),                        # imu_num*3 = 30
            gR2.cpu(),                               # 3
        ), dim=0).to(gR2.device)                     # 267

        x, self.vr1hc = self.vrnet.rnn(x.view(1, 1, -1), self.vr1hc)
        x = self.vrnet.linear2(x.squeeze())          # vRR_V, vRR_H(3), stationary_prob(K)

        # ---- get translation estimation ----
        vRR_V = x[0].item()
        vRR_H = x[1:4].cpu()
        stationary_prob = x[4:].sigmoid().cpu()

        vWR = pose[0].mm(vRR_H.unsqueeze(-1)).squeeze(-1)
        vWR[1] = vRR_V

        cjoint = torch.cat((torch.zeros(1, 3), pRJ.mm(pose[0].t())))[self.j_contact, :]
        stationary_weight = (stationary_prob * 5 - 3).clip(0, 1)
        vWR = self._sanitize_torch(vWR, "vWR", clip_val=20.0)
        cjoint = self._sanitize_torch(cjoint, "cjoint", clip_val=20.0)
        stationary_weight = self._sanitize_torch(stationary_weight, "stationary_weight", clip_val=1.0)
        self.last_cjoint = self._sanitize_torch(self.last_cjoint, "last_cjoint", clip_val=20.0)

        velocity = (
            stationary_weight.unsqueeze(0).mm(self.last_cjoint - cjoint)[0] / self.dt
            + self.beta_velocity * vWR
        ) / (self.beta_velocity + stationary_weight.sum())

        if not torch.isfinite(velocity).all():
            print("[warn] velocity non-finite, zeroing")
            velocity = torch.zeros_like(velocity)
        else:
            max_v = 10.0
            v_norm = velocity.norm().item()
            if v_norm > max_v:
                velocity = velocity * (max_v / (v_norm + 1e-8))

        self.last_cjoint = cjoint

        # physics optimization
        if not self.is_init:
            self.is_init = True
            self.physics_model.set_state_R(
                pose.numpy(),
                np.array([0, self.floor_y - cjoint[:, 1].min().item(), 0]),
                np.zeros(75),
            )
        else:
            # get current state
            pose_cur, tran_cur, qdot = self.physics_model.get_state_R()
            cjoint_cur = np.vstack([self.physics_model.get_position(j) for j in self.j_contact])
            cvel_cur = np.vstack([self.physics_model.get_linear_velocity(j) for j in self.j_contact])
            cJ_cur = np.vstack([self.physics_model.get_linear_Jacobian(j) for j in self.j_contact])
            cJdot_cur = np.vstack([self.physics_model.get_linear_Jacobian_dot(j) for j in self.j_contact])
            M = self.physics_model.mass_matrix()
            h = self.physics_model.inverse_dynamics(np.zeros(75))
            stationary = stationary_prob.numpy() > 0.7

            # dual RSPD controller
            R_delta = art.math.axis_angle_to_rotation_matrix(
                torch.from_numpy(qdot[3:]) * self.alpha_pd * self.dt
            )
            delta_pose = art.math.rotation_matrix_to_axis_angle(
                torch.from_numpy(pose_cur).bmm(R_delta).transpose(1, 2).bmm(pose)
            ).ravel().numpy()
            thetaddotdes = (
                self.kp_pose * delta_pose - self.kd_pose * qdot[3:]
            ) / (1 + self.kd_pose * self.alpha_pd * self.dt)
            thetaddotdes = self._sanitize_numpy(thetaddotdes, "thetaddotdes", clip_val=500.0)

            cjoint_np = tran_cur + velocity.numpy() * self.dt + cjoint.numpy()
            cjoint_np = art.math.lerp(
                cjoint_np,
                cjoint_cur,
                stationary_weight.view(5, 1).numpy(),
            )
            cjoint_np = self._sanitize_numpy(cjoint_np, "cjoint_np", clip_val=20.0)
            delta_tran = cjoint_np - cjoint_cur - cvel_cur * self.alpha_pd * self.dt
            rddotdes = (
                self.kp_tran * delta_tran - self.kd_tran * cvel_cur
            ).ravel() / (1 + self.kd_tran * self.alpha_pd * self.dt)
            rddotdes = self._sanitize_numpy(rddotdes, "rddotdes_stage1", clip_val=500.0)

            # unconstrained tracking
            cjoint_cur[0, 1] -= 0.15
            cjoint_np[0, 1] -= 0.15
            k = np.ones((5, 3)) * self.beta_cjoint
            k[self.contact] *= 10

            A = np.vstack((
                np.hstack((np.zeros((72, 3)), np.eye(72))),
                np.sqrt(k.reshape(15, 1)) * cJ_cur,
                np.sqrt(self.beta_torque) * M,
            ))
            b = np.concatenate((
                thetaddotdes,
                np.sqrt(k.reshape(15)) * (-cJdot_cur @ qdot + rddotdes),
                np.sqrt(self.beta_torque) * (-h),
            ))

            # ✅ 安全 lsqr（stage1）
            qddot = self._safe_lsqr(A, b, name="qddot_stage1", clip_val=500.0, damp=1e-2)
            residual_force = M[:6] @ qddot + h[:6]

            # determine potential contact
            vdist = np.abs(cjoint_cur[np.newaxis, :, 1] - cjoint_cur[:, np.newaxis, 1])
            contact = stationary & (self.contact | (cjoint_cur[:, 1] < self.floor_y + 0.05))
            if np.any(contact):
                contact |= stationary & (vdist[contact].min(axis=0) < 0.05)
            potential_contact = stationary & ~contact

            if contact[0] or potential_contact[0]:  # root joint
                lleg = self.physics_model.get_position(4) - self.physics_model.get_position(1)
                rleg = self.physics_model.get_position(5) - self.physics_model.get_position(2)

                ang_l = self._safe_leg_angle(lleg)
                ang_r = self._safe_leg_angle(rleg)

                # 角度算不出来（NaN）就别做这个判据
                if np.isfinite(ang_l) and np.isfinite(ang_r):
                    if min(ang_l, ang_r) < np.pi / 4:
                        contact[0], potential_contact[0] = False, False

            # explain residual force by contacts
            force, err, forceB = self._explain_residual_force(
                contact, cJ_cur, cjoint_cur, residual_force
            )
            for i in np.argsort(cjoint_cur[:, 1]):  # add potential contact from lowest to highest
                if err > 400 and potential_contact[i]:
                    contact[i] = True
                    force_new, err_new, forceB_new = self._explain_residual_force(
                        contact, cJ_cur, cjoint_cur, residual_force
                    )
                    self.contact_counter[i] = (
                        self.contact_counter[i] + 1 if err_new / err < 0.6 else 0
                    )
                    if self.contact_counter[i] >= 5:
                        force, err, forceB = force_new, err_new, forceB_new
                    else:
                        contact[i] = False
                else:
                    self.contact_counter[i] = 0

            # update contact position
            cjoint = cjoint_np.copy()
            near_ground = cjoint[:, 1] < self.floor_y + 0.15
            for i in np.where(contact & near_ground)[0]:
                cjoint[i, 1] = art.math.lerp(cjoint[i, 1], self.floor_y, 0.1)
            cjoint[cjoint[:, 1] < self.floor_y, 1] = self.floor_y

            # re-optimization and update state
            delta_tran = cjoint - cjoint_cur - cvel_cur * self.alpha_pd * self.dt
            rddotdes = (
                self.kp_tran * delta_tran - self.kd_tran * cvel_cur
            ).ravel() / (1 + self.kd_tran * self.alpha_pd * self.dt)
            rddotdes = self._sanitize_numpy(rddotdes, "rddotdes_stage2", clip_val=500.0)

            if np.any(contact):
                J = cJ_cur.reshape(5, 3, 75)[contact].reshape(-1, 75)
                B = art.math.block_diagonal_matrix_np(forceB)
                force_vec = B @ force
                torque = J.T @ force_vec
            else:
                torque = np.zeros(75)
            torque = self._sanitize_numpy(torque, "contact_torque", clip_val=1e5)

            A = np.vstack((
                np.hstack((np.zeros((72, 3)), np.eye(72))),
                np.sqrt(k.reshape(15, 1)) * cJ_cur,
                np.sqrt(self.beta_torque * 3) * M,
            ))
            b = np.concatenate((
                thetaddotdes,
                np.sqrt(k.reshape(15)) * (-cJdot_cur @ qdot + rddotdes),
                np.sqrt(self.beta_torque * 3) * (-h + torque),
            ))

            # ✅ 安全 lsqr（真正进入 update_state 的那次）
            qddot = self._safe_lsqr(A, b, name="qddot_stage2", clip_val=500.0, damp=1e-2)

            # 再兜底一层，防止任何非有限值流入 C++
            if not np.all(np.isfinite(qddot)):
                print("[warn] qddot non-finite before update_state, zeroing")
                qddot[:] = 0.0

            self.physics_model.update_state(qddot, self.dt)
            self.contact = contact

        
        refined_pose, refined_tran, qdot = self.physics_model.get_state_R()
        return torch.from_numpy(refined_pose), torch.from_numpy(refined_tran)

    @torch.no_grad()
    def forward_frame_fast(self, a, w, R):
        """
        A lower-latency path that keeps the learned pose/velocity estimation
        but skips the CPU-heavy physics optimization and contact solving.
        """
        imu_num = self.imu_num
        root_idx = imu_num - 1
        n_nonroot = imu_num - 1

        a = self._sanitize_torch(a.detach().to(torch.float32), "imu_acc", clip_val=100.0)
        w = self._sanitize_torch(w.detach().to(torch.float32), "imu_gyro", clip_val=100.0)
        R = self._sanitize_torch(R.detach().to(torch.float32), "imu_rot", clip_val=10.0)
        R = art.math.normalize_rotation_matrix(R)

        aRB = a.mm(R[root_idx])
        wRB = w.mm(R[root_idx])
        RRB = R[root_idx].t().matmul(R[:root_idx])
        gR0 = -R[root_idx, 1]

        x = torch.cat((aRB.reshape(-1), wRB.reshape(-1), RRB.reshape(-1), gR0), dim=0)
        x, self.pl1hc = self.plnet.rnn(x.view(1, 1, -1), self.pl1hc)
        x = self.plnet.linear2(x.squeeze())

        gR1 = art.math.normalize_tensor(x[15:])
        if not torch.isfinite(gR1).all():
            print("[warn] gR1 non-finite, fallback to gR0")
            gR1 = gR0.clone()
        RRB = art.math.from_to_rotation_matrix(gR0, gR1).matmul(RRB)

        x = torch.cat((RRB.reshape(-1), gR1, x[:15]), dim=0)
        x, self.ik1hc = self.iknet.net1.rnn(x.view(1, 1, -1), self.ik1hc)
        x = self.iknet.net1.linear2(x.squeeze())

        gR2 = art.math.normalize_tensor(x[69:])
        if not torch.isfinite(gR2).all():
            print("[warn] gR2 non-finite, fallback to gR1")
            gR2 = gR1.clone()
        RRB = art.math.from_to_rotation_matrix(gR1, gR2).matmul(RRB)

        x = torch.cat((RRB.reshape(-1), gR2, x[:69]), dim=0)
        x, self.ik2hc = self.iknet.net2.rnn(x.view(1, 1, -1), self.ik2hc)
        x = self.iknet.net2.linear2(x.squeeze())

        RRJ = art.math.r6d_to_rotation_matrix(x).cpu()
        RRJ = self._sanitize_torch(RRJ, "RRJ_fast", clip_val=10.0)
        RRJ = art.math.normalize_rotation_matrix(RRJ)
        glb_pose = torch.eye(3).repeat(1, 24, 1, 1)
        glb_pose[:, self.j_reduce] = RRJ.view(1, 15, 3, 3)

        pose = self.body_model.inverse_kinematics_R(glb_pose).view(24, 3, 3)
        pose = self._sanitize_torch(pose, "pose_ik_fast", clip_val=10.0)
        pose = art.math.normalize_rotation_matrix(pose)
        pose[self.j_ignore, ...] = torch.eye(3)

        pRJ = self.body_model.forward_kinematics(pose.unsqueeze(0))[1][0, 1:]
        pRJ = self._sanitize_torch(pRJ, "pRJ_fast", clip_val=10.0)

        R_align = art.math.from_to_rotation_matrix(gR2, gR0)
        if R_align.dim() == 3:
            R_align = R_align.squeeze(0)
        pose[0] = (R[root_idx] @ R_align).cpu()

        aRB2 = a.cpu().mm(pose[0])
        wRB2 = w.cpu().mm(pose[0])
        x = torch.cat((
            RRJ.reshape(-1),
            pRJ.reshape(-1),
            aRB2.reshape(-1),
            wRB2.reshape(-1),
            gR2.cpu(),
        ), dim=0).to(gR2.device)
        x, self.vr1hc = self.vrnet.rnn(x.view(1, 1, -1), self.vr1hc)
        x = self.vrnet.linear2(x.squeeze())

        vRR_V = x[0].item()
        vRR_H = x[1:4].cpu()
        stationary_prob = x[4:].sigmoid().cpu()

        vWR = pose[0].mm(vRR_H.unsqueeze(-1)).squeeze(-1)
        vWR[1] = vRR_V
        cjoint = torch.cat((torch.zeros(1, 3), pRJ.mm(pose[0].t())))[self.j_contact, :]
        stationary_weight = (stationary_prob * 5 - 3).clip(0, 1)
        vWR = self._sanitize_torch(vWR, "vWR_fast", clip_val=20.0)
        cjoint = self._sanitize_torch(cjoint, "cjoint_fast", clip_val=20.0)
        stationary_weight = self._sanitize_torch(stationary_weight, "stationary_weight_fast", clip_val=1.0)
        self.last_cjoint = self._sanitize_torch(self.last_cjoint, "last_cjoint_fast", clip_val=20.0)

        velocity = (
            stationary_weight.unsqueeze(0).mm(self.last_cjoint - cjoint)[0] / self.dt
            + self.beta_velocity * vWR
        ) / (self.beta_velocity + stationary_weight.sum())

        if not torch.isfinite(velocity).all():
            print("[warn] velocity non-finite in fast path, zeroing")
            velocity = torch.zeros_like(velocity)
        else:
            max_v = 10.0
            v_norm = velocity.norm().item()
            if v_norm > max_v:
                velocity = velocity * (max_v / (v_norm + 1e-8))

        self.last_cjoint = cjoint

        if not hasattr(self, "fast_tran"):
            self.fast_tran = torch.zeros(3)
            self.fast_initialized = False

        floor_tran_y = self.floor_y - cjoint[:, 1].min().item()
        if not self.fast_initialized:
            self.fast_tran = torch.tensor([0.0, floor_tran_y, 0.0], dtype=torch.float32)
            self.fast_initialized = True
        else:
            self.fast_tran = self.fast_tran + velocity * self.dt
            self.fast_tran[1] = max(self.fast_tran[1].item(), floor_tran_y)

        self.fast_tran = self._sanitize_torch(self.fast_tran, "fast_tran", clip_val=20.0)
        return pose, self.fast_tran.clone()
    @torch.no_grad()
    def forward_frame_fast0(self, a, w, R):
        """
        A lower-latency path that keeps the learned pose/velocity estimation
        but skips the CPU-heavy physics optimization and contact solving.
        """
        imu_num = self.imu_num
        root_idx = imu_num - 1
        n_nonroot = imu_num - 1

        a = self._sanitize_torch(a.detach().to(torch.float32), "imu_acc", clip_val=100.0)
        w = self._sanitize_torch(w.detach().to(torch.float32), "imu_gyro", clip_val=100.0)
        R = self._sanitize_torch(R.detach().to(torch.float32), "imu_rot", clip_val=10.0)
        R = art.math.normalize_rotation_matrix(R)

        aRB = a.mm(R[root_idx])
        wRB = w.mm(R[root_idx])
        RRB = R[root_idx].t().matmul(R[:root_idx])
        gR0 = -R[root_idx, 1]

        x = torch.cat((aRB.reshape(-1), wRB.reshape(-1), RRB.reshape(-1), gR0), dim=0)
        x, self.pl1hc = self.plnet.rnn(x.view(1, 1, -1), self.pl1hc)
        x = self.plnet.linear2(x.squeeze())

        gR1 = art.math.normalize_tensor(x[15:])
        if not torch.isfinite(gR1).all():
            print("[warn] gR1 non-finite, fallback to gR0")
            gR1 = gR0.clone()
        RRB = art.math.from_to_rotation_matrix(gR0, gR1).matmul(RRB)

        x = torch.cat((RRB.reshape(-1), gR1, x[:15]), dim=0)
        x, self.ik1hc = self.iknet.net1.rnn(x.view(1, 1, -1), self.ik1hc)
        x = self.iknet.net1.linear2(x.squeeze())

        gR2 = art.math.normalize_tensor(x[69:])
        if not torch.isfinite(gR2).all():
            print("[warn] gR2 non-finite, fallback to gR1")
            gR2 = gR1.clone()
        RRB = art.math.from_to_rotation_matrix(gR1, gR2).matmul(RRB)

        x = torch.cat((RRB.reshape(-1), gR2, x[:69]), dim=0)
        x, self.ik2hc = self.iknet.net2.rnn(x.view(1, 1, -1), self.ik2hc)
        x = self.iknet.net2.linear2(x.squeeze())

        RRJ = art.math.r6d_to_rotation_matrix(x).cpu()
        RRJ = self._sanitize_torch(RRJ, "RRJ_fast", clip_val=10.0)
        RRJ = art.math.normalize_rotation_matrix(RRJ)
        glb_pose = torch.eye(3).repeat(1, 24, 1, 1)
        glb_pose[:, self.j_reduce] = RRJ.view(1, 15, 3, 3)

        pose = self.body_model.inverse_kinematics_R(glb_pose).view(24, 3, 3)
        pose = self._sanitize_torch(pose, "pose_ik_fast", clip_val=10.0)
        pose = art.math.normalize_rotation_matrix(pose)
        pose[self.j_ignore, ...] = torch.eye(3)

        pRJ = self.body_model.forward_kinematics(pose.unsqueeze(0))[1][0, 1:]
        pRJ = self._sanitize_torch(pRJ, "pRJ_fast", clip_val=10.0)

        R_align = art.math.from_to_rotation_matrix(gR2, gR0)
        if R_align.dim() == 3:
            R_align = R_align.squeeze(0)
        pose[0] = (R[root_idx] @ R_align).cpu()

        aRB2 = a.cpu().mm(pose[0])
        wRB2 = w.cpu().mm(pose[0])
        x = torch.cat((
            RRJ.reshape(-1),
            pRJ.reshape(-1),
            aRB2.reshape(-1),
            wRB2.reshape(-1),
            gR2.cpu(),
        ), dim=0).to(gR2.device)
        x, self.vr1hc = self.vrnet.rnn(x.view(1, 1, -1), self.vr1hc)
        x = self.vrnet.linear2(x.squeeze())

        vRR_V = x[0].item()
        vRR_H = x[1:4].cpu()
        stationary_prob = x[4:].sigmoid().cpu()

        vWR = pose[0].mm(vRR_H.unsqueeze(-1)).squeeze(-1)
        vWR[1] = vRR_V
        cjoint = torch.cat((torch.zeros(1, 3), pRJ.mm(pose[0].t())))[self.j_contact, :]
        stationary_weight = (stationary_prob * 5 - 3).clip(0, 1)
        vWR = self._sanitize_torch(vWR, "vWR_fast", clip_val=20.0)
        cjoint = self._sanitize_torch(cjoint, "cjoint_fast", clip_val=20.0)
        stationary_weight = self._sanitize_torch(stationary_weight, "stationary_weight_fast", clip_val=1.0)
        self.last_cjoint = self._sanitize_torch(self.last_cjoint, "last_cjoint_fast", clip_val=20.0)

        velocity = (
            stationary_weight.unsqueeze(0).mm(self.last_cjoint - cjoint)[0] / self.dt
            + self.beta_velocity * vWR
        ) / (self.beta_velocity + stationary_weight.sum())

        if not torch.isfinite(velocity).all():
            print("[warn] velocity non-finite in fast path, zeroing")
            velocity = torch.zeros_like(velocity)
        else:
            max_v = 10.0
            v_norm = velocity.norm().item()
            if v_norm > max_v:
                velocity = velocity * (max_v / (v_norm + 1e-8))

        self.last_cjoint = cjoint

        if not hasattr(self, "fast_tran"):
            self.fast_tran = torch.zeros(3)
            self.fast_initialized = False

        floor_tran_y = self.floor_y - cjoint[:, 1].min().item()
        if not self.fast_initialized:
            self.fast_tran = torch.tensor([0.0, floor_tran_y, 0.0], dtype=torch.float32)
            self.fast_initialized = True
        else:
            self.fast_tran = self.fast_tran + velocity * self.dt
            self.fast_tran[1] = max(self.fast_tran[1].item(), floor_tran_y)

        self.fast_tran = self._sanitize_torch(self.fast_tran, "fast_tran", clip_val=20.0)
        return pose, self.fast_tran.clone()

    def forward(self, x, fast=True):
        imu_num = self.imu_num 
        n_nonroot = imu_num - 1

        a_dim = imu_num * 3
        w_dim = imu_num * 3
        aw_dim = a_dim + w_dim              # 60
        rrb_dim = n_nonroot * 9             # 81

        # ---- parse input ----
        # x: list of (x_, y_) with x_ shape [T, 144]
        RRB = [x_[:, aw_dim:aw_dim + rrb_dim].view(-1, n_nonroot, 3, 3) for x_, y_ in x]
        gR0 = [x_[:, aw_dim + rrb_dim:aw_dim + rrb_dim + 3] for x_, y_ in x]

        # PL-s1 (label前18维：pRB(15) + gR(3) 没变)
        x1 = [(x_, y_[:18]) for x_, y_ in x]
        x1 = self.plnet(x1)  # -> [pRB(15), gR(3)]

        # IK-s1
        pRB = [x_[:, :15] for x_ in x1]
        gR1 = [art.math.normalize_tensor(x_[:, 15:].clone().detach()) for x_ in x1]
        RRB = [art.math.from_to_rotation_matrix(gR0_, gR1_).unsqueeze(1).matmul(RRB_)
            for gR0_, gR1_, RRB_ in zip(gR0, gR1, RRB)]
        x2 = [torch.cat((RRB_.flatten(1), gR1_, pRB_), dim=1) for RRB_, gR1_, pRB_ in zip(RRB, gR1, pRB)]
        x2 = self.iknet.net1(x2)  # -> [pRJ(69), gR(3)]

        # IK-s2
        pRJ = [x_[:, :69] for x_ in x2]
        gR2 = [art.math.normalize_tensor(x_[:, 69:].clone().detach()) for x_ in x2]
        RRB = [art.math.from_to_rotation_matrix(gR1_, gR2_).unsqueeze(1).matmul(RRB_)
            for gR1_, gR2_, RRB_ in zip(gR1, gR2, RRB)]
        x3 = [torch.cat((RRB_.flatten(1), gR2_, pRJ_), dim=1) for RRB_, gR2_, pRJ_ in zip(RRB, gR2, pRJ)]
        x3 = self.iknet.net2(x3)  # -> RRJ r6d (15*6=90)

        # VR-s1
        if fast:
            RRJ = [art.math.r6d_to_rotation_matrix(x3_.detach()).view(-1, 135) for x3_ in x3]

            # 原来 36 -> 60； 12 -> 20（a+w 共 2*imu_num 个 3D 向量）
            awRB = [
                x_[:, :aw_dim].view(-1, 2 * imu_num, 3)
                    .bmm(art.math.from_to_rotation_matrix(gR2_, gR0_))
                    .view(-1, aw_dim)
                for gR2_, gR0_, (x_, y_) in zip(gR2, gR0, x)
            ]

            x4 = [
                (
                    torch.cat((RRJ_, pRJ_.detach(), awRB_, gR2_), dim=1),
                    torch.zeros_like(y_[-9:]) if torch.isnan(y_[-9]) else y_[-9:]
                )
                for RRJ_, pRJ_, awRB_, gR2_, (x_, y_) in zip(RRJ, pRJ, awRB, gR2, x)
            ]
        else:
            RRJ, pRJ2, aRB_new, wRB_new, pose = [], [], [], [], []

            # 原来 6 -> 10；18/36 -> 30/60
            aRB = [x_[:, :a_dim].view(-1, imu_num, 3) for x_, y_ in x]
            wRB = [x_[:, a_dim:aw_dim].view(-1, imu_num, 3) for x_, y_ in x]

            for i in range(len(x3)):
                x3_ = x3[i].clone().detach().cpu()
                RRJ_ = art.math.r6d_to_rotation_matrix(x3_).view(-1, 15, 3, 3)

                glb_pose_ = torch.eye(3).repeat(RRJ_.shape[0], 24, 1, 1)
                glb_pose_[:, self.j_reduce] = RRJ_
                pose_ = self.body_model.inverse_kinematics_R(glb_pose_).view(-1, 24, 3, 3)
                pose_[:, self.j_ignore] = torch.eye(3)

                pRJ_ = self.body_model.forward_kinematics(pose_)[1][:, 1:]
                aRB_ = aRB[i].bmm(art.math.from_to_rotation_matrix(gR2[i], gR0[i]))
                wRB_ = wRB[i].bmm(art.math.from_to_rotation_matrix(gR2[i], gR0[i]))

                RRJ.append(RRJ_)
                pRJ2.append(pRJ_)
                aRB_new.append(aRB_)
                wRB_new.append(wRB_)
                pose.append(pose_)

            no_translation = [torch.isnan(y_[-9]).item() for x_, y_ in x]
            x4 = [
                (
                    torch.cat((RRJ_.flatten(1).to(aRB_.device),
                            pRJ_.flatten(1).to(aRB_.device),
                            aRB_.flatten(1), wRB_.flatten(1),
                            gR2_), dim=1),
                    x_[1][-9:] if not nt_ else torch.zeros_like(x_[1][-9:])
                )
                for RRJ_, pRJ_, aRB_, wRB_, gR2_, x_, nt_ in zip(RRJ, pRJ2, aRB_new, wRB_new, gR2, x, no_translation)
            ]

        x4 = self.vrnet(x4)  # -> vRR_V, vRR_H, stationary_prob
        result = [torch.cat((x1_, x2_, x3_, x4_), dim=1) for x1_, x2_, x3_, x4_ in zip(x1, x2, x3, x4)]
        return result
    



class Full_GR_OV_6(torch.nn.Module):
    dt = 1 / 60
    mu = 0.7           # environment fiction coefficient
    kp_pose = 3600     # kp in pose PD controller
    kd_pose = 60       # kd in pose PD controller
    kp_tran = 3600     # kp in tran PD controller
    kd_tran = 60       # kd in tran PD controller
    alpha_pd = 0.0     # relaxation in stable PD controller
    floor_y = -0.97    # floor height
    beta_velocity = 1
    beta_cjoint = 1
    beta_extforce = 0.4
    beta_torque = 1e-3 / 80
    v_imu = (1961, 5424, 1176, 4662, 411, 3021)
    j_reduce = (1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19)
    j_ignore = (0, 7, 8, 10, 11, 20, 21, 22, 23)
    j_contact = (0, 10, 11, 22, 23)

    # j_imu = [18, 19, 4, 5, 12, 0]
    v_imu = (1961, 5424, 3187, 6585, 1305, 4298)

    class Visualization:
        enable = False
        show_residual_force = False
        show_contact_force = True
        show_block = True
        show_contact = False
        show_stationary = False
        show_torque = False

    def __init__(self):
        from articulate.utils.torch import RNN, RNNWithInit
        super(Full_GR_OV_6, self).__init__()
        self.plnet = RNNWithInit(input_linear=False,
                                 input_size=84,
                                 output_size=18,
                                 hidden_size=512,
                                 num_rnn_layer=3,
                                 dropout=0.4)
        self.iknet = torch.nn.ModuleDict({
            'net1': RNN(input_linear=False,
                        input_size=63,
                        output_size=72,
                        hidden_size=512,
                        num_rnn_layer=3,
                        dropout=0.4),
            'net2': RNN(input_linear=False,
                        input_size=117,
                        output_size=90,
                        hidden_size=512,
                        num_rnn_layer=3,
                        dropout=0.4)
        })
        self.vrnet = RNNWithInit(input_linear=False,
                                 input_size=243,
                                 output_size=9,
                                 hidden_size=512,
                                 num_rnn_layer=3,
                                 dropout=0.4)

        # for training
        # self.plnet.load_state_dict(torch.load('data/weight_6/Pose-GR/PL/best_weights.pt'))
        # self.iknet.load_state_dict(torch.load('data/weight_6/Pose-GR/IK/best_weights.pt'))
        # self.vrnet.load_state_dict(torch.load('data/weight_6/Tran-OV/VR/best_weights.pt'))

        # for testing
        # self.load_state_dict(torch.load(r'C:\Users\15482\Desktop\GlobalPose\data\weight_6\Full-GR-OV\full\weights.pt'))
        # self.load_state_dict(torch.load(r'C:\Users\15482\Desktop\GlobalPose\data\weights_6_finetune_with_loose_data\weights.pt'))
        self.load_state_dict(torch.load(r'C:\Users\15482\Desktop\GlobalPose\data\weight_6_loose_e2e\weights.pt'))

        self.B = np.array([[self.mu, -self.mu, 0,       0       ],
                           [1,       1,        1,       1       ],
                           [0,       0,        self.mu, -self.mu]]) / np.sqrt(1 + self.mu ** 2)  # basis of friction cone
        self.body_model = art.ParametricModel('models/SMPL_male.pkl', vert_mask=self.v_imu)
        self.physics_model = cart.get_dynamic_model('models/SMPL_male.pkl')
        # self.rnn_initialize()  # using T-pose
        self.eval()

        if self.Visualization.enable:
            from articulate.utils.unity import MotionViewer
            MotionViewer.colors = [(1, 1, 1)]
            self.viewer = MotionViewer(1)
            self.viewer.connect()
            self.scene = Scene(self.viewer)
            self.force_lpf = [art.LowPassFilter(0.3) for _ in range(5)]
            self.torque_lpf = art.LowPassFilter(0.3)
            self.tran_offset = torch.zeros(3)
            if self.Visualization.show_torque:
                self.viewer.show_torque(0, [1, 2, 4, 5, 16, 17, 18, 19])

    @torch.no_grad()
    def rnn_initialize(self, init_pose=None, init_vel=None):
        r"""
        Initialize the hidden states of the RNNs.

        :param init_pose: Pose in shape [24, 3, 3]. T-pose by default.
        :param init_vel: Root world-space velocity in shape [3]. Zero by default.
        """
        init_pose = torch.eye(3).expand(1, 24, 3, 3) if init_pose is None else init_pose.cpu().view(1, 24, 3, 3)
        init_vel = torch.zeros(3) if init_vel is None else init_vel.cpu().view(3)
        vRR_V = init_vel[1].view(1).clone()
        init_vel[1] = 0
        _, j, v = self.body_model.forward_kinematics(init_pose, calc_mesh=True)
        pRL, gR = (v[0, :5] - v[0, 5:]).mm(init_pose[0, 0]).ravel(), -init_pose[0, 0, 1]
        x1 = torch.cat((pRL, gR)).to(self.plnet.init_net[0].weight.device)
        h, vRR_H, c = -j[:, :, 1].min().view(1), init_pose[0, 0].t().mm(init_vel.unsqueeze(-1)).squeeze(-1), torch.zeros(5)
        x2 = torch.cat((vRR_V, vRR_H, c)).to(self.vrnet.init_net[0].weight.device)
        self.pl1hc = [_.contiguous() for _ in self.plnet.init_net(x1).view(1, 2, self.plnet.num_layers, self.plnet.hidden_size).permute(1, 2, 0, 3)]
        self.vr1hc = [_.contiguous() for _ in self.vrnet.init_net(x2).view(1, 2, self.vrnet.num_layers, self.vrnet.hidden_size).permute(1, 2, 0, 3)]
        self.ik1hc = None
        self.ik2hc = None
        self.last_cjoint = torch.tensor([0, h + self.floor_y, 0]) + j[0, self.j_contact]
        self.is_init = False
        self.contact = np.zeros(5, dtype=bool)
        self.contact_counter = np.zeros(5, dtype=int)

    def _safe_lsqr(self, A, b, name="qddot", clip_val=500.0, damp=1e-2, iters=100):
        A = np.asarray(A, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)

        if not np.all(np.isfinite(A)):
            print(f"[safe_lsqr] {name} A non-finite, sanitizing")
            A = np.nan_to_num(A, nan=0.0, posinf=clip_val, neginf=-clip_val)

        if not np.all(np.isfinite(b)):
            print(f"[safe_lsqr] {name} b non-finite, sanitizing")
            b = np.nan_to_num(b, nan=0.0, posinf=clip_val, neginf=-clip_val)

        sol, *_ = lsqr(csc_array(A), b, damp=damp, iter_lim=iters)
        sol = np.asarray(sol, dtype=np.float64)

        if not np.all(np.isfinite(sol)):
            print(f"[safe_lsqr] {name} non-finite, zeroed")
            sol[:] = 0.0
            return sol

        np.clip(sol, -clip_val, clip_val, out=sol)
        return sol

    def _sanitize_torch(self, x, name, clip_val=None):
        if torch.isfinite(x).all():
            return x
        print(f"[warn] {name} non-finite, sanitizing")
        pos = clip_val if clip_val is not None else 0.0
        neg = -clip_val if clip_val is not None else 0.0
        x = torch.nan_to_num(x, nan=0.0, posinf=pos, neginf=neg)
        if clip_val is not None:
            x = x.clamp(-clip_val, clip_val)
        return x

    def _sanitize_numpy(self, x, name, clip_val=None):
        x = np.asarray(x, dtype=np.float64)
        if np.all(np.isfinite(x)):
            return x
        print(f"[warn] {name} non-finite, sanitizing")
        pos = clip_val if clip_val is not None else 0.0
        neg = -clip_val if clip_val is not None else 0.0
        x = np.nan_to_num(x, nan=0.0, posinf=pos, neginf=neg)
        if clip_val is not None:
            np.clip(x, -clip_val, clip_val, out=x)
        return x

    @torch.no_grad()
    def _explain_residual_force(self, contact, contact_Jacobian, contact_position, residual_force):
        if np.any(contact):
            J = contact_Jacobian.reshape(5, 3, 75)[contact, :, :6].reshape(-1, 6)
            B, lb = [], []
            for i in np.where(contact)[0]:
                if i > 3 and contact_position[i, 1] > self.floor_y + 0.15:      # hand grasp
                    B.append(np.eye(3))
                    lb.append(-np.ones(3) * np.inf)
                else:
                    B.append(self.B)
                    lb.append(np.zeros(4))
            JTB = J.T @ art.math.block_diagonal_matrix_np(B)
            P = csc_array(JTB.T @ JTB + self.beta_extforce * np.eye(JTB.shape[1]))
            q = -JTB.T @ residual_force
            force = solve_qp(P, q, lb=np.concatenate(lb), solver='osqp')
            if force is None:
                force = np.zeros(0)
                error = np.linalg.norm(residual_force)
            else:
                error = np.linalg.norm(JTB @ force - residual_force)
        else:
            B = None
            force = np.zeros(0)
            error = np.linalg.norm(residual_force)
        return force, error, B

    @torch.no_grad()
    def forward_frame(self, a, w, R):
        a = self._sanitize_torch(a.detach().to(torch.float32), "imu_acc_6", clip_val=100.0)
        w = self._sanitize_torch(w.detach().to(torch.float32), "imu_gyro_6", clip_val=100.0)
        R = self._sanitize_torch(R.detach().to(torch.float32), "imu_rot_6", clip_val=10.0)

        aRB = a.mm(R[5])
        wRB = w.mm(R[5])
        RRB = R[5].t().matmul(R[:5])
        gR0 = -R[5, 1]

        # PL-s1
        x = torch.cat((aRB.ravel(), wRB.ravel(), RRB.ravel(), gR0))
        x, self.pl1hc = self.plnet.rnn(x.view(1, 1, -1), self.pl1hc)
        x = self.plnet.linear2(x.squeeze())        # pRB, gR
        gR1 = self._sanitize_torch(art.math.normalize_tensor(x[15:]), "gR1_6", clip_val=10.0)
        RRB = art.math.from_to_rotation_matrix(gR0, gR1).matmul(RRB)

        # IK-s1
        x = torch.cat((RRB.ravel(), gR1, x[:15]))
        x, self.ik1hc = self.iknet.net1.rnn(x.view(1, 1, -1), self.ik1hc)
        x = self.iknet.net1.linear2(x.squeeze())   # pRJ, gR
        gR2 = self._sanitize_torch(art.math.normalize_tensor(x[69:]), "gR2_6", clip_val=10.0)
        RRB = art.math.from_to_rotation_matrix(gR1, gR2).matmul(RRB)

        # IK-s2
        x = torch.cat((RRB.ravel(), gR2, x[:69]))
        x, self.ik2hc = self.iknet.net2.rnn(x.view(1, 1, -1), self.ik2hc)
        x = self.iknet.net2.linear2(x.squeeze())   # RRJ

        # get pose estimation
        RRJ = self._sanitize_torch(art.math.r6d_to_rotation_matrix(x).cpu(), "RRJ_6", clip_val=10.0)
        glb_pose = torch.eye(3).repeat(1, 24, 1, 1)
        glb_pose[:, self.j_reduce] = RRJ.view(1, 15, 3, 3)
        pose = self._sanitize_torch(self.body_model.inverse_kinematics_R(glb_pose).view(24, 3, 3), "pose_ik_6", clip_val=10.0)
        pose[self.j_ignore, ...] = torch.eye(3)
        pRJ = self._sanitize_torch(self.body_model.forward_kinematics(pose.unsqueeze(0))[1][0, 1:], "pRJ_6", clip_val=10.0)
        pose[0] = R[5].mm(art.math.from_to_rotation_matrix(gR2, gR0).squeeze()).cpu()

        # VR-s1
        aRB = a.cpu().mm(pose[0])
        wRB = w.cpu().mm(pose[0])
        x = torch.cat((RRJ.ravel(), pRJ.ravel(), aRB.ravel(), wRB.ravel(), gR2.cpu())).to(gR2.device)
        x, self.vr1hc = self.vrnet.rnn(x.view(1, 1, -1), self.vr1hc)
        x = self.vrnet.linear2(x.squeeze())  # vRR_V, vRR_H, stationary_prob

        # get translation estimation
        vRR_V, vRR_H, stationary_prob = x[0].item(), x[1:4].cpu(), x[4:].sigmoid().cpu()
        vWR = self._sanitize_torch(pose[0].mm(vRR_H.unsqueeze(-1)).squeeze(-1), "vWR_6", clip_val=20.0)
        vWR[1] = vRR_V
        cjoint = self._sanitize_torch(torch.cat((torch.zeros(1, 3), pRJ.mm(pose[0].t())))[self.j_contact, :], "cjoint_6", clip_val=20.0)
        stationary_weight = self._sanitize_torch((stationary_prob * 5 - 3).clip(0, 1), "stationary_weight_6", clip_val=1.0)
        self.last_cjoint = self._sanitize_torch(self.last_cjoint, "last_cjoint_6", clip_val=20.0)
        velocity = (stationary_weight.unsqueeze(0).mm(self.last_cjoint - cjoint)[0] / self.dt + self.beta_velocity * vWR) / (self.beta_velocity + stationary_weight.sum())
        self.last_cjoint = cjoint

        # physics optimization
        if not self.is_init:
            self.is_init = True
            self.physics_model.set_state_R(pose.numpy(), np.array([0, self.floor_y - cjoint[:, 1].min().item(), 0]), np.zeros(75))
        else:
            # get current state
            pose_cur, tran_cur, qdot = self.physics_model.get_state_R()
            cjoint_cur = np.vstack([self.physics_model.get_position(j) for j in self.j_contact])
            cvel_cur = np.vstack([self.physics_model.get_linear_velocity(j) for j in self.j_contact])
            cJ_cur = np.vstack([self.physics_model.get_linear_Jacobian(j) for j in self.j_contact])
            cJdot_cur = np.vstack([self.physics_model.get_linear_Jacobian_dot(j) for j in self.j_contact])
            M = self.physics_model.mass_matrix()
            h = self.physics_model.inverse_dynamics(np.zeros(75))
            stationary = stationary_prob.numpy() > 0.7
            qdot = self._sanitize_numpy(qdot, "qdot_6", clip_val=500.0)
            pose_cur = self._sanitize_numpy(pose_cur, "pose_cur_6", clip_val=10.0)
            tran_cur = self._sanitize_numpy(tran_cur, "tran_cur_6", clip_val=20.0)
            cjoint_cur = self._sanitize_numpy(cjoint_cur, "cjoint_cur_6", clip_val=20.0)
            cvel_cur = self._sanitize_numpy(cvel_cur, "cvel_cur_6", clip_val=50.0)
            cJ_cur = self._sanitize_numpy(cJ_cur, "cJ_cur_6", clip_val=1e3)
            cJdot_cur = self._sanitize_numpy(cJdot_cur, "cJdot_cur_6", clip_val=1e3)
            M = self._sanitize_numpy(M, "mass_matrix_6", clip_val=1e5)
            h = self._sanitize_numpy(h, "inverse_dynamics_6", clip_val=1e5)

            # dual RSPD controller
            # R = art.math.axis_angle_to_rotation_matrix(torch.from_numpy(qdot[3:]) * self.alpha_pd * self.dt)
            # delta_pose = art.math.rotation_matrix_to_axis_angle(torch.from_numpy(pose_cur).bmm(R).transpose(1, 2).bmm(pose)).ravel().numpy()
            pose_dtype = pose.dtype
            qdot_t = torch.from_numpy(qdot[3:]).to(dtype=pose_dtype)
            pose_cur_t = torch.from_numpy(pose_cur).to(dtype=pose_dtype)
            R = art.math.axis_angle_to_rotation_matrix(qdot_t * self.alpha_pd * self.dt)
            delta_pose = art.math.rotation_matrix_to_axis_angle(
                    pose_cur_t.bmm(R).transpose(1, 2).bmm(pose)
                ).ravel().numpy()

            delta_pose = self._sanitize_numpy(delta_pose, "delta_pose_6", clip_val=10.0)
            thetaddotdes = (self.kp_pose * delta_pose - self.kd_pose * qdot[3:]) / (1 + self.kd_pose * self.alpha_pd * self.dt)
            thetaddotdes = self._sanitize_numpy(thetaddotdes, "thetaddotdes_6", clip_val=500.0)
            cjoint = tran_cur + velocity.numpy() * self.dt + cjoint.numpy()
            cjoint = self._sanitize_numpy(cjoint, "cjoint_target_6", clip_val=20.0)
            cjoint = art.math.lerp(cjoint, cjoint_cur, stationary_weight.view(5, 1).numpy())
            cjoint = self._sanitize_numpy(cjoint, "cjoint_lerp_6", clip_val=20.0)
            delta_tran = cjoint - cjoint_cur - cvel_cur * self.alpha_pd * self.dt
            rddotdes = (self.kp_tran * delta_tran - self.kd_tran * cvel_cur).ravel() / (1 + self.kd_tran * self.alpha_pd * self.dt)
            rddotdes = self._sanitize_numpy(rddotdes, "rddotdes_stage1_6", clip_val=500.0)

            # unconstrained tracking
            cjoint_cur[0, 1] -= 0.15
            cjoint[0, 1] -= 0.15
            k = np.ones((5, 3)) * self.beta_cjoint
            k[self.contact] *= 10
            A = np.vstack((np.hstack((np.zeros((72, 3)), np.eye(72))), np.sqrt(k.reshape(15, 1)) * cJ_cur, np.sqrt(self.beta_torque) * M))
            b = np.concatenate((thetaddotdes, np.sqrt(k.reshape(15)) * (-cJdot_cur @ qdot + rddotdes), np.sqrt(self.beta_torque) * (-h)))
            qddot = self._safe_lsqr(A, b, name="qddot_stage1_6", clip_val=500.0, damp=1e-2)
            residual_force = M[:6] @ qddot + h[:6]
            residual_force = self._sanitize_numpy(residual_force, "residual_force_6", clip_val=1e5)

            # determine potential contact
            vdist = np.abs(cjoint_cur[np.newaxis, :, 1] - cjoint_cur[:, np.newaxis, 1])
            contact = stationary & (self.contact | (cjoint_cur[:, 1] < self.floor_y + 0.05))
            if np.any(contact):
                contact |= stationary & (vdist[contact].min(axis=0) < 0.05)
            potential_contact = stationary & ~contact
            if contact[0] or potential_contact[0]:  # root joint
                lleg = self.physics_model.get_position(4) - self.physics_model.get_position(1)
                rleg = self.physics_model.get_position(5) - self.physics_model.get_position(2)
                lleg_norm = np.linalg.norm(lleg)
                rleg_norm = np.linalg.norm(rleg)
                if lleg_norm > 1e-8 and rleg_norm > 1e-8:
                    lcos = np.clip(-lleg[1] / lleg_norm, -1.0, 1.0)
                    rcos = np.clip(-rleg[1] / rleg_norm, -1.0, 1.0)
                    if min(np.arccos(lcos), np.arccos(rcos)) < np.pi / 4:
                        contact[0], potential_contact[0] = False, False

            # explain residual force by contacts
            force, err, forceB = self._explain_residual_force(contact, cJ_cur, cjoint_cur, residual_force)
            for i in np.argsort(cjoint_cur[:, 1]):  # add potential contact from lowest to highest
                if err > 400 and potential_contact[i]:
                    contact[i] = True
                    force_new, err_new, forceB_new = self._explain_residual_force(contact, cJ_cur, cjoint_cur, residual_force)
                    self.contact_counter[i] = self.contact_counter[i] + 1 if err_new / err < 0.6 else 0
                    if self.contact_counter[i] >= 5:
                        force, err, forceB = force_new, err_new, forceB_new
                    else:
                        contact[i] = False
                else:
                    self.contact_counter[i] = 0

            # update contact position
            near_ground = cjoint[:, 1] < self.floor_y + 0.15
            # object_y = []
            # for i in np.where(contact & ~near_ground)[0]:
            #     is_added = False
            #     for y in object_y:
            #         if not is_added and abs(sum(y) / len(y) - cjoint[i, 1]) < 0.05:
            #             y.append(cjoint[i, 1])
            #             is_added = True
            #     if not is_added:
            #         object_y.append([cjoint[i, 1]])
            # object_y = np.array([sum(y) / len(y) for y in object_y])
            # for i in np.where(contact)[0]:
            #     cjoint[i, 1] = art.math.lerp(cjoint[i, 1], self.floor_y, 0.1) if near_ground[i] else object_y[np.abs(cjoint[i, 1] - object_y).argmin()]
            for i in np.where(contact & near_ground)[0]:
                cjoint[i, 1] = art.math.lerp(cjoint[i, 1], self.floor_y, 0.1)
            cjoint[cjoint[:, 1] < self.floor_y, 1] = self.floor_y

            # re-optimization and update state
            delta_tran = cjoint - cjoint_cur - cvel_cur * self.alpha_pd * self.dt
            rddotdes = (self.kp_tran * delta_tran - self.kd_tran * cvel_cur).ravel() / (1 + self.kd_tran * self.alpha_pd * self.dt)
            rddotdes = self._sanitize_numpy(rddotdes, "rddotdes_stage2_6", clip_val=500.0)
            if np.any(contact):
                expected_force_dim = sum(b.shape[1] for b in forceB) if forceB is not None else 0
                if force.shape[0] == expected_force_dim and expected_force_dim > 0:
                    J = cJ_cur.reshape(5, 3, 75)[contact].reshape(-1, 75)
                    B = art.math.block_diagonal_matrix_np(forceB)
                    force = B @ force
                    torque = J.T @ force
                    torque = self._sanitize_numpy(torque, "contact_torque_6", clip_val=1e5)
                else:
                    # If the QP solver failed to produce a valid force vector, skip
                    # contact torque for this frame instead of crashing.
                    contact[:] = False
                    force = np.zeros(0)
                    torque = np.zeros(75)
            else:
                torque = np.zeros(75)
            A = np.vstack((np.hstack((np.zeros((72, 3)), np.eye(72))), np.sqrt(k.reshape(15, 1)) * cJ_cur, np.sqrt(self.beta_torque * 3) * M))
            b = np.concatenate((thetaddotdes, np.sqrt(k.reshape(15)) * (-cJdot_cur @ qdot + rddotdes), np.sqrt(self.beta_torque * 3) * (-h + torque)))
            qddot = self._safe_lsqr(A, b, name="qddot_stage2_6", clip_val=500.0, damp=1e-2)
            qddot = self._sanitize_numpy(qddot, "qddot_stage2_6_final", clip_val=500.0)
            self.physics_model.update_state(qddot, self.dt)
            self.contact = contact

            # visualization
            if self.Visualization.enable:
                force = iter(force.reshape(-1, 3))
                self.viewer.clear_line(render=False)
                self.viewer.clear_point(render=False)
                self.viewer.update(torch.from_numpy(pose_cur), torch.from_numpy(tran_cur) - self.tran_offset, render=False)
                cjoint -= self.tran_offset.numpy()
                for i in np.where(contact)[0]:
                    if self.Visualization.show_block and not near_ground[i]:
                        if i < 3 or True:   # always show block
                            self.scene.add_block(Scene.SupportBlock, cjoint[i] - [0, 0.2, 0], lifetime=120, render=False)
                        else:
                            self.scene.add_block(Scene.GraspBlock, cjoint[i], lifetime=30, render=False)
                    if self.Visualization.show_contact_force:
                        self.viewer.draw_line(cjoint[i], cjoint[i] + self.force_lpf[i](next(force)) * 0.002, [1, 0, 0], 0.02, render=False)
                    if self.Visualization.show_contact:
                        self.viewer.draw_point(cjoint[i], [0, 1, 0], 0.15, render=False)
                if self.Visualization.show_stationary:
                    for i in np.where(stationary & ~contact)[0]:
                        self.viewer.draw_point(cjoint[i], [0.4, 0.4, 1], 0.15, render=False)
                if self.Visualization.show_block:
                    self.scene.update(render=False)
                if self.Visualization.show_residual_force:
                    self.viewer.draw_line_from_joint(0, 0, residual_force[:3] * 0.001, (0, 0, 1), 0.02, render=False)
                if self.Visualization.show_torque:
                    self.viewer.show_torque(0, [1, 2, 4, 5, 16, 17, 18, 19], render=False)
                    tau = M @ qddot + h - torque
                    tau = self.torque_lpf(tau)
                    self.viewer.update_torque(tau[3:], render=False)
                else:
                    self.viewer.hide_torque(render=False)
                self.viewer.render()

        refined_pose, refined_tran, qdot = self.physics_model.get_state_R()
        return torch.from_numpy(refined_pose), torch.from_numpy(refined_tran)

    def forward(self, x, fast=True):
        # PL-s1
        RRB = [x_[:, 36:81].view(-1, 5, 3, 3) for x_, y_ in x]
        gR0 = [x_[:, 81:] for x_, y_ in x]
        x1 = [(x_, y_[:18]) for x_, y_ in x]
        x1 = self.plnet(x1)       # pRB, gR

        # IK-s1
        pRB = [x_[:, :15] for x_ in x1]
        gR1 = [art.math.normalize_tensor(x_[:, 15:].clone().detach()) for x_ in x1]
        RRB = [art.math.from_to_rotation_matrix(gR0_, gR1_).unsqueeze(1).matmul(RRB_) for gR0_, gR1_, RRB_ in zip(gR0, gR1, RRB)]
        x2 = [torch.cat((RRB_.flatten(1), gR1_, pRB_), dim=1) for RRB_, gR1_, pRB_ in zip(RRB, gR1, pRB)]
        x2 = self.iknet.net1(x2)   # pRJ, gR

        # IK-s2
        pRJ = [x_[:, :69] for x_ in x2]
        gR2 = [art.math.normalize_tensor(x_[:, 69:].clone().detach()) for x_ in x2]
        RRB = [art.math.from_to_rotation_matrix(gR1_, gR2_).unsqueeze(1).matmul(RRB_) for gR1_, gR2_, RRB_ in zip(gR1, gR2, RRB)]
        x3 = [torch.cat((RRB_.flatten(1), gR2_, pRJ_), dim=1) for RRB_, gR2_, pRJ_ in zip(RRB, gR2, pRJ)]
        x3 = self.iknet.net2(x3)   # RRJ

        # VR-s1
        if fast:   # faster approximation
            RRJ = [art.math.r6d_to_rotation_matrix(x3_.detach()).view(-1, 135) for x3_ in x3]
            awRB = [x_[:, :36].view(-1, 12, 3).bmm(art.math.from_to_rotation_matrix(gR2_, gR0_)).view(-1, 36) for gR2_, gR0_, (x_, y_) in zip(gR2, gR0, x)]
            x4 = [(torch.cat((RRJ_, pRJ_.detach(), awRB_, gR2_), dim=1), torch.zeros_like(y_[-9:]) if torch.isnan(y_[-9]) else y_[-9:]) for RRJ_, pRJ_, awRB_, gR2_, (x_, y_) in zip(RRJ, pRJ, awRB, gR2, x)]
        else:
            RRJ, pRJ, aRB_new, wRB_new, pose = [], [], [], [], []
            aRB = [x_[:, :18].view(-1, 6, 3) for x_, y_ in x]
            wRB = [x_[:, 18:36].view(-1, 6, 3) for x_, y_ in x]
            for i in range(len(x3)):
                x3_ = x3[i].clone().detach().cpu()
                RRJ_ = art.math.r6d_to_rotation_matrix(x3_).view(-1, 15, 3, 3)
                glb_pose_ = torch.eye(3).repeat(RRJ_.shape[0], 24, 1, 1)
                glb_pose_[:, self.j_reduce] = RRJ_
                pose_ = self.body_model.inverse_kinematics_R(glb_pose_).view(-1, 24, 3, 3)
                pose_[:, self.j_ignore] = torch.eye(3)
                pRJ_ = self.body_model.forward_kinematics(pose_)[1][:, 1:]
                aRB_ = aRB[i].bmm(art.math.from_to_rotation_matrix(gR2[i], gR0[i]))
                wRB_ = wRB[i].bmm(art.math.from_to_rotation_matrix(gR2[i], gR0[i]))
                RRJ.append(RRJ_)
                pRJ.append(pRJ_)
                aRB_new.append(aRB_)
                wRB_new.append(wRB_)
                pose.append(pose_)
            no_translation = [torch.isnan(y_[-9]).item() for x_, y_ in x]
            x4 = [(torch.cat((RRJ_.flatten(1).to(aRB_.device), pRJ_.flatten(1).to(aRB_.device), aRB_.flatten(1), wRB_.flatten(1), gR2_), dim=1), x_[1][-9:] if not nt_ else torch.zeros_like(x_[1][-9:])) for RRJ_, pRJ_, aRB_, wRB_, gR2_, x_, nt_ in zip(RRJ, pRJ, aRB_new, wRB_new, gR2, x, no_translation)]
        x4 = self.vrnet(x4)   # vRR_V, vRR_H, stationary_prob
        result = [torch.cat((x1_, x2_, x3_, x4_), dim=1) for x1_, x2_, x3_, x4_ in zip(x1, x2, x3, x4)]
        return result



class Full_GR_OV_4(torch.nn.Module):
    dt = 1 / 60
    mu = 0.7           # environment fiction coefficient
    kp_pose = 3600     # kp in pose PD controller
    kd_pose = 60       # kd in pose PD controller
    kp_tran = 3600     # kp in tran PD controller
    kd_tran = 60       # kd in tran PD controller
    alpha_pd = 0.0     # relaxation in stable PD controller
    floor_y = -0.97    # floor height
    beta_velocity = 1
    beta_cjoint = 1
    beta_extforce = 0.4
    beta_torque = 1e-3 / 80
    v_imu = (1961, 5424, 1176, 4662, 411, 3021)
    j_reduce = (1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19)
    j_ignore = (0, 7, 8, 10, 11, 20, 21, 22, 23)
    j_contact = (0, 10, 11, 22, 23)

    # j_imu = [18, 19, 4, 5, 12, 0]
    v_imu = (1961, 5424, 1305, 3021)

    class Visualization:
        enable = False
        show_residual_force = False
        show_contact_force = True
        show_block = True
        show_contact = False
        show_stationary = False
        show_torque = False

    def __init__(self):
        from articulate.utils.torch import RNN, RNNWithInit
        super(Full_GR_OV_4, self).__init__()
        self.plnet = RNNWithInit(input_linear=False,
                                 input_size=54,
                                 output_size=12,
                                 hidden_size=512,
                                 num_rnn_layer=3,
                                 dropout=0.4)
        self.iknet = torch.nn.ModuleDict({
            'net1': RNN(input_linear=False,
                        input_size=27 + 3 + 9,
                        output_size=72,
                        hidden_size=512,
                        num_rnn_layer=3,
                        dropout=0.4),
            'net2': RNN(input_linear=False,
                        input_size=99,
                        output_size=90,
                        hidden_size=512,
                        num_rnn_layer=3,
                        dropout=0.4)
        })
        self.vrnet = RNNWithInit(input_linear=False,
                                 input_size=231,
                                 output_size=9,
                                 hidden_size=512,
                                 num_rnn_layer=3,
                                 dropout=0.4)

        # for training
        # self.plnet.load_state_dict(torch.load('data/weight_4/Pose-GR/PL/best_weights.pt'))
        # self.iknet.load_state_dict(torch.load('data/weight_4/Pose-GR/IK/best_weights.pt'))
        # self.vrnet.load_state_dict(torch.load('data/weight_4/Tran-OV/VR/best_weights.pt'))

        # for testing
        # self.load_state_dict(torch.load('data/weight_4/Full-GR-OV/full/weights.pt'))
        self.load_state_dict(torch.load('data/weights_4_finetune_with_loose_data/weights.pt'))

        self.B = np.array([[self.mu, -self.mu, 0,       0       ],
                           [1,       1,        1,       1       ],
                           [0,       0,        self.mu, -self.mu]]) / np.sqrt(1 + self.mu ** 2)  # basis of friction cone
        self.body_model = art.ParametricModel('models/SMPL_male.pkl', vert_mask=self.v_imu)
        self.physics_model = cart.get_dynamic_model('models/SMPL_male.pkl')
        # self.rnn_initialize()  # using T-pose
        self.eval()

        if self.Visualization.enable:
            from articulate.utils.unity import MotionViewer
            MotionViewer.colors = [(1, 1, 1)]
            self.viewer = MotionViewer(1)
            self.viewer.connect()
            self.scene = Scene(self.viewer)
            self.force_lpf = [art.LowPassFilter(0.3) for _ in range(5)]
            self.torque_lpf = art.LowPassFilter(0.3)
            self.tran_offset = torch.zeros(3)
            if self.Visualization.show_torque:
                self.viewer.show_torque(0, [1, 2, 4, 5, 16, 17, 18, 19])

    @torch.no_grad()
    def rnn_initialize(self, init_pose=None, init_vel=None):
        r"""
        Initialize the hidden states of the RNNs.

        :param init_pose: Pose in shape [24, 3, 3]. T-pose by default.
        :param init_vel: Root world-space velocity in shape [3]. Zero by default.
        """
        init_pose = torch.eye(3).expand(1, 24, 3, 3) if init_pose is None else init_pose.cpu().view(1, 24, 3, 3)
        init_vel = torch.zeros(3) if init_vel is None else init_vel.cpu().view(3)
        vRR_V = init_vel[1].view(1).clone()
        init_vel[1] = 0
        _, j, v = self.body_model.forward_kinematics(init_pose, calc_mesh=True)
        pRL, gR = (v[0, :3] - v[0, 3:]).mm(init_pose[0, 0]).ravel(), -init_pose[0, 0, 1]
        x1 = torch.cat((pRL, gR)).to(self.plnet.init_net[0].weight.device)
        h, vRR_H, c = -j[:, :, 1].min().view(1), init_pose[0, 0].t().mm(init_vel.unsqueeze(-1)).squeeze(-1), torch.zeros(5)
        x2 = torch.cat((vRR_V, vRR_H, c)).to(self.vrnet.init_net[0].weight.device)
        self.pl1hc = [_.contiguous() for _ in self.plnet.init_net(x1).view(1, 2, self.plnet.num_layers, self.plnet.hidden_size).permute(1, 2, 0, 3)]
        self.vr1hc = [_.contiguous() for _ in self.vrnet.init_net(x2).view(1, 2, self.vrnet.num_layers, self.vrnet.hidden_size).permute(1, 2, 0, 3)]
        self.ik1hc = None
        self.ik2hc = None
        self.last_cjoint = torch.tensor([0, h + self.floor_y, 0]) + j[0, self.j_contact]
        self.is_init = False
        self.contact = np.zeros(5, dtype=bool)
        self.contact_counter = np.zeros(5, dtype=int)

    @torch.no_grad()
    def _explain_residual_force(self, contact, contact_Jacobian, contact_position, residual_force):
        if np.any(contact):
            J = contact_Jacobian.reshape(5, 3, 75)[contact, :, :6].reshape(-1, 6)
            B, lb = [], []
            for i in np.where(contact)[0]:
                if i > 3 and contact_position[i, 1] > self.floor_y + 0.15:      # hand grasp
                    B.append(np.eye(3))
                    lb.append(-np.ones(3) * np.inf)
                else:
                    B.append(self.B)
                    lb.append(np.zeros(4))
            JTB = J.T @ art.math.block_diagonal_matrix_np(B)
            P = csc_array(JTB.T @ JTB + self.beta_extforce * np.eye(JTB.shape[1]))
            q = -JTB.T @ residual_force
            force = solve_qp(P, q, lb=np.concatenate(lb), solver='osqp')
            error = np.linalg.norm(JTB @ force - residual_force)
        else:
            B = None
            force = np.zeros(0)
            error = np.linalg.norm(residual_force)
        return force, error, B

    @torch.no_grad()
    def forward_frame(self, a, w, R):
        root_idx = 3

        # a, w: [4, 3], R: [4, 3, 3]
        aRB = a.mm(R[root_idx])                              # [4, 3]
        wRB = w.mm(R[root_idx])                              # [4, 3]
        RRB = R[root_idx].t().matmul(R[:root_idx])           # [3, 3, 3]
        gR0 = -R[root_idx, 1]                                # [3]

        # PL-s1
        x = torch.cat((aRB.ravel(), wRB.ravel(), RRB.ravel(), gR0))   # 12+12+27+3 = 54
        x, self.pl1hc = self.plnet.rnn(x.view(1, 1, -1), self.pl1hc)
        x = self.plnet.linear2(x.squeeze())                  # pRB(9), gR(3)
        gR1 = art.math.normalize_tensor(x[9:])
        RRB = art.math.from_to_rotation_matrix(gR0, gR1).matmul(RRB)

        # IK-s1
        x = torch.cat((RRB.ravel(), gR1, x[:9]))             # 27+3+9 = 39
        x, self.ik1hc = self.iknet.net1.rnn(x.view(1, 1, -1), self.ik1hc)
        x = self.iknet.net1.linear2(x.squeeze())             # pRJ(69), gR(3)
        gR2 = art.math.normalize_tensor(x[69:])
        RRB = art.math.from_to_rotation_matrix(gR1, gR2).matmul(RRB)

        # IK-s2
        x = torch.cat((RRB.ravel(), gR2, x[:69]))            # 27+3+69 = 99
        x, self.ik2hc = self.iknet.net2.rnn(x.view(1, 1, -1), self.ik2hc)
        x = self.iknet.net2.linear2(x.squeeze())             # RRJ(90)

        # get pose estimation
        RRJ = art.math.r6d_to_rotation_matrix(x).cpu()
        glb_pose = torch.eye(3).repeat(1, 24, 1, 1)
        glb_pose[:, self.j_reduce] = RRJ.view(1, 15, 3, 3)
        pose = self.body_model.inverse_kinematics_R(glb_pose).view(24, 3, 3)
        pose[self.j_ignore, ...] = torch.eye(3)
        pRJ = self.body_model.forward_kinematics(pose.unsqueeze(0))[1][0, 1:]
        pose[0] = R[root_idx].mm(art.math.from_to_rotation_matrix(gR2, gR0).squeeze()).cpu()

        # VR-s1
        aRB = a.cpu().mm(pose[0])                            # [4, 3]
        wRB = w.cpu().mm(pose[0])                            # [4, 3]
        x = torch.cat((RRJ.ravel(), pRJ.ravel(), aRB.ravel(), wRB.ravel(), gR2.cpu())).to(gR2.device)
        # RRJ:135 + pRJ:69 + aRB:12 + wRB:12 + gR2:3 = 231
        x, self.vr1hc = self.vrnet.rnn(x.view(1, 1, -1), self.vr1hc)
        x = self.vrnet.linear2(x.squeeze())                  # vRR_V, vRR_H, stationary_prob

        # get translation estimation
        vRR_V, vRR_H, stationary_prob = x[0].item(), x[1:4].cpu(), x[4:].sigmoid().cpu()
        vWR = pose[0].mm(vRR_H.unsqueeze(-1)).squeeze(-1)
        vWR[1] = vRR_V
        cjoint = torch.cat((torch.zeros(1, 3), pRJ.mm(pose[0].t())))[self.j_contact, :]
        stationary_weight = (stationary_prob * 5 - 3).clip(0, 1)
        velocity = (stationary_weight.unsqueeze(0).mm(self.last_cjoint - cjoint)[0] / self.dt + self.beta_velocity * vWR) / (self.beta_velocity + stationary_weight.sum())
        self.last_cjoint = cjoint

        # physics optimization
        if not self.is_init:
            self.is_init = True
            self.physics_model.set_state_R(pose.numpy(), np.array([0, self.floor_y - cjoint[:, 1].min().item(), 0]), np.zeros(75))
        else:
            pose_cur, tran_cur, qdot = self.physics_model.get_state_R()
            cjoint_cur = np.vstack([self.physics_model.get_position(j) for j in self.j_contact])
            cvel_cur = np.vstack([self.physics_model.get_linear_velocity(j) for j in self.j_contact])
            cJ_cur = np.vstack([self.physics_model.get_linear_Jacobian(j) for j in self.j_contact])
            cJdot_cur = np.vstack([self.physics_model.get_linear_Jacobian_dot(j) for j in self.j_contact])
            M = self.physics_model.mass_matrix()
            h = self.physics_model.inverse_dynamics(np.zeros(75))
            stationary = stationary_prob.numpy() > 0.7

            R_pd = art.math.axis_angle_to_rotation_matrix(torch.from_numpy(qdot[3:]) * self.alpha_pd * self.dt)
            delta_pose = art.math.rotation_matrix_to_axis_angle(torch.from_numpy(pose_cur).bmm(R_pd).transpose(1, 2).bmm(pose)).ravel().numpy()
            thetaddotdes = (self.kp_pose * delta_pose - self.kd_pose * qdot[3:]) / (1 + self.kd_pose * self.alpha_pd * self.dt)

            cjoint = tran_cur + velocity.numpy() * self.dt + cjoint.numpy()
            cjoint = art.math.lerp(cjoint, cjoint_cur, stationary_weight.view(5, 1).numpy())
            delta_tran = cjoint - cjoint_cur - cvel_cur * self.alpha_pd * self.dt
            rddotdes = (self.kp_tran * delta_tran - self.kd_tran * cvel_cur).ravel() / (1 + self.kd_tran * self.alpha_pd * self.dt)

            cjoint_cur[0, 1] -= 0.15
            cjoint[0, 1] -= 0.15
            k = np.ones((5, 3)) * self.beta_cjoint
            k[self.contact] *= 10
            A = np.vstack((np.hstack((np.zeros((72, 3)), np.eye(72))), np.sqrt(k.reshape(15, 1)) * cJ_cur, np.sqrt(self.beta_torque) * M))
            b = np.concatenate((thetaddotdes, np.sqrt(k.reshape(15)) * (-cJdot_cur @ qdot + rddotdes), np.sqrt(self.beta_torque) * (-h)))
            qddot = lsqr(csc_array(A), b)[0]
            residual_force = M[:6] @ qddot + h[:6]

            vdist = np.abs(cjoint_cur[np.newaxis, :, 1] - cjoint_cur[:, np.newaxis, 1])
            contact = stationary & (self.contact | (cjoint_cur[:, 1] < self.floor_y + 0.05))
            if np.any(contact):
                contact |= stationary & (vdist[contact].min(axis=0) < 0.05)
            potential_contact = stationary & ~contact

            if contact[0] or potential_contact[0]:
                lleg = self.physics_model.get_position(4) - self.physics_model.get_position(1)
                rleg = self.physics_model.get_position(5) - self.physics_model.get_position(2)
                if min(np.arccos(-lleg[1] / np.linalg.norm(lleg)), np.arccos(-rleg[1] / np.linalg.norm(rleg))) < np.pi / 4:
                    contact[0], potential_contact[0] = False, False

            force, err, forceB = self._explain_residual_force(contact, cJ_cur, cjoint_cur, residual_force)
            for i in np.argsort(cjoint_cur[:, 1]):
                if err > 400 and potential_contact[i]:
                    contact[i] = True
                    force_new, err_new, forceB_new = self._explain_residual_force(contact, cJ_cur, cjoint_cur, residual_force)
                    self.contact_counter[i] = self.contact_counter[i] + 1 if err_new / err < 0.6 else 0
                    if self.contact_counter[i] >= 5:
                        force, err, forceB = force_new, err_new, forceB_new
                    else:
                        contact[i] = False
                else:
                    self.contact_counter[i] = 0

            near_ground = cjoint[:, 1] < self.floor_y + 0.15
            for i in np.where(contact & near_ground)[0]:
                cjoint[i, 1] = art.math.lerp(cjoint[i, 1], self.floor_y, 0.1)
            cjoint[cjoint[:, 1] < self.floor_y, 1] = self.floor_y

            delta_tran = cjoint - cjoint_cur - cvel_cur * self.alpha_pd * self.dt
            rddotdes = (self.kp_tran * delta_tran - self.kd_tran * cvel_cur).ravel() / (1 + self.kd_tran * self.alpha_pd * self.dt)
            if np.any(contact):
                J = cJ_cur.reshape(5, 3, 75)[contact].reshape(-1, 75)
                B = art.math.block_diagonal_matrix_np(forceB)
                force = B @ force
                torque = J.T @ force
            else:
                torque = np.zeros(75)

            A = np.vstack((np.hstack((np.zeros((72, 3)), np.eye(72))), np.sqrt(k.reshape(15, 1)) * cJ_cur, np.sqrt(self.beta_torque * 3) * M))
            b = np.concatenate((thetaddotdes, np.sqrt(k.reshape(15)) * (-cJdot_cur @ qdot + rddotdes), np.sqrt(self.beta_torque * 3) * (-h + torque)))
            qddot = lsqr(csc_array(A), b)[0]
            self.physics_model.update_state(qddot, self.dt)
            self.contact = contact

            if self.Visualization.enable:
                force = iter(force.reshape(-1, 3))
                self.viewer.clear_line(render=False)
                self.viewer.clear_point(render=False)
                self.viewer.update(torch.from_numpy(pose_cur), torch.from_numpy(tran_cur) - self.tran_offset, render=False)
                cjoint -= self.tran_offset.numpy()
                for i in np.where(contact)[0]:
                    if self.Visualization.show_block and not near_ground[i]:
                        if i < 3 or True:
                            self.scene.add_block(Scene.SupportBlock, cjoint[i] - [0, 0.2, 0], lifetime=120, render=False)
                        else:
                            self.scene.add_block(Scene.GraspBlock, cjoint[i], lifetime=30, render=False)
                    if self.Visualization.show_contact_force:
                        self.viewer.draw_line(cjoint[i], cjoint[i] + self.force_lpf[i](next(force)) * 0.002, [1, 0, 0], 0.02, render=False)
                    if self.Visualization.show_contact:
                        self.viewer.draw_point(cjoint[i], [0, 1, 0], 0.15, render=False)
                if self.Visualization.show_stationary:
                    for i in np.where(stationary & ~contact)[0]:
                        self.viewer.draw_point(cjoint[i], [0.4, 0.4, 1], 0.15, render=False)
                if self.Visualization.show_block:
                    self.scene.update(render=False)
                if self.Visualization.show_residual_force:
                    self.viewer.draw_line_from_joint(0, 0, residual_force[:3] * 0.001, (0, 0, 1), 0.02, render=False)
                if self.Visualization.show_torque:
                    self.viewer.show_torque(0, [1, 2, 4, 5, 16, 17, 18, 19], render=False)
                    tau = M @ qddot + h - torque
                    tau = self.torque_lpf(tau)
                    self.viewer.update_torque(tau[3:], render=False)
                else:
                    self.viewer.hide_torque(render=False)
                self.viewer.render()

        refined_pose, refined_tran, qdot = self.physics_model.get_state_R()
        return torch.from_numpy(refined_pose), torch.from_numpy(refined_tran)
    def _get_dims(self, imu_num: int):
        root_idx = imu_num - 1
        num_leaf = root_idx

        D = {
            "imu_num": imu_num,
            "root_idx": root_idx,
            "num_leaf": num_leaf,

            "dim_a": imu_num * 3,
            "dim_w": imu_num * 3,
            "dim_rrb": num_leaf * 9,
            "dim_prb": num_leaf * 3,

            "dim_gr": 3,
            "dim_prj": 23 * 3,       # 69
            "dim_rrj_r6d": 15 * 6,   # 90
            "dim_rrj_mat": 15 * 9,   # 135
            "dim_vr_out": 9,
        }

        D["x_in"] = D["dim_a"] + D["dim_w"] + D["dim_rrb"] + D["dim_gr"]
        D["pl_out"] = D["dim_prb"] + D["dim_gr"]
        D["ik1_in"] = D["dim_rrb"] + D["dim_gr"] + D["dim_prb"]
        D["ik1_out"] = D["dim_prj"] + D["dim_gr"]
        D["ik2_in"] = D["dim_rrb"] + D["dim_gr"] + D["dim_prj"]
        D["ik2_out"] = D["dim_rrj_r6d"]
        D["vr_in"] = D["dim_rrj_mat"] + D["dim_prj"] + D["dim_a"] + D["dim_w"] + D["dim_gr"]

        return D


    def forward(self, x, fast=True, imu_num=None):
        if imu_num is None:
            imu_num = len(self.v_imu)

        D = self._get_dims(imu_num)

        a0 = 0
        a1 = a0 + D["dim_a"]
        w0 = a1
        w1 = w0 + D["dim_w"]
        rrb0 = w1
        rrb1 = rrb0 + D["dim_rrb"]
        gr0 = rrb1
        gr1 = gr0 + D["dim_gr"]

        # PL-s1
        RRB = [x_[:, rrb0:rrb1].view(-1, D["num_leaf"], 3, 3) for x_, y_ in x]
        gR0 = [x_[:, gr0:gr1] for x_, y_ in x]

        x1 = [(x_, y_[:D["pl_out"]]) for x_, y_ in x]
        x1 = self.plnet(x1)   # pRB, gR

        # IK-s1
        pRB = [x_[:, :D["dim_prb"]] for x_ in x1]
        gR1 = [art.math.normalize_tensor(x_[:, D["dim_prb"]:].clone().detach()) for x_ in x1]
        RRB = [
            art.math.from_to_rotation_matrix(gR0_, gR1_).unsqueeze(1).matmul(RRB_)
            for gR0_, gR1_, RRB_ in zip(gR0, gR1, RRB)
        ]
        x2 = [
            torch.cat((RRB_.flatten(1), gR1_, pRB_), dim=1)
            for RRB_, gR1_, pRB_ in zip(RRB, gR1, pRB)
        ]
        x2 = self.iknet.net1(x2)   # pRJ, gR

        # IK-s2
        pRJ = [x_[:, :D["dim_prj"]] for x_ in x2]
        gR2 = [art.math.normalize_tensor(x_[:, D["dim_prj"]:].clone().detach()) for x_ in x2]
        RRB = [
            art.math.from_to_rotation_matrix(gR1_, gR2_).unsqueeze(1).matmul(RRB_)
            for gR1_, gR2_, RRB_ in zip(gR1, gR2, RRB)
        ]
        x3 = [
            torch.cat((RRB_.flatten(1), gR2_, pRJ_), dim=1)
            for RRB_, gR2_, pRJ_ in zip(RRB, gR2, pRJ)
        ]
        x3 = self.iknet.net2(x3)   # RRJ (r6d)

        # VR-s1
        if fast:
            RRJ = [art.math.r6d_to_rotation_matrix(x3_.detach()).view(-1, D["dim_rrj_mat"]) for x3_ in x3]

            awRB = [
                x_[:, :D["dim_a"] + D["dim_w"]]
                .view(-1, 2 * D["imu_num"], 3)
                .bmm(art.math.from_to_rotation_matrix(gR2_, gR0_))
                .view(-1, D["dim_a"] + D["dim_w"])
                for gR2_, gR0_, (x_, y_) in zip(gR2, gR0, x)
            ]

            x4 = [
                (
                    torch.cat((RRJ_, pRJ_.detach(), awRB_, gR2_), dim=1),
                    torch.zeros_like(y_[-D["dim_vr_out"]:]) if torch.isnan(y_[-D["dim_vr_out"]]) else y_[-D["dim_vr_out"]:]
                )
                for RRJ_, pRJ_, awRB_, gR2_, (x_, y_) in zip(RRJ, pRJ, awRB, gR2, x)
            ]
        else:
            RRJ, pRJ_new, aRB_new, wRB_new, pose = [], [], [], [], []

            aRB = [x_[:, :D["dim_a"]].view(-1, D["imu_num"], 3) for x_, y_ in x]
            wRB = [x_[:, D["dim_a"]:D["dim_a"] + D["dim_w"]].view(-1, D["imu_num"], 3) for x_, y_ in x]

            for i in range(len(x3)):
                x3_ = x3[i].clone().detach().cpu()
                RRJ_ = art.math.r6d_to_rotation_matrix(x3_).view(-1, 15, 3, 3)
                glb_pose_ = torch.eye(3).repeat(RRJ_.shape[0], 24, 1, 1)
                glb_pose_[:, self.j_reduce] = RRJ_
                pose_ = self.body_model.inverse_kinematics_R(glb_pose_).view(-1, 24, 3, 3)
                pose_[:, self.j_ignore] = torch.eye(3)
                pRJ_ = self.body_model.forward_kinematics(pose_)[1][:, 1:]
                aRB_ = aRB[i].bmm(art.math.from_to_rotation_matrix(gR2[i], gR0[i]))
                wRB_ = wRB[i].bmm(art.math.from_to_rotation_matrix(gR2[i], gR0[i]))
                RRJ.append(RRJ_)
                pRJ_new.append(pRJ_)
                aRB_new.append(aRB_)
                wRB_new.append(wRB_)
                pose.append(pose_)

            no_translation = [torch.isnan(y_[-D["dim_vr_out"]]).item() for x_, y_ in x]
            x4 = [
                (
                    torch.cat((
                        RRJ_.flatten(1).to(aRB_.device),
                        pRJ_.flatten(1).to(aRB_.device),
                        aRB_.flatten(1),
                        wRB_.flatten(1),
                        gR2_
                    ), dim=1),
                    x_[1][-D["dim_vr_out"]:] if not nt_ else torch.zeros_like(x_[1][-D["dim_vr_out"]:])
                )
                for RRJ_, pRJ_, aRB_, wRB_, gR2_, x_, nt_ in zip(RRJ, pRJ_new, aRB_new, wRB_new, gR2, x, no_translation)
            ]

        x4 = self.vrnet(x4)   # vRR_V, vRR_H, stationary_prob
        result = [torch.cat((x1_, x2_, x3_, x4_), dim=1) for x1_, x2_, x3_, x4_ in zip(x1, x2, x3, x4)]
        return result


class Full_GR_OV_8(torch.nn.Module):
    dt = 1 / 60
    mu = 0.7           # environment fiction coefficient
    kp_pose = 3600     # kp in pose PD controller
    kd_pose = 60       # kd in pose PD controller
    kp_tran = 3600     # kp in tran PD controller
    kd_tran = 60       # kd in tran PD controller
    alpha_pd = 0.0     # relaxation in stable PD controller
    floor_y = -0.97    # floor height
    beta_velocity = 1
    beta_cjoint = 1
    beta_extforce = 0.4
    beta_torque = 1e-3 / 80
    j_reduce = (1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19)
    j_ignore = (0, 7, 8, 10, 11, 20, 21, 22, 23)
    j_contact = (0, 10, 11, 22, 23)
    leaf_idx = [0, 1, 5, 6, 4]
    # j_imu = [18, 19, 4, 5, 12, 0]
    v_imu = (1961, 5424, 1505, 4917, 1305, 3187, 6585, 4298)
    imu_num = len(v_imu)

    class Visualization:
        enable = False
        show_residual_force = False
        show_contact_force = True
        show_block = True
        show_contact = False
        show_stationary = False
        show_torque = False

    def __init__(self):
        from articulate.utils.torch import RNN, RNNWithInit
        super(Full_GR_OV_8, self).__init__()
        self.plnet = RNNWithInit(input_linear=False,
                                 input_size=self.imu_num*6+(self.imu_num-1)*9+3,
                                 output_size=18,
                                 hidden_size=512,
                                 num_rnn_layer=3,
                                 dropout=0.4)
        self.iknet = torch.nn.ModuleDict({
            'net1': RNN(input_linear=False,
                        input_size=(self.imu_num-1)*9 + 3 + 15,
                        output_size=72,
                        hidden_size=512,
                        num_rnn_layer=3,
                        dropout=0.4),
            'net2': RNN(input_linear=False,
                        input_size=(self.imu_num-1)*9 + 3 + 69,
                        output_size=90,
                        hidden_size=512,
                        num_rnn_layer=3,
                        dropout=0.4)
        })
        self.vrnet = RNNWithInit(input_linear=False,
                                 input_size=135+69+6*self.imu_num+3,
                                 output_size=9,
                                 hidden_size=512,
                                 num_rnn_layer=3,
                                 dropout=0.4)

        # for training
        # self.plnet.load_state_dict(torch.load('data/weight_8/Pose-GR/PL/best_weights.pt'))
        # self.iknet.load_state_dict(torch.load('data/weight_8/Pose-GR/IK/best_weights.pt'))
        # self.vrnet.load_state_dict(torch.load('data/weight_8/Tran-OV/VR/best_weights.pt'))

        # for testing
        self.load_state_dict(torch.load('data/weight_8/Full-GR-OV/full/weights.pt'))

        self.B = np.array([[self.mu, -self.mu, 0,       0       ],
                           [1,       1,        1,       1       ],
                           [0,       0,        self.mu, -self.mu]]) / np.sqrt(1 + self.mu ** 2)  # basis of friction cone
        self.body_model = art.ParametricModel('models/SMPL_male.pkl', vert_mask=self.v_imu)
        self.physics_model = cart.get_dynamic_model('models/SMPL_male.pkl')
        # self.rnn_initialize()  # using T-pose
        self.eval()

        if self.Visualization.enable:
            from articulate.utils.unity import MotionViewer
            MotionViewer.colors = [(1, 1, 1)]
            self.viewer = MotionViewer(1)
            self.viewer.connect()
            self.scene = Scene(self.viewer)
            self.force_lpf = [art.LowPassFilter(0.3) for _ in range(5)]
            self.torque_lpf = art.LowPassFilter(0.3)
            self.tran_offset = torch.zeros(3)
            if self.Visualization.show_torque:
                self.viewer.show_torque(0, [1, 2, 4, 5, 16, 17, 18, 19])

    @torch.no_grad()
    def rnn_initialize(self, init_pose=None, init_vel=None):
        """
        Initialize the hidden states of the RNNs.

        :param init_pose: Pose in shape [24, 3, 3]. T-pose by default.
        :param init_vel: Root world-space velocity in shape [3]. Zero by default.
        """
        init_pose = torch.eye(3).expand(1, 24, 3, 3) if init_pose is None else init_pose.cpu().view(1, 24, 3, 3)
        init_vel = torch.zeros(3) if init_vel is None else init_vel.cpu().view(3)

        root_idx = self.imu_num - 1  # 7 for 8-IMU

        vRR_V = init_vel[1].view(1).clone()
        init_vel[1] = 0

        _, j, v = self.body_model.forward_kinematics(init_pose, calc_mesh=True)

        # use supervised leaf points, not all non-root IMUs
        pRL = (v[0, self.leaf_idx] - v[0, root_idx:root_idx + 1]).mm(init_pose[0, 0]).ravel()   # 5*3 = 15
        gR = -init_pose[0, 0, 1]                                                                  # 3
        x1 = torch.cat((pRL, gR)).to(self.plnet.init_net[0].weight.device)                        # 18

        h = -j[:, :, 1].min().view(1)
        vRR_H = init_pose[0, 0].t().mm(init_vel.unsqueeze(-1)).squeeze(-1)
        c = torch.zeros(5)
        x2 = torch.cat((vRR_V, vRR_H, c)).to(self.vrnet.init_net[0].weight.device)                # 9

        self.pl1hc = [_.contiguous() for _ in self.plnet.init_net(x1).view(1, 2, self.plnet.num_layers, self.plnet.hidden_size).permute(1, 2, 0, 3)]
        self.vr1hc = [_.contiguous() for _ in self.vrnet.init_net(x2).view(1, 2, self.vrnet.num_layers, self.vrnet.hidden_size).permute(1, 2, 0, 3)]
        self.ik1hc = None
        self.ik2hc = None
        self.last_cjoint = torch.tensor([0, h + self.floor_y, 0]) + j[0, self.j_contact]
        self.is_init = False
        self.contact = np.zeros(5, dtype=bool)
        self.contact_counter = np.zeros(5, dtype=int)

    @torch.no_grad()
    def _explain_residual_force(self, contact, contact_Jacobian, contact_position, residual_force):
        if np.any(contact):
            J = contact_Jacobian.reshape(5, 3, 75)[contact, :, :6].reshape(-1, 6)
            B, lb = [], []
            for i in np.where(contact)[0]:
                if i > 3 and contact_position[i, 1] > self.floor_y + 0.15:      # hand grasp
                    B.append(np.eye(3))
                    lb.append(-np.ones(3) * np.inf)
                else:
                    B.append(self.B)
                    lb.append(np.zeros(4))
            JTB = J.T @ art.math.block_diagonal_matrix_np(B)
            P = csc_array(JTB.T @ JTB + self.beta_extforce * np.eye(JTB.shape[1]))
            q = -JTB.T @ residual_force
            force = solve_qp(P, q, lb=np.concatenate(lb), solver='osqp')
            error = np.linalg.norm(JTB @ force - residual_force)
        else:
            B = None
            force = np.zeros(0)
            error = np.linalg.norm(residual_force)
        return force, error, B

    @torch.no_grad()
    def forward_frame(self, a, w, R):
        root_idx = self.imu_num - 1   # 7 for 8-IMU

        # a, w: [8, 3], R: [8, 3, 3]
        aRB = a.mm(R[root_idx])                              # [8, 3]
        wRB = w.mm(R[root_idx])                              # [8, 3]
        RRB = R[root_idx].t().matmul(R[:root_idx])          # [7, 3, 3]
        gR0 = -R[root_idx, 1]                                # [3]

        # PL-s1
        # 24 + 24 + 63 + 3 = 114
        x = torch.cat((aRB.ravel(), wRB.ravel(), RRB.ravel(), gR0))
        x, self.pl1hc = self.plnet.rnn(x.view(1, 1, -1), self.pl1hc)
        x = self.plnet.linear2(x.squeeze())                  # pRB(15), gR(3)
        gR1 = art.math.normalize_tensor(x[15:])
        RRB = art.math.from_to_rotation_matrix(gR0, gR1).matmul(RRB)

        # IK-s1
        # 63 + 3 + 15 = 81
        x = torch.cat((RRB.ravel(), gR1, x[:15]))
        x, self.ik1hc = self.iknet.net1.rnn(x.view(1, 1, -1), self.ik1hc)
        x = self.iknet.net1.linear2(x.squeeze())             # pRJ(69), gR(3)
        gR2 = art.math.normalize_tensor(x[69:])
        RRB = art.math.from_to_rotation_matrix(gR1, gR2).matmul(RRB)

        # IK-s2
        # 63 + 3 + 69 = 135
        x = torch.cat((RRB.ravel(), gR2, x[:69]))
        x, self.ik2hc = self.iknet.net2.rnn(x.view(1, 1, -1), self.ik2hc)
        x = self.iknet.net2.linear2(x.squeeze())             # RRJ(90)

        # get pose estimation
        RRJ = art.math.r6d_to_rotation_matrix(x).cpu()
        glb_pose = torch.eye(3).repeat(1, 24, 1, 1)
        glb_pose[:, self.j_reduce] = RRJ.view(1, 15, 3, 3)
        pose = self.body_model.inverse_kinematics_R(glb_pose).view(24, 3, 3)
        pose[self.j_ignore, ...] = torch.eye(3)
        pRJ = self.body_model.forward_kinematics(pose.unsqueeze(0))[1][0, 1:]
        pose[0] = R[root_idx].mm(art.math.from_to_rotation_matrix(gR2, gR0).squeeze()).cpu()

        # VR-s1
        aRB = a.cpu().mm(pose[0])                            # [8, 3]
        wRB = w.cpu().mm(pose[0])                            # [8, 3]
        # RRJ:135 + pRJ:69 + aRB:24 + wRB:24 + gR2:3 = 255
        x = torch.cat((RRJ.ravel(), pRJ.ravel(), aRB.ravel(), wRB.ravel(), gR2.cpu())).to(gR2.device)
        x, self.vr1hc = self.vrnet.rnn(x.view(1, 1, -1), self.vr1hc)
        x = self.vrnet.linear2(x.squeeze())                  # vRR_V, vRR_H, stationary_prob

        # get translation estimation
        vRR_V, vRR_H, stationary_prob = x[0].item(), x[1:4].cpu(), x[4:].sigmoid().cpu()
        vWR = pose[0].mm(vRR_H.unsqueeze(-1)).squeeze(-1)
        vWR[1] = vRR_V
        cjoint = torch.cat((torch.zeros(1, 3), pRJ.mm(pose[0].t())))[self.j_contact, :]
        stationary_weight = (stationary_prob * 5 - 3).clip(0, 1)
        velocity = (stationary_weight.unsqueeze(0).mm(self.last_cjoint - cjoint)[0] / self.dt + self.beta_velocity * vWR) / (self.beta_velocity + stationary_weight.sum())
        self.last_cjoint = cjoint

        # physics optimization
        if not self.is_init:
            self.is_init = True
            self.physics_model.set_state_R(
                pose.numpy(),
                np.array([0, self.floor_y - cjoint[:, 1].min().item(), 0]),
                np.zeros(75)
            )
        else:
            pose_cur, tran_cur, qdot = self.physics_model.get_state_R()
            cjoint_cur = np.vstack([self.physics_model.get_position(j) for j in self.j_contact])
            cvel_cur = np.vstack([self.physics_model.get_linear_velocity(j) for j in self.j_contact])
            cJ_cur = np.vstack([self.physics_model.get_linear_Jacobian(j) for j in self.j_contact])
            cJdot_cur = np.vstack([self.physics_model.get_linear_Jacobian_dot(j) for j in self.j_contact])
            M = self.physics_model.mass_matrix()
            h = self.physics_model.inverse_dynamics(np.zeros(75))
            stationary = stationary_prob.numpy() > 0.7

            R_pd = art.math.axis_angle_to_rotation_matrix(torch.from_numpy(qdot[3:]) * self.alpha_pd * self.dt)
            delta_pose = art.math.rotation_matrix_to_axis_angle(
                torch.from_numpy(pose_cur).bmm(R_pd).transpose(1, 2).bmm(pose)
            ).ravel().numpy()
            thetaddotdes = (self.kp_pose * delta_pose - self.kd_pose * qdot[3:]) / (1 + self.kd_pose * self.alpha_pd * self.dt)

            cjoint = tran_cur + velocity.numpy() * self.dt + cjoint.numpy()
            cjoint = art.math.lerp(cjoint, cjoint_cur, stationary_weight.view(5, 1).numpy())
            delta_tran = cjoint - cjoint_cur - cvel_cur * self.alpha_pd * self.dt
            rddotdes = (self.kp_tran * delta_tran - self.kd_tran * cvel_cur).ravel() / (1 + self.kd_tran * self.alpha_pd * self.dt)

            cjoint_cur[0, 1] -= 0.15
            cjoint[0, 1] -= 0.15
            k = np.ones((5, 3)) * self.beta_cjoint
            k[self.contact] *= 10
            A = np.vstack((
                np.hstack((np.zeros((72, 3)), np.eye(72))),
                np.sqrt(k.reshape(15, 1)) * cJ_cur,
                np.sqrt(self.beta_torque) * M
            ))
            b = np.concatenate((
                thetaddotdes,
                np.sqrt(k.reshape(15)) * (-cJdot_cur @ qdot + rddotdes),
                np.sqrt(self.beta_torque) * (-h)
            ))
            qddot = lsqr(csc_array(A), b)[0]
            residual_force = M[:6] @ qddot + h[:6]

            vdist = np.abs(cjoint_cur[np.newaxis, :, 1] - cjoint_cur[:, np.newaxis, 1])
            contact = stationary & (self.contact | (cjoint_cur[:, 1] < self.floor_y + 0.05))
            if np.any(contact):
                contact |= stationary & (vdist[contact].min(axis=0) < 0.05)
            potential_contact = stationary & ~contact

            if contact[0] or potential_contact[0]:
                lleg = self.physics_model.get_position(4) - self.physics_model.get_position(1)
                rleg = self.physics_model.get_position(5) - self.physics_model.get_position(2)
                if min(np.arccos(-lleg[1] / np.linalg.norm(lleg)), np.arccos(-rleg[1] / np.linalg.norm(rleg))) < np.pi / 4:
                    contact[0], potential_contact[0] = False, False

            force, err, forceB = self._explain_residual_force(contact, cJ_cur, cjoint_cur, residual_force)
            for i in np.argsort(cjoint_cur[:, 1]):
                if err > 400 and potential_contact[i]:
                    contact[i] = True
                    force_new, err_new, forceB_new = self._explain_residual_force(contact, cJ_cur, cjoint_cur, residual_force)
                    self.contact_counter[i] = self.contact_counter[i] + 1 if err_new / err < 0.6 else 0
                    if self.contact_counter[i] >= 5:
                        force, err, forceB = force_new, err_new, forceB_new
                    else:
                        contact[i] = False
                else:
                    self.contact_counter[i] = 0

            near_ground = cjoint[:, 1] < self.floor_y + 0.15
            for i in np.where(contact & near_ground)[0]:
                cjoint[i, 1] = art.math.lerp(cjoint[i, 1], self.floor_y, 0.1)
            cjoint[cjoint[:, 1] < self.floor_y, 1] = self.floor_y

            delta_tran = cjoint - cjoint_cur - cvel_cur * self.alpha_pd * self.dt
            rddotdes = (self.kp_tran * delta_tran - self.kd_tran * cvel_cur).ravel() / (1 + self.kd_tran * self.alpha_pd * self.dt)
            if np.any(contact):
                J = cJ_cur.reshape(5, 3, 75)[contact].reshape(-1, 75)
                B = art.math.block_diagonal_matrix_np(forceB)
                force = B @ force
                torque = J.T @ force
            else:
                torque = np.zeros(75)

            A = np.vstack((
                np.hstack((np.zeros((72, 3)), np.eye(72))),
                np.sqrt(k.reshape(15, 1)) * cJ_cur,
                np.sqrt(self.beta_torque * 3) * M
            ))
            b = np.concatenate((
                thetaddotdes,
                np.sqrt(k.reshape(15)) * (-cJdot_cur @ qdot + rddotdes),
                np.sqrt(self.beta_torque * 3) * (-h + torque)
            ))
            qddot = lsqr(csc_array(A), b)[0]
            self.physics_model.update_state(qddot, self.dt)
            self.contact = contact

            if self.Visualization.enable:
                force = iter(force.reshape(-1, 3))
                self.viewer.clear_line(render=False)
                self.viewer.clear_point(render=False)
                self.viewer.update(torch.from_numpy(pose_cur), torch.from_numpy(tran_cur) - self.tran_offset, render=False)
                cjoint -= self.tran_offset.numpy()
                for i in np.where(contact)[0]:
                    if self.Visualization.show_block and not near_ground[i]:
                        if i < 3 or True:
                            self.scene.add_block(Scene.SupportBlock, cjoint[i] - [0, 0.2, 0], lifetime=120, render=False)
                        else:
                            self.scene.add_block(Scene.GraspBlock, cjoint[i], lifetime=30, render=False)
                    if self.Visualization.show_contact_force:
                        self.viewer.draw_line(cjoint[i], cjoint[i] + self.force_lpf[i](next(force)) * 0.002, [1, 0, 0], 0.02, render=False)
                    if self.Visualization.show_contact:
                        self.viewer.draw_point(cjoint[i], [0, 1, 0], 0.15, render=False)
                if self.Visualization.show_stationary:
                    for i in np.where(stationary & ~contact)[0]:
                        self.viewer.draw_point(cjoint[i], [0.4, 0.4, 1], 0.15, render=False)
                if self.Visualization.show_block:
                    self.scene.update(render=False)
                if self.Visualization.show_residual_force:
                    self.viewer.draw_line_from_joint(0, 0, residual_force[:3] * 0.001, (0, 0, 1), 0.02, render=False)
                if self.Visualization.show_torque:
                    self.viewer.show_torque(0, [1, 2, 4, 5, 16, 17, 18, 19], render=False)
                    tau = M @ qddot + h - torque
                    tau = self.torque_lpf(tau)
                    self.viewer.update_torque(tau[3:], render=False)
                else:
                    self.viewer.hide_torque(render=False)
                self.viewer.render()

        refined_pose, refined_tran, qdot = self.physics_model.get_state_R()
        return torch.from_numpy(refined_pose), torch.from_numpy(refined_tran)
        
        def _get_dims(self, imu_num: int):
            root_idx = imu_num - 1
            num_input_leaf = root_idx
            num_supervised_leaf = len(self.leaf_idx)

            D = {
                "imu_num": imu_num,
                "root_idx": root_idx,
                "num_input_leaf": num_input_leaf,
                "num_supervised_leaf": num_supervised_leaf,

                "dim_a": imu_num * 3,
                "dim_w": imu_num * 3,
                "dim_rrb": num_input_leaf * 9,
                "dim_prb": num_supervised_leaf * 3,

                "dim_gr": 3,
                "dim_prj": 23 * 3,
                "dim_rrj_r6d": 15 * 6,
                "dim_rrj_mat": 15 * 9,
                "dim_vr_out": 9,
            }

            D["x_in"] = D["dim_a"] + D["dim_w"] + D["dim_rrb"] + D["dim_gr"]
            D["pl_out"] = D["dim_prb"] + D["dim_gr"]
            D["ik1_in"] = D["dim_rrb"] + D["dim_gr"] + D["dim_prb"]
            D["ik1_out"] = D["dim_prj"] + D["dim_gr"]
            D["ik2_in"] = D["dim_rrb"] + D["dim_gr"] + D["dim_prj"]
            D["ik2_out"] = D["dim_rrj_r6d"]
            D["vr_in"] = D["dim_rrj_mat"] + D["dim_prj"] + D["dim_a"] + D["dim_w"] + D["dim_gr"]

            return D



    def _get_dims(self, imu_num: int):
        root_idx = imu_num - 1
        num_input_leaf = root_idx
        num_supervised_leaf = len(self.leaf_idx)

        D = {
            "imu_num": imu_num,
            "root_idx": root_idx,
            "num_input_leaf": num_input_leaf,
            "num_supervised_leaf": num_supervised_leaf,

            "dim_a": imu_num * 3,
            "dim_w": imu_num * 3,
            "dim_rrb": num_input_leaf * 9,
            "dim_prb": num_supervised_leaf * 3,

            "dim_gr": 3,
            "dim_prj": 23 * 3,
            "dim_rrj_r6d": 15 * 6,
            "dim_rrj_mat": 15 * 9,
            "dim_vr_out": 9,
        }

        D["x_in"] = D["dim_a"] + D["dim_w"] + D["dim_rrb"] + D["dim_gr"]
        D["pl_out"] = D["dim_prb"] + D["dim_gr"]
        D["ik1_in"] = D["dim_rrb"] + D["dim_gr"] + D["dim_prb"]
        D["ik1_out"] = D["dim_prj"] + D["dim_gr"]
        D["ik2_in"] = D["dim_rrb"] + D["dim_gr"] + D["dim_prj"]
        D["ik2_out"] = D["dim_rrj_r6d"]
        D["vr_in"] = D["dim_rrj_mat"] + D["dim_prj"] + D["dim_a"] + D["dim_w"] + D["dim_gr"]

        return D


    def forward(self, x, fast=True, imu_num=None):
        if imu_num is None:
            imu_num = len(self.v_imu)

        D = self._get_dims(imu_num)

        a0 = 0
        a1 = a0 + D["dim_a"]
        w0 = a1
        w1 = w0 + D["dim_w"]
        rrb0 = w1
        rrb1 = rrb0 + D["dim_rrb"]
        gr0 = rrb1
        gr1 = gr0 + D["dim_gr"]

        # PL-s1
        RRB = [x_[:, rrb0:rrb1].view(-1, D["num_input_leaf"], 3, 3) for x_, y_ in x]
        gR0 = [x_[:, gr0:gr1] for x_, y_ in x]

        x1 = [(x_, y_[:D["pl_out"]]) for x_, y_ in x]
        x1 = self.plnet(x1)   # pRB, gR

        # IK-s1
        pRB = [x_[:, :D["dim_prb"]] for x_ in x1]
        gR1 = [art.math.normalize_tensor(x_[:, D["dim_prb"]:].clone().detach()) for x_ in x1]
        RRB = [
            art.math.from_to_rotation_matrix(gR0_, gR1_).unsqueeze(1).matmul(RRB_)
            for gR0_, gR1_, RRB_ in zip(gR0, gR1, RRB)
        ]
        x2 = [
            torch.cat((RRB_.flatten(1), gR1_, pRB_), dim=1)
            for RRB_, gR1_, pRB_ in zip(RRB, gR1, pRB)
        ]
        x2 = self.iknet.net1(x2)   # pRJ, gR

        # IK-s2
        pRJ = [x_[:, :D["dim_prj"]] for x_ in x2]
        gR2 = [art.math.normalize_tensor(x_[:, D["dim_prj"]:].clone().detach()) for x_ in x2]
        RRB = [
            art.math.from_to_rotation_matrix(gR1_, gR2_).unsqueeze(1).matmul(RRB_)
            for gR1_, gR2_, RRB_ in zip(gR1, gR2, RRB)
        ]
        x3 = [
            torch.cat((RRB_.flatten(1), gR2_, pRJ_), dim=1)
            for RRB_, gR2_, pRJ_ in zip(RRB, gR2, pRJ)
        ]
        x3 = self.iknet.net2(x3)   # RRJ (r6d)

        # VR-s1
        if fast:
            RRJ = [art.math.r6d_to_rotation_matrix(x3_.detach()).view(-1, D["dim_rrj_mat"]) for x3_ in x3]

            awRB = [
                x_[:, :D["dim_a"] + D["dim_w"]]
                .view(-1, 2 * D["imu_num"], 3)
                .bmm(art.math.from_to_rotation_matrix(gR2_, gR0_))
                .view(-1, D["dim_a"] + D["dim_w"])
                for gR2_, gR0_, (x_, y_) in zip(gR2, gR0, x)
            ]

            x4 = [
                (
                    torch.cat((RRJ_, pRJ_.detach(), awRB_, gR2_), dim=1),
                    torch.zeros_like(y_[-D["dim_vr_out"]:]) if torch.isnan(y_[-D["dim_vr_out"]]) else y_[-D["dim_vr_out"]:]
                )
                for RRJ_, pRJ_, awRB_, gR2_, (x_, y_) in zip(RRJ, pRJ, awRB, gR2, x)
            ]
        else:
            RRJ, pRJ_new, aRB_new, wRB_new, pose = [], [], [], [], []

            aRB = [x_[:, :D["dim_a"]].view(-1, D["imu_num"], 3) for x_, y_ in x]
            wRB = [x_[:, D["dim_a"]:D["dim_a"] + D["dim_w"]].view(-1, D["imu_num"], 3) for x_, y_ in x]

            for i in range(len(x3)):
                x3_ = x3[i].clone().detach().cpu()
                RRJ_ = art.math.r6d_to_rotation_matrix(x3_).view(-1, 15, 3, 3)
                glb_pose_ = torch.eye(3).repeat(RRJ_.shape[0], 24, 1, 1)
                glb_pose_[:, self.j_reduce] = RRJ_
                pose_ = self.body_model.inverse_kinematics_R(glb_pose_).view(-1, 24, 3, 3)
                pose_[:, self.j_ignore] = torch.eye(3)
                pRJ_ = self.body_model.forward_kinematics(pose_)[1][:, 1:]
                aRB_ = aRB[i].bmm(art.math.from_to_rotation_matrix(gR2[i], gR0[i]))
                wRB_ = wRB[i].bmm(art.math.from_to_rotation_matrix(gR2[i], gR0[i]))
                RRJ.append(RRJ_)
                pRJ_new.append(pRJ_)
                aRB_new.append(aRB_)
                wRB_new.append(wRB_)
                pose.append(pose_)

            no_translation = [torch.isnan(y_[-D["dim_vr_out"]]).item() for x_, y_ in x]
            x4 = [
                (
                    torch.cat((
                        RRJ_.flatten(1).to(aRB_.device),
                        pRJ_.flatten(1).to(aRB_.device),
                        aRB_.flatten(1),
                        wRB_.flatten(1),
                        gR2_
                    ), dim=1),
                    x_[1][-D["dim_vr_out"]:] if not nt_ else torch.zeros_like(x_[1][-D["dim_vr_out"]:])
                )
                for RRJ_, pRJ_, aRB_, wRB_, gR2_, x_, nt_ in zip(RRJ, pRJ_new, aRB_new, wRB_new, gR2, x, no_translation)
            ]

        x4 = self.vrnet(x4)   # vRR_V, vRR_H, stationary_prob
        result = [torch.cat((x1_, x2_, x3_, x4_), dim=1) for x1_, x2_, x3_, x4_ in zip(x1, x2, x3, x4)]
        return result



class Full_GR_OV_3(torch.nn.Module):
    dt = 1 / 60
    mu = 0.7
    kp_pose = 3600
    kd_pose = 60
    kp_tran = 3600
    kd_tran = 60
    alpha_pd = 0.0
    floor_y = -0.97
    beta_velocity = 1
    beta_cjoint = 1
    beta_extforce = 0.4
    beta_torque = 1e-3 / 80

    j_reduce = (1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19)
    j_ignore = (0, 7, 8, 10, 11, 20, 21, 22, 23)
    j_contact = (0, 10, 11, 22, 23)

    # 3 IMUs: left wrist, right wrist, root
    v_imu = (1961, 5424, 4298)

    class Visualization:
        enable = False
        show_residual_force = False
        show_contact_force = True
        show_block = True
        show_contact = False
        show_stationary = False
        show_torque = False

    def __init__(self):
        from articulate.utils.torch import RNN, RNNWithInit
        super(Full_GR_OV_3, self).__init__()

        self.plnet = RNNWithInit(
            input_linear=False,
            input_size=39,
            output_size=9,
            hidden_size=512,
            num_rnn_layer=3,
            dropout=0.4
        )

        self.iknet = torch.nn.ModuleDict({
            'net1': RNN(
                input_linear=False,
                input_size=27,
                output_size=72,
                hidden_size=512,
                num_rnn_layer=3,
                dropout=0.4
            ),
            'net2': RNN(
                input_linear=False,
                input_size=90,
                output_size=90,
                hidden_size=512,
                num_rnn_layer=3,
                dropout=0.4
            )
        })

        self.vrnet = RNNWithInit(
            input_linear=False,
            input_size=225,
            output_size=9,
            hidden_size=512,
            num_rnn_layer=3,
            dropout=0.4
        )

        # for training
        # self.plnet.load_state_dict(torch.load('data/weight_3/Pose-GR/PL/best_weights.pt'))
        # self.iknet.load_state_dict(torch.load('data/weight_3/Pose-GR/IK/best_weights.pt'))
        # self.vrnet.load_state_dict(torch.load('data/weight_3/Tran-OV/VR/best_weights.pt'))

        # for testing
        self.load_state_dict(torch.load('data/weight_3/Full-GR-OV/full/weights.pt'))

        self.B = np.array([
            [self.mu, -self.mu, 0,        0       ],
            [1,        1,       1,        1       ],
            [0,        0,       self.mu, -self.mu]
        ]) / np.sqrt(1 + self.mu ** 2)

        self.body_model = art.ParametricModel('models/SMPL_male.pkl', vert_mask=self.v_imu)
        self.physics_model = cart.get_dynamic_model('models/SMPL_male.pkl')
        self.eval()

        if self.Visualization.enable:
            from articulate.utils.unity import MotionViewer
            MotionViewer.colors = [(1, 1, 1)]
            self.viewer = MotionViewer(1)
            self.viewer.connect()
            self.scene = Scene(self.viewer)
            self.force_lpf = [art.LowPassFilter(0.3) for _ in range(5)]
            self.torque_lpf = art.LowPassFilter(0.3)
            self.tran_offset = torch.zeros(3)
            if self.Visualization.show_torque:
                self.viewer.show_torque(0, [1, 2, 4, 5, 16, 17, 18, 19])
    @torch.no_grad()
    def rnn_initialize(self, init_pose=None, init_vel=None):
        """
        Initialize the hidden states of the RNNs.
        """
        init_pose = torch.eye(3).expand(1, 24, 3, 3) if init_pose is None else init_pose.cpu().view(1, 24, 3, 3)
        init_vel = torch.zeros(3) if init_vel is None else init_vel.cpu().view(3)

        root_idx = 2

        vRR_V = init_vel[1].view(1).clone()
        init_vel[1] = 0

        _, j, v = self.body_model.forward_kinematics(init_pose, calc_mesh=True)

        # 3 IMU: 2 non-root points
        pRL = (v[0, :root_idx] - v[0, root_idx:]).mm(init_pose[0, 0]).ravel()   # 2*3 = 6
        gR = -init_pose[0, 0, 1]                                                 # 3
        x1 = torch.cat((pRL, gR)).to(self.plnet.init_net[0].weight.device)       # 9

        h = -j[:, :, 1].min().view(1)
        vRR_H = init_pose[0, 0].t().mm(init_vel.unsqueeze(-1)).squeeze(-1)
        c = torch.zeros(5)
        x2 = torch.cat((vRR_V, vRR_H, c)).to(self.vrnet.init_net[0].weight.device)   # 9

        self.pl1hc = [_.contiguous() for _ in self.plnet.init_net(x1).view(1, 2, self.plnet.num_layers, self.plnet.hidden_size).permute(1, 2, 0, 3)]
        self.vr1hc = [_.contiguous() for _ in self.vrnet.init_net(x2).view(1, 2, self.vrnet.num_layers, self.vrnet.hidden_size).permute(1, 2, 0, 3)]
        self.ik1hc = None
        self.ik2hc = None
        self.last_cjoint = torch.tensor([0, h + self.floor_y, 0]) + j[0, self.j_contact]
        self.is_init = False
        self.contact = np.zeros(5, dtype=bool)
        self.contact_counter = np.zeros(5, dtype=int)

    @torch.no_grad()
    def _explain_residual_force(self, contact, contact_Jacobian, contact_position, residual_force):
        if np.any(contact):
            J = contact_Jacobian.reshape(5, 3, 75)[contact, :, :6].reshape(-1, 6)
            B, lb = [], []
            for i in np.where(contact)[0]:
                if i > 3 and contact_position[i, 1] > self.floor_y + 0.15:      # hand grasp
                    B.append(np.eye(3))
                    lb.append(-np.ones(3) * np.inf)
                else:
                    B.append(self.B)
                    lb.append(np.zeros(4))
            JTB = J.T @ art.math.block_diagonal_matrix_np(B)
            P = csc_array(JTB.T @ JTB + self.beta_extforce * np.eye(JTB.shape[1]))
            q = -JTB.T @ residual_force
            force = solve_qp(P, q, lb=np.concatenate(lb), solver='osqp')
            error = np.linalg.norm(JTB @ force - residual_force)
        else:
            B = None
            force = np.zeros(0)
            error = np.linalg.norm(residual_force)
        return force, error, B



    def _get_dims(self, imu_num: int):
        root_idx = imu_num - 1
        num_leaf = root_idx

        D = {
            "imu_num": imu_num,
            "root_idx": root_idx,
            "num_leaf": num_leaf,

            "dim_a": imu_num * 3,
            "dim_w": imu_num * 3,
            "dim_rrb": num_leaf * 9,
            "dim_prb": num_leaf * 3,

            "dim_gr": 3,
            "dim_prj": 23 * 3,       # 69
            "dim_rrj_r6d": 15 * 6,   # 90
            "dim_rrj_mat": 15 * 9,   # 135
            "dim_vr_out": 9,
        }

        D["x_in"] = D["dim_a"] + D["dim_w"] + D["dim_rrb"] + D["dim_gr"]
        D["pl_out"] = D["dim_prb"] + D["dim_gr"]
        D["ik1_in"] = D["dim_rrb"] + D["dim_gr"] + D["dim_prb"]
        D["ik1_out"] = D["dim_prj"] + D["dim_gr"]
        D["ik2_in"] = D["dim_rrb"] + D["dim_gr"] + D["dim_prj"]
        D["ik2_out"] = D["dim_rrj_r6d"]
        D["vr_in"] = D["dim_rrj_mat"] + D["dim_prj"] + D["dim_a"] + D["dim_w"] + D["dim_gr"]

        return D


    
    @torch.no_grad()
    def forward_frame(self, a, w, R):
        root_idx = 2

        # a, w: [3, 3], R: [3, 3, 3]
        aRB = a.mm(R[root_idx])                            # [3, 3]
        wRB = w.mm(R[root_idx])                            # [3, 3]
        RRB = R[root_idx].t().matmul(R[:root_idx])         # [2, 3, 3]
        gR0 = -R[root_idx, 1]                              # [3]

        # PL-s1
        x = torch.cat((aRB.ravel(), wRB.ravel(), RRB.ravel(), gR0))   # 9+9+18+3 = 39
        x, self.pl1hc = self.plnet.rnn(x.view(1, 1, -1), self.pl1hc)
        x = self.plnet.linear2(x.squeeze())                  # pRB(6), gR(3)
        gR1 = art.math.normalize_tensor(x[6:])
        RRB = art.math.from_to_rotation_matrix(gR0, gR1).matmul(RRB)

        # IK-s1
        x = torch.cat((RRB.ravel(), gR1, x[:6]))             # 18+3+6 = 27
        x, self.ik1hc = self.iknet.net1.rnn(x.view(1, 1, -1), self.ik1hc)
        x = self.iknet.net1.linear2(x.squeeze())             # pRJ(69), gR(3)
        gR2 = art.math.normalize_tensor(x[69:])
        RRB = art.math.from_to_rotation_matrix(gR1, gR2).matmul(RRB)

        # IK-s2
        x = torch.cat((RRB.ravel(), gR2, x[:69]))            # 18+3+69 = 90
        x, self.ik2hc = self.iknet.net2.rnn(x.view(1, 1, -1), self.ik2hc)
        x = self.iknet.net2.linear2(x.squeeze())             # RRJ(90)

        # get pose estimation
        RRJ = art.math.r6d_to_rotation_matrix(x).cpu()
        glb_pose = torch.eye(3).repeat(1, 24, 1, 1)
        glb_pose[:, self.j_reduce] = RRJ.view(1, 15, 3, 3)
        pose = self.body_model.inverse_kinematics_R(glb_pose).view(24, 3, 3)
        pose[self.j_ignore, ...] = torch.eye(3)
        pRJ = self.body_model.forward_kinematics(pose.unsqueeze(0))[1][0, 1:]
        pose[0] = R[root_idx].mm(art.math.from_to_rotation_matrix(gR2, gR0).squeeze()).cpu()

        # VR-s1
        aRB = a.cpu().mm(pose[0])                            # [3, 3]
        wRB = w.cpu().mm(pose[0])                            # [3, 3]
        x = torch.cat((RRJ.ravel(), pRJ.ravel(), aRB.ravel(), wRB.ravel(), gR2.cpu())).to(gR2.device)
        # RRJ:135 + pRJ:69 + aRB:9 + wRB:9 + gR2:3 = 225
        x, self.vr1hc = self.vrnet.rnn(x.view(1, 1, -1), self.vr1hc)
        x = self.vrnet.linear2(x.squeeze())

        vRR_V, vRR_H, stationary_prob = x[0].item(), x[1:4].cpu(), x[4:].sigmoid().cpu()
        vWR = pose[0].mm(vRR_H.unsqueeze(-1)).squeeze(-1)
        vWR[1] = vRR_V
        cjoint = torch.cat((torch.zeros(1, 3), pRJ.mm(pose[0].t())))[self.j_contact, :]
        stationary_weight = (stationary_prob * 5 - 3).clip(0, 1)
        velocity = (stationary_weight.unsqueeze(0).mm(self.last_cjoint - cjoint)[0] / self.dt + self.beta_velocity * vWR) / (self.beta_velocity + stationary_weight.sum())
        self.last_cjoint = cjoint

        if not self.is_init:
            self.is_init = True
            self.physics_model.set_state_R(pose.numpy(), np.array([0, self.floor_y - cjoint[:, 1].min().item(), 0]), np.zeros(75))
        else:
            pose_cur, tran_cur, qdot = self.physics_model.get_state_R()
            cjoint_cur = np.vstack([self.physics_model.get_position(j) for j in self.j_contact])
            cvel_cur = np.vstack([self.physics_model.get_linear_velocity(j) for j in self.j_contact])
            cJ_cur = np.vstack([self.physics_model.get_linear_Jacobian(j) for j in self.j_contact])
            cJdot_cur = np.vstack([self.physics_model.get_linear_Jacobian_dot(j) for j in self.j_contact])
            M = self.physics_model.mass_matrix()
            h = self.physics_model.inverse_dynamics(np.zeros(75))
            stationary = stationary_prob.numpy() > 0.7

            R_pd = art.math.axis_angle_to_rotation_matrix(torch.from_numpy(qdot[3:]) * self.alpha_pd * self.dt)
            delta_pose = art.math.rotation_matrix_to_axis_angle(torch.from_numpy(pose_cur).bmm(R_pd).transpose(1, 2).bmm(pose)).ravel().numpy()
            thetaddotdes = (self.kp_pose * delta_pose - self.kd_pose * qdot[3:]) / (1 + self.kd_pose * self.alpha_pd * self.dt)

            cjoint = tran_cur + velocity.numpy() * self.dt + cjoint.numpy()
            cjoint = art.math.lerp(cjoint, cjoint_cur, stationary_weight.view(5, 1).numpy())
            delta_tran = cjoint - cjoint_cur - cvel_cur * self.alpha_pd * self.dt
            rddotdes = (self.kp_tran * delta_tran - self.kd_tran * cvel_cur).ravel() / (1 + self.kd_tran * self.alpha_pd * self.dt)

            cjoint_cur[0, 1] -= 0.15
            cjoint[0, 1] -= 0.15
            k = np.ones((5, 3)) * self.beta_cjoint
            k[self.contact] *= 10
            A = np.vstack((np.hstack((np.zeros((72, 3)), np.eye(72))), np.sqrt(k.reshape(15, 1)) * cJ_cur, np.sqrt(self.beta_torque) * M))
            b = np.concatenate((thetaddotdes, np.sqrt(k.reshape(15)) * (-cJdot_cur @ qdot + rddotdes), np.sqrt(self.beta_torque) * (-h)))
            qddot = lsqr(csc_array(A), b)[0]
            residual_force = M[:6] @ qddot + h[:6]

            vdist = np.abs(cjoint_cur[np.newaxis, :, 1] - cjoint_cur[:, np.newaxis, 1])
            contact = stationary & (self.contact | (cjoint_cur[:, 1] < self.floor_y + 0.05))
            if np.any(contact):
                contact |= stationary & (vdist[contact].min(axis=0) < 0.05)
            potential_contact = stationary & ~contact

            if contact[0] or potential_contact[0]:
                lleg = self.physics_model.get_position(4) - self.physics_model.get_position(1)
                rleg = self.physics_model.get_position(5) - self.physics_model.get_position(2)
                if min(np.arccos(-lleg[1] / np.linalg.norm(lleg)), np.arccos(-rleg[1] / np.linalg.norm(rleg))) < np.pi / 4:
                    contact[0], potential_contact[0] = False, False

            force, err, forceB = self._explain_residual_force(contact, cJ_cur, cjoint_cur, residual_force)
            for i in np.argsort(cjoint_cur[:, 1]):
                if err > 400 and potential_contact[i]:
                    contact[i] = True
                    force_new, err_new, forceB_new = self._explain_residual_force(contact, cJ_cur, cjoint_cur, residual_force)
                    self.contact_counter[i] = self.contact_counter[i] + 1 if err_new / err < 0.6 else 0
                    if self.contact_counter[i] >= 5:
                        force, err, forceB = force_new, err_new, forceB_new
                    else:
                        contact[i] = False
                else:
                    self.contact_counter[i] = 0

            near_ground = cjoint[:, 1] < self.floor_y + 0.15
            for i in np.where(contact & near_ground)[0]:
                cjoint[i, 1] = art.math.lerp(cjoint[i, 1], self.floor_y, 0.1)
            cjoint[cjoint[:, 1] < self.floor_y, 1] = self.floor_y

            delta_tran = cjoint - cjoint_cur - cvel_cur * self.alpha_pd * self.dt
            rddotdes = (self.kp_tran * delta_tran - self.kd_tran * cvel_cur).ravel() / (1 + self.kd_tran * self.alpha_pd * self.dt)
            if np.any(contact):
                J = cJ_cur.reshape(5, 3, 75)[contact].reshape(-1, 75)
                B = art.math.block_diagonal_matrix_np(forceB)
                force = B @ force
                torque = J.T @ force
            else:
                torque = np.zeros(75)

            A = np.vstack((np.hstack((np.zeros((72, 3)), np.eye(72))), np.sqrt(k.reshape(15, 1)) * cJ_cur, np.sqrt(self.beta_torque * 3) * M))
            b = np.concatenate((thetaddotdes, np.sqrt(k.reshape(15)) * (-cJdot_cur @ qdot + rddotdes), np.sqrt(self.beta_torque * 3) * (-h + torque)))
            qddot = lsqr(csc_array(A), b)[0]
            self.physics_model.update_state(qddot, self.dt)
            self.contact = contact

        refined_pose, refined_tran, qdot = self.physics_model.get_state_R()
        return torch.from_numpy(refined_pose), torch.from_numpy(refined_tran)




import numpy as np
import torch
import torch.nn as nn

# 这里假定 art, cart, Scene, solve_qp, lsqr, csc_array 等依然在模块其他位置 import 好了
# from articulate import something as art
# from articulate.utils import cart
# from some_module import Scene
# from some_qp_lib import solve_qp
# from scipy.sparse.linalg import lsqr
# from scipy.sparse import csc_array
import numpy as np
import torch
import torch.nn as nn



class IMUSTEncoder(nn.Module):
    """
    SpatioTemporal encoder over tokens (t, k): L = T*K
    """
    def __init__(self, in_dim: int, d_model=256, nhead=8, num_layers=4, dropout=0.1, max_t=512):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.sensor_id_emb = nn.Embedding(10, d_model)   # sensor id 0..9
        self.time_emb = nn.Embedding(max_t, d_model)     # time id 0..max_t-1

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x_vis, vis_ids, src_key_padding_mask=None):
        """
        x_vis: [B,T,K,C]
        vis_ids: [B,K]
        src_key_padding_mask: [B, T*K]  True=padding
        return memory: [B, T*K, D]
        """
        B, T, K, C = x_vis.shape

        # tokens: flatten (t,k)
        x = x_vis.reshape(B, T * K, C)  # [B, L, C], L=T*K

        # sensor ids per token spatial
        sid = vis_ids[:, None, :].expand(B, T, K).reshape(B, T * K)  # [B, L]

        # time ids per token
        tid = torch.arange(T, device=x_vis.device)[None, :, None].expand(B, T, K).reshape(B, T * K)  # [B, L]

        h = self.in_proj(x) + self.sensor_id_emb(sid) + self.time_emb(tid)  # [B,L,D]
        mem = self.enc(h, src_key_padding_mask=src_key_padding_mask)
        return mem


class IMUSTDecoder(nn.Module):
    """
    Decoder queries over tokens (t, n): Lq = T*10
    """
    def __init__(self, out_dim: int, d_model=256, nhead=8, num_layers=4, dropout=0.1, max_t=512):
        super().__init__()
        self.query_sensor_emb = nn.Embedding(10, d_model)  # target sensor id 0..9
        self.query_time_emb = nn.Embedding(max_t, d_model)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.dec = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, out_dim)

    def forward(self, memory, T: int, memory_key_padding_mask=None):
        """
        memory: [B, L, D] where L=T*K
        memory_key_padding_mask: [B, L] True=padding
        return pred_full: [B, T, 10, C]
        """
        B, L, D = memory.shape
        device = memory.device

        # build queries for all (t, n)
        t_ids = torch.arange(T, device=device)                   # [T]
        n_ids = torch.arange(10, device=device)                  # [10]
        tt = t_ids[:, None].expand(T, 10).reshape(T * 10)        # [T*10]
        nn_ids = n_ids[None, :].expand(T, 10).reshape(T * 10)    # [T*10]

        tgt = (self.query_time_emb(tt) + self.query_sensor_emb(nn_ids))  # [T*10, D]
        tgt = tgt[None, :, :].expand(B, T * 10, D)                        # [B, T*10, D]

        h = self.dec(
            tgt=tgt,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # [B, T*10, D]
        out = self.out_proj(h)  # [B, T*10, C]
        return out.reshape(B, T, 10, -1)


class MaskedIMUAutoEncoder(nn.Module):
    """
    SpatioTemporal token AE:
      encoder sees (t,k) visible tokens
      decoder queries (t,n) for all 10 sensors
    """
    def __init__(self, feat_dim, d_model=256, nhead=8, enc_layers=4, dec_layers=4, dropout=0.1, max_t=512):
        super().__init__()
        self.encoder = IMUSTEncoder(feat_dim, d_model, nhead, enc_layers, dropout, max_t=max_t)
        self.decoder = IMUSTDecoder(feat_dim, d_model, nhead, dec_layers, dropout, max_t=max_t)

    def forward(self, x_vis, vis_ids, mem_pad_mask=None):
        """
        x_vis: [B,T,K,C]
        vis_ids: [B,K]
        mem_pad_mask: [B,T*K] True=padding
        return pred_full: [B,T,10,C]
        """
        B, T, K, C = x_vis.shape
        memory = self.encoder(x_vis, vis_ids, src_key_padding_mask=mem_pad_mask)  # [B,T*K,D]
        pred_full = self.decoder(memory, T=T, memory_key_padding_mask=mem_pad_mask)  # [B,T,10,C]
        return pred_full
