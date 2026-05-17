__all__ = ['PNPGR_HC']


import torch
import articulate as art
from articulate.utils.torch import RNN, RNNWithInit
from dynamics import PhysicsOptimizer


class PNPGR_HC(torch.nn.Module):
    v_imu = (1961, 5424, 1176, 4662, 411, 3021)
    j_reduce = (1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19)
    j_ignore = (0, 7, 8, 10, 11, 20, 21, 22, 23)
    j_contact = (0, 10, 11, 22, 23)

    def __init__(self):
        super(PNPGR_HC, self).__init__()
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
                                 input_size=357,
                                 output_size=9,
                                 hidden_size=512,
                                 num_rnn_layer=3,
                                 dropout=0.4)

        self.plnet.load_state_dict(torch.load('data/weights/PNP-GR/PL/best_weights.pt'))
        self.iknet.load_state_dict(torch.load('data/weights/PNP-GR/IK/best_weights.pt'))
        self.vrnet.load_state_dict(torch.load('data/weights/PNP-GR-HC/VR/best_weights.pt'))

        self.body_model = art.ParametricModel('models/SMPL_male.pkl', vert_mask=self.v_imu)
        self.dynamics_optimizer = PhysicsOptimizer()
        self.rnn_initialize()  # using T-pose
        self.eval()

    @torch.no_grad()
    def rnn_initialize(self, init_pose=None, init_vel=None):
        init_pose = torch.eye(3).expand(1, 24, 3, 3) if init_pose is None else init_pose.cpu().view(1, 24, 3, 3)
        init_vel = torch.zeros(3) if init_vel is None else init_vel.cpu().view(3)
        _, j, v = self.body_model.forward_kinematics(init_pose, calc_mesh=True)
        pRL, gR = (v[0, :5] - v[0, 5:]).mm(init_pose[0, 0]).ravel(), -init_pose[0, 0, 1]
        x1 = torch.cat((pRL, gR)).to(self.plnet.init_net[0].weight.device)
        h, vRR, c = -j[:, :, 1].min().view(1), init_vel, torch.zeros(5)
        x2 = torch.cat((h, vRR, c)).to(self.vrnet.init_net[0].weight.device)
        self.pl1hc = [_.contiguous() for _ in self.plnet.init_net(x1).view(1, 2, self.plnet.num_layers, self.plnet.hidden_size).permute(1, 2, 0, 3)]
        self.vr1hc = [_.contiguous() for _ in self.vrnet.init_net(x2).view(1, 2, self.vrnet.num_layers, self.vrnet.hidden_size).permute(1, 2, 0, 3)]
        self.ik1hc = None
        self.ik2hc = None
        self.last_RRJ = None
        self.last_pRJ = None
        self.last_tran = None
        self.dynamics_optimizer.reset_states()

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
        RRJdot = torch.zeros(15, 3, 3) if self.last_RRJ is None else (RRJ - self.last_RRJ) * 60
        pRJdot = torch.zeros(23, 3) if self.last_pRJ is None else (pRJ - self.last_pRJ) * 60
        RRJdot = art.math.vee(RRJdot.bmm(RRJ.transpose(1, 2)))
        aRB = a.cpu().mm(pose[0])
        wRB = w.cpu().mm(pose[0])

        x = torch.cat((RRJ.ravel(), RRJdot.ravel(), pRJ.ravel(), pRJdot.ravel(), aRB.ravel(), wRB.ravel(), gR2.cpu())).to(gR2.device)
        x, self.vr1hc = self.vrnet.rnn(x.view(1, 1, -1), self.vr1hc)
        x = self.vrnet.linear2(x.squeeze())  # h, vRR
        h, vRR, c = x[0].item(), x[1:4].cpu(), x[4:].cpu()
        vWR = pose[0].mm(vRR.unsqueeze(-1)).squeeze(-1)
        tran = self.last_tran + vWR / 60 if self.last_tran is not None else torch.tensor([0, h - 0.97, 0])
        tran[1] = art.math.lerp(tran[1], h - 0.97, 0.05)

        self.last_RRJ = RRJ
        self.last_pRJ = pRJ
        self.last_tran = tran
        self.contact = c.sigmoid().cpu()

        return pose, tran

        # # physics-based optimization
        # av = av.mm(RIR.t())
        # pose_opt, tran_opt = self.dynamics_optimizer.optimize_frame(pose.cpu(), av.cpu(), c.cpu(), a.cpu())
        # return pose_opt, tran_opt
