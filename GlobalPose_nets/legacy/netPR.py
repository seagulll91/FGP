__all__ = ['PNPPR']


import torch
import articulate as art
from articulate.utils.torch import RNN, RNNWithInit
from dynamics import PhysicsOptimizer


class PNPPR(torch.nn.Module):
    v_imu = (1961, 5424, 1176, 4662, 411, 3021)
    j_reduce = (1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19)
    j_ignore = (0, 7, 8, 10, 11, 20, 21, 22, 23)

    def __init__(self):
        super(PNPPR, self).__init__()
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
        self.vrnet = torch.nn.ModuleDict({
            'net1': RNN(input_linear=False,
                        input_size=144,
                        output_size=72,
                        hidden_size=512,
                        num_rnn_layer=3,
                        dropout=0.4),
            'net2': RNN(input_linear=False,
                        input_size=144,
                        output_size=2,
                        hidden_size=512,
                        num_rnn_layer=3,
                        dropout=0.4)
        })

        self.plnet.load_state_dict(torch.load('data/weights/PNP-PR/PL/best_weights.pt'))
        self.iknet.load_state_dict(torch.load('data/weights/PNP-PR/IK/best_weights.pt'))
        # self.vrnet.load_state_dict(torch.load(os.path.join(weight_dir, 'VR/best_weights.pt')))

        self.body_model = art.ParametricModel('models/SMPL_male.pkl', vert_mask=self.v_imu)
        self.dynamics_optimizer = PhysicsOptimizer()
        self.rnn_initialize()  # using T-pose
        self.eval()

    @torch.no_grad()
    def rnn_initialize(self, init_pose=None):
        init_pose = torch.eye(3).expand(1, 24, 3, 3) if init_pose is None else init_pose.cpu().view(1, 24, 3, 3)
        RWR, up = init_pose[0, 0], torch.tensor([0, 1, 0.], device=init_pose.device)
        RWP = art.math.from_to_rotation_matrix(RWR[:, 1], up).squeeze().mm(RWR)
        pl = self.body_model.forward_kinematics(init_pose, calc_mesh=True)[2].view(6, 3)
        pl = (pl[:5] - pl[5:]).mm(RWP).ravel().to(self.plnet.init_net[0].weight.device)
        g = -RWR[1].to(self.plnet.init_net[0].weight.device)
        self.pl1hc = [_.contiguous() for _ in self.plnet.init_net(torch.cat((pl, g))).view(1, 2, self.plnet.num_layers, self.plnet.hidden_size).permute(1, 2, 0, 3)]
        self.ik1hc = None
        self.ik2hc = None
        self.vr1hc = None
        self.vr2hc = None
        self.dynamics_optimizer.reset_states()

    @torch.no_grad()
    def forward_frame(self, a, w, R):
        RWP = art.math.from_to_rotation_matrix(R[5, :, 1], torch.tensor([0, 1, 0.], device=R.device)).squeeze().mm(R[5])
        aPB = a.mm(RWP)
        wPB = w.mm(RWP)
        RPB = RWP.t().matmul(R[:5])
        gR0 = -R[5, 1]

        # PL-s1
        x = torch.cat((aPB.ravel(), wPB.ravel(), RPB.ravel(), gR0))
        x, self.pl1hc = self.plnet.rnn(x.view(1, 1, -1), self.pl1hc)
        x = self.plnet.linear2(x.squeeze())        # pPB, gR
        gR1 = art.math.normalize_tensor(x[15:])

        # IK-s1
        x = torch.cat((RPB.ravel(), gR1, x[:15]))
        x, self.ik1hc = self.iknet.net1.rnn(x.view(1, 1, -1), self.ik1hc)
        x = self.iknet.net1.linear2(x.squeeze())   # pPJ, gR
        gR2 = art.math.normalize_tensor(x[69:])

        # IK-s2
        x = torch.cat((RPB.ravel(), gR2, x[:69]))
        x, self.ik2hc = self.iknet.net2.rnn(x.view(1, 1, -1), self.ik2hc)
        x = self.iknet.net2.linear2(x.squeeze())   # RPJ

        # get pose estimation
        reduced_glb_pose = art.math.r6d_to_rotation_matrix(x).view(1, 15, 3, 3).cpu()
        Rroot = R[5].mm(art.math.from_to_rotation_matrix(gR2, gR0).squeeze()).cpu()
        glb_pose = torch.eye(3).repeat(1, 24, 1, 1)
        glb_pose[:, self.j_reduce] = RWP.cpu().matmul(reduced_glb_pose)
        glb_pose[:, 0] = Rroot
        pose = self.body_model.inverse_kinematics_R(glb_pose).view(24, 3, 3)
        pose[self.j_ignore, ...] = torch.eye(3)
        pose[0] = Rroot

        return pose, torch.zeros(3)

        # joint = self.body_model.forward_kinematics(pose.view(1, 24, 3, 3).to(self.device))[1].view(24, 3)
        # aj = joint[1:].mm(RIR)
        # imu = torch.cat((aRB_sta.ravel() / 20, RRB_sta.ravel(), wRR_sta.ravel() / 4, aj.ravel()))
        #
        # # VR-s1
        # x, self.vr1hc = self.vrnet_net1.rnn(imu.unsqueeze(0), self.vr1hc)
        # x = self.vrnet_net1.linear2(x.squeeze(0))
        # av = x.view(24, 3) * 2
        #
        # # VR-s2
        # x, self.vr2hc = self.vrnet_net2.rnn(imu.unsqueeze(0), self.vr2hc)
        # x = self.vrnet_net2.linear2(x.squeeze(0))
        # c = x.view(2)
        #
        # # physics-based optimization
        # av = av.mm(RIR.t())
        # pose_opt, tran_opt = self.dynamics_optimizer.optimize_frame(pose.cpu(), av.cpu(), c.cpu(), a.cpu())
        # return pose_opt, tran_opt
