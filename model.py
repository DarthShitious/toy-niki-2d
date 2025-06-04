import torch
import torch.nn as nn
from FrEIA.framework import InputNode, OutputNode, Node, GraphINN
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom

class SimpleINN2D(nn.Module):
    def __init__(self, input_dim, rig_dim):
        super().__init__()
        self.input_dim = input_dim
        self.rig_dim = rig_dim
        self.latent_dim = input_dim - rig_dim  # extra dimensions for latent uncertainty

        nodes = [InputNode(input_dim, name='input')]

        for k in range(8):  # Deeper network for more expressive power
            nodes.append(Node(nodes[-1], PermuteRandom, {'seed': k}))
            nodes.append(Node(nodes[-1], GLOWCouplingBlock,
                              {'subnet_constructor': self.subnet_fc, 'clamp': 1.5}))

        nodes.append(OutputNode(nodes[-1], name='output'))
        self.inn = GraphINN(nodes)

    def subnet_fc(self, c_in, c_out):
        return nn.Sequential(
            nn.Linear(c_in, 512), nn.ReLU(),
            nn.Linear(512, c_out)
        )

    def forward(self, x, reverse=False):
        if not reverse:
            z, _ = self.inn(x)
            return z
        else:
            x_recon, _ = self.inn(x, rev=True)
            return x_recon

    def predict_rig(self, z):
        return z[:, :self.rig_dim]

    def predict_latent(self, z):
        return z[:, self.rig_dim:]