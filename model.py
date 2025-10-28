import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    """A single GCN layer using pre-normalized adjacency (D^{-1/2}(A+I)D^{-1/2})."""
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x, adj_norm):
        # x: [N, Fin], adj_norm: [N, N]
        x = self.lin(x)
        return adj_norm @ x

class GCNEncoder(nn.Module):
    """Shallow GCN encoder with PReLU, as in DGI transductive setup."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.gcn = GCNLayer(in_dim, out_dim)
        self.act = nn.PReLU(out_dim)

    def forward(self, x, adj_norm):
        h = self.gcn(x, adj_norm)
        return self.act(h)

class Readout(nn.Module):
    """R(H) = sigmoid(mean(H))"""
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, h):
        return self.sigmoid(h.mean(dim=0, keepdim=True))

class BilinearDiscriminator(nn.Module):
    """D(h_i, s) = σ(h_i^T W s)."""
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Bilinear(dim, dim, 1, bias=False)

    def forward(self, h, s):
        score = self.w(h, s.expand_as(h))
        return torch.sigmoid(score)

class DGI(nn.Module):
    def __init__(self, encoder, out_dim):
        super().__init__()
        self.encoder = encoder
        self.readout = Readout()
        self.disc = BilinearDiscriminator(out_dim)

    @torch.no_grad()
    def corrupt(self, x):
        idx = torch.randperm(x.size(0), device=x.device)
        return x[idx]

    def forward(self, x, adj_norm):
        h = self.encoder(x, adj_norm)
        s = self.readout(h)
        x_tilde = self.corrupt(x)
        h_tilde = self.encoder(x_tilde, adj_norm)
        return h, h_tilde, s

    def loss(self, h, h_tilde, s):
        pos = self.disc(h, s)
        neg = self.disc(h_tilde, s)
        l_pos = F.binary_cross_entropy(pos, torch.ones_like(pos))
        l_neg = F.binary_cross_entropy(neg, torch.zeros_like(neg))
        return l_pos + l_neg

    @torch.no_grad()
    def embed(self, x, adj_norm):
        return self.encoder(x, adj_norm)
