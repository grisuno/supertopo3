#!/usr/bin/env python3
# _*_ coding: utf8 _*_
"""
app.py

Autor: Gris Iscomeback
Correo electrónico: grisiscomeback[at]gmail[dot]com
Fecha de creación: 08/01/2026
Licencia: GPL v3

Descripción:  TopoBrain-Physical v3 fixed number of nodes to message passing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from dataclasses import dataclass
from typing import Tuple
from torch.utils.data import Dataset, DataLoader

@dataclass
class Config:
    device: str = "cpu"
    seed: int = 42
    dt: float = 0.05
    omega_range: Tuple[float, float] = (0.8, 2.5)
    n_samples: int = 2000
    seq_len: int = 15
    pred_len: int = 1
    grid_size: int = 4
    radial_bins: int = 4
    message_nodes: int = 8  
    embed_dim: int = 12
    hidden_dim: int = 24
    batch_size: int = 64
    epochs: int = 80
    lr: float = 0.01
    weight_decay: float = 1e-4
    grok_threshold: float = 5e-4
    curriculum_stages: int = 3
    expand_after_stage: int = 2
    target_grid: int = 8
    target_radial: int = 8

class CyclotronDataset(Dataset):
    def __init__(self, cfg: Config, omega: float, n_samples: int):
        self.cfg = cfg
        self.omega = omega
        self.n_samples = n_samples
        self.data, self.targets = self._generate()

    def _generate(self):
        dt = self.cfg.dt
        seq_len = self.cfg.seq_len
        data, targets = [], []
        for _ in range(self.n_samples):
            x0 = np.random.uniform(-1, 1)
            y0 = np.random.uniform(-1, 1)
            vx0 = np.random.uniform(-1, 1)
            vy0 = np.random.uniform(-1, 1)
            R = math.sqrt(vx0**2 + vy0**2) / self.omega
            cx = x0 + vy0 / self.omega
            cy = y0 - vx0 / self.omega
            traj = np.zeros((seq_len + 1, 4))
            for t in range(seq_len + 1):
                phase = self.omega * t * dt
                x = cx - R * math.sin(phase)
                y = cy + R * math.cos(phase)
                vx = -self.omega * R * math.cos(phase)
                vy = -self.omega * R * math.sin(phase)
                traj[t] = [x, y, vx, vy]
            data.append(traj[:-1])
            targets.append(traj[-1])
        data_array = np.array(data, dtype=np.float32)
        targets_array = np.array(targets, dtype=np.float32)
        return torch.from_numpy(data_array), torch.from_numpy(targets_array)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class StableMax(nn.Module):
    def __init__(self, beta=0.7, epsilon=1e-8):
        super().__init__()
        self.beta = beta
        self.epsilon = epsilon
        
    def forward(self, x, dim=-1):
        x_max = x.max(dim=dim, keepdim=True)[0]
        x_centered = x - x_max
        stable_exp = torch.sign(x_centered) * torch.pow(
            torch.abs(x_centered) + self.epsilon, self.beta
        ) + 1.0
        sum_stable = stable_exp.sum(dim=dim, keepdim=True)
        return stable_exp / sum_stable.clamp(min=self.epsilon)

class OrthogonalAdamW(torch.optim.AdamW):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, topo_threshold=0.1):
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        self.topo_threshold = topo_threshold
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None and p.dim() > 0:
                    grad = p.grad
                    param_flat = p.view(-1)
                    grad_flat = grad.view(-1)
                    
                    grad_norm = torch.norm(grad_flat)
                    param_norm = torch.norm(param_flat)
                    
                    if grad_norm > self.topo_threshold * param_norm:
                        dot_product = torch.dot(grad_flat, param_flat)
                        norm_sq = (param_norm ** 2).clamp(min=1e-8)
                        parallel_component = (dot_product / norm_sq) * param_flat
                        orthogonal_grad = grad_flat - parallel_component
                        p.grad.copy_(orthogonal_grad.view_as(p))
        return super().step(closure)

class TopoBrainPhysical(nn.Module):
    def __init__(self, cfg: Config, grid_size: int = None, radial_bins: int = None):
        super().__init__()
        self.cfg = cfg
        self.grid_size = grid_size if grid_size is not None else cfg.grid_size
        self.radial_bins = radial_bins if radial_bins is not None else cfg.radial_bins
        self.msg_angular = 4  
        self.msg_radial = 2   
        self.num_nodes = self.msg_angular * self.msg_radial  
        self.embed_dim = cfg.embed_dim
        self.stable_max_ang = StableMax(beta=0.7)
        self.stable_max_rad = StableMax(beta=0.8)

        self.encoder = nn.Sequential(
            nn.Linear(4, cfg.hidden_dim),
            nn.Tanh(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.Tanh(),
            nn.Linear(cfg.hidden_dim, 3)  
        )

        self.node_net = nn.Sequential(
            nn.Linear(self.embed_dim, cfg.hidden_dim),
            nn.Tanh(),
            nn.Linear(cfg.hidden_dim, self.embed_dim)
        )

        
        self.register_buffer('angular_adj', self._angular_adjacency())
        self.register_buffer('radial_adj', self._radial_adjacency())
        self.angular_logit = nn.Parameter(torch.zeros(self.msg_angular))
        self.radial_logit = nn.Parameter(torch.zeros(self.msg_radial))
        
        self.readout = nn.Sequential(
            nn.Linear(self.embed_dim * self.num_nodes, cfg.hidden_dim),
            nn.Tanh(),
            nn.Linear(cfg.hidden_dim, 4)
        )

    def _angular_adjacency(self):
        """Grafo FIXED of 4 nodes angulars"""
        adj = torch.zeros(self.msg_angular, self.msg_angular)
        for i in range(self.msg_angular):
            adj[i, (i-1) % self.msg_angular] = 1.0
            adj[i, (i+1) % self.msg_angular] = 1.0
        return adj

    def _radial_adjacency(self):
        """Grafo FIXED of 2 nodes radials"""
        adj = torch.zeros(self.msg_radial, self.msg_radial)
        adj[0, 1] = 1.0
        adj[1, 0] = 1.0
        return adj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        device = x.device
        all_z = []
        for t in range(L):
            step_input = x[:, t, :]
            z = self.encoder(step_input)
            all_z.append(z)
        z = all_z[-1]
        log_R, phi, v_norm = z[:, 0], z[:, 1], z[:, 2]
        R_normalized = torch.sigmoid(log_R)
        phi_normalized = (phi + math.pi) / (2 * math.pi)
        R_idx = R_normalized * (self.msg_radial - 1)  
        phi_idx = phi_normalized * self.msg_angular    
        grid = torch.zeros(B, self.msg_radial, self.msg_angular, self.embed_dim, device=device)
        for b in range(B):
            r = R_idx[b].item()
            p = phi_idx[b].item()
            r0, p0 = int(r), int(p)
            dr, dp = r - r0, p - p0
            for dr_o in [0, 1]:
                for dp_o in [0, 1]:
                    wr = (1 - dr) if dr_o == 0 else dr
                    wp = (1 - dp) if dp_o == 0 else dp
                    r_i = min(r0 + dr_o, self.msg_radial - 1)
                    p_i = (p0 + dp_o) % self.msg_angular
                    weight = wr * wp
                    if weight > 0:
                        node_embed = torch.cat([z[b], torch.tensor([weight], device=device)])
                        if node_embed.size(0) < self.embed_dim:
                            node_embed = F.pad(node_embed, (0, self.embed_dim - node_embed.size(0)))
                        elif node_embed.size(0) > self.embed_dim:
                            node_embed = node_embed[:self.embed_dim]
                        grid[b, r_i, p_i] += node_embed

        h = grid.view(B, -1, self.embed_dim)
        ang_adj = torch.softmax(self.angular_logit, dim=0).unsqueeze(0) * self.angular_adj
        ang_adj = ang_adj / ang_adj.sum(1, keepdim=True).clamp(min=1e-6)
        h_ang = torch.bmm(ang_adj.unsqueeze(0).expand(B, -1, -1), 
                        h.view(B, self.msg_angular, -1)).view(B, -1, self.embed_dim)
        rad_adj = torch.softmax(self.radial_logit, dim=0).unsqueeze(0) * self.radial_adj
        rad_adj = rad_adj / rad_adj.sum(1, keepdim=True).clamp(min=1e-6)
        h_rad = torch.bmm(rad_adj.unsqueeze(0).expand(B, -1, -1), 
                        h.view(B, self.msg_radial, -1)).view(B, -1, self.embed_dim)
        h = h + h_ang + h_rad
        h = self.node_net(h)

        return self.readout(h.view(B, -1))
        
def expand_grid_weights_topobrain(src_model: TopoBrainPhysical, target_grid: int, target_radial: int) -> TopoBrainPhysical:
    """
    Expansion SIMPLE: Only copy weights
    The topology message passing is Fixed.
    """
    cfg = src_model.cfg
    tgt_model = TopoBrainPhysical(cfg, grid_size=target_grid, radial_bins=target_radial).to(cfg.device)
    
    with torch.no_grad():
        tgt_model.encoder.load_state_dict(src_model.encoder.state_dict())
        tgt_model.angular_logit.copy_(src_model.angular_logit.data)
        tgt_model.radial_logit.copy_(src_model.radial_logit.data)
        tgt_model.readout[0].weight.copy_(src_model.readout[0].weight.data)
        tgt_model.readout[0].bias.copy_(src_model.readout[0].bias.data)
        tgt_model.readout[2].weight.copy_(src_model.readout[2].weight.data)
        tgt_model.readout[2].bias.copy_(src_model.readout[2].bias.data)
        tgt_model.node_net.load_state_dict(src_model.node_net.state_dict())
    
    print(f"[+] Expansion SIMPLE: {src_model.grid_size}x{src_model.radial_bins} → {target_grid}x{target_radial}")
    print(f"   - Topology message passing fixed (4 * 2 nodos)")

    return tgt_model

def train_stage(cfg: Config, model: nn.Module, omega: float, epochs: int) -> Tuple[float, bool]:
    dataset = CyclotronDataset(cfg, omega=omega, n_samples=cfg.n_samples)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    optimizer = OrthogonalAdamW(
        model.parameters(), 
        lr=cfg.lr, 
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.99)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    best_mse = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            pred = model(x)
            loss = F.mse_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(loader)
        if avg_loss < best_mse:
            best_mse = avg_loss
        if best_mse < cfg.grok_threshold:
            return best_mse, True
    return best_mse, False

def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    print("TopoBrain-Physical v3: Nodes Fixed to Message Passing")
    print(f"Device: {cfg.device}")
    base_model = TopoBrainPhysical(cfg).to(cfg.device)
    omegas = np.linspace(cfg.omega_range[0], cfg.omega_range[1], cfg.curriculum_stages)
    grokked = False
    for stage, omega in enumerate(omegas, 1):
        print(f"\n--- Stage {stage}/{cfg.curriculum_stages}  ω={omega:.2f} ---")
        mse, grokked = train_stage(cfg, base_model, omega, epochs=cfg.epochs // cfg.curriculum_stages)
        print(f"MSE: {mse:.6f}  Grokked: {grokked}")
        if grokked and stage >= cfg.expand_after_stage:
            print("[+] Grokking achieved. Expanding torus resolution.")
            break
    print(f"\nExpanding discretization ({cfg.grid_size}x{cfg.radial_bins}) → ({cfg.target_grid}x{cfg.target_radial})")
    print(f"   Message passing tetap di {base_model.msg_angular}×{base_model.msg_radial} nodes (TETAP)")
    expanded_model = expand_grid_weights_topobrain(
        base_model, cfg.target_grid, cfg.target_radial
    )
    test_omega = 2.0
    test_dataset = CyclotronDataset(cfg, omega=test_omega, n_samples=500)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    expanded_model.eval()
    total_mse = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            pred = expanded_model(x)
            total_mse += F.mse_loss(pred, y).item()
    avg_mse = total_mse / len(test_loader)
    print(f"\n Zero-shot MSE on ω={test_omega:.2f}: {avg_mse:.6f}")
    
    if avg_mse < 0.01:
        print("[+] Expansion SUCCEDED: MSE Preserved!")
    elif avg_mse < 0.1:
        print("[-]  Expansion partial: MSE degradated")
    else:
        print("[x] Expansion failed: MSE High")

    torch.save({
        'model_state_dict': expanded_model.state_dict(),
        'angular_bins': cfg.target_grid,
        'radial_bins': cfg.target_radial,
    }, 'topobrain_physical_v3.pth')

if __name__ == "__main__":
    main()
