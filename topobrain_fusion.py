#!/usr/bin/env python3
"""

TopoBrain Fusion Engine: Combining 1-Node and 8-Node Architectures


FUSION STRATEGY: Prediction-Level Ensemble
Since 1-node (embed=12) and 8-node (embed x 8=96) have incompatible weight
sizes, we implement a two-branch ensemble that combines predictions at
the output level using learnable fusion weights.

The fusion leverages:
- Precision from 8-node model (better training distribution fit)
- Generalization from 1-node model (better high-ω extrapolation)

Key Technical Details:
- Base models are frozen (eval mode, no_grad)
- Fusion weights are learnable via gradient descent
- Spectral adaptation gate learns frequency-dependent weighting
- Final prediction: y_fusion = α(ω)·y_1node + (1-α(ω))·y_8node

Author: grisun0
Date: 2026-01-14

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import json
import os
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
from torch.utils.data import Dataset, DataLoader
from datetime import datetime


@dataclass
class Config:
    device: str = "cpu"
    seed: int = 42
    dt: float = 0.05
    omega_range: Tuple[float, float] = (0.8, 2.5)
    n_samples: int = 1500
    seq_len: int = 15
    grid_size: int = 4
    radial_bins: int = 4
    embed_dim: int = 12
    hidden_dim: int = 24
    batch_size: int = 64
    epochs: int = 60
    lr: float = 0.01
    weight_decay: float = 1e-4
    grok_threshold: float = 5e-4
    curriculum_stages: int = 3
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
                 weight_decay=1e-2, topo_threshold=0.1):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
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
                        p.grad.copy_((grad_flat - parallel_component).view_as(p))
        return super().step(closure)


class TopoBrainPhysical(nn.Module):
    def __init__(self, cfg: Config, msg_angular: int = 4, msg_radial: int = 2):
        super().__init__()
        self.cfg = cfg
        self.msg_angular = msg_angular
        self.msg_radial = msg_radial
        self.num_nodes = msg_angular * msg_radial
        self.embed_dim = cfg.embed_dim

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
        adj = torch.zeros(self.msg_angular, self.msg_angular)
        for i in range(self.msg_angular):
            adj[i, (i-1) % self.msg_angular] = 1.0
            adj[i, (i+1) % self.msg_angular] = 1.0
        return adj

    def _radial_adjacency(self):
        adj = torch.zeros(self.msg_radial, self.msg_radial)
        if self.msg_radial == 1:
            adj[0, 0] = 1.0
        else:
            for i in range(self.msg_radial):
                adj[i, (i-1) % self.msg_radial] = 1.0
                adj[i, (i+1) % self.msg_radial] = 1.0
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

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ======================================================================
# FUSION ENSEMBLE (Predictions Level)
# ======================================================================

class FusionEnsemble(nn.Module):
    """
    Two-branch ensemble combining 1-node and 8-node predictions.
    
    Fusion strategy: Dynamic weighting based on frequency ω
    - Low ω (near training): favor 8-node (precision)
    - High ω (extrapolation): favor 1-node (generalization)
    
    Architecture:
    - model_1node: Frozen pretrained 1-node model
    - model_8node: Frozen pretrained 8-node model  
    - spectral_gate: Learnable network mapping ω → blending weight
    - fusion_weight: Scalar learnable parameter for additional flexibility
    """
    
    def __init__(self, model_1node: TopoBrainPhysical, model_8node: TopoBrainPhysical):
        super().__init__()
        
        # Freeze base models - they are not trainable
        self.model_1node = model_1node
        self.model_8node = model_8node
        
        # Make sure base models are in eval mode and don't compute gradients
        self.model_1node.eval()
        self.model_8node.eval()
        for param in self.model_1node.parameters():
            param.requires_grad = False
        for param in self.model_8node.parameters():
            param.requires_grad = False
        
        # Learnable fusion weights (initialized to balance)
        # This parameter CAN be trained
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        
        # Spectral adaptation layers
        # Maps frequency ω to a blending weight
        self.spectral_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, omega: float) -> torch.Tensor:
        """
        Forward pass with frequency-adaptive fusion.
        
        Args:
            x: Input tensor (batch, seq_len, features)
            omega: Current frequency for adaptive fusion
            
        Returns:
            Fused prediction: y_fusion = α(ω)·y_1node + (1-α(ω))·y_8node
        """
        # Get predictions from both models (with no_grad to prevent gradient flow into base models)
        with torch.no_grad():
            pred_1 = self.model_1node(x)
            pred_8 = self.model_8node(x)
        
        # Compute frequency-based weight using spectral gate
        # omega_tensor: (batch_size, 1) to match model input shape
        omega_tensor = torch.full((pred_1.size(0), 1), omega, device=x.device, dtype=torch.float32)
        spectral_weight = self.spectral_gate(omega_tensor)  # (batch_size, 1)
        
        # Squeeze to get scalar per sample, then broadcast
        w_1 = spectral_weight.squeeze(-1)  # (batch_size,)
        w_8 = 1.0 - w_1
        
        # Ensemble prediction: weighted average of predictions
        # Broadcasting: w_1 (batch,) * pred_1 (batch, 4)
        fused_pred = w_1.unsqueeze(-1) * pred_1 + w_8.unsqueeze(-1) * pred_8
        
        return fused_pred
    
    def get_fusion_info(self, omega: float) -> Dict:
        """Get information about fusion weights at given frequency."""
        omega_tensor = torch.tensor([[omega]], dtype=torch.float32)
        with torch.no_grad():
            spectral_weight = self.spectral_gate(omega_tensor)
        
        return {
            'omega': omega,
            'weight_1node': spectral_weight.item(),
            'weight_8node': 1.0 - spectral_weight.item(),
            'learned_weight': torch.sigmoid(self.fusion_weight).item()
        }


def train_model(model: TopoBrainPhysical, cfg: Config) -> TopoBrainPhysical:
    """Train a model to grokking on cyclotron dynamics."""
    dataset = CyclotronDataset(cfg, omega=0.80, n_samples=cfg.n_samples)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    optimizer = OrthogonalAdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)
    best_mse = float('inf')
    
    print(f"    Training {model.msg_angular}×{model.msg_radial} ({model.num_nodes} nodes)...")
    print(f"    Parameters: {model.count_parameters():,}")
    
    for epoch in range(cfg.epochs):
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
            print(f"    Grokking achieved at epoch {epoch+1}: MSE={best_mse:.6f}")
            break
    
    if best_mse >= cfg.grok_threshold:
        print(f"    Final MSE: {best_mse:.6f} (threshold: {cfg.grok_threshold:.6f})")
    
    return model


def evaluate_model(model: nn.Module, cfg: Config, omega_list: List[float], model_type: str = "base") -> Dict:
    """Evaluate model on multiple frequencies."""
    model.eval()
    results = {}
    
    for omega in omega_list:
        dataset = CyclotronDataset(cfg, omega=omega, n_samples=500)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)
        
        total_mse = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(cfg.device), y.to(cfg.device)
                
                if model_type == "fusion":
                    pred = model(x, omega)
                else:
                    pred = model(x)
                
                total_mse += F.mse_loss(pred, y).item()
        
        avg_mse = total_mse / len(loader)
        results[omega] = avg_mse
    
    # Calculate average across all frequencies
    freq_results = {k: v for k, v in results.items() if isinstance(k, float)}
    results['average'] = np.mean(list(freq_results.values()))
    return results


def run_fusion_experiment():
    """Main experiment: train, fuse, and evaluate."""
    
    cfg = Config(device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"TOPOBRAIN FUSION EXPERIMENT: 1-Node + 8-Node Ensemble")
    print(f"{'='*70}")
    print(f"Device: {cfg.device}")
    print(f"Embed dim: {cfg.embed_dim}")
    
    # Evaluation frequencies (spread across training and extrapolation regimes)
    omega_list = [0.9, 1.2, 1.5, 2.0, 2.2]
    
    # ======================================================================
    # Step 1: Train base models
    # ======================================================================
    print(f"\n[STEP 1] Training base models...")
    print("-" * 50)
    
    print("\n  Training 1-node model (specialized for generalization)...")
    model_1node = TopoBrainPhysical(cfg, msg_angular=1, msg_radial=1).to(cfg.device)
    model_1node = train_model(model_1node, cfg)
    results_1node = evaluate_model(model_1node, cfg, omega_list, "base")
    
    print("\n  Training 8-node model (specialized for precision)...")
    model_8node = TopoBrainPhysical(cfg, msg_angular=4, msg_radial=2).to(cfg.device)
    model_8node = train_model(model_8node, cfg)
    results_8node = evaluate_model(model_8node, cfg, omega_list, "base")
    
    # Save checkpoints
    os.makedirs("supertopobrain3", exist_ok=True)
    torch.save({
        'model_state_dict': model_1node.state_dict(),
        'msg_angular': 1,
        'msg_radial': 1
    }, "supertopobrain3/checkpoint_1node.pth")
    torch.save({
        'model_state_dict': model_8node.state_dict(),
        'msg_angular': 4,
        'msg_radial': 2
    }, "supertopobrain3/checkpoint_8node.pth")
    print("\n  Checkpoints saved to supertopobrain3/checkpoint_*.pth")
    
    # ======================================================================
    # Step 2: Create fusion ensemble
    # ======================================================================
    print(f"\n[STEP 2] Creating fusion ensemble...")
    print("-" * 50)
    
    fusion_model = FusionEnsemble(model_1node, model_8node).to(cfg.device)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in fusion_model.parameters() if p.requires_grad)
    print(f"  Trainable fusion parameters: {trainable_params}")
    
    # Show initial fusion weights at different frequencies
    print("\n  Initial fusion weights by frequency:")
    for omega in [0.9, 1.2, 1.5, 2.0, 2.2]:
        info = fusion_model.get_fusion_info(omega)
        print(f"    ω={omega}: 1-node weight={info['weight_1node']:.3f}, 8-node weight={info['weight_8node']:.3f}")
    
    # ======================================================================
    # Step 3: Fine-tune fusion weights
    # ======================================================================
    print(f"\n[STEP 3] Fine-tuning fusion weights...")
    print("-" * 50)
    
    # Set base models to eval (already done in __init__, but being explicit)
    fusion_model.model_1node.eval()
    fusion_model.model_8node.eval()
    
    # Optimizer for ONLY the fusion weights
    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=0.1)
    
    # Training on mix of frequencies (focusing on extrapolation regime)
    training_omegas = [1.5, 2.0, 2.2]
    
    for epoch in range(100):
        optimizer.zero_grad()
        
        total_loss = 0
        num_batches = 0
        
        for omega in training_omegas:
            dataset = CyclotronDataset(cfg, omega=omega, n_samples=256)
            x, y = dataset.data.to(cfg.device), dataset.targets.to(cfg.device)
            
            # Forward pass through fusion model
            pred = fusion_model(x, omega)
            
            # Compute loss - gradients flow only through spectral_gate and fusion_weight
            loss = F.mse_loss(pred, y)
            total_loss += loss
            num_batches += 1
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 25 == 0:
            avg_loss = total_loss.item() / num_batches
            print(f"    Epoch {epoch+1}: Avg Loss={avg_loss:.6f}")
            print(f"      Updated fusion weights:")
            for omega in [0.9, 1.5, 2.0, 2.2]:
                info = fusion_model.get_fusion_info(omega)
                print(f"        ω={omega}: w(1-node)={info['weight_1node']:.3f}, w(8-node)={info['weight_8node']:.3f}")
    
    # ======================================================================
    # Step 4: Evaluate all models
    # ======================================================================
    print(f"\n[STEP 4] Evaluating all models...")
    print("-" * 50)
    
    results_fusion = evaluate_model(fusion_model, cfg, omega_list, "fusion")
    
    # ======================================================================
    # Results Summary
    # ======================================================================
    print(f"\n{'='*70}")
    print("FUSION EXPERIMENT RESULTS")
    print(f"{'='*70}")
    
    # Print comparison table
    print(f"\n{'Model':<20} | {'ω=0.9':<10} | {'ω=1.2':<10} | {'ω=1.5':<10} | {'ω=2.0':<10} | {'ω=2.2':<10} | {'Avg':<10}")
    print("-" * 100)
    
    for name, res in [("1-node", results_1node), ("8-node", results_8node), ("Fusion", results_fusion)]:
        row = f"{name:<20} | {res.get(0.9, 'N/A'):<10.4f} | {res.get(1.2, 'N/A'):<10.4f} | "
        row += f"{res.get(1.5, 'N/A'):<10.4f} | {res.get(2.0, 'N/A'):<10.4f} | {res.get(2.2, 'N/A'):<10.4f} | "
        row += f"{res['average']:<10.4f}"
        print(row)
    
    # ======================================================================
    # Improvement Analysis
    # ======================================================================
    print(f"\n{'='*70}")
    print("IMPROVEMENT ANALYSIS")
    print(f"{'='*70}")
    
    fusion_avg = results_fusion['average']
    node1_avg = results_1node['average']
    node8_avg = results_8node['average']
    
    print(f"\n  Baseline 8-node average MSE: {node8_avg:.6f}")
    print(f"  Baseline 1-node average MSE: {node1_avg:.6f}")
    print(f"  Fusion average MSE:          {fusion_avg:.6f}")
    
    print(f"\n  Fusion vs 8-node: {((node8_avg - fusion_avg) / node8_avg * 100):+.2f}%")
    print(f"  Fusion vs 1-node: {((node1_avg - fusion_avg) / node1_avg * 100):+.2f}%")
    
    if fusion_avg < node8_avg and fusion_avg < node1_avg:
        print("\n  SUCCESS: Fusion is better than both baselines!")
        success_flag = True
    elif fusion_avg < node8_avg:
        print("\n  PARTIAL: Fusion beats 8-node but not 1-node")
        success_flag = False
    elif fusion_avg < node1_avg:
        print("\n  PARTIAL: Fusion beats 1-node but not 8-node")
        success_flag = False
    else:
        print("\n  FAILED: Fusion doesn't improve over baselines")
        success_flag = False
    
    # ======================================================================
    # Frequency-by-Frequency Comparison
    # ======================================================================
    print(f"\n{'='*70}")
    print("FREQUENCY-BY-FREQUENCY COMPARISON")
    print(f"{'='*70}")
    
    fusion_wins = 0
    for omega in omega_list:
        m1 = results_1node[omega]
        m8 = results_8node[omega]
        mf = results_fusion[omega]
        
        best_baseline = min(m1, m8)
        if mf < best_baseline:
            winner = "FUSION"
            fusion_wins += 1
        elif m1 <= m8:
            winner = "1-node"
        else:
            winner = "8-node"
        
        beat_str = "✓" if mf < best_baseline else "✗"
        print(f"  ω={omega}: 1-node={m1:.6f}, 8-node={m8:.6f}, Fusion={mf:.6f} -> Winner: {winner} {beat_str}")
    
    print(f"\n  Fusion wins: {fusion_wins}/{len(omega_list)} frequencies")
    
    # ======================================================================
    # Save Results
    # ======================================================================
    all_results = {
        '1node': results_1node,
        '8node': results_8node,
        'fusion': results_fusion,
        'config': {
            'epochs': cfg.epochs,
            'grok_threshold': cfg.grok_threshold,
            'n_samples': cfg.n_samples,
            'embed_dim': cfg.embed_dim
        },
        'analysis': {
            'fusion_success': success_flag,
            'fusion_wins': f"{fusion_wins}/{len(omega_list)}"
        }
    }
    
    with open("supertopobrain3/fusion_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("OUTPUT FILES")
    print(f"{'='*70}")
    print(f"  - supertopobrain3/checkpoint_1node.pth")
    print(f"  - supertopobrain3/checkpoint_8node.pth")
    print(f"  - supertopobrain3/checkpoint_fusion.pth")
    print(f"  - supertopobrain3/fusion_results.json")
    
    # Save fusion model
    torch.save({
        'model_state_dict': fusion_model.state_dict(),
        'model_1node_state': model_1node.state_dict(),
        'model_8node_state': model_8node.state_dict()
    }, "supertopobrain3/checkpoint_fusion.pth")
    
    return all_results


if __name__ == "__main__":
    run_fusion_experiment()
