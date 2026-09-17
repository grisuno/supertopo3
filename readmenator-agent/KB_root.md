# Subsystem: root

## app.py
- Layer: utility
- Doc: app.py  Autor: Gris Iscomeback Correo electrónico: grisiscomeback[at]gmail[dot]com Fecha de creación: 08/01/2026 Licenci
- Language: py
- Symbols:
  - `Config` (class, line 23) `class Config`
  - `CyclotronDataset` (class, line 46) `class CyclotronDataset(Dataset)`
  - `StableMax` (class, line 85) `class StableMax(Module)`
  - `OrthogonalAdamW` (class, line 100) `class OrthogonalAdamW(AdamW)`
  - `TopoBrainPhysical` (class, line 131) `class TopoBrainPhysical(Module)`
  - `expand_grid_weights_topobrain` (method, line 234) `def expand_grid_weights_topobrain(src_model, target_grid, target_radial)`
  - `train_stage` (method, line 257) `def train_stage(cfg, model, omega, epochs)`
  - `main` (method, line 287) `def main()`
  - `__init__` (method, line 47) `def __init__(self, cfg, omega, n_samples)`
  - `_generate` (method, line 53) `def _generate(self)`
  - `__len__` (method, line 79) `def __len__(self)`
  - `__getitem__` (method, line 82) `def __getitem__(self, idx)`
  - `__init__` (method, line 86) `def __init__(self, beta, epsilon)`
  - `forward` (method, line 91) `def forward(self, x, dim)`
  - `__init__` (method, line 101) `def __init__(self, params, lr, betas, eps, weight_decay, amsgrad, topo_threshold)`
  - `step` (method, line 108) `def step(self, closure)`
  - `__init__` (method, line 132) `def __init__(self, cfg, grid_size, radial_bins)`
  - `_angular_adjacency` (method, line 170) `def _angular_adjacency(self)`
  - `_radial_adjacency` (method, line 178) `def _radial_adjacency(self)`
  - `forward` (method, line 185) `def forward(self, x)`

## install.sh
- Layer: utility
- Language: sh

## topobrain_fusion.py
- Layer: utility
- Doc: TopoBrain Fusion Engine: Combining 1-Node and 8-Node Architectures   FUSION STRATEGY: Prediction-Level Ensemble Since 1-
- Language: py
- Symbols:
  - `Config` (class, line 41) `class Config`
  - `CyclotronDataset` (class, line 62) `class CyclotronDataset(Dataset)`
  - `StableMax` (class, line 102) `class StableMax(Module)`
  - `OrthogonalAdamW` (class, line 118) `class OrthogonalAdamW(AdamW)`
  - `TopoBrainPhysical` (class, line 146) `class TopoBrainPhysical(Module)`
  - `FusionEnsemble` (class, line 255) `class FusionEnsemble(Module)`
  - `train_model` (method, line 343) `def train_model(model, cfg)`
  - `evaluate_model` (method, line 381) `def evaluate_model(model, cfg, omega_list, model_type)`
  - `run_fusion_experiment` (method, line 411) `def run_fusion_experiment()`
  - `__init__` (method, line 63) `def __init__(self, cfg, omega, n_samples)`
  - `_generate` (method, line 69) `def _generate(self)`
  - `__len__` (method, line 95) `def __len__(self)`
  - `__getitem__` (method, line 98) `def __getitem__(self, idx)`
  - `__init__` (method, line 103) `def __init__(self, beta, epsilon)`
  - `forward` (method, line 108) `def forward(self, x, dim)`
  - `__init__` (method, line 119) `def __init__(self, params, lr, betas, eps, weight_decay, topo_threshold)`
  - `step` (method, line 125) `def step(self, closure)`
  - `__init__` (method, line 147) `def __init__(self, cfg, msg_angular, msg_radial)`
  - `_angular_adjacency` (method, line 180) `def _angular_adjacency(self)`
  - `_radial_adjacency` (method, line 187) `def _radial_adjacency(self)`
  - `forward` (method, line 197) `def forward(self, x)`
  - `count_parameters` (method, line 247) `def count_parameters(self)`
  - `__init__` (method, line 270) `def __init__(self, model_1node, model_8node)`
  - `forward` (method, line 298) `def forward(self, x, omega)`
  - `get_fusion_info` (method, line 329) `def get_fusion_info(self, omega)`
