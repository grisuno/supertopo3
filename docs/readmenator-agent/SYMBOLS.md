# Symbols

| Symbol | Kind | File:Line | Signature |
|--------|------|-----------|-----------|
| `Config` | class | `app.py:23` | `class Config` |
| `CyclotronDataset` | class | `app.py:46` | `class CyclotronDataset(Dataset)` |
| `OrthogonalAdamW` | class | `app.py:100` | `class OrthogonalAdamW(AdamW)` |
| `StableMax` | class | `app.py:85` | `class StableMax(Module)` |
| `TopoBrainPhysical` | class | `app.py:131` | `class TopoBrainPhysical(Module)` |
| `__getitem__` | method | `app.py:82` | `def __getitem__(self, idx)` |
| `__init__` | method | `app.py:47` | `def __init__(self, cfg, omega, n_samples)` |
| `__init__` | method | `app.py:86` | `def __init__(self, beta, epsilon)` |
| `__init__` | method | `app.py:101` | `def __init__(self, params, lr, betas, eps, weight_decay, amsgrad, topo_threshold)` |
| `__init__` | method | `app.py:132` | `def __init__(self, cfg, grid_size, radial_bins)` |
| `__len__` | method | `app.py:79` | `def __len__(self)` |
| `_angular_adjacency` | method | `app.py:170` | `def _angular_adjacency(self)` |
| `_generate` | method | `app.py:53` | `def _generate(self)` |
| `_radial_adjacency` | method | `app.py:178` | `def _radial_adjacency(self)` |
| `expand_grid_weights_topobrain` | method | `app.py:234` | `def expand_grid_weights_topobrain(src_model, target_grid, target_radial)` |
| `forward` | method | `app.py:91` | `def forward(self, x, dim)` |
| `forward` | method | `app.py:185` | `def forward(self, x)` |
| `main` | method | `app.py:287` | `def main()` |
| `step` | method | `app.py:108` | `def step(self, closure)` |
| `train_stage` | method | `app.py:257` | `def train_stage(cfg, model, omega, epochs)` |
| `Config` | class | `topobrain_fusion.py:41` | `class Config` |
| `CyclotronDataset` | class | `topobrain_fusion.py:62` | `class CyclotronDataset(Dataset)` |
| `FusionEnsemble` | class | `topobrain_fusion.py:255` | `class FusionEnsemble(Module)` |
| `OrthogonalAdamW` | class | `topobrain_fusion.py:118` | `class OrthogonalAdamW(AdamW)` |
| `StableMax` | class | `topobrain_fusion.py:102` | `class StableMax(Module)` |
| `TopoBrainPhysical` | class | `topobrain_fusion.py:146` | `class TopoBrainPhysical(Module)` |
| `__getitem__` | method | `topobrain_fusion.py:98` | `def __getitem__(self, idx)` |
| `__init__` | method | `topobrain_fusion.py:63` | `def __init__(self, cfg, omega, n_samples)` |
| `__init__` | method | `topobrain_fusion.py:103` | `def __init__(self, beta, epsilon)` |
| `__init__` | method | `topobrain_fusion.py:119` | `def __init__(self, params, lr, betas, eps, weight_decay, topo_threshold)` |
| `__init__` | method | `topobrain_fusion.py:147` | `def __init__(self, cfg, msg_angular, msg_radial)` |
| `__init__` | method | `topobrain_fusion.py:270` | `def __init__(self, model_1node, model_8node)` |
| `__len__` | method | `topobrain_fusion.py:95` | `def __len__(self)` |
| `_angular_adjacency` | method | `topobrain_fusion.py:180` | `def _angular_adjacency(self)` |
| `_generate` | method | `topobrain_fusion.py:69` | `def _generate(self)` |
| `_radial_adjacency` | method | `topobrain_fusion.py:187` | `def _radial_adjacency(self)` |
| `count_parameters` | method | `topobrain_fusion.py:247` | `def count_parameters(self)` |
| `evaluate_model` | method | `topobrain_fusion.py:381` | `def evaluate_model(model, cfg, omega_list, model_type)` |
| `forward` | method | `topobrain_fusion.py:108` | `def forward(self, x, dim)` |
| `forward` | method | `topobrain_fusion.py:197` | `def forward(self, x)` |
| `forward` | method | `topobrain_fusion.py:298` | `def forward(self, x, omega)` |
| `get_fusion_info` | method | `topobrain_fusion.py:329` | `def get_fusion_info(self, omega)` |
| `run_fusion_experiment` | method | `topobrain_fusion.py:411` | `def run_fusion_experiment()` |
| `step` | method | `topobrain_fusion.py:125` | `def step(self, closure)` |
| `train_model` | method | `topobrain_fusion.py:343` | `def train_model(model, cfg)` |
