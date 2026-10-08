# API

## app.py
- `CyclotronDataset.__init__` (method) `app.py:47` `def __init__(self, cfg, omega, n_samples)`
- `StableMax.__init__` (method) `app.py:86` `def __init__(self, beta, epsilon)`
- `StableMax.forward` (method) `app.py:91` `def forward(self, x, dim)`
- `OrthogonalAdamW.__init__` (method) `app.py:101` `def __init__(self, params, lr, betas, eps, weight_decay, amsgrad, topo_threshold)`
- `OrthogonalAdamW.step` (method) `app.py:108` `def step(self, closure)`
- `TopoBrainPhysical.__init__` (method) `app.py:132` `def __init__(self, cfg, grid_size, radial_bins)`
- `TopoBrainPhysical.forward` (method) `app.py:185` `def forward(self, x)`
- `TopoBrainPhysical.expand_grid_weights_topobrain` (method) `app.py:234` `def expand_grid_weights_topobrain(src_model, target_grid, target_radial)` -- Expansion SIMPLE: Only copy weights The topology message passing is Fixed.
- `TopoBrainPhysical.train_stage` (method) `app.py:257` `def train_stage(cfg, model, omega, epochs)`
- `TopoBrainPhysical.main` (method) `app.py:287` `def main()`

## topobrain_fusion.py
- `CyclotronDataset.__init__` (method) `topobrain_fusion.py:63` `def __init__(self, cfg, omega, n_samples)`
- `StableMax.__init__` (method) `topobrain_fusion.py:103` `def __init__(self, beta, epsilon)`
- `StableMax.forward` (method) `topobrain_fusion.py:108` `def forward(self, x, dim)`
- `OrthogonalAdamW.__init__` (method) `topobrain_fusion.py:119` `def __init__(self, params, lr, betas, eps, weight_decay, topo_threshold)`
- `OrthogonalAdamW.step` (method) `topobrain_fusion.py:125` `def step(self, closure)`
- `TopoBrainPhysical.__init__` (method) `topobrain_fusion.py:147` `def __init__(self, cfg, msg_angular, msg_radial)`
- `TopoBrainPhysical.forward` (method) `topobrain_fusion.py:197` `def forward(self, x)`
- `TopoBrainPhysical.count_parameters` (method) `topobrain_fusion.py:247` `def count_parameters(self)`
- `FusionEnsemble.__init__` (method) `topobrain_fusion.py:270` `def __init__(self, model_1node, model_8node)`
- `FusionEnsemble.forward` (method) `topobrain_fusion.py:298` `def forward(self, x, omega)` -- Forward pass with frequency-adaptive fusion.
- `FusionEnsemble.get_fusion_info` (method) `topobrain_fusion.py:329` `def get_fusion_info(self, omega)` -- Get information about fusion weights at given frequency.
- `FusionEnsemble.train_model` (method) `topobrain_fusion.py:343` `def train_model(model, cfg)` -- Train a model to grokking on cyclotron dynamics.
- `FusionEnsemble.evaluate_model` (method) `topobrain_fusion.py:381` `def evaluate_model(model, cfg, omega_list, model_type)` -- Evaluate model on multiple frequencies.
- `FusionEnsemble.run_fusion_experiment` (method) `topobrain_fusion.py:411` `def run_fusion_experiment()` -- Main experiment: train, fuse, and evaluate.
