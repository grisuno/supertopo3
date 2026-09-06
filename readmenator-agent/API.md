# API

## app.py

### expand_grid_weights_topobrain `def expand_grid_weights_topobrain(src_model, target_grid, target_radial)`
- Defined: `app.py:234`
- Doc: Expansion SIMPLE: Only copy weights

### train_stage `def train_stage(cfg, model, omega, epochs)`
- Defined: `app.py:257`

### main `def main()`
- Defined: `app.py:287`

### __init__ `def __init__(self, cfg, omega, n_samples)`
- Defined: `app.py:47`

### _generate `def _generate(self)`
- Defined: `app.py:53`

### __len__ `def __len__(self)`
- Defined: `app.py:79`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `app.py:82`

### __init__ `def __init__(self, beta, epsilon)`
- Defined: `app.py:86`

### forward `def forward(self, x, dim)`
- Defined: `app.py:91`

### __init__ `def __init__(self, params, lr, betas, eps, weight_decay, amsgrad, topo_threshold)`
- Defined: `app.py:101`

### step `def step(self, closure)`
- Defined: `app.py:108`

### __init__ `def __init__(self, cfg, grid_size, radial_bins)`
- Defined: `app.py:132`

### _angular_adjacency `def _angular_adjacency(self)`
- Defined: `app.py:170`
- Doc: Grafo FIXED of 4 nodes angulars

### _radial_adjacency `def _radial_adjacency(self)`
- Defined: `app.py:178`
- Doc: Grafo FIXED of 2 nodes radials

### forward `def forward(self, x)`
- Defined: `app.py:185`

## topobrain_fusion.py

### train_model `def train_model(model, cfg)`
- Defined: `topobrain_fusion.py:343`
- Doc: Train a model to grokking on cyclotron dynamics.

### evaluate_model `def evaluate_model(model, cfg, omega_list, model_type)`
- Defined: `topobrain_fusion.py:381`
- Doc: Evaluate model on multiple frequencies.

### run_fusion_experiment `def run_fusion_experiment()`
- Defined: `topobrain_fusion.py:411`
- Doc: Main experiment: train, fuse, and evaluate.

### __init__ `def __init__(self, cfg, omega, n_samples)`
- Defined: `topobrain_fusion.py:63`

### _generate `def _generate(self)`
- Defined: `topobrain_fusion.py:69`

### __len__ `def __len__(self)`
- Defined: `topobrain_fusion.py:95`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `topobrain_fusion.py:98`

### __init__ `def __init__(self, beta, epsilon)`
- Defined: `topobrain_fusion.py:103`

### forward `def forward(self, x, dim)`
- Defined: `topobrain_fusion.py:108`

### __init__ `def __init__(self, params, lr, betas, eps, weight_decay, topo_threshold)`
- Defined: `topobrain_fusion.py:119`

### step `def step(self, closure)`
- Defined: `topobrain_fusion.py:125`

### __init__ `def __init__(self, cfg, msg_angular, msg_radial)`
- Defined: `topobrain_fusion.py:147`

### _angular_adjacency `def _angular_adjacency(self)`
- Defined: `topobrain_fusion.py:180`

### _radial_adjacency `def _radial_adjacency(self)`
- Defined: `topobrain_fusion.py:187`

### forward `def forward(self, x)`
- Defined: `topobrain_fusion.py:197`

### count_parameters `def count_parameters(self)`
- Defined: `topobrain_fusion.py:247`

### __init__ `def __init__(self, model_1node, model_8node)`
- Defined: `topobrain_fusion.py:270`

### forward `def forward(self, x, omega)`
- Defined: `topobrain_fusion.py:298`
- Doc: Forward pass with frequency-adaptive fusion.

### get_fusion_info `def get_fusion_info(self, omega)`
- Defined: `topobrain_fusion.py:329`
- Doc: Get information about fusion weights at given frequency.
