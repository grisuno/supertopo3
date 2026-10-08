# root

*Community 0 | 3 files | cohesion 1.00*

## Definition

This community groups 3 file(s) rooted at `root` with dominant language py (cohesion 1.00). Central symbols: `Config`, `CyclotronDataset`, `FusionEnsemble`, `OrthogonalAdamW`, `StableMax`, `TopoBrainPhysical`, `__getitem__`, `__init__`. Core file: `topobrain_fusion.py` (25 symbols). Documented purpose: Autor: Gris Iscomeback Correo electrónico: grisiscomeback[at]gmail[dot]com Fecha de creación: 08/01/2026 Licencia: GPL v3  Descripción:  TopoBrain-Physical v3 f.

## Files

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `app.py` | py | utility | 20 | yes |
| `install.sh` | sh | utility | 0 | no |
| `topobrain_fusion.py` | py | utility | 25 | yes |

## Key Symbols

- `Config` (class, `app.py:23`) `class Config`
- `CyclotronDataset` (class, `app.py:46`) `class CyclotronDataset(Dataset)`
- `__init__` (method, `app.py:47`) `def __init__(self, cfg, omega, n_samples)`
- `_generate` (method, `app.py:53`) `def _generate(self)`
- `__len__` (method, `app.py:79`) `def __len__(self)`
- `__getitem__` (method, `app.py:82`) `def __getitem__(self, idx)`
- `StableMax` (class, `app.py:85`) `class StableMax(Module)`
- `__init__` (method, `app.py:86`) `def __init__(self, beta, epsilon)`
- `forward` (method, `app.py:91`) `def forward(self, x, dim)`
- `OrthogonalAdamW` (class, `app.py:100`) `class OrthogonalAdamW(AdamW)`
- `__init__` (method, `app.py:101`) `def __init__(self, params, lr, betas, eps, weight_decay, amsgrad, topo_threshold`
- `step` (method, `app.py:108`) `def step(self, closure)`
- `TopoBrainPhysical` (class, `app.py:131`) `class TopoBrainPhysical(Module)`
- `__init__` (method, `app.py:132`) `def __init__(self, cfg, grid_size, radial_bins)`
- `_angular_adjacency` (method, `app.py:170`) `def _angular_adjacency(self)` - Grafo FIXED of 4 nodes angulars
- `_radial_adjacency` (method, `app.py:178`) `def _radial_adjacency(self)` - Grafo FIXED of 2 nodes radials
- `forward` (method, `app.py:185`) `def forward(self, x)`
- `expand_grid_weights_topobrain` (method, `app.py:234`) `def expand_grid_weights_topobrain(src_model, target_grid, target_radial)` - Expansion SIMPLE: Only copy weights
- `train_stage` (method, `app.py:257`) `def train_stage(cfg, model, omega, epochs)`
- `main` (method, `app.py:287`) `def main()`
- `Config` (class, `topobrain_fusion.py:41`) `class Config`
- `CyclotronDataset` (class, `topobrain_fusion.py:62`) `class CyclotronDataset(Dataset)`
- `__init__` (method, `topobrain_fusion.py:63`) `def __init__(self, cfg, omega, n_samples)`
- `_generate` (method, `topobrain_fusion.py:69`) `def _generate(self)`
- `__len__` (method, `topobrain_fusion.py:95`) `def __len__(self)`
- `__getitem__` (method, `topobrain_fusion.py:98`) `def __getitem__(self, idx)`
- `StableMax` (class, `topobrain_fusion.py:102`) `class StableMax(Module)`
- `__init__` (method, `topobrain_fusion.py:103`) `def __init__(self, beta, epsilon)`
- `forward` (method, `topobrain_fusion.py:108`) `def forward(self, x, dim)`
- `OrthogonalAdamW` (class, `topobrain_fusion.py:118`) `class OrthogonalAdamW(AdamW)`

## Internal vs External Edges

- Internal resolved imports (EXTRACTED): 0
- Cross-boundary resolved imports (EXTRACTED): 0

## Connections

- No cross-community bridges recorded. This community is self-contained.

## Risks

- No scoped security, taint, cycle, or layer risks.

## Open Questions

- Why do 1 file(s) lack file-level docs (e.g. `install.sh`)? What purpose do they serve?
- What would break if the most connected file in root changed?
- Should root be split, given cohesion 1.00?

## Sources

- `app.py`
- `install.sh`
- `topobrain_fusion.py`
