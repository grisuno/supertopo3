```text
❯ python3 supertopo3.py
TopoBrain-Physical v3: Nodes Fixed to Message Passing
Device: cpu

--- Stage 1/3  ω=0.80 ---
MSE: 0.000360  Grokked: True

--- Stage 2/3  ω=1.65 ---
MSE: 0.000499  Grokked: True
[+] Grokking achieved. Expanding torus resolution.

Expanding discretization (4x4) → (8x8)
   Message passing tetap di 4×2 nodes (TETAP)
[+] Expansion SIMPLE: 4x4 → 8x8
   - Topology message passing fixed (4 * 2 nodos)

 Zero-shot MSE on ω=2.00: 0.020778
[-]  Expansion partial: MSE degradated


python3 topobrain_fusion_ensemble.py

======================================================================
TOPOBRAIN FUSION EXPERIMENT: 1-Node + 8-Node Ensemble
======================================================================
Device: cpu
Embed dim: 12

[STEP 1] Training base models...
--------------------------------------------------

  Training 1-node model (specialized for generalization)...
    Training 1×1 (1 nodes)...
    Parameters: 1,821
    Grokking achieved at epoch 24: MSE=0.000480

  Training 8-node model (specialized for precision)...
    Training 4×2 (8 nodes)...
    Parameters: 3,841
    Grokking achieved at epoch 28: MSE=0.000493

  Checkpoints saved to supertopobrain3/checkpoint_*.pth

[STEP 2] Creating fusion ensemble...
--------------------------------------------------
  Trainable fusion parameters: 50

  Initial fusion weights by frequency:
    ω=0.9: 1-node weight=0.403, 8-node weight=0.597
    ω=1.2: 1-node weight=0.381, 8-node weight=0.619
    ω=1.5: 1-node weight=0.364, 8-node weight=0.636
    ω=2.0: 1-node weight=0.345, 8-node weight=0.655
    ω=2.2: 1-node weight=0.341, 8-node weight=0.659

[STEP 3] Fine-tuning fusion weights...
--------------------------------------------------
    Epoch 25: Avg Loss=0.106211
      Updated fusion weights:
        ω=0.9: w(1-node)=0.007, w(8-node)=0.993
        ω=1.5: w(1-node)=0.005, w(8-node)=0.995
        ω=2.0: w(1-node)=0.005, w(8-node)=0.995
        ω=2.2: w(1-node)=0.005, w(8-node)=0.995
    Epoch 50: Avg Loss=0.101904
      Updated fusion weights:
        ω=0.9: w(1-node)=0.054, w(8-node)=0.946
        ω=1.5: w(1-node)=0.048, w(8-node)=0.952
        ω=2.0: w(1-node)=0.049, w(8-node)=0.951
        ω=2.2: w(1-node)=0.050, w(8-node)=0.950
    Epoch 75: Avg Loss=0.103395
      Updated fusion weights:
        ω=0.9: w(1-node)=0.053, w(8-node)=0.947
        ω=1.5: w(1-node)=0.043, w(8-node)=0.957
        ω=2.0: w(1-node)=0.042, w(8-node)=0.958
        ω=2.2: w(1-node)=0.042, w(8-node)=0.958
    Epoch 100: Avg Loss=0.105762
      Updated fusion weights:
        ω=0.9: w(1-node)=0.096, w(8-node)=0.904
        ω=1.5: w(1-node)=0.059, w(8-node)=0.941
        ω=2.0: w(1-node)=0.049, w(8-node)=0.951
        ω=2.2: w(1-node)=0.046, w(8-node)=0.954

[STEP 4] Evaluating all models...
--------------------------------------------------

======================================================================
FUSION EXPERIMENT RESULTS
======================================================================

Model                | ω=0.9      | ω=1.2      | ω=1.5      | ω=2.0      | ω=2.2      | Avg       
----------------------------------------------------------------------------------------------------
1-node               | 0.0015     | 0.0203     | 0.0574     | 0.1447     | 0.1727     | 0.0793    
8-node               | 0.0016     | 0.0173     | 0.0473     | 0.1189     | 0.1488     | 0.0668    
Fusion               | 0.0015     | 0.0164     | 0.0461     | 0.1255     | 0.1556     | 0.0690    

======================================================================
IMPROVEMENT ANALYSIS
======================================================================

  Baseline 8-node average MSE: 0.066757
  Baseline 1-node average MSE: 0.079309
  Fusion average MSE:          0.069031

  Fusion vs 8-node: -3.41%
  Fusion vs 1-node: +12.96%

  PARTIAL: Fusion beats 1-node but not 8-node

======================================================================
FREQUENCY-BY-FREQUENCY COMPARISON
======================================================================
  ω=0.9: 1-node=0.001493, 8-node=0.001599, Fusion=0.001527 -> Winner: 1-node ✗
  ω=1.2: 1-node=0.020280, 8-node=0.017264, Fusion=0.016417 -> Winner: FUSION ✓
  ω=1.5: 1-node=0.057362, 8-node=0.047297, Fusion=0.046086 -> Winner: FUSION ✓
  ω=2.0: 1-node=0.144726, 8-node=0.118851, Fusion=0.125518 -> Winner: 8-node ✗
  ω=2.2: 1-node=0.172684, 8-node=0.148776, Fusion=0.155609 -> Winner: 8-node ✗

  Fusion wins: 2/5 frequencies

======================================================================
OUTPUT FILES
======================================================================
  - supertopobrain3/checkpoint_1node.pth
  - supertopobrain3/checkpoint_8node.pth
  - supertopobrain3/checkpoint_fusion.pth
  - supertopobrain3/fusion_results.json

```
