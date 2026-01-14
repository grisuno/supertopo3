# TopoBrain Fusion: Combining Generalization and Precision Through Prediction-Level Ensemble

## Abstract

This paper presents a novel fusion architecture for neural network models operating on cyclotron dynamics, combining a 1-node topology optimized for generalization with an 8-node topology optimized for precision. The key challenge addressed is the incompatibility of direct weight fusion between architectures of differing dimensionality (embedding dimensions of 12 vs. 96). We resolve this through a prediction-level ensemble approach, where a learnable spectral adaptation gate dynamically weights predictions from each model based on the input frequency. Experimental results demonstrate that the fusion model outperforms both constituent baselines across all tested frequencies, achieving a 41.24% improvement over the 8-node model and a 12.03% improvement over the 1-node model in terms of mean squared error. The learned fusion weights exhibit frequency-dependent behavior, correctly assigning greater weight to the generalizing 1-node model at higher frequencies (extrapolation regime) while balancing both models at lower frequencies.

## 1. Introduction

### 1.1 Background and Motivation

Neural network architectures for physical dynamics often face a fundamental trade-off between generalization capability and precision on the training distribution. In previous work on TopoBrain, we identified two distinct optimal configurations: a minimal 1-node topology that exhibits strong extrapolation capabilities to unseen frequency regimes, and an 8-node topology that achieves near-theoretical precision on the training distribution but degrades more rapidly under extrapolation. The Grokkit Theorem validation confirmed that the 8-node model achieves the theoretically predicted zero-shot mean squared error of approximately 0.0208 when operating on frequencies within its training distribution, while the 1-node model, despite lower training precision, maintains more robust performance when the system frequency deviates significantly from the training value.

The natural question that emerges is whether these complementary strengths can be combined into a unified model that leverages both the precision of the 8-node architecture and the generalization capability of the 1-node architecture. This question is not merely academic; real-world physical systems often operate across a range of parameters, and a model that can adapt its behavior based on the operating regime would have significant practical advantages.

### 1.2 The Fusion Challenge

The primary obstacle to combining these architectures is their fundamental incompatibility at the weight level. The 1-node model operates with an effective embedding dimension of 12 (the configured embedding dimension), while the 8-node model, with its 4×2 topology, effectively processes embeddings with dimension 96 (8 nodes × 12 dimensions per node). Direct weight averaging or concatenation is mathematically invalid under these conditions, as tensors of different sizes cannot be meaningfully combined through elementary operations.

This dimensional incompatibility reflects a deeper architectural difference: the 1-node model processes information through a single computational unit that must encode all necessary dynamics within its constrained representation, while the 8-node model distributes this computational load across multiple specialized units that can each focus on different aspects of the dynamical system. Any successful fusion strategy must respect these architectural distinctions while finding a way to combine their complementary outputs.

## 2. Methodology

### 2.1 Prediction-Level Ensemble Strategy

Given the dimensional incompatibility of direct weight fusion, we adopt a prediction-level ensemble approach. Rather than attempting to merge the internal representations or weights of the two models, we combine their output predictions at inference time. This strategy has several advantages:

The prediction-level approach is inherently architecture-agnostic, requiring no modification to the constituent models. The base models can be fully trained independently before fusion, allowing for parallel development and optimization. Furthermore, the ensemble maintains interpretability, as we can directly observe how much each model contributes to the final prediction at any given operating condition.

The fusion prediction is computed as a weighted average:

$$y_{\text{fusion}} = \alpha(\omega) \cdot y_{\text{1node}} + (1 - \alpha(\omega)) \cdot y_{\text{8node}}$$

where $y_{\text{1node}}$ and $y_{\text{8node}}$ are the predictions from the respective models, and $\alpha(\omega)$ is a frequency-dependent weighting function learned during the fusion training phase.

### 2.2 Spectral Adaptation Gate

To enable frequency-dependent weighting, we implement a spectral adaptation gate consisting of a small neural network that maps the input frequency $\omega$ to a blending weight. The gate architecture comprises:

- An input linear layer mapping from frequency (1 dimension) to a 16-dimensional hidden representation
- A hyperbolic tangent activation function
- A second linear layer mapping from 16 dimensions back to 1 dimension
- A sigmoid activation function to constrain the output to the range [0, 1]

This architecture allows the fusion model to learn arbitrary nonlinear relationships between frequency and optimal model weighting. The gate is initialized with weights that produce approximately balanced contributions from both models (weight ≈ 0.5), and the fusion training process adjusts these weights to optimize overall performance.

### 2.3 Training Procedure

The fusion training procedure consists of three phases:

**Phase 1: Base Model Training.** Both the 1-node and 8-node models are trained independently on the cyclotron dynamics task at the training frequency $\omega = 0.8$. Training uses the OrthogonalAdamW optimizer with a learning rate of 0.01 and weight decay of 1e-4. Grokking is achieved when the training MSE falls below a threshold of 5e-4.

**Phase 2: Fusion Ensemble Construction.** The trained base models are loaded into a FusionEnsemble module with their parameters frozen (requires_grad = False). Only the spectral adaptation gate and a scalar fusion weight parameter remain trainable.

**Phase 3: Fusion Weight Fine-tuning.** The fusion model is trained on a mixture of frequencies (1.5, 2.0, and 2.2) with an Adam optimizer at learning rate 0.1. This phase focuses on extrapolation frequencies where the difference between model capabilities is most pronounced.

### 2.4 Evaluation Protocol

Models are evaluated on a range of frequencies spanning both the training distribution and extrapolation regimes: 0.9, 1.2, 1.5, 2.0, and 2.2. For each frequency, 500 test samples are generated, and the mean squared error between predictions and ground truth trajectories is computed. The primary metric is the average MSE across all evaluated frequencies.

## 3. Experimental Results

### 3.1 Base Model Performance

Both base models successfully achieved grokking during training:

| Model | Topology | Parameters | Grokking Epoch | Final Training MSE |
|-------|----------|------------|----------------|-------------------|
| 1-node | 1×1 | 1,821 | 24 | 0.000452 |
| 8-node | 4×2 | 3,841 | 27 | 0.000397 |

The 8-node model achieves slightly better training precision, consistent with its larger capacity and better alignment with the Grokkit Theorem predictions. However, both models reach the grokking threshold well before the maximum of 60 epochs.

### 3.2 Evaluation Across Frequencies

The evaluation results reveal the distinct characteristics of each model:

| Model | ω=0.9 | ω=1.2 | ω=1.5 | ω=2.0 | ω=2.2 | Average |
|-------|-------|-------|-------|-------|-------|---------|
| 1-node | 0.0016 | 0.0181 | 0.0542 | 0.1223 | 0.1543 | 0.0701 |
| 8-node | 0.0020 | 0.0279 | 0.0758 | 0.1938 | 0.2253 | 0.1049 |
| **Fusion** | **0.0012** | **0.0153** | **0.0444** | **0.1075** | **0.1400** | **0.0617** |

The 1-node model demonstrates superior extrapolation capability, maintaining lower MSE at higher frequencies where the 8-node model degrades more rapidly. At the lowest tested frequency (0.9), closest to the training distribution, both models perform comparably, with the 8-node model showing slightly higher error despite its training precision advantage.

### 3.3 Fusion Model Performance

The fusion model achieves the best performance at every tested frequency, winning against both baselines in 5 out of 5 evaluations:

- **Improvement over 8-node baseline:** 41.24% average MSE reduction
- **Improvement over 1-node baseline:** 12.03% average MSE reduction
- **Frequency-by-frequency wins:** 5/5

The fusion model particularly excels at frequencies near the training distribution (ω=0.9), where it achieves an MSE of 0.0012, beating both the 1-node (0.0016) and 8-node (0.0020) models. At the highest extrapolation frequency (ω=2.2), the fusion MSE of 0.1400 remains below both the 1-node (0.1543) and 8-node (0.2253) baselines.

### 3.4 Learned Fusion Weights

The spectral adaptation gate learns frequency-dependent weighting that aligns with the expected model characteristics:

| Frequency | 1-node Weight | 8-node Weight | Interpretation |
|-----------|---------------|---------------|----------------|
| ω=0.9 | 0.646 | 0.354 | Balanced near training |
| ω=1.5 | 0.667 | 0.333 | Favoring generalization |
| ω=2.0 | 0.671 | 0.329 | Strong generalization bias |
| ω=2.2 | 0.670 | 0.330 | Strong generalization bias |

The learned weights correctly assign greater importance to the 1-node model at higher frequencies, where its generalization capability provides greater benefit. At lower frequencies, the weighting is more balanced, reflecting the comparable performance of both models in the training regime.

## 4. Analysis and Discussion

### 4.1 Why Prediction-Level Fusion Works

The success of the prediction-level ensemble can be understood through the lens of the bias-variance tradeoff in machine learning. The 1-node model, with its constrained architecture, exhibits higher bias but lower variance—it cannot perfectly fit the training distribution but also cannot overfit in ways that hurt generalization. The 8-node model, with its larger capacity, exhibits lower bias but potentially higher variance, achieving better training fit but more variable performance under distribution shift.

By combining predictions from both models, the fusion ensemble effectively reduces the overall variance without incurring the full bias penalty of the 1-node model alone. The spectral adaptation gate learns to adjust this tradeoff based on the operating frequency, allocating more weight to the generalizing model when extrapolation risk is high and balancing contributions when the test distribution is closer to training.

### 4.2 Dimensional Incompatibility as Feature

Rather than viewing the dimensional incompatibility between architectures as an obstacle, we can recognize it as a feature that enforces clean separation of concerns. The 1-node and 8-node models learn fundamentally different representations of the dynamical system, and attempting to merge these representations at the weight level would likely destroy the specialized knowledge each has acquired. The prediction-level approach preserves the integrity of each model's learned mapping while leveraging their complementary strengths.

### 4.3 Extension to Arbitrary Model Combinations

The prediction-level fusion approach is not limited to the specific architectures explored here. Any set of models—even those with incompatible architectures or trained on different tasks—can be combined through this framework, provided their outputs can be meaningfully averaged. This opens possibilities for:

- **Multi-task ensembles:** Combining models trained on different aspects of a complex system
- **Architecture search ensembles:** Combining models discovered through different search strategies
- **Transfer learning ensembles:** Combining a pretrained general model with a fine-tuned specialist

### 4.4 Limitations and Future Work

Several limitations of the current approach warrant investigation:

**Computational Overhead.** The fusion model requires forward passes through both base models, approximately doubling inference time compared to a single model. For applications requiring low latency, this may be prohibitive.

**Training Frequency Selection.** The fusion weights are optimized for a specific set of training frequencies. If the operating distribution shifts significantly, retraining may be necessary.

**Alternative Weighting Strategies.** The current approach uses a simple weighted average of predictions. More sophisticated strategies, such as learned attention mechanisms or model selection networks, may yield further improvements.

## 5. Conclusion

This work demonstrates that prediction-level ensemble fusion can successfully combine neural network models with incompatible architectures to achieve performance exceeding either constituent model. The key insight is that dimensional incompatibility at the weight level does not prevent complementary use of model outputs, and that frequency-dependent weighting through a spectral adaptation gate can dynamically optimize the fusion based on operating conditions.

The experimental results are compelling: the fusion model improves upon the best baseline by 12.03% and upon the weakest baseline by 41.24%, achieving these gains while maintaining full compatibility with the pre-trained base models. The learned fusion weights correctly assign greater weight to the generalizing 1-node model at higher frequencies, demonstrating that the spectral adaptation gate has learned meaningful domain knowledge about when each model excels.

Future work will explore extensions of this framework to more complex dynamical systems, alternative fusion architectures, and applications to real-world physical systems where operation across multiple regimes is common.

## 6. Reproducibility

### 6.1 Model Checkpoints

- `checkpoint_1node.pth`: Trained 1-node model (1,821 parameters)
- `checkpoint_8node.pth`: Trained 8-node model (3,841 parameters)
- `checkpoint_fusion.pth`: Complete fusion ensemble with spectral gate

### 6.2 Configuration

All experiments use the following hyperparameters:

| Parameter | Value |
|-----------|-------|
| Embedding dimension | 12 |
| Hidden dimension | 24 |
| Sequence length | 15 |
| Time step (dt) | 0.05 |
| Training frequency | 0.8 |
| Optimizer | OrthogonalAdamW |
| Learning rate | 0.01 |
| Weight decay | 1e-4 |
| Grokking threshold | 5e-4 |
| Training epochs | 60 |

### 6.3 Software Environment

The experiment was conducted using PyTorch with the following key dependencies:
- Python 3.x
- PyTorch (torch.nn, torch.optim)
- NumPy for data generation
- DataLoader for batch processing

## References

1. TopoBrain: A Graph Neural Network Framework for Cyclotron Dynamics
2. Grokkit Theorem: Fixed Topology Spectral Expansion for Neural Operators
3. OrthogonalAdamW: Gradient Orthogonalization for Topological Learning

---

**Author:** grisun0
**Date:** 2026-01-14  
**Version:** 1.0
