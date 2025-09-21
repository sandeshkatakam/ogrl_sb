# Offline Goal-Conditioned RL with KL Divergence Decomposition

## Algorithm Overview

This implementation realizes a sophisticated approach to offline goal-conditioned RL by decomposing the problem using the KL divergence decomposition:

```
KL(q(s_0:T, a_0:T-1) || p(s_0:T, a_0:T-1)) = 
    KL(q(s_0:T) || p(s_0:T)) + 
    E_q(s_0:T)[KL(q(a_0:T-1|s_0:T) || p(a_0:T-1|s_0:T))]
```

## Components Implemented

### 1. Action Reconstruction Model (`action_reconstruction_model.py`)
- **Purpose**: Learn `p(a_0:T-1|s_0:T)` using autoregressive GRU
- **Architecture**: 
  - Input: State sequence + Goal sequence
  - GRU layers with configurable hidden dimensions
  - Output: Action logits for each timestep
- **Training**: Supervised learning on offline trajectories
- **Key Features**:
  - Autoregressive action generation
  - Goal-conditioned input
  - Temperature-controlled sampling

### 2. Schrödinger Bridge (`schrodinger_bridge.py`)
- **Purpose**: Learn `q(s_0:T)` using IPF-Sinkhorn algorithm
- **Algorithm**: 
  - Discretizes continuous state space into grid
  - Uses Iterative Proportional Fitting (IPF) with Sinkhorn iterations
  - Connects initial state distribution to goal state distribution
- **Key Features**:
  - Discrete state space discretization
  - IPF-Sinkhorn optimization
  - Trajectory sampling from learned bridge

### 3. Integrated Model (`integrated_model.py`)
- **Purpose**: Combine both components for complete trajectory generation
- **Integration Strategy**:
  1. Sample state trajectory from `q(s_0:T)` (Schrödinger Bridge)
  2. Sample actions from `p(a_0:T-1|s_0:T)` (GRU model)
  3. Combine to generate complete trajectories
- **Key Features**:
  - End-to-end training pipeline
  - Evaluation metrics
  - Visualization capabilities

## Mathematical Foundation

### KL Divergence Decomposition
The algorithm is based on the chain rule for KL divergence:

```
KL(q(s_0:T, a_0:T-1) || p(s_0:T, a_0:T-1)) = 
    KL(q(s_0:T) || p(s_0:T)) + 
    E_q(s_0:T)[KL(q(a_0:T-1|s_0:T) || p(a_0:T-1|s_0:T))]
```

Where:
- `q(s_0:T)`: State trajectory distribution learned via Schrödinger Bridge
- `p(a_0:T-1|s_0:T)`: Action distribution learned via autoregressive GRU
- `p(s_0:T, a_0:T-1)`: Target distribution from offline data

### Schrödinger Bridge
The Schrödinger Bridge solves the optimal transport problem:
- **Input**: Initial state distribution `p_0(s_0)` and goal state distribution `p_T(s_T)`
- **Output**: Optimal transition matrices connecting the distributions
- **Algorithm**: IPF-Sinkhorn iterations for discrete case

### Action Model
The autoregressive GRU learns the conditional action distribution:
- **Input**: `(s_0, s_1, ..., s_T, goal)`
- **Output**: `(a_0, a_1, ..., a_{T-1})`
- **Loss**: Negative log-likelihood of actions given states

## Implementation Details

### State Discretization
- Continuous 2D state space discretized into `grid_size × grid_size` grid
- Default: 20×20 = 400 discrete states
- Handles continuous-to-discrete conversion and vice versa

### IPF-Sinkhorn Algorithm
- **Forward Pass**: Update transitions to match target marginals
- **Backward Pass**: Update transitions to match initial marginals
- **Convergence**: Based on maximum change in transition matrices
- **Regularization**: Sinkhorn regularization parameter for numerical stability

### Action Model Training
- **Architecture**: Multi-layer GRU with dropout
- **Input Embedding**: State + Goal concatenation
- **Output Head**: Linear layers with softmax for action probabilities
- **Optimization**: Adam optimizer with learning rate scheduling

## Usage Example

```python
from integrated_model import IntegratedOfflineRLModel, IntegratedModelConfig
from data_loader import create_data_loaders

# Create model
config = IntegratedModelConfig()
model = IntegratedOfflineRLModel(config)

# Train on offline data
training_info = model.train(train_loader, val_loader)

# Generate trajectories
trajectories = model.generate_trajectories(
    num_trajectories=100,
    temperature=1.0
)

# Evaluate performance
metrics = model.evaluate_trajectories(trajectories)
```

## Key Advantages

1. **Theoretical Foundation**: Based on rigorous KL divergence decomposition
2. **Modular Design**: Separate components for state and action modeling
3. **Optimal Transport**: Schrödinger Bridge ensures optimal state transitions
4. **Goal Conditioning**: Natural support for goal-conditioned behavior
5. **Scalability**: Can be extended to more complex environments
6. **Interpretability**: Clear separation of state and action learning

## Performance Characteristics

- **State Modeling**: Schrödinger Bridge learns optimal state trajectories
- **Action Modeling**: GRU learns realistic action sequences
- **Integration**: Seamless combination of both components
- **Evaluation**: Comprehensive metrics for trajectory quality

## Future Extensions

1. **Continuous State Spaces**: Extend to continuous Schrödinger Bridge
2. **Higher Dimensions**: Scale to higher-dimensional state spaces
3. **Multi-Goal**: Support for multiple simultaneous goals
4. **Uncertainty Quantification**: Add uncertainty estimates
5. **Online Adaptation**: Real-time model updates

## Files Structure

```
ogrl_sb/
├── action_reconstruction_model.py    # GRU-based action model
├── schrodinger_bridge.py            # IPF-Sinkhorn bridge implementation
├── integrated_model.py              # Combined model
├── ogrl_toy_expt_algorithm.ipynb   # Complete demonstration
├── toy_env.py                       # 2D navigation environment
├── trajectory_generator.py          # Data collection utilities
├── data_loader.py                   # PyTorch data loading
└── ALGORITHM_SUMMARY.md            # This summary
```

This implementation provides a complete, theoretically grounded approach to offline goal-conditioned RL using the KL divergence decomposition framework.
