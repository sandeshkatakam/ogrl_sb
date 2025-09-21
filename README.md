# Offline Goal-Conditioned RL Experiment Setup

This repository provides a complete setup for offline goal-conditioned RL experiments using a simple 2D navigation environment.

## Features

- **Simple Toy Environment**: 2D navigation with goal conditioning
- **Trajectory Generation**: Collect offline trajectories with different policies
- **Composable Data Loader**: PyTorch data loader for offline RL
- **Goal Conditioning**: Support for goal relabeling and different sampling strategies
- **Clean Architecture**: Modular, composable design

## Components

### 1. Toy Environment (`toy_env.py`)
- Simple 2D navigation environment
- Goal-conditioned observations
- Configurable grid size, max steps, and noise
- Support for custom start/goal positions

### 2. Trajectory Generator (`trajectory_generator.py`)
- Generate offline trajectories with different policies
- Expert policy implementation
- Random policy for comparison
- Save/load trajectory datasets

### 3. Data Loader (`data_loader.py`)
- PyTorch Dataset and DataLoader for offline RL
- Goal-conditioned dataset with relabeling
- Multiple goal sampling strategies (uniform, future, final)
- Batch processing and device support

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from toy_env import make_toy_env
from trajectory_generator import create_expert_dataset
from data_loader import create_data_loaders

# Create environment
env = make_toy_env(goal_conditioned=True)

# Generate expert dataset
trajectories = create_expert_dataset(env, num_trajectories=1000)

# Create data loaders
train_loader, val_loader = create_data_loaders(
    trajectories=trajectories,
    batch_size=64,
    goal_conditioned=True,
    goal_relabeling=True
)

# Use with your offline RL algorithm
for batch in train_loader:
    # Your training code here
    pass
```

## Usage Examples

### Basic Environment Usage

```python
from toy_env import make_toy_env

# Create environment
env = make_toy_env(grid_size=10, max_steps=50, goal_conditioned=True)

# Reset with specific goal
obs, info = env.reset(options={'goal': [8.0, 8.0]})
print(f"Goal: {info['goal']}")

# Take actions
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### Data Collection

```python
from trajectory_generator import TrajectoryGenerator, ExpertPolicy

# Create generator with expert policy
expert_policy = ExpertPolicy(env, noise=0.1)
generator = TrajectoryGenerator(env, policy=expert_policy)

# Generate trajectories
trajectories = generator.generate_dataset(
    num_trajectories=500,
    max_steps=50,
    save_path="expert_data.pkl"
)
```

### Data Loading

```python
from data_loader import GoalConditionedDataset, OfflineRLDataLoader

# Create goal-conditioned dataset
dataset = GoalConditionedDataset(
    trajectories=trajectories,
    goal_relabeling=True,
    goal_sampling_strategy='future'  # HER-style relabeling
)

# Create data loader
loader = OfflineRLDataLoader(dataset, batch_size=64, shuffle=True)

# Use in training loop
for batch in loader:
    observations = batch.observations  # [batch_size, 4] (state + goal)
    actions = batch.actions           # [batch_size]
    rewards = batch.rewards           # [batch_size]
    # ... your training code
```

## Goal Sampling Strategies

The data loader supports different goal sampling strategies:

- **`uniform`**: Sample goals uniformly from all possible goals
- **`future`**: Sample goals from future states in the same trajectory (HER-style)
- **`final`**: Use the final state of each trajectory as the goal

## Environment Details

- **State Space**: [x, y] position (2D continuous)
- **Action Space**: 5 discrete actions (up, down, left, right, no-op)
- **Goal Space**: [gx, gy] target position (2D continuous)
- **Reward**: Sparse reward (1.0 for reaching goal, 0.0 otherwise)
- **Termination**: When goal is reached or max steps exceeded

## Customization

### Custom Policies

```python
def my_policy(obs):
    # Your policy implementation
    return action

generator = TrajectoryGenerator(env, policy=my_policy)
```

### Custom Goal Samplers

```python
def my_goal_sampler():
    # Your goal sampling logic
    return goal

generator = TrajectoryGenerator(env, goal_sampler=my_goal_sampler)
```

### Custom Transforms

```python
def my_transform(obs):
    # Your observation transformation
    return transformed_obs

dataset = OfflineRLDataset(trajectories, transform=my_transform)
```

## File Structure

```
ogrl_sb/
├── toy_env.py              # Environment implementation
├── trajectory_generator.py # Data collection utilities
├── data_loader.py          # PyTorch data loading
├── ogrl_toy_expts.ipynb   # Example notebook
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Dependencies

- numpy>=1.21.0
- torch>=1.9.0
- gymnasium>=0.26.0
- matplotlib>=3.5.0

## License

MIT License