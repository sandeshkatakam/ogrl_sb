"""
Composable Data Loader for Offline RL Trajectories
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple, Optional, Callable, Union
import random
from dataclasses import dataclass
from trajectory_generator import Trajectory, TrajectoryGenerator


@dataclass
class Batch:
    """A batch of transitions for training."""
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    rewards: torch.Tensor
    terminals: torch.Tensor
    goals: torch.Tensor
    batch_size: int
    
    def to(self, device: torch.device) -> 'Batch':
        """Move batch to device."""
        return Batch(
            observations=self.observations.to(device),
            actions=self.actions.to(device),
            next_observations=self.next_observations.to(device),
            rewards=self.rewards.to(device),
            terminals=self.terminals.to(device),
            goals=self.goals.to(device),
            batch_size=self.batch_size
        )


class OfflineRLDataset(Dataset):
    """
    PyTorch Dataset for offline RL trajectories.
    """
    
    def __init__(self, 
                 trajectories: List[Trajectory],
                 max_length: Optional[int] = None,
                 goal_conditioned: bool = True,
                 transform: Optional[Callable] = None):
        """
        Initialize the dataset.
        
        Args:
            trajectories: List of trajectories
            max_length: Maximum trajectory length (if None, uses all)
            goal_conditioned: Whether to include goals in observations
            transform: Optional transform to apply to observations
        """
        self.trajectories = trajectories
        self.max_length = max_length
        self.goal_conditioned = goal_conditioned
        self.transform = transform
        
        # Flatten all transitions
        self.transitions = []
        for traj in trajectories:
            traj_length = min(traj.length, max_length) if max_length else traj.length
            
            for i in range(traj_length):
                obs = traj.observations[i]
                action = traj.actions[i]
                next_obs = traj.next_observations[i]
                reward = traj.rewards[i]
                terminal = traj.terminals[i]
                goal = traj.goals[i]
                
                # Apply transform if provided
                if transform is not None:
                    obs = transform(obs)
                    next_obs = transform(next_obs)
                
                self.transitions.append({
                    'observation': obs,
                    'action': action,
                    'next_observation': next_obs,
                    'reward': reward,
                    'terminal': terminal,
                    'goal': goal
                })
    
    def __len__(self) -> int:
        return len(self.transitions)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.transitions[idx]
    
    def get_trajectory_indices(self) -> List[int]:
        """Get indices for each trajectory start."""
        indices = []
        current_idx = 0
        
        for traj in self.trajectories:
            traj_length = min(traj.length, self.max_length) if self.max_length else traj.length
            indices.append(current_idx)
            current_idx += traj_length
        
        return indices


class GoalConditionedDataset(OfflineRLDataset):
    """
    Goal-conditioned version of the offline RL dataset.
    """
    
    def __init__(self, 
                 trajectories: List[Trajectory],
                 max_length: Optional[int] = None,
                 transform: Optional[Callable] = None,
                 goal_relabeling: bool = True,
                 goal_sampling_strategy: str = 'uniform'):
        """
        Initialize goal-conditioned dataset.
        
        Args:
            trajectories: List of trajectories
            max_length: Maximum trajectory length
            transform: Optional transform to apply
            goal_relabeling: Whether to relabel goals (HER-style)
            goal_sampling_strategy: Strategy for sampling goals ('uniform', 'future', 'final')
        """
        super().__init__(trajectories, max_length, goal_conditioned=True, transform=transform)
        self.goal_relabeling = goal_relabeling
        self.goal_sampling_strategy = goal_sampling_strategy
        
        if goal_relabeling:
            self._relabel_goals()
    
    def _relabel_goals(self):
        """Relabel goals using the specified strategy."""
        if self.goal_sampling_strategy == 'uniform':
            self._uniform_goal_relabeling()
        elif self.goal_sampling_strategy == 'future':
            self._future_goal_relabeling()
        elif self.goal_sampling_strategy == 'final':
            self._final_goal_relabeling()
        else:
            raise ValueError(f"Unknown goal sampling strategy: {self.goal_sampling_strategy}")
    
    def _uniform_goal_relabeling(self):
        """Uniformly sample goals from all possible goals in the dataset."""
        all_goals = []
        for traj in self.trajectories:
            all_goals.extend(traj.goals)
        
        for transition in self.transitions:
            transition['goal'] = random.choice(all_goals)
    
    def _future_goal_relabeling(self):
        """Sample goals from future states in the same trajectory (HER-style)."""
        traj_starts = self.get_trajectory_indices()
        
        for i, traj_start in enumerate(traj_starts):
            traj = self.trajectories[i]
            traj_length = min(traj.length, self.max_length) if self.max_length else traj.length
            
            for j in range(traj_length):
                idx = traj_start + j
                
                # Sample future goal from the same trajectory
                future_idx = random.randint(j, traj_length - 1)
                future_goal = traj.goals[future_idx]
                
                self.transitions[idx]['goal'] = future_goal
    
    def _final_goal_relabeling(self):
        """Use the final state of each trajectory as the goal."""
        traj_starts = self.get_trajectory_indices()
        
        for i, traj_start in enumerate(traj_starts):
            traj = self.trajectories[i]
            traj_length = min(traj.length, self.max_length) if self.max_length else traj.length
            
            # Use the final goal of the trajectory
            final_goal = traj.goals[traj_length - 1]
            
            for j in range(traj_length):
                idx = traj_start + j
                self.transitions[idx]['goal'] = final_goal


def collate_fn(batch: List[Dict[str, Any]]) -> Batch:
    """
    Collate function for creating batches from transitions.
    
    Args:
        batch: List of transition dictionaries
    
    Returns:
        Batch object with batched tensors
    """
    observations = torch.stack([torch.tensor(trans['observation'], dtype=torch.float32) 
                              for trans in batch])
    actions = torch.tensor([trans['action'] for trans in batch], dtype=torch.long)
    next_observations = torch.stack([torch.tensor(trans['next_observation'], dtype=torch.float32) 
                                   for trans in batch])
    rewards = torch.tensor([trans['reward'] for trans in batch], dtype=torch.float32)
    terminals = torch.tensor([trans['terminal'] for trans in batch], dtype=torch.bool)
    goals = torch.stack([torch.tensor(trans['goal'], dtype=torch.float32) 
                        for trans in batch])
    
    return Batch(
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        rewards=rewards,
        terminals=terminals,
        goals=goals,
        batch_size=len(batch)
    )


class OfflineRLDataLoader:
    """
    Composable data loader for offline RL.
    """
    
    def __init__(self, 
                 dataset: OfflineRLDataset,
                 batch_size: int = 256,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 pin_memory: bool = True):
        """
        Initialize the data loader.
        
        Args:
            dataset: The offline RL dataset
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory for faster GPU transfer
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn
        )
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)
    
    def get_batch(self) -> Batch:
        """Get a single batch."""
        return next(iter(self.dataloader))


def create_data_loaders(trajectories: List[Trajectory],
                       train_ratio: float = 0.8,
                       batch_size: int = 256,
                       goal_conditioned: bool = True,
                       goal_relabeling: bool = True,
                       goal_sampling_strategy: str = 'future',
                       **kwargs) -> Tuple[OfflineRLDataLoader, OfflineRLDataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        trajectories: List of trajectories
        train_ratio: Ratio of trajectories for training
        batch_size: Batch size
        goal_conditioned: Whether to use goal conditioning
        goal_relabeling: Whether to relabel goals
        goal_sampling_strategy: Strategy for goal sampling
        **kwargs: Additional arguments for data loaders
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Split trajectories
    num_train = int(len(trajectories) * train_ratio)
    train_trajectories = trajectories[:num_train]
    val_trajectories = trajectories[num_train:]
    
    # Create datasets
    if goal_conditioned:
        train_dataset = GoalConditionedDataset(
            train_trajectories,
            goal_relabeling=goal_relabeling,
            goal_sampling_strategy=goal_sampling_strategy
        )
        val_dataset = GoalConditionedDataset(
            val_trajectories,
            goal_relabeling=goal_relabeling,
            goal_sampling_strategy=goal_sampling_strategy
        )
    else:
        train_dataset = OfflineRLDataset(train_trajectories, goal_conditioned=False)
        val_dataset = OfflineRLDataset(val_trajectories, goal_conditioned=False)
    
    # Create data loaders
    train_loader = OfflineRLDataLoader(train_dataset, batch_size=batch_size, **kwargs)
    val_loader = OfflineRLDataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    
    return train_loader, val_loader


class TrajectoryBuffer:
    """
    Buffer for storing and sampling trajectories.
    """
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.trajectories = []
        self._trajectory_indices = []
    
    def add_trajectory(self, trajectory: Trajectory):
        """Add a trajectory to the buffer."""
        if len(self.trajectories) >= self.max_size:
            # Remove oldest trajectory
            self.trajectories.pop(0)
            self._trajectory_indices.pop(0)
        
        self.trajectories.append(trajectory)
        self._trajectory_indices.append(len(self.trajectories) - 1)
    
    def sample_trajectory(self) -> Trajectory:
        """Sample a random trajectory."""
        return random.choice(self.trajectories)
    
    def sample_trajectories(self, num_trajectories: int) -> List[Trajectory]:
        """Sample multiple random trajectories."""
        return random.sample(self.trajectories, min(num_trajectories, len(self.trajectories)))
    
    def get_all_trajectories(self) -> List[Trajectory]:
        """Get all trajectories."""
        return self.trajectories.copy()
    
    def __len__(self) -> int:
        return len(self.trajectories)


if __name__ == "__main__":
    # Test the data loader
    from toy_env import make_toy_env
    from trajectory_generator import create_expert_dataset
    
    # Create environment and generate data
    env = make_toy_env(goal_conditioned=True)
    trajectories = create_expert_dataset(env, num_trajectories=100)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        trajectories,
        batch_size=32,
        goal_conditioned=True,
        goal_relabeling=True
    )
    
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    
    # Test a batch
    batch = next(iter(train_loader))
    print(f"Batch shape: {batch.observations.shape}")
    print(f"Actions: {batch.actions.shape}")
    print(f"Goals: {batch.goals.shape}")
