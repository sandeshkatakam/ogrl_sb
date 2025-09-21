"""
Sequence-based Data Loader for Offline RL Trajectories

This module provides sequence-based data loading for the action reconstruction model.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from trajectory_generator import Trajectory


@dataclass
class SequenceBatch:
    """A batch of trajectory sequences for training."""
    states: torch.Tensor          # [batch_size, seq_len, state_dim]
    actions: torch.Tensor         # [batch_size, seq_len-1]
    goals: torch.Tensor           # [batch_size, goal_dim]
    batch_size: int
    
    def to(self, device: torch.device) -> 'SequenceBatch':
        """Move batch to device."""
        return SequenceBatch(
            states=self.states.to(device),
            actions=self.actions.to(device),
            goals=self.goals.to(device),
            batch_size=self.batch_size
        )


class SequenceDataset(Dataset):
    """
    Dataset for trajectory sequences.
    """
    
    def __init__(self, 
                 trajectories: List[Trajectory],
                 max_length: int = 50,
                 min_length: int = 5):
        """
        Initialize the sequence dataset.
        
        Args:
            trajectories: List of trajectories
            max_length: Maximum sequence length
            min_length: Minimum sequence length
        """
        self.trajectories = trajectories
        self.max_length = max_length
        self.min_length = min_length
        
        # Filter trajectories by length
        self.valid_trajectories = [
            traj for traj in trajectories 
            if min_length <= traj.length <= max_length
        ]
        
        print(f"Filtered {len(self.valid_trajectories)}/{len(trajectories)} trajectories")
        
        if len(self.valid_trajectories) == 0:
            print(f"Warning: No valid trajectories found. Adjusting length constraints...")
            # Use all trajectories and adjust length during collation
            self.valid_trajectories = trajectories
            self.max_length = max(traj.length for traj in trajectories)
            self.min_length = 1
    
    def __len__(self) -> int:
        return len(self.valid_trajectories)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        traj = self.valid_trajectories[idx]
        
        # Extract states (first 2 dimensions for position)
        states = np.array(traj.observations)[:, :2]  # [seq_len, 2]
        
        # Extract actions
        actions = np.array(traj.actions)  # [seq_len-1]
        
        # Extract goal (same for all timesteps)
        goal = np.array(traj.goals[0])  # [2]
        
        return {
            'states': states,
            'actions': actions,
            'goal': goal,
            'length': len(actions)
        }


def sequence_collate_fn(batch: List[Dict[str, Any]]) -> SequenceBatch:
    """
    Collate function for creating sequence batches.
    
    Args:
        batch: List of trajectory dictionaries
    
    Returns:
        SequenceBatch object with batched tensors
    """
    # Pad sequences to the same length
    max_length = max(item['length'] for item in batch)
    
    states_list = []
    actions_list = []
    goals_list = []
    
    for item in batch:
        states = item['states']
        actions = item['actions']
        goal = item['goal']
        
        # Pad states
        if len(states) < max_length + 1:
            pad_length = max_length + 1 - len(states)
            states_padded = np.pad(states, ((0, pad_length), (0, 0)), mode='edge')
        else:
            states_padded = states[:max_length + 1]
        
        # Pad actions
        if len(actions) < max_length:
            pad_length = max_length - len(actions)
            actions_padded = np.pad(actions, (0, pad_length), mode='constant', constant_values=0)
        else:
            actions_padded = actions[:max_length]
        
        states_list.append(states_padded)
        actions_list.append(actions_padded)
        goals_list.append(goal)
    
    # Convert to tensors
    states_tensor = torch.tensor(np.array(states_list), dtype=torch.float32)
    actions_tensor = torch.tensor(np.array(actions_list), dtype=torch.long)
    goals_tensor = torch.tensor(np.array(goals_list), dtype=torch.float32)
    
    return SequenceBatch(
        states=states_tensor,
        actions=actions_tensor,
        goals=goals_tensor,
        batch_size=len(batch)
    )


class SequenceDataLoader:
    """
    Data loader for trajectory sequences.
    """
    
    def __init__(self, 
                 dataset: SequenceDataset,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 pin_memory: bool = True):
        """
        Initialize the sequence data loader.
        
        Args:
            dataset: The sequence dataset
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
            collate_fn=sequence_collate_fn
        )
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)
    
    def get_batch(self) -> SequenceBatch:
        """Get a single batch."""
        return next(iter(self.dataloader))


def create_sequence_data_loaders(trajectories: List[Trajectory],
                                train_ratio: float = 0.8,
                                batch_size: int = 32,
                                max_length: int = 50,
                                min_length: int = 5,
                                **kwargs) -> Tuple[SequenceDataLoader, SequenceDataLoader]:
    """
    Create train and validation sequence data loaders.
    
    Args:
        trajectories: List of trajectories
        train_ratio: Ratio of trajectories for training
        batch_size: Batch size
        max_length: Maximum sequence length
        min_length: Minimum sequence length
        **kwargs: Additional arguments for data loaders
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Split trajectories
    num_train = int(len(trajectories) * train_ratio)
    train_trajectories = trajectories[:num_train]
    val_trajectories = trajectories[num_train:]
    
    # Create datasets
    train_dataset = SequenceDataset(train_trajectories, max_length, min_length)
    val_dataset = SequenceDataset(val_trajectories, max_length, min_length)
    
    # Create data loaders
    train_loader = SequenceDataLoader(train_dataset, batch_size=batch_size, **kwargs)
    val_loader = SequenceDataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the sequence data loader
    from toy_env import make_toy_env
    from trajectory_generator import create_expert_dataset
    
    # Create environment and generate data
    env = make_toy_env(goal_conditioned=True)
    trajectories = create_expert_dataset(env, num_trajectories=50)
    
    # Create sequence data loaders
    train_loader, val_loader = create_sequence_data_loaders(
        trajectories,
        batch_size=8,
        max_length=30
    )
    
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    
    # Test a batch
    batch = next(iter(train_loader))
    print(f"Batch shapes:")
    print(f"  States: {batch.states.shape}")
    print(f"  Actions: {batch.actions.shape}")
    print(f"  Goals: {batch.goals.shape}")
