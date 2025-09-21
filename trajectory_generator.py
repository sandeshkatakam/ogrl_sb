"""
Trajectory Generator for Offline RL Data Collection
"""

import numpy as np
import gymnasium as gym
from typing import List, Dict, Any, Tuple, Optional, Callable
import pickle
import os
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class Transition:
    """A single transition in an MDP trajectory."""
    observation: np.ndarray
    action: int
    next_observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    goal: np.ndarray
    info: Dict[str, Any]


@dataclass
class Trajectory:
    """A complete trajectory in an MDP."""
    observations: List[np.ndarray]
    actions: List[int]
    next_observations: List[np.ndarray]
    rewards: List[float]
    terminals: List[bool]
    truncateds: List[bool]
    goals: List[np.ndarray]
    infos: List[Dict[str, Any]]
    success: bool
    length: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trajectory to dictionary format."""
        return {
            'observations': np.array(self.observations),
            'actions': np.array(self.actions),
            'next_observations': np.array(self.next_observations),
            'rewards': np.array(self.rewards),
            'terminals': np.array(self.terminals),
            'truncateds': np.array(self.truncateds),
            'goals': np.array(self.goals),
            'success': self.success,
            'length': self.length
        }


class TrajectoryGenerator:
    """
    Generates offline trajectories for RL environments.
    """
    
    def __init__(self, 
                 env: gym.Env,
                 policy: Optional[Callable[[np.ndarray], int]] = None,
                 goal_sampler: Optional[Callable[[], np.ndarray]] = None,
                 start_sampler: Optional[Callable[[], np.ndarray]] = None):
        """
        Initialize the trajectory generator.
        
        Args:
            env: The environment to generate trajectories from
            policy: Policy function that takes observation and returns action
            goal_sampler: Function that samples goals (if None, uses random)
            start_sampler: Function that samples start states (if None, uses random)
        """
        self.env = env
        self.policy = policy or self._random_policy
        self.goal_sampler = goal_sampler or self._random_goal_sampler
        self.start_sampler = start_sampler or self._random_start_sampler
    
    def _random_policy(self, obs: np.ndarray) -> int:
        """Random policy for data collection."""
        return self.env.action_space.sample()
    
    def _random_goal_sampler(self) -> np.ndarray:
        """Sample random goal."""
        if hasattr(self.env, 'grid_size'):
            return np.random.uniform(0, self.env.grid_size, size=2)
        else:
            # Fallback for other environments
            return np.random.uniform(-1, 1, size=2)
    
    def _random_start_sampler(self) -> np.ndarray:
        """Sample random start state."""
        if hasattr(self.env, 'grid_size'):
            return np.random.uniform(0, self.env.grid_size, size=2)
        else:
            # Fallback for other environments
            return np.random.uniform(-1, 1, size=2)
    
    def generate_trajectory(self, 
                          max_steps: int = 100,
                          goal: Optional[np.ndarray] = None,
                          start: Optional[np.ndarray] = None,
                          seed: Optional[int] = None) -> Trajectory:
        """
        Generate a single trajectory.
        
        Args:
            max_steps: Maximum number of steps in the trajectory
            goal: Specific goal to use (if None, samples randomly)
            start: Specific start state to use (if None, samples randomly)
            seed: Random seed for reproducibility
        
        Returns:
            Generated trajectory
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Sample goal and start if not provided
        if goal is None:
            goal = self.goal_sampler()
        if start is None:
            start = self.start_sampler()
        
        # Reset environment with specific goal and start
        obs, info = self.env.reset(options={'goal': goal, 'start': start})
        
        # Initialize trajectory
        observations = [obs.copy()]
        actions = []
        next_observations = []
        rewards = []
        terminals = []
        truncateds = []
        goals = [goal.copy()]
        infos = [info.copy()]
        
        done = False
        step_count = 0
        
        while not done and step_count < max_steps:
            # Get action from policy
            action = self.policy(obs)
            
            # Take step
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Store transition
            actions.append(action)
            next_observations.append(next_obs.copy())
            rewards.append(reward)
            terminals.append(terminated)
            truncateds.append(truncated)
            goals.append(goal.copy())
            infos.append(info.copy())
            
            # Update for next iteration
            obs = next_obs
            observations.append(obs.copy())
            done = terminated or truncated
            step_count += 1
        
        # Create trajectory
        trajectory = Trajectory(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            rewards=rewards,
            terminals=terminals,
            truncateds=truncateds,
            goals=goals,
            infos=infos,
            success=info.get('success', False),
            length=len(actions)
        )
        
        return trajectory
    
    def generate_dataset(self, 
                        num_trajectories: int = 1000,
                        max_steps: int = 100,
                        save_path: Optional[str] = None,
                        seed: Optional[int] = None) -> List[Trajectory]:
        """
        Generate a dataset of trajectories.
        
        Args:
            num_trajectories: Number of trajectories to generate
            max_steps: Maximum steps per trajectory
            save_path: Path to save the dataset (optional)
            seed: Random seed for reproducibility
        
        Returns:
            List of generated trajectories
        """
        if seed is not None:
            np.random.seed(seed)
        
        trajectories = []
        
        print(f"Generating {num_trajectories} trajectories...")
        
        for i in range(num_trajectories):
            if i % 100 == 0:
                print(f"Generated {i}/{num_trajectories} trajectories")
            
            trajectory = self.generate_trajectory(max_steps=max_steps)
            trajectories.append(trajectory)
        
        print(f"Generated {len(trajectories)} trajectories")
        
        # Save if path provided
        if save_path is not None:
            self.save_dataset(trajectories, save_path)
        
        return trajectories
    
    def save_dataset(self, trajectories: List[Trajectory], path: str) -> None:
        """Save trajectories to disk."""
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Convert to dictionary format for easier loading
        dataset = {
            'trajectories': [traj.to_dict() for traj in trajectories],
            'num_trajectories': len(trajectories),
            'env_info': {
                'observation_space': str(self.env.observation_space),
                'action_space': str(self.env.action_space),
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"Dataset saved to {path}")
    
    @staticmethod
    def load_dataset(path: str) -> List[Trajectory]:
        """Load trajectories from disk."""
        with open(path, 'rb') as f:
            dataset = pickle.load(f)
        
        trajectories = []
        for traj_dict in dataset['trajectories']:
            trajectory = Trajectory(
                observations=traj_dict['observations'].tolist(),
                actions=traj_dict['actions'].tolist(),
                next_observations=traj_dict['next_observations'].tolist(),
                rewards=traj_dict['rewards'].tolist(),
                terminals=traj_dict['terminals'].tolist(),
                truncateds=traj_dict['truncateds'].tolist(),
                goals=traj_dict['goals'].tolist(),
                infos=[],  # Info is not saved
                success=traj_dict['success'],
                length=traj_dict['length']
            )
            trajectories.append(trajectory)
        
        return trajectories


class ExpertPolicy:
    """
    Simple expert policy for the toy environment.
    Moves directly towards the goal.
    """
    
    def __init__(self, env: gym.Env, noise: float = 0.1):
        self.env = env
        self.noise = noise
    
    def __call__(self, obs: np.ndarray) -> int:
        """Get action from expert policy."""
        # Extract current position and goal from observation
        if len(obs) == 4:  # Goal-conditioned observation
            pos = obs[:2]
            goal = obs[2:]
        else:  # Regular observation
            pos = obs
            goal = self.env.get_goal_observation()
        
        # Calculate direction to goal
        direction = goal - pos
        
        # Add noise
        if self.noise > 0:
            direction += np.random.normal(0, self.noise, size=2)
        
        # Choose action based on direction
        if abs(direction[0]) > abs(direction[1]):
            return 2 if direction[0] < 0 else 3  # left or right
        else:
            return 1 if direction[1] < 0 else 0  # down or up


def create_expert_dataset(env: gym.Env, 
                         num_trajectories: int = 1000,
                         save_path: str = "expert_dataset.pkl",
                         noise: float = 0.1) -> List[Trajectory]:
    """
    Create an expert dataset for the toy environment.
    
    Args:
        env: The environment
        num_trajectories: Number of trajectories to generate
        save_path: Path to save the dataset
        noise: Noise level for the expert policy
    
    Returns:
        List of expert trajectories
    """
    expert_policy = ExpertPolicy(env, noise=noise)
    generator = TrajectoryGenerator(env, policy=expert_policy)
    
    trajectories = generator.generate_dataset(
        num_trajectories=num_trajectories,
        save_path=save_path
    )
    
    # Print statistics
    success_rate = sum(traj.success for traj in trajectories) / len(trajectories)
    avg_length = np.mean([traj.length for traj in trajectories])
    
    print(f"Expert dataset created:")
    print(f"  Success rate: {success_rate:.2%}")
    print(f"  Average length: {avg_length:.1f}")
    print(f"  Total trajectories: {len(trajectories)}")
    
    return trajectories


if __name__ == "__main__":
    # Test the trajectory generator
    from toy_env import make_toy_env
    
    env = make_toy_env(goal_conditioned=True)
    generator = TrajectoryGenerator(env)
    
    # Generate a single trajectory
    trajectory = generator.generate_trajectory(max_steps=20)
    print(f"Generated trajectory with {trajectory.length} steps")
    print(f"Success: {trajectory.success}")
    
    # Generate a small dataset
    trajectories = generator.generate_dataset(num_trajectories=10, max_steps=20)
    print(f"Generated {len(trajectories)} trajectories")
