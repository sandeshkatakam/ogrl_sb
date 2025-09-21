"""
Simple 2D Navigation Environment for Offline Goal-Conditioned RL
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt


class Toy2DNavigationEnv(gym.Env):
    """
    A simple 2D navigation environment for testing offline goal-conditioned RL algorithms.
    
    The agent starts at a random position and must navigate to a goal position.
    State: [x, y] position
    Action: [dx, dy] movement (discretized to 4 directions)
    Goal: [gx, gy] target position
    """
    
    def __init__(self, 
                 grid_size: int = 10,
                 max_steps: int = 50,
                 goal_tolerance: float = 0.5,
                 action_noise: float = 0.0):
        super().__init__()
        
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.goal_tolerance = goal_tolerance
        self.action_noise = action_noise
        
        # State space: [x, y] position
        self.observation_space = spaces.Box(
            low=0.0, high=grid_size, shape=(2,), dtype=np.float32
        )
        
        # Action space: 4 discrete directions + no-op
        self.action_space = spaces.Discrete(5)  # up, down, left, right, no-op
        
        # Action mapping
        self.action_map = {
            0: np.array([0, 1]),   # up
            1: np.array([0, -1]),  # down
            2: np.array([-1, 0]),  # left
            3: np.array([1, 0]),   # right
            4: np.array([0, 0]),   # no-op
        }
        
        self.reset()
    
    def reset(self, 
              seed: Optional[int] = None, 
              options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        if options is None:
            options = {}
        
        # Set goal if provided, otherwise random
        if 'goal' in options:
            self.goal = np.array(options['goal'], dtype=np.float32)
        else:
            self.goal = self.np_random.uniform(0, self.grid_size, size=2).astype(np.float32)
        
        # Set start position if provided, otherwise random
        if 'start' in options:
            self.state = np.array(options['start'], dtype=np.float32)
        else:
            self.state = self.np_random.uniform(0, self.grid_size, size=2).astype(np.float32)
        
        self.step_count = 0
        self.done = False
        
        info = {
            'goal': self.goal.copy(),
            'success': False,
            'distance_to_goal': np.linalg.norm(self.state - self.goal)
        }
        
        return self.state.copy(), info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        if self.done:
            return self.state, 0.0, True, True, {}
        
        # Get movement vector
        movement = self.action_map[action].astype(np.float32)
        
        # Add noise if specified
        if self.action_noise > 0:
            noise = self.np_random.normal(0, self.action_noise, size=2)
            movement += noise
        
        # Update state
        new_state = self.state + movement
        
        # Clip to grid bounds
        new_state = np.clip(new_state, 0, self.grid_size)
        
        # Check if goal reached
        distance_to_goal = np.linalg.norm(new_state - self.goal)
        goal_reached = distance_to_goal <= self.goal_tolerance
        
        # Calculate reward
        if goal_reached:
            reward = 1.0
            terminated = True
        else:
            # Sparse reward: only reward when reaching goal
            reward = 0.0
            terminated = False
        
        # Check if max steps exceeded
        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        
        # Update state
        self.state = new_state
        self.done = terminated or truncated
        
        info = {
            'goal': self.goal.copy(),
            'success': goal_reached,
            'distance_to_goal': distance_to_goal,
            'step_count': self.step_count
        }
        
        return self.state.copy(), reward, terminated, truncated, info
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the environment."""
        if mode == 'human':
            plt.figure(figsize=(6, 6))
            plt.xlim(0, self.grid_size)
            plt.ylim(0, self.grid_size)
            
            # Plot agent position
            plt.scatter(self.state[0], self.state[1], c='blue', s=100, label='Agent', marker='o')
            
            # Plot goal position
            plt.scatter(self.goal[0], self.goal[1], c='red', s=100, label='Goal', marker='*')
            
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f'Step {self.step_count}, Distance: {np.linalg.norm(self.state - self.goal):.2f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
        
        return None
    
    def get_goal_observation(self) -> np.ndarray:
        """Get the current goal as an observation."""
        return self.goal.copy()
    
    def set_goal(self, goal: np.ndarray) -> None:
        """Set a new goal."""
        self.goal = np.array(goal, dtype=np.float32)


class GoalConditionedWrapper(gym.Wrapper):
    """
    Wrapper to make the environment goal-conditioned by concatenating goal to observation.
    """
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        
        # Update observation space to include goal
        original_obs_space = env.observation_space
        goal_space = spaces.Box(
            low=0.0, high=env.grid_size, shape=(2,), dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=np.concatenate([original_obs_space.low, goal_space.low]),
            high=np.concatenate([original_obs_space.high, goal_space.high]),
            shape=(4,),  # [x, y, gx, gy]
            dtype=np.float32
        )
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        goal = info['goal']
        goal_conditioned_obs = np.concatenate([obs, goal])
        return goal_conditioned_obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        goal = info['goal']
        goal_conditioned_obs = np.concatenate([obs, goal])
        return goal_conditioned_obs, reward, terminated, truncated, info


def make_toy_env(grid_size: int = 10, 
                 max_steps: int = 50,
                 goal_conditioned: bool = True,
                 **kwargs) -> gym.Env:
    """
    Create a toy 2D navigation environment.
    
    Args:
        grid_size: Size of the grid
        max_steps: Maximum steps per episode
        goal_conditioned: Whether to return goal-conditioned observations
        **kwargs: Additional arguments passed to the environment
    
    Returns:
        The environment instance
    """
    env = Toy2DNavigationEnv(grid_size=grid_size, max_steps=max_steps, **kwargs)
    
    if goal_conditioned:
        env = GoalConditionedWrapper(env)
    
    return env


if __name__ == "__main__":
    # Test the environment
    env = make_toy_env(goal_conditioned=True)
    
    print("Environment created successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Test episode
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Goal: {info['goal']}")
    
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step}: action={action}, obs={obs}, reward={reward}, done={terminated or truncated}")
        
        if terminated or truncated:
            break
