"""
Schrödinger Bridge Implementation with IPF-Sinkhorn Algorithm

This module implements the discrete Schrödinger Bridge optimization using
the Iterative Proportional Fitting (IPF) algorithm with Sinkhorn iterations.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class SchrodingerBridgeConfig:
    """Configuration for the Schrödinger Bridge."""
    grid_size: int = 20  # Grid resolution for discretization
    max_iterations: int = 100  # Maximum IPF iterations
    sinkhorn_iterations: int = 50  # Sinkhorn iterations per IPF step
    convergence_threshold: float = 1e-6  # Convergence threshold
    regularization: float = 0.01  # Sinkhorn regularization parameter
    max_trajectory_length: int = 50  # Maximum trajectory length


class StateDiscretizer:
    """Discretizes continuous state space into a grid."""
    
    def __init__(self, 
                 grid_size: int = 20,
                 state_bounds: Tuple[float, float] = (0.0, 10.0)):
        self.grid_size = grid_size
        self.state_bounds = state_bounds
        self.state_min, self.state_max = state_bounds
        
        # Create grid points
        self.grid_points = np.linspace(self.state_min, self.state_max, grid_size)
        self.grid_spacing = self.grid_points[1] - self.grid_points[0]
        
        # Create 2D grid
        x_grid, y_grid = np.meshgrid(self.grid_points, self.grid_points, indexing='ij')
        self.grid_coords = np.stack([x_grid.flatten(), y_grid.flatten()], axis=1)
        self.num_states = len(self.grid_coords)
    
    def discretize_state(self, state: np.ndarray) -> int:
        """Convert continuous state to discrete grid index."""
        # Clip state to bounds
        state = np.clip(state, self.state_min, self.state_max)
        
        # Find closest grid point
        distances = np.linalg.norm(self.grid_coords - state, axis=1)
        return np.argmin(distances)
    
    def continuous_state(self, discrete_idx: int) -> np.ndarray:
        """Convert discrete grid index to continuous state."""
        return self.grid_coords[discrete_idx]
    
    def discretize_trajectory(self, trajectory: np.ndarray) -> List[int]:
        """Discretize a trajectory of states."""
        return [self.discretize_state(state) for state in trajectory]


class SchrodingerBridge:
    """
    Discrete Schrödinger Bridge implementation using IPF-Sinkhorn algorithm.
    
    The bridge connects initial state distribution p_0(s_0) to goal state 
    distribution p_T(s_T) through intermediate distributions p_t(s_t).
    """
    
    def __init__(self, config: SchrodingerBridgeConfig):
        self.config = config
        self.discretizer = StateDiscretizer(
            grid_size=config.grid_size,
            state_bounds=(0.0, 10.0)  # Match toy environment bounds
        )
        self.num_states = self.discretizer.num_states
        self.T = config.max_trajectory_length
        
        # Initialize transition matrices and distributions
        self.transition_matrices = None
        self.marginal_distributions = None
        self.joint_distributions = None
        
    def compute_transition_matrix(self, 
                                states: np.ndarray, 
                                next_states: np.ndarray) -> np.ndarray:
        """
        Compute empirical transition matrix from trajectory data.
        
        Args:
            states: [N, 2] - Current states
            next_states: [N, 2] - Next states
        
        Returns:
            P: [num_states, num_states] - Transition matrix
        """
        P = np.zeros((self.num_states, self.num_states))
        
        for state, next_state in zip(states, next_states):
            i = self.discretizer.discretize_state(state)
            j = self.discretizer.discretize_state(next_state)
            P[i, j] += 1
        
        # Normalize rows
        row_sums = P.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        P = P / row_sums[:, np.newaxis]
        
        return P
    
    def extract_marginals_from_data(self, 
                                   trajectories: List[np.ndarray],
                                   goals: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract initial and goal state distributions from trajectory data.
        
        Args:
            trajectories: List of state trajectories
            goals: List of goal states
        
        Returns:
            p_0: [num_states] - Initial state distribution
            p_T: [num_states] - Goal state distribution
        """
        # Extract initial states
        initial_states = np.array([traj[0] for traj in trajectories])
        initial_discrete = [self.discretizer.discretize_state(s) for s in initial_states]
        
        # Extract goal states
        goal_discrete = [self.discretizer.discretize_state(g) for g in goals]
        
        # Compute distributions
        p_0 = np.zeros(self.num_states)
        p_T = np.zeros(self.num_states)
        
        for idx in initial_discrete:
            p_0[idx] += 1
        p_0 = p_0 / p_0.sum()
        
        for idx in goal_discrete:
            p_T[idx] += 1
        p_T = p_T / p_T.sum()
        
        return p_0, p_T
    
    def sinkhorn_iteration(self, 
                          K: np.ndarray, 
                          a: np.ndarray, 
                          b: np.ndarray,
                          num_iterations: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Sinkhorn iterations to find scaling factors.
        
        Args:
            K: [m, n] - Cost matrix
            a: [m] - Target row marginals
            b: [n] - Target column marginals
            num_iterations: Number of Sinkhorn iterations
        
        Returns:
            u: [m] - Row scaling factors
            v: [n] - Column scaling factors
        """
        u = np.ones(K.shape[0]) / K.shape[0]
        v = np.ones(K.shape[1]) / K.shape[1]
        
        for _ in range(num_iterations):
            # Update u
            u = a / (K @ v + 1e-8)
            # Update v
            v = b / (K.T @ u + 1e-8)
        
        return u, v
    
    def ipf_sinkhorn_step(self, 
                         P: np.ndarray, 
                         target_marginals: np.ndarray,
                         current_marginals: np.ndarray) -> np.ndarray:
        """
        One step of IPF with Sinkhorn iterations.
        
        Args:
            P: [num_states, num_states] - Current transition matrix
            target_marginals: [num_states] - Target marginal distribution
            current_marginals: [num_states] - Current marginal distribution
        
        Returns:
            P_new: [num_states, num_states] - Updated transition matrix
        """
        # Compute scaling factors using Sinkhorn
        u, v = self.sinkhorn_iteration(
            P, target_marginals, current_marginals, self.config.sinkhorn_iterations
        )
        
        # Update transition matrix
        P_new = P * np.outer(u, v)
        
        return P_new
    
    def solve_bridge(self, 
                    trajectories: List[np.ndarray],
                    goals: List[np.ndarray]) -> np.ndarray:
        """
        Solve the Schrödinger Bridge problem.
        
        Args:
            trajectories: List of state trajectories
            goals: List of goal states
        
        Returns:
            P_bridge: [T, num_states, num_states] - Bridge transition matrices
        """
        print("Solving Schrödinger Bridge...")
        
        # Extract initial and goal distributions
        p_0, p_T = self.extract_marginals_from_data(trajectories, goals)
        
        # Compute empirical transition matrix from data
        all_states = np.concatenate([traj[:-1] for traj in trajectories])
        all_next_states = np.concatenate([traj[1:] for traj in trajectories])
        P_empirical = self.compute_transition_matrix(all_states, all_next_states)
        
        # Initialize bridge with empirical transitions
        P_bridge = np.tile(P_empirical, (self.T, 1, 1))
        
        # IPF iterations
        for iteration in range(self.config.max_iterations):
            P_old = P_bridge.copy()
            
            # Forward pass: update transitions to match target marginals
            current_marginals = p_0.copy()
            for t in range(self.T):
                # Update transition matrix at time t
                P_bridge[t] = self.ipf_sinkhorn_step(
                    P_bridge[t], p_T, current_marginals
                )
                
                # Update current marginals
                current_marginals = P_bridge[t].T @ current_marginals
            
            # Backward pass: update transitions to match initial marginals
            current_marginals = p_T.copy()
            for t in range(self.T-1, -1, -1):
                # Update transition matrix at time t
                P_bridge[t] = self.ipf_sinkhorn_step(
                    P_bridge[t], p_0, current_marginals
                )
                
                # Update current marginals
                current_marginals = P_bridge[t] @ current_marginals
            
            # Check convergence
            max_change = np.max(np.abs(P_bridge - P_old))
            if max_change < self.config.convergence_threshold:
                print(f"Converged after {iteration + 1} iterations")
                break
            
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}: Max change = {max_change:.6f}")
        
        self.transition_matrices = P_bridge
        return P_bridge
    
    def sample_trajectory(self, 
                         initial_state: Optional[np.ndarray] = None,
                         goal_state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Sample a trajectory from the learned bridge.
        
        Args:
            initial_state: Initial state (if None, sample from p_0)
            goal_state: Goal state (if None, sample from p_T)
        
        Returns:
            trajectory: [T+1, 2] - Sampled state trajectory
        """
        if self.transition_matrices is None:
            raise ValueError("Bridge not solved yet. Call solve_bridge first.")
        
        # Sample initial state
        if initial_state is None:
            # Sample from initial distribution
            p_0, _ = self.extract_marginals_from_data([], [])
            # Fix: If p_0 contains NaN or sums to zero, use uniform distribution
            if np.isnan(p_0).any() or p_0.sum() == 0:
                p_0 = np.ones(self.num_states) / self.num_states
            else:
                p_0 = np.nan_to_num(p_0, nan=0.0)
                p_0 = p_0 / p_0.sum()
            initial_idx = np.random.choice(self.num_states, p=p_0)
        else:
            initial_idx = self.discretizer.discretize_state(initial_state)
        
        # Sample trajectory
        trajectory_indices = [initial_idx]
        current_idx = initial_idx
        
        for t in range(self.T):
            # Sample next state
            next_probs = self.transition_matrices[t][current_idx]
            # Fix: Ensure probabilities sum to 1 and contain no NaN
            next_probs = np.nan_to_num(next_probs, nan=0.0)
            if next_probs.sum() == 0:
                next_probs = np.ones(self.num_states) / self.num_states
            else:
                next_probs = next_probs / next_probs.sum()
            next_idx = np.random.choice(self.num_states, p=next_probs)
            trajectory_indices.append(next_idx)
            current_idx = next_idx
        
        # Convert to continuous states
        trajectory = np.array([self.discretizer.continuous_state(idx) 
                              for idx in trajectory_indices])
        
        return trajectory
    
    def visualize_bridge(self, 
                        trajectories: List[np.ndarray],
                        goals: List[np.ndarray],
                        num_samples: int = 10):
        """Visualize the learned bridge."""
        if self.transition_matrices is None:
            print("Bridge not solved yet. Call solve_bridge first.")
            return
        
        # Sample trajectories from bridge
        bridge_trajectories = []
        for _ in range(num_samples):
            traj = self.sample_trajectory()
            bridge_trajectories.append(traj)
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original trajectories
        ax1 = axes[0]
        for i, traj in enumerate(trajectories[:5]):  # Show first 5
            ax1.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.7, linewidth=2)
            ax1.scatter(traj[0, 0], traj[0, 1], c='green', s=100, marker='o', zorder=5)
            ax1.scatter(traj[-1, 0], traj[-1, 1], c='red', s=100, marker='s', zorder=5)
        
        ax1.set_title('Original Trajectories')
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        ax1.grid(True, alpha=0.3)
        
        # Bridge trajectories
        ax2 = axes[1]
        for i, traj in enumerate(bridge_trajectories[:5]):  # Show first 5
            ax2.plot(traj[:, 0], traj[:, 1], 'r-', alpha=0.7, linewidth=2)
            ax2.scatter(traj[0, 0], traj[0, 1], c='green', s=100, marker='o', zorder=5)
            ax2.scatter(traj[-1, 0], traj[-1, 1], c='red', s=100, marker='s', zorder=5)
        
        ax2.set_title('Schrödinger Bridge Trajectories')
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def create_schrodinger_bridge(config: SchrodingerBridgeConfig) -> SchrodingerBridge:
    """Create a Schrödinger Bridge instance."""
    return SchrodingerBridge(config)


if __name__ == "__main__":
    # Test the Schrödinger Bridge
    config = SchrodingerBridgeConfig()
    bridge = create_schrodinger_bridge(config)
    
    # Create dummy trajectory data
    trajectories = [
        np.array([[1, 1], [2, 1], [3, 1], [4, 2], [5, 3], [6, 4], [7, 5], [8, 6]]),
        np.array([[2, 2], [3, 2], [4, 3], [5, 4], [6, 5], [7, 6], [8, 7]]),
    ]
    goals = [np.array([8, 8]), np.array([8, 8])]
    
    # Solve bridge
    P_bridge = bridge.solve_bridge(trajectories, goals)
    print(f"Bridge solved. Shape: {P_bridge.shape}")
    
    # Sample trajectory
    traj = bridge.sample_trajectory()
    print(f"Sampled trajectory shape: {traj.shape}")
    
    print("Schrödinger Bridge test passed!")
