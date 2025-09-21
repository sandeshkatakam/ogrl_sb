"""
Integrated Model for Offline Goal-Conditioned RL

This module integrates the action reconstruction model and Schrödinger Bridge
to generate complete trajectories following the KL divergence decomposition.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from action_reconstruction_model import ActionReconstructionModel, ActionModelConfig
from schrodinger_bridge import SchrodingerBridge, SchrodingerBridgeConfig
from data_loader import OfflineRLDataLoader, Batch
from sequence_data_loader import SequenceDataLoader, create_sequence_data_loaders
from trajectory_generator import Trajectory


@dataclass
class IntegratedModelConfig:
    """Configuration for the integrated model."""
    # Action model config
    action_state_dim: int = 2
    action_goal_dim: int = 2
    action_action_dim: int = 5
    action_hidden_dim: int = 128
    action_num_layers: int = 2
    action_dropout: float = 0.1
    
    # Schrödinger Bridge config
    bridge_grid_size: int = 20
    bridge_max_iterations: int = 100
    bridge_sinkhorn_iterations: int = 50
    bridge_convergence_threshold: float = 1e-6
    bridge_regularization: float = 0.01
    bridge_max_trajectory_length: int = 50
    
    # Training config
    learning_rate: float = 1e-3
    num_epochs: int = 100
    batch_size: int = 64
    device: str = 'cpu'


class IntegratedOfflineRLModel:
    """
    Integrated model that combines action reconstruction and Schrödinger Bridge.
    
    This model implements the KL divergence decomposition:
    KL(q(s_0:T, a_0:T-1) || p(s_0:T, a_0:T-1)) = 
        KL(q(s_0:T) || p(s_0:T)) + 
        E_q(s_0:T)[KL(q(a_0:T-1|s_0:T) || p(a_0:T-1|s_0:T))]
    
    Where:
    - q(s_0:T) is learned via Schrödinger Bridge
    - p(a_0:T-1|s_0:T) is learned via autoregressive GRU
    """
    
    def __init__(self, config: IntegratedModelConfig):
        self.config = config
        
        # Initialize action model
        action_config = ActionModelConfig(
            state_dim=config.action_state_dim,
            goal_dim=config.action_goal_dim,
            action_dim=config.action_action_dim,
            hidden_dim=config.action_hidden_dim,
            num_layers=config.action_num_layers,
            dropout=config.action_dropout
        )
        self.action_model = ActionReconstructionModel(action_config)
        
        # Initialize Schrödinger Bridge
        bridge_config = SchrodingerBridgeConfig(
            grid_size=config.bridge_grid_size,
            max_iterations=config.bridge_max_iterations,
            sinkhorn_iterations=config.bridge_sinkhorn_iterations,
            convergence_threshold=config.bridge_convergence_threshold,
            regularization=config.bridge_regularization,
            max_trajectory_length=config.bridge_max_trajectory_length
        )
        self.schrodinger_bridge = SchrodingerBridge(bridge_config)
        
        # Training state
        self.is_trained = False
        self.training_history = {'action_loss': [], 'bridge_convergence': []}
    
    def prepare_training_data(self, dataloader: OfflineRLDataLoader) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Extract state trajectories and goals from the dataloader.
        
        Args:
            dataloader: Data loader with offline trajectories
        
        Returns:
            trajectories: List of state trajectories [T, 2]
            goals: List of goal states [2]
        """
        trajectories = []
        goals = []
        
        for batch in dataloader:
            # Extract states (first 2 dimensions) and goals
            print("Batch.States:", batch.states.shape)
            print("Batch.observations:", batch.observations.shape)
            batch_states = batch.states[:, :, :2].numpy()  # [batch_size, seq_len, 2]
            batch_goals = batch.goals[:, :2].numpy()  # [batch_size, 2]
            
            for i in range(batch_states.shape[0]):
                # Get trajectory length (exclude padding)
                traj_length = batch.observations.shape[1]
                
                # Extract state trajectory
                state_traj = batch_states[i, :traj_length]
                trajectories.append(state_traj)
                
                # Extract goal (same for all timesteps)
                goal = batch_goals[i]
                goals.append(goal)
        
        return trajectories, goals
    
    def train_action_model(self, 
                          train_loader: SequenceDataLoader,
                          val_loader: SequenceDataLoader) -> List[float]:
        """
        Train the action reconstruction model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        
        Returns:
            training_losses: List of training losses
        """
        print("Training action reconstruction model...")
        
        from action_reconstruction_model import ActionModelTrainer
        
        trainer = ActionModelTrainer(
            model=self.action_model,
            learning_rate=self.config.learning_rate,
            device=self.config.device
        )
        
        training_losses = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=self.config.num_epochs,
            verbose=True
        )
        
        self.training_history['action_loss'] = training_losses
        return training_losses
    
    def train_schrodinger_bridge(self, 
                                trajectories: List[np.ndarray],
                                goals: List[np.ndarray]) -> np.ndarray:
        """
        Train the Schrödinger Bridge.
        
        Args:
            trajectories: List of state trajectories
            goals: List of goal states
        
        Returns:
            bridge_matrices: Learned bridge transition matrices
        """
        print("Training Schrödinger Bridge...")
        
        bridge_matrices = self.schrodinger_bridge.solve_bridge(trajectories, goals)
        
        # Store convergence info
        self.training_history['bridge_convergence'] = [True]  # Simplified
        
        return bridge_matrices
    
    def train(self, 
              train_loader: SequenceDataLoader,
              val_loader: SequenceDataLoader) -> Dict[str, Any]:
        """
        Train the complete integrated model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        
        Returns:
            training_info: Dictionary with training results
        """
        print("Training integrated offline goal-conditioned RL model...")
        
        # Prepare training data for Schrödinger Bridge
        trajectories, goals = self.prepare_training_data_from_sequences(train_loader)
        
        # Train action model
        action_losses = self.train_action_model(train_loader, val_loader)
        
        # Train Schrödinger Bridge
        bridge_matrices = self.train_schrodinger_bridge(trajectories, goals)
        
        self.is_trained = True
        
        training_info = {
            'action_losses': action_losses,
            'bridge_converged': True,
            'num_trajectories': len(trajectories),
            'bridge_shape': bridge_matrices.shape
        }
        
        return training_info
    
    def prepare_training_data_from_sequences(self, dataloader: SequenceDataLoader) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Extract state trajectories and goals from the sequence dataloader.
        
        Args:
            dataloader: Sequence data loader with offline trajectories
        
        Returns:
            trajectories: List of state trajectories [T, 2]
            goals: List of goal states [2]
        """
        trajectories = []
        goals = []
        
        for batch in dataloader:
            # Extract states and goals
            print("batch.states:", batch.states.shape)
            print("batch.goals:", batch.goals.shape)
            batch_states = batch.states.numpy()  # [batch_size, seq_len, 2]
            batch_goals = batch.goals.numpy()  # [batch_size, 2]
            
            for i in range(batch_states.shape[0]):
                # Get trajectory length (exclude padding)
                traj_length = batch.actions.shape[1] + 1  # actions is seq_len-1
                
                # Extract state trajectory
                state_traj = batch_states[i, :traj_length]
                trajectories.append(state_traj)
                
                # Extract goal
                goal = batch_goals[i]
                goals.append(goal)
        
        return trajectories, goals
    
    def generate_trajectory(self, 
                           initial_state: Optional[np.ndarray] = None,
                           goal_state: Optional[np.ndarray] = None,
                           temperature: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a complete trajectory using both components.
        
        Args:
            initial_state: Initial state (if None, sample from bridge)
            goal_state: Goal state (if None, sample from bridge)
            temperature: Sampling temperature for actions
        
        Returns:
            states: [T+1, 2] - Generated state trajectory
            actions: [T] - Generated action sequence
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Sample state trajectory from Schrödinger Bridge
        state_trajectory = self.schrodinger_bridge.sample_trajectory(
            initial_state=initial_state,
            goal_state=goal_state
        )
        
        # Prepare data for action model
        states_tensor = torch.tensor(state_trajectory, dtype=torch.float32).unsqueeze(0)  # [1, T+1, 2]
        
        if goal_state is None:
            # Use the last state as goal (or sample from bridge)
            goal_tensor = torch.tensor(state_trajectory[-1], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, 2]
        else:
            goal_tensor = torch.tensor(goal_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, 2]
        
        goal_tensor = goal_tensor.expand(-1, states_tensor.shape[1], -1)  # [1, T+1, 2]
        
        # Sample actions from action model
        with torch.no_grad():
            actions = self.action_model.sample_actions(
                states_tensor, goal_tensor, temperature=temperature
            )
        
        actions = actions.squeeze(0).numpy()  # [T]
        
        return state_trajectory, actions
    
    def generate_trajectories(self, 
                             num_trajectories: int = 100,
                             initial_states: Optional[List[np.ndarray]] = None,
                             goal_states: Optional[List[np.ndarray]] = None,
                             temperature: float = 1.0) -> List[Trajectory]:
        """
        Generate multiple trajectories.
        
        Args:
            num_trajectories: Number of trajectories to generate
            initial_states: List of initial states (if None, sample from bridge)
            goal_states: List of goal states (if None, sample from bridge)
            temperature: Sampling temperature for actions
        
        Returns:
            trajectories: List of generated trajectories
        """
        trajectories = []
        
        for i in range(num_trajectories):
            # Get initial and goal states
            initial_state = initial_states[i] if initial_states else None
            goal_state = goal_states[i] if goal_states else None
            
            # Generate trajectory
            states, actions = self.generate_trajectory(
                initial_state=initial_state,
                goal_state=goal_state,
                temperature=temperature
            )
            
            # Create trajectory object
            trajectory = Trajectory(
                observations=[states[j] for j in range(len(states))],
                actions=actions.tolist(),
                next_observations=[states[j+1] for j in range(len(actions))],
                rewards=[0.0] * len(actions),  # Placeholder
                terminals=[False] * (len(actions) - 1) + [True],
                truncateds=[False] * len(actions),
                goals=[goal_state if goal_state is not None else states[-1]] * len(states),
                infos=[{}] * len(states),
                success=False,  # Placeholder
                length=len(actions)
            )
            
            trajectories.append(trajectory)
        
        return trajectories
    
    def evaluate_trajectories(self, 
                             generated_trajectories: List[Trajectory],
                             goal_tolerance: float = 0.5) -> Dict[str, float]:
        """
        Evaluate generated trajectories.
        
        Args:
            generated_trajectories: List of generated trajectories
            goal_tolerance: Tolerance for goal reaching
        
        Returns:
            metrics: Dictionary with evaluation metrics
        """
        if not generated_trajectories:
            return {}
        
        # Compute metrics
        success_rate = 0.0
        avg_length = 0.0
        goal_distances = []
        
        for traj in generated_trajectories:
            # Check if goal was reached
            final_state = np.array(traj.observations[-1])
            goal = np.array(traj.goals[0])
            distance = np.linalg.norm(final_state - goal)
            goal_distances.append(distance)
            
            if distance <= goal_tolerance:
                success_rate += 1.0
            
            avg_length += traj.length
        
        success_rate /= len(generated_trajectories)
        avg_length /= len(generated_trajectories)
        avg_goal_distance = np.mean(goal_distances)
        
        metrics = {
            'success_rate': success_rate,
            'avg_length': avg_length,
            'avg_goal_distance': avg_goal_distance,
            'num_trajectories': len(generated_trajectories)
        }
        
        return metrics
    
    def visualize_generation(self, 
                           num_samples: int = 5,
                           initial_states: Optional[List[np.ndarray]] = None,
                           goal_states: Optional[List[np.ndarray]] = None):
        """Visualize generated trajectories."""
        if not self.is_trained:
            print("Model not trained yet. Call train() first.")
            return
        
        # Generate trajectories
        trajectories = self.generate_trajectories(
            num_trajectories=num_samples,
            initial_states=initial_states,
            goal_states=goal_states
        )
        
        # Plot
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        for i, traj in enumerate(trajectories):
            states = np.array(traj.observations)
            goal = np.array(traj.goals[0])
            
            # Plot trajectory
            ax.plot(states[:, 0], states[:, 1], f'C{i}-', alpha=0.7, linewidth=2, 
                   label=f'Trajectory {i+1}')
            ax.scatter(states[0, 0], states[0, 1], c=f'C{i}', s=100, marker='o', zorder=5)
            ax.scatter(states[-1, 0], states[-1, 1], c=f'C{i}', s=100, marker='s', zorder=5)
            ax.scatter(goal[0], goal[1], c=f'C{i}', s=100, marker='*', zorder=5)
        
        ax.set_title('Generated Trajectories from Integrated Model')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.show()


def create_integrated_model(config: IntegratedModelConfig) -> IntegratedOfflineRLModel:
    """Create an integrated offline RL model."""
    return IntegratedOfflineRLModel(config)


if __name__ == "__main__":
    # Test the integrated model
    config = IntegratedModelConfig()
    model = create_integrated_model(config)
    
    print("Integrated model created successfully!")
    print(f"Action model: {model.action_model}")
    print(f"Schrödinger Bridge: {model.schrodinger_bridge}")
    print("Model ready for training!")
