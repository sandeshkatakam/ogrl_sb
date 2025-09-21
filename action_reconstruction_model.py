"""
Action Reconstruction Model for Offline Goal-Conditioned RL

This module implements p(a_0:T-1|s_0:T) using an autoregressive GRU model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ActionModelConfig:
    """Configuration for the action reconstruction model."""
    state_dim: int = 2  # Dimension of state (x, y)
    goal_dim: int = 2   # Dimension of goal (gx, gy)
    action_dim: int = 5 # Number of discrete actions
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    max_seq_length: int = 50


class ActionReconstructionModel(nn.Module):
    """
    Autoregressive GRU model for learning p(a_0:T-1|s_0:T).
    
    The model takes a sequence of states and goals and predicts the corresponding
    action sequence autoregressively.
    """
    
    def __init__(self, config: ActionModelConfig):
        super().__init__()
        self.config = config
        
        # Input embedding: state + goal
        input_dim = config.state_dim + config.goal_dim
        self.input_embedding = nn.Linear(input_dim, config.hidden_dim)
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True
        )
        
        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.action_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, 
                states: torch.Tensor, 
                goals: torch.Tensor,
                actions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the action reconstruction model.
        
        Args:
            states: [batch_size, seq_len, state_dim] - State sequences
            goals: [batch_size, seq_len, goal_dim] - Goal sequences (same goal repeated)
            actions: [batch_size, seq_len-1] - Action sequences (for training)
        
        Returns:
            logits: [batch_size, seq_len-1, action_dim] - Action logits
            log_probs: [batch_size, seq_len-1, action_dim] - Action log probabilities
        """
        batch_size, seq_len, _ = states.shape
        
        # Concatenate states and goals
        state_goal = torch.cat([states, goals], dim=-1)  # [batch_size, seq_len, state_dim + goal_dim]
        
        # Embed input
        embedded = self.input_embedding(state_goal)  # [batch_size, seq_len, hidden_dim]
        
        # Pass through GRU
        gru_output, _ = self.gru(embedded)  # [batch_size, seq_len, hidden_dim]
        
        # Predict actions (exclude last state since we predict actions for transitions)
        action_states = gru_output[:, :-1, :]  # [batch_size, seq_len-1, hidden_dim]
        logits = self.action_head(action_states)  # [batch_size, seq_len-1, action_dim]
        
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        return logits, log_probs
    
    def sample_actions(self, 
                      states: torch.Tensor, 
                      goals: torch.Tensor,
                      temperature: float = 1.0) -> torch.Tensor:
        """
        Sample actions from the model.
        
        Args:
            states: [batch_size, seq_len, state_dim] - State sequences
            goals: [batch_size, seq_len, goal_dim] - Goal sequences
            temperature: Sampling temperature
        
        Returns:
            actions: [batch_size, seq_len-1] - Sampled actions
        """
        with torch.no_grad():
            logits, _ = self.forward(states, goals)
            logits = logits / temperature
            
            # Sample actions
            probs = F.softmax(logits, dim=-1)
            if torch.isnan(probs).any():
                print("NaN detected in action probabilities!")
                print("Logits:", logits)
                print("Probabilities:", probs)
                # Optionally, replace NaNs with zeros or a small value
                probs = torch.nan_to_num(probs, nan=0.0)
            # probs = F.softmax(logits, dim=-1)
            actions = torch.multinomial(probs.view(-1, self.config.action_dim), 1)
            actions = actions.view(states.shape[0], -1)
            
            return actions
    
    def compute_loss(self, 
                    states: torch.Tensor, 
                    goals: torch.Tensor,
                    actions: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log-likelihood loss.
        
        Args:
            states: [batch_size, seq_len, state_dim] - State sequences
            goals: [batch_size, seq_len, goal_dim] - Goal sequences
            actions: [batch_size, seq_len-1] - Target actions
        
        Returns:
            loss: Scalar loss value
        """
        logits, log_probs = self.forward(states, goals)
        
        # Compute NLL loss
        loss = F.nll_loss(
            log_probs.view(-1, self.config.action_dim),
            actions.view(-1),
            reduction='mean'
        )
        
        return loss


class ActionModelTrainer:
    """Trainer for the action reconstruction model."""
    
    def __init__(self, 
                 model: ActionReconstructionModel,
                 learning_rate: float = 1e-3,
                 device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
    
    def train_epoch(self, dataloader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            # Move batch to device
            batch = batch.to(self.device)
            
            # Extract data - handle both sequence and transition-based batches
            if hasattr(batch, 'states'):
                # Sequence-based batch
                states = batch.states  # [batch_size, seq_len, 2]
                goals = batch.goals.unsqueeze(1).expand(-1, states.shape[1], -1)  # [batch_size, seq_len, 2]
                actions = batch.actions  # [batch_size, seq_len-1]
            else:
                # Transition-based batch (legacy)
                states = batch.observations[:, :, :2]  # [batch_size, seq_len, 2] - only state part
                goals = batch.goals[:, :2]  # [batch_size, 2] - goal (same for all timesteps)
                goals = goals.unsqueeze(1).expand(-1, states.shape[1], -1)  # [batch_size, seq_len, 2]
                actions = batch.actions  # [batch_size, seq_len-1]
            
            # Forward pass
            self.optimizer.zero_grad()
            loss = self.model.compute_loss(states, goals, actions)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        self.scheduler.step(avg_loss)
        
        return avg_loss
    
    def evaluate(self, dataloader) -> float:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                
                # Extract data - handle both sequence and transition-based batches
                if hasattr(batch, 'states'):
                    # Sequence-based batch
                    states = batch.states  # [batch_size, seq_len, 2]
                    goals = batch.goals.unsqueeze(1).expand(-1, states.shape[1], -1)  # [batch_size, seq_len, 2]
                    actions = batch.actions  # [batch_size, seq_len-1]
                else:
                    # Transition-based batch (legacy)
                    states = batch.observations[:, :, :2]
                    goals = batch.goals[:, :2]
                    goals = goals.unsqueeze(1).expand(-1, states.shape[1], -1)
                    actions = batch.actions
                
                # Compute loss
                loss = self.model.compute_loss(states, goals, actions)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def train(self, 
              train_loader, 
              val_loader, 
              num_epochs: int = 100,
              verbose: bool = True) -> List[float]:
        """Train the model."""
        train_losses = []
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            train_losses.append(train_loss)
            
            # Validate
            val_loss = self.evaluate(val_loader)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}: "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        return train_losses


def create_action_model(config: ActionModelConfig) -> ActionReconstructionModel:
    """Create an action reconstruction model."""
    return ActionReconstructionModel(config)


if __name__ == "__main__":
    # Test the action model
    config = ActionModelConfig()
    model = create_action_model(config)
    
    # Create dummy data
    batch_size, seq_len = 4, 10
    states = torch.randn(batch_size, seq_len, config.state_dim)
    goals = torch.randn(batch_size, seq_len, config.goal_dim)
    actions = torch.randint(0, config.action_dim, (batch_size, seq_len-1))
    
    # Test forward pass
    logits, log_probs = model(states, goals)
    print(f"Logits shape: {logits.shape}")
    print(f"Log probs shape: {log_probs.shape}")
    
    # Test sampling
    sampled_actions = model.sample_actions(states, goals)
    print(f"Sampled actions shape: {sampled_actions.shape}")
    
    # Test loss
    loss = model.compute_loss(states, goals, actions)
    print(f"Loss: {loss.item():.4f}")
    
    print("Action model test passed!")
