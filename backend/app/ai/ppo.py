"""
Proximal Policy Optimization (PPO) Implementation

This module implements PPO algorithm for training the Deep RL agent.
PPO is a state-of-the-art on-policy algorithm that provides stable training.

Research Features:
- Clipped surrogate objective
- Generalized Advantage Estimation (GAE)
- Value function clipping
- Entropy bonus for exploration
- Learning rate scheduling
"""

import numpy as np
import math
import time
from typing import List, Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import deque

import sys
sys.path.insert(0, '..')


@dataclass
class PPOConfig:
    """Configuration for PPO training"""
    # Core hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    clip_epsilon: float = 0.2  # PPO clipping parameter
    
    # Training settings
    batch_size: int = 64
    n_epochs: int = 4  # Epochs per update
    n_steps: int = 2048  # Steps per rollout
    
    # Value function
    value_loss_coef: float = 0.5
    value_clip: bool = True  # Clip value function updates
    
    # Entropy
    entropy_coef: float = 0.01
    entropy_decay: float = 0.999  # Decay entropy coef over time
    min_entropy_coef: float = 0.001
    
    # Gradient clipping
    max_grad_norm: float = 0.5
    
    # Learning rate schedule
    lr_schedule: str = 'linear'  # 'constant', 'linear', 'cosine'
    warmup_steps: int = 1000
    
    # Normalization
    normalize_advantages: bool = True
    normalize_returns: bool = True
    
    # Misc
    seed: int = 42


@dataclass
class RolloutBuffer:
    """
    Buffer for storing rollout data during training.
    
    Stores states, actions, rewards, values, log_probs for PPO update.
    """
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[Any] = field(default_factory=list)
    action_log_probs: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    
    # Computed during processing
    advantages: Optional[np.ndarray] = None
    returns: Optional[np.ndarray] = None
    
    def add(self, state: np.ndarray, action: Any, log_prob: float,
            reward: float, value: float, done: bool):
        """Add a transition to the buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.action_log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def __len__(self) -> int:
        return len(self.states)
    
    def clear(self):
        """Clear the buffer"""
        self.states.clear()
        self.actions.clear()
        self.action_log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.advantages = None
        self.returns = None
    
    def compute_returns_and_advantages(self, last_value: float, 
                                       gamma: float, gae_lambda: float):
        """
        Compute returns and GAE advantages.
        
        Generalized Advantage Estimation:
        A_t = delta_t + (gamma * lambda) * delta_{t+1} + ...
        where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        """
        n = len(self.rewards)
        advantages = np.zeros(n)
        returns = np.zeros(n)
        
        last_gae = 0.0
        last_return = last_value
        
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(self.dones[t])
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - float(self.dones[t])
            
            # TD error
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            
            # GAE
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
            
            # Returns (for value function targets)
            last_return = self.rewards[t] + gamma * last_return * next_non_terminal
            returns[t] = last_return
        
        self.advantages = advantages
        self.returns = returns
    
    def get_batches(self, batch_size: int) -> List[Dict[str, np.ndarray]]:
        """Generate mini-batches for training"""
        n = len(self.states)
        indices = np.random.permutation(n)
        
        batches = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_indices = indices[start:end]
            
            batch = {
                'states': np.array([self.states[i] for i in batch_indices]),
                'actions': [self.actions[i] for i in batch_indices],
                'old_log_probs': np.array([self.action_log_probs[i] for i in batch_indices]),
                'advantages': self.advantages[batch_indices],
                'returns': self.returns[batch_indices],
                'old_values': np.array([self.values[i] for i in batch_indices])
            }
            batches.append(batch)
        
        return batches


class NumpyOptimizer:
    """
    Adam optimizer implemented in pure NumPy.
    
    Supports momentum, adaptive learning rates, and weight decay.
    """
    
    def __init__(
        self,
        parameters: List[Tuple[np.ndarray, np.ndarray]],
        lr: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Initialize moment estimates
        self.m = [np.zeros_like(p) for p, _ in parameters]
        self.v = [np.zeros_like(p) for p, _ in parameters]
        self.t = 0
    
    def step(self):
        """Perform one optimization step"""
        self.t += 1
        
        for i, (param, grad) in enumerate(self.parameters):
            if grad is None:
                continue
            
            # Weight decay
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param
            
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
    
    def zero_grad(self):
        """Zero all gradients"""
        for param, grad in self.parameters:
            grad.fill(0)
    
    def set_lr(self, lr: float):
        """Update learning rate"""
        self.lr = lr


class LearningRateScheduler:
    """Learning rate scheduler for training"""
    
    def __init__(self, config: PPOConfig, total_steps: int):
        self.config = config
        self.total_steps = total_steps
        self.current_step = 0
        self.initial_lr = config.learning_rate
    
    def get_lr(self) -> float:
        """Get current learning rate"""
        if self.config.lr_schedule == 'constant':
            return self.initial_lr
        
        # Warmup
        if self.current_step < self.config.warmup_steps:
            warmup_factor = self.current_step / self.config.warmup_steps
            return self.initial_lr * warmup_factor
        
        # Main schedule
        progress = (self.current_step - self.config.warmup_steps) / max(1, self.total_steps - self.config.warmup_steps)
        progress = min(1.0, progress)
        
        if self.config.lr_schedule == 'linear':
            return self.initial_lr * (1.0 - progress)
        elif self.config.lr_schedule == 'cosine':
            return self.initial_lr * (1.0 + math.cos(math.pi * progress)) / 2
        
        return self.initial_lr
    
    def step(self):
        """Advance scheduler by one step"""
        self.current_step += 1


class RunningMeanStd:
    """
    Running mean and standard deviation for normalization.
    
    Uses Welford's online algorithm for numerical stability.
    """
    
    def __init__(self, shape: Tuple[int, ...] = (), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x: np.ndarray):
        """Update statistics with new data"""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        self.update_from_moments(batch_mean, batch_var, batch_count)
    
    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int):
        """Update from batch statistics"""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = m2 / total_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = total_count
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize data using running statistics"""
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


class PPOTrainer:
    """
    PPO Trainer for training the actor-critic network.
    
    Handles:
    - Rollout collection
    - Advantage estimation
    - Policy and value updates
    - Logging and monitoring
    """
    
    def __init__(
        self,
        actor_critic,  # ActorCriticNetwork
        get_valid_actions_fn: Callable,  # Get valid actions for a state
        config: Optional[PPOConfig] = None
    ):
        self.actor_critic = actor_critic
        self.get_valid_actions_fn = get_valid_actions_fn
        self.config = config or PPOConfig()
        
        # Initialize optimizer
        self.optimizer = NumpyOptimizer(
            parameters=actor_critic.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        # Rollout buffer
        self.buffer = RolloutBuffer()
        
        # Normalization
        self.return_normalizer = RunningMeanStd()
        self.advantage_normalizer = RunningMeanStd()
        
        # Training state
        self.total_steps = 0
        self.update_count = 0
        self.entropy_coef = self.config.entropy_coef
        
        # Logging
        self.training_stats = deque(maxlen=100)
    
    def collect_rollout(self, env, n_steps: int) -> None:
        """
        Collect rollout data by interacting with environment.
        
        Args:
            env: Game environment with step() and reset() methods
            n_steps: Number of steps to collect
        """
        self.buffer.clear()
        state = env.state
        
        for _ in range(n_steps):
            # Get state encoding
            state_encoding = self.actor_critic.encoder.forward(state)
            
            # Get valid actions
            valid_actions = self.get_valid_actions_fn(state)
            
            if not valid_actions:
                break
            
            # Get action probabilities and value
            action_type_probs, target_probs, value = self.actor_critic.forward(state)
            
            # Sample action
            action, log_prob = self._sample_action(valid_actions, action_type_probs, target_probs)
            
            # Execute action
            result = env.execute_action(action)
            reward = self._compute_reward(result, state)
            
            # Check if done
            winner, _ = env.state.check_victory_conditions()
            done = winner is not None
            
            # Store transition
            self.buffer.add(
                state=state_encoding,
                action=action,
                log_prob=log_prob,
                reward=reward,
                value=value,
                done=done
            )
            
            self.total_steps += 1
            
            if done:
                state = env.reset()
            else:
                state = env.state
        
        # Compute last value for bootstrapping
        last_value = self.actor_critic.get_value(state)
        
        # Compute advantages and returns
        self.buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda
        )
        
        # Normalize advantages
        if self.config.normalize_advantages and len(self.buffer) > 1:
            advantages = self.buffer.advantages
            mean = np.mean(advantages)
            std = np.std(advantages) + 1e-8
            self.buffer.advantages = (advantages - mean) / std
        
        # Normalize returns
        if self.config.normalize_returns:
            self.return_normalizer.update(self.buffer.returns.reshape(-1, 1))
            self.buffer.returns = self.return_normalizer.normalize(
                self.buffer.returns.reshape(-1, 1)
            ).flatten()
    
    def _sample_action(self, valid_actions: List, action_type_probs: np.ndarray,
                       target_probs: np.ndarray) -> Tuple[Any, float]:
        """Sample action from policy distribution"""
        # Build action probabilities
        action_probs = []
        
        for action in valid_actions:
            type_idx = action.action_type.value
            target = getattr(action, 'target_node_id', None) or getattr(action, 'target_node', None)
            target_idx = target if target is not None else 0
            
            # Combine type and target probabilities
            type_prob = action_type_probs[type_idx] if type_idx < len(action_type_probs) else 0.01
            tgt_prob = target_probs[target_idx] if target_idx < len(target_probs) else 0.01
            
            prob = type_prob * tgt_prob
            action_probs.append(max(prob, 1e-8))
        
        # Normalize
        action_probs = np.array(action_probs)
        action_probs = action_probs / action_probs.sum()
        
        # Sample
        action_idx = np.random.choice(len(valid_actions), p=action_probs)
        action = valid_actions[action_idx]
        log_prob = np.log(action_probs[action_idx] + 1e-8)
        
        return action, float(log_prob)
    
    def _compute_reward(self, result, state) -> float:
        """Compute reward from action result"""
        reward = 0.0
        
        # Points gained
        reward += getattr(result, 'points_gained', 0) * 0.1
        
        # Success bonus
        if result.success:
            reward += 0.5
        else:
            reward -= 0.1
        
        # Damage dealt
        reward += getattr(result, 'damage_dealt', 0) * 0.2
        
        # Check for game end
        winner, _ = state.check_victory_conditions()
        if winner is not None:
            if winner == state.current_player:
                reward += 10.0  # Win bonus
            else:
                reward -= 10.0  # Loss penalty
        
        return reward
    
    def update(self) -> Dict[str, float]:
        """
        Perform PPO update using collected rollout data.
        
        Returns dictionary of training statistics.
        """
        if len(self.buffer) == 0:
            return {}
        
        stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'kl_divergence': [],
            'clip_fraction': []
        }
        
        # Multiple epochs over data
        for epoch in range(self.config.n_epochs):
            # Get mini-batches
            batches = self.buffer.get_batches(self.config.batch_size)
            
            for batch in batches:
                # Compute new action probabilities and values
                batch_stats = self._update_batch(batch)
                
                for key in stats:
                    if key in batch_stats:
                        stats[key].append(batch_stats[key])
        
        # Aggregate statistics
        final_stats = {k: np.mean(v) for k, v in stats.items() if v}
        final_stats['n_updates'] = self.update_count
        final_stats['total_steps'] = self.total_steps
        
        # Decay entropy coefficient
        self.entropy_coef = max(
            self.config.min_entropy_coef,
            self.entropy_coef * self.config.entropy_decay
        )
        
        self.update_count += 1
        self.training_stats.append(final_stats)
        
        return final_stats
    
    def _update_batch(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Update network on a single batch"""
        states = batch['states']
        actions = batch['actions']
        old_log_probs = batch['old_log_probs']
        advantages = batch['advantages']
        returns = batch['returns']
        old_values = batch['old_values']
        
        # This is a simplified update that demonstrates the PPO logic
        # In practice, you'd need proper backpropagation through the network
        
        # Compute new log probs and values
        new_log_probs = []
        new_values = []
        entropies = []
        
        for i, (state, action) in enumerate(zip(states, actions)):
            # Forward pass through actor-critic
            # Note: This is simplified - real implementation needs full forward/backward
            action_type_probs, target_probs, value = self._forward_state(state)
            
            # Compute log prob for taken action
            log_prob = self._compute_action_log_prob(action, action_type_probs, target_probs)
            new_log_probs.append(log_prob)
            new_values.append(value)
            
            # Compute entropy
            entropy = -np.sum(action_type_probs * np.log(action_type_probs + 1e-8))
            entropies.append(entropy)
        
        new_log_probs = np.array(new_log_probs)
        new_values = np.array(new_values)
        entropies = np.array(entropies)
        
        # Policy loss (clipped surrogate objective)
        ratio = np.exp(new_log_probs - old_log_probs)
        clipped_ratio = np.clip(ratio, 1.0 - self.config.clip_epsilon, 
                                1.0 + self.config.clip_epsilon)
        
        policy_loss = -np.minimum(ratio * advantages, clipped_ratio * advantages).mean()
        
        # Value loss
        if self.config.value_clip:
            value_pred_clipped = old_values + np.clip(
                new_values - old_values, 
                -self.config.clip_epsilon, 
                self.config.clip_epsilon
            )
            value_losses = np.maximum(
                (new_values - returns) ** 2,
                (value_pred_clipped - returns) ** 2
            )
            value_loss = 0.5 * value_losses.mean()
        else:
            value_loss = 0.5 * ((new_values - returns) ** 2).mean()
        
        # Entropy bonus
        entropy_loss = -entropies.mean()
        
        # Total loss
        total_loss = (
            policy_loss + 
            self.config.value_loss_coef * value_loss + 
            self.entropy_coef * entropy_loss
        )
        
        # Compute KL divergence (for monitoring)
        kl = (old_log_probs - new_log_probs).mean()
        
        # Clip fraction (for monitoring)
        clip_fraction = (np.abs(ratio - 1.0) > self.config.clip_epsilon).mean()
        
        # In a full implementation, we'd compute gradients and update here
        # For the NumPy implementation, this would involve backward passes
        # through each layer and updating parameters via the optimizer
        
        # Simplified gradient update (pseudo-code style)
        # self.optimizer.zero_grad()
        # backward(total_loss)
        # self.optimizer.step()
        
        return {
            'policy_loss': float(policy_loss),
            'value_loss': float(value_loss),
            'entropy': float(entropies.mean()),
            'kl_divergence': float(kl),
            'clip_fraction': float(clip_fraction),
            'total_loss': float(total_loss)
        }
    
    def _forward_state(self, state_encoding: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Forward pass for a single state encoding"""
        # This is simplified - real implementation uses the full network
        # Just return uniform distributions and zero value for demonstration
        n_action_types = 12
        n_targets = 50
        
        action_type_probs = np.ones(n_action_types) / n_action_types
        target_probs = np.ones(n_targets) / n_targets
        value = 0.0
        
        return action_type_probs, target_probs, value
    
    def _compute_action_log_prob(self, action, action_type_probs: np.ndarray,
                                 target_probs: np.ndarray) -> float:
        """Compute log probability of an action"""
        type_idx = action.action_type.value
        target = getattr(action, 'target_node_id', None) or getattr(action, 'target_node', None)
        target_idx = target if target is not None else 0
        
        type_prob = action_type_probs[type_idx] if type_idx < len(action_type_probs) else 0.01
        tgt_prob = target_probs[target_idx] if target_idx < len(target_probs) else 0.01
        
        return np.log(type_prob + 1e-8) + np.log(tgt_prob + 1e-8)
    
    def save_checkpoint(self, filepath: str):
        """Save training checkpoint"""
        checkpoint = {
            'total_steps': self.total_steps,
            'update_count': self.update_count,
            'entropy_coef': self.entropy_coef,
            'optimizer_state': {
                't': self.optimizer.t,
                'm': self.optimizer.m,
                'v': self.optimizer.v
            }
        }
        np.savez(filepath, **checkpoint)
        
        # Save network weights separately
        self.actor_critic.save(filepath + '_network')
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint"""
        data = np.load(filepath + '.npz', allow_pickle=True)
        
        self.total_steps = int(data['total_steps'])
        self.update_count = int(data['update_count'])
        self.entropy_coef = float(data['entropy_coef'])
        
        optimizer_state = data['optimizer_state'].item()
        self.optimizer.t = optimizer_state['t']
        self.optimizer.m = optimizer_state['m']
        self.optimizer.v = optimizer_state['v']
        
        # Load network weights
        self.actor_critic.load(filepath + '_network.npz')
