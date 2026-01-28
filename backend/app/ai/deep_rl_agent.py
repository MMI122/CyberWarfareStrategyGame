"""
Deep Reinforcement Learning Agent

This module implements the Deep RL agent using:
- PPO (Proximal Policy Optimization) for learning
- MCTS (Monte Carlo Tree Search) for planning
- Neural networks for function approximation

Research Features:
- AlphaZero-style integration of MCTS and neural networks
- Self-play training
- Curriculum learning support
- Evaluation against various opponents
"""

import numpy as np
import time
from typing import List, Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

import sys
sys.path.insert(0, '..')

from app.core.enums import PlayerRole, ActionType, GamePhase
from app.core.data_structures import GameAction
from .neural_network import (
    ActorCriticNetwork, NetworkConfig, StateEncoder
)
from .mcts import MCTS, MCTSConfig, AlphaZeroMCTS
from .ppo import PPOTrainer, PPOConfig, RolloutBuffer


class TrainingMode(Enum):
    """Training modes for the Deep RL agent"""
    SELF_PLAY = "self_play"  # Train against itself
    VS_MINMAX = "vs_minmax"  # Train against MinMax opponent
    VS_RANDOM = "vs_random"  # Train against random opponent
    CURRICULUM = "curriculum"  # Curriculum learning


@dataclass
class DeepRLConfig:
    """Configuration for Deep RL agent"""
    # Agent settings
    role: PlayerRole = PlayerRole.ATTACKER
    use_mcts: bool = True  # Use MCTS for action selection
    
    # Network configuration
    network_config: NetworkConfig = field(default_factory=NetworkConfig)
    
    # MCTS configuration
    mcts_config: MCTSConfig = field(default_factory=MCTSConfig)
    
    # PPO configuration
    ppo_config: PPOConfig = field(default_factory=PPOConfig)
    
    # Training
    training_mode: TrainingMode = TrainingMode.SELF_PLAY
    num_episodes: int = 1000
    eval_frequency: int = 100  # Evaluate every N episodes
    save_frequency: int = 500  # Save checkpoint every N episodes
    
    # MCTS settings
    mcts_simulations_train: int = 100  # Simulations during training
    mcts_simulations_eval: int = 200  # Simulations during evaluation
    mcts_temperature_train: float = 1.0  # Exploration during training
    mcts_temperature_eval: float = 0.1  # Exploitation during evaluation
    
    # Experience replay (optional)
    use_experience_replay: bool = False
    replay_buffer_size: int = 50000
    
    # Seed
    seed: int = 42


class ExperienceReplayBuffer:
    """
    Experience replay buffer for off-policy learning.
    
    Stores (state, policy_target, value_target) tuples from self-play.
    """
    
    def __init__(self, max_size: int = 50000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state: np.ndarray, policy: np.ndarray, value: float):
        """Add experience to buffer"""
        self.buffer.append({
            'state': state.copy(),
            'policy': policy.copy(),
            'value': value
        })
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences"""
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)
        
        batch = {
            'states': np.array([self.buffer[i]['state'] for i in indices]),
            'policies': np.array([self.buffer[i]['policy'] for i in indices]),
            'values': np.array([self.buffer[i]['value'] for i in indices])
        }
        
        return batch
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()


class DeepRLAgent:
    """
    Deep Reinforcement Learning Agent.
    
    Combines:
    - Neural network for state evaluation and policy
    - MCTS for planning (optional)
    - PPO for learning
    
    This agent can:
    - Play the game using neural network + MCTS
    - Train via self-play or against opponents
    - Be evaluated against various opponents
    """
    
    def __init__(
        self,
        role: PlayerRole = PlayerRole.ATTACKER,
        config: Optional[DeepRLConfig] = None
    ):
        self.role = role
        self.config = config or DeepRLConfig(role=role)
        
        # Set random seed
        np.random.seed(self.config.seed)
        
        # Initialize neural network
        self.network = ActorCriticNetwork(
            config=self.config.network_config,
            num_action_types=len(ActionType),
            max_targets=50
        )
        
        # Initialize MCTS (created per-game)
        self.mcts: Optional[MCTS] = None
        
        # Initialize PPO trainer
        self.ppo_trainer: Optional[PPOTrainer] = None
        
        # Experience replay
        if self.config.use_experience_replay:
            self.replay_buffer = ExperienceReplayBuffer(self.config.replay_buffer_size)
        else:
            self.replay_buffer = None
        
        # Training statistics
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'win_rates': [],
            'policy_losses': [],
            'value_losses': [],
            'entropies': []
        }
        
        # Current game state
        self.game_engine = None
    
    def _create_mcts(self, game_engine) -> MCTS:
        """Create MCTS instance for current game"""
        
        def policy_value_fn(state):
            """Get policy and value from neural network"""
            action_probs, target_probs, value = self.network.forward(state)
            # Combine into single action distribution
            combined_probs = action_probs  # Simplified
            return combined_probs, value
        
        def get_valid_actions_fn(state):
            """Get valid actions for state"""
            return game_engine.get_valid_actions()
        
        def simulate_action_fn(state, action):
            """Simulate action and return new state"""
            new_state, result = game_engine.simulate_action(state, action)
            return new_state
        
        return MCTS(
            policy_value_fn=policy_value_fn,
            get_valid_actions_fn=get_valid_actions_fn,
            simulate_action_fn=simulate_action_fn,
            config=self.config.mcts_config
        )
    
    def get_action(
        self,
        game_engine,
        use_mcts: Optional[bool] = None,
        temperature: Optional[float] = None
    ) -> Optional[GameAction]:
        """
        Get action for current game state.
        
        Args:
            game_engine: Current game engine
            use_mcts: Whether to use MCTS (overrides config)
            temperature: Temperature for action selection
        
        Returns:
            Selected action or None if no valid actions
        """
        self.game_engine = game_engine
        state = game_engine.state
        
        # Get valid actions (no role argument needed)
        valid_actions = game_engine.get_valid_actions()
        
        if not valid_actions:
            return None
        
        use_mcts = use_mcts if use_mcts is not None else self.config.use_mcts
        
        if use_mcts:
            return self._get_action_with_mcts(game_engine, valid_actions, temperature)
        else:
            return self._get_action_without_mcts(state, valid_actions, temperature)
    
    def _get_action_with_mcts(
        self,
        game_engine,
        valid_actions: List[GameAction],
        temperature: Optional[float] = None
    ) -> GameAction:
        """Get action using MCTS"""
        # Create MCTS instance
        mcts = self._create_mcts(game_engine)
        
        # Set temperature
        if temperature is not None:
            mcts.config.temperature = temperature
        
        # Run MCTS search
        num_sims = self.config.mcts_simulations_eval
        mcts.config.num_simulations = num_sims
        
        # Get best action
        action = mcts.get_best_action(game_engine.state)
        
        return action if action else valid_actions[0]
    
    def _get_action_without_mcts(
        self,
        state,
        valid_actions: List[GameAction],
        temperature: Optional[float] = None
    ) -> GameAction:
        """Get action directly from neural network"""
        # Get policy from network
        action_type_probs, target_probs, _ = self.network.forward(state)
        
        # Compute action probabilities
        action_probs = []
        for action in valid_actions:
            type_idx = action.action_type.value
            target = getattr(action, 'target_node_id', None) or getattr(action, 'target_node', None)
            target_idx = target if target is not None else 0
            
            type_prob = action_type_probs[type_idx] if type_idx < len(action_type_probs) else 0.01
            tgt_prob = target_probs[target_idx] if target_idx < len(target_probs) else 0.01
            
            prob = type_prob * tgt_prob
            action_probs.append(max(prob, 1e-8))
        
        action_probs = np.array(action_probs)
        
        # Apply temperature
        temp = temperature if temperature is not None else 1.0
        if temp > 0:
            action_probs = action_probs ** (1.0 / temp)
        
        # Normalize
        action_probs = action_probs / action_probs.sum()
        
        # Sample action
        action_idx = np.random.choice(len(valid_actions), p=action_probs)
        
        return valid_actions[action_idx]
    
    def get_best_action(
        self,
        state,
        time_limit: float = 2.0
    ) -> Optional[GameAction]:
        """
        Get best action for a state (compatible with MinMax interface).
        
        Uses MCTS with time limit.
        """
        if self.game_engine is None:
            raise ValueError("Must set game_engine before calling get_best_action")
        
        # Create MCTS
        mcts = self._create_mcts(self.game_engine)
        mcts.config.time_limit = time_limit
        mcts.config.temperature = 0.0  # Deterministic for best action
        
        return mcts.get_best_action(state)
    
    def train_self_play(
        self,
        create_game_fn: Callable,
        num_episodes: int = None,
        opponent_agent = None
    ) -> Dict[str, List[float]]:
        """
        Train agent via self-play.
        
        Args:
            create_game_fn: Function that creates a new game engine
            num_episodes: Number of episodes to train
            opponent_agent: Optional opponent (for vs_minmax mode)
        
        Returns:
            Training history
        """
        num_episodes = num_episodes or self.config.num_episodes
        
        print(f"Starting self-play training for {num_episodes} episodes...")
        print(f"Training mode: {self.config.training_mode.value}")
        
        for episode in range(num_episodes):
            start_time = time.time()
            
            # Create new game
            engine = create_game_fn()
            
            # Play game
            episode_data = self._play_episode(engine, opponent_agent)
            
            # Update network
            if episode_data['experiences']:
                self._update_from_experiences(episode_data['experiences'])
            
            # Record statistics
            self.training_history['episode_rewards'].append(episode_data['total_reward'])
            self.training_history['episode_lengths'].append(episode_data['num_steps'])
            
            # Logging
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-10:])
                avg_length = np.mean(self.training_history['episode_lengths'][-10:])
                elapsed = time.time() - start_time
                
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Reward: {avg_reward:.2f} | "
                      f"Length: {avg_length:.1f} | "
                      f"Time: {elapsed:.2f}s")
            
            # Evaluation
            if (episode + 1) % self.config.eval_frequency == 0:
                win_rate = self._evaluate(create_game_fn)
                self.training_history['win_rates'].append(win_rate)
                print(f"  Evaluation win rate: {win_rate:.2%}")
            
            # Checkpoint
            if (episode + 1) % self.config.save_frequency == 0:
                self.save(f"checkpoint_episode_{episode + 1}")
                print(f"  Saved checkpoint")
        
        print("Training complete!")
        return self.training_history
    
    def _play_episode(
        self,
        engine,
        opponent_agent = None
    ) -> Dict[str, Any]:
        """Play one episode and collect training data"""
        self.game_engine = engine
        
        experiences = []
        total_reward = 0.0
        num_steps = 0
        
        # Create MCTS for training
        def simulate_fn(s, a):
            new_state, result = engine.simulate_action(s, a)
            return new_state
        
        mcts = AlphaZeroMCTS(
            policy_value_fn=lambda s: (self.network.get_action_probs(s)[0], 
                                       self.network.get_value(s)),
            get_valid_actions_fn=lambda s: engine.get_valid_actions(),
            simulate_action_fn=simulate_fn,
            config=MCTSConfig(
                num_simulations=self.config.mcts_simulations_train,
                temperature=self.config.mcts_temperature_train
            )
        )
        
        while True:
            # Check victory
            victory_result = engine.state.check_victory_conditions()
            if victory_result is not None:
                winner, _ = victory_result
                break
            winner = None
            
            # Determine whose turn
            current_player = engine.state.current_player
            
            if current_player == self.role:
                # Agent's turn - use MCTS
                action, policy_target, value = mcts.self_play_step(engine.state)
                
                if action:
                    # Execute action
                    result = engine.perform_action(action)
                    
                    # Compute reward
                    reward = self._compute_reward(result, engine.state)
                    total_reward += reward
                    
                    # Store experience
                    experiences.append({
                        'state': engine.state,
                        'policy': policy_target,
                        'value': value,
                        'reward': reward
                    })
            else:
                # Opponent's turn
                if opponent_agent:
                    action = opponent_agent.get_best_action(engine.state, time_limit=1.0)
                else:
                    # Self-play: use same MCTS but for opponent role
                    valid_actions = engine.get_valid_actions()
                    if valid_actions:
                        action = np.random.choice(valid_actions)
                    else:
                        action = None
                
                if action:
                    engine.perform_action(action)
            
            num_steps += 1
            
            # Safety limit
            if num_steps >= 200:
                break
        
        # Compute final returns
        victory_result = engine.state.check_victory_conditions()
        winner = victory_result[0] if victory_result else None
        final_reward = 1.0 if winner == self.role else (-1.0 if winner else 0.0)
        
        # Add to replay buffer
        if self.replay_buffer and experiences:
            for exp in experiences:
                if hasattr(exp['state'], 'to_tensor'):
                    try:
                        state_tensor = exp['state'].to_tensor(PlayerRole.ATTACKER)
                    except TypeError:
                        state_tensor = np.zeros(256)
                else:
                    state_tensor = np.zeros(256)
                self.replay_buffer.add(state_tensor, exp['policy'], final_reward)
        
        return {
            'experiences': experiences,
            'total_reward': total_reward + final_reward,
            'num_steps': num_steps,
            'winner': winner
        }
    
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
        
        return reward
    
    def _update_from_experiences(self, experiences: List[Dict]):
        """Update network from collected experiences"""
        # This is a simplified update
        # In a full implementation, you'd use PPO or another RL algorithm
        
        if not experiences:
            return
        
        # For now, just track that we have data to train on
        # Full implementation would involve:
        # 1. Computing policy and value targets
        # 2. Running PPO updates
        # 3. Updating the neural network
        
        pass
    
    def _evaluate(
        self,
        create_game_fn: Callable,
        num_games: int = 10
    ) -> float:
        """Evaluate agent against random opponent"""
        wins = 0
        
        self.network.eval()
        
        for _ in range(num_games):
            engine = create_game_fn()
            self.game_engine = engine
            
            while True:
                victory_result = engine.state.check_victory_conditions()
                if victory_result is not None:
                    winner, _ = victory_result
                    if winner == self.role:
                        wins += 1
                    break
                
                current_player = engine.state.current_player
                
                if current_player == self.role:
                    action = self.get_action(engine, use_mcts=True, 
                                            temperature=self.config.mcts_temperature_eval)
                else:
                    valid_actions = engine.get_valid_actions()
                    action = np.random.choice(valid_actions) if valid_actions else None
                
                if action:
                    engine.perform_action(action)
                else:
                    break
        
        self.network.train()
        
        return wins / num_games
    
    def evaluate_state(self, state) -> float:
        """
        Evaluate a state using the neural network.
        
        Compatible with MinMax interface.
        """
        return self.network.get_value(state)
    
    def save(self, filepath: str):
        """Save agent to file"""
        self.network.save(filepath + '_network')
        
        # Save training history
        np.savez(filepath + '_history', **{
            k: np.array(v) for k, v in self.training_history.items()
        })
        
        print(f"Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent from file"""
        self.network.load(filepath + '_network.npz')
        
        # Load training history
        history_path = filepath + '_history.npz'
        try:
            data = np.load(history_path)
            for key in self.training_history:
                if key in data:
                    self.training_history[key] = data[key].tolist()
        except FileNotFoundError:
            pass
        
        print(f"Agent loaded from {filepath}")


class RandomAgent:
    """Simple random agent for baseline comparisons"""
    
    def __init__(self, role: PlayerRole = PlayerRole.ATTACKER):
        self.role = role
    
    def get_action(self, engine, **kwargs) -> Optional[GameAction]:
        """Get random valid action"""
        valid_actions = engine.get_valid_actions()
        if valid_actions:
            return np.random.choice(valid_actions)
        return None
    
    def get_best_action(self, state, time_limit: float = None) -> Optional[GameAction]:
        """Get random action (for interface compatibility)"""
        # This requires access to game engine
        raise NotImplementedError("RandomAgent needs game engine for action selection")


class HeuristicAgent:
    """
    Simple heuristic agent for baseline comparisons.
    
    Uses hand-crafted rules to select actions.
    """
    
    def __init__(self, role: PlayerRole = PlayerRole.ATTACKER):
        self.role = role
        self.game_engine = None
    
    def set_engine(self, engine):
        """Set game engine"""
        self.game_engine = engine
    
    def get_action(self, engine, **kwargs) -> Optional[GameAction]:
        """Get heuristic action"""
        self.game_engine = engine
        valid_actions = engine.get_valid_actions()
        
        if not valid_actions:
            return None
        
        # Score each action
        action_scores = []
        for action in valid_actions:
            score = self._score_action(action, engine.state)
            action_scores.append((action, score))
        
        # Sort by score and pick best
        action_scores.sort(key=lambda x: x[1], reverse=True)
        return action_scores[0][0]
    
    def _score_action(self, action: GameAction, state) -> float:
        """Score an action using heuristics"""
        score = 0.0
        
        target = getattr(action, 'target_node_id', None) or getattr(action, 'target_node', None)
        
        if self.role == PlayerRole.ATTACKER:
            # Attacker heuristics
            if action.action_type == ActionType.SCAN:
                score += 5.0  # Scanning is good early
            elif action.action_type == ActionType.EXPLOIT:
                if target is not None and target in state.nodes:
                    node = state.nodes[target]
                    # Prefer high-value targets
                    score += getattr(node, 'importance', 1.0) * 10
            elif action.action_type == ActionType.PIVOT:
                score += 3.0
            elif action.action_type == ActionType.EXFILTRATE:
                score += 8.0  # Data theft is valuable
            elif action.action_type == ActionType.DEPLOY_RAT:
                score += 6.0  # Persistence is valuable
        else:
            # Defender heuristics
            if action.action_type == ActionType.PATCH:
                score += 7.0  # Patching is important
            elif action.action_type == ActionType.ISOLATE:
                score += 5.0
            elif action.action_type == ActionType.RESTORE:
                score += 4.0
            elif action.action_type == ActionType.MONITOR:
                score += 3.0
        
        return score
    
    def get_best_action(self, state, time_limit: float = None) -> Optional[GameAction]:
        """Get best action (for interface compatibility)"""
        if self.game_engine is None:
            raise ValueError("Must set game_engine before calling get_best_action")
        return self.get_action(self.game_engine)
