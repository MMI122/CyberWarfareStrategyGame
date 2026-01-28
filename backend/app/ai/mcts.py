"""
Monte Carlo Tree Search (MCTS) Implementation

This module implements MCTS with neural network guidance for the Deep RL agent.
MCTS is used for planning and action selection during both training and inference.

Research Features:
- UCB1 with neural network prior (PUCT formula from AlphaZero)
- Virtual loss for parallel tree traversal
- Progressive widening for large action spaces
- Transposition table for graph-like game trees
"""

import numpy as np
import math
import time
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict

import sys
sys.path.insert(0, '..')

from app.core.enums import PlayerRole, ActionType, GamePhase
from app.core.data_structures import GameAction


@dataclass
class MCTSConfig:
    """Configuration for MCTS"""
    num_simulations: int = 100  # Number of MCTS simulations per move
    c_puct: float = 1.5  # Exploration constant for PUCT
    dirichlet_alpha: float = 0.3  # Dirichlet noise parameter
    dirichlet_epsilon: float = 0.25  # Dirichlet noise weight
    temperature: float = 1.0  # Temperature for action selection
    virtual_loss: float = 3.0  # Virtual loss for parallelization
    max_depth: int = 50  # Maximum search depth
    use_transposition: bool = True  # Use transposition table
    progressive_widening: bool = True  # Use progressive widening
    pw_alpha: float = 0.5  # Progressive widening alpha
    pw_beta: float = 0.5  # Progressive widening beta
    time_limit: Optional[float] = None  # Time limit in seconds


@dataclass
class MCTSNode:
    """
    Node in the MCTS tree.
    
    Stores visit counts, value estimates, and children.
    """
    state_hash: int  # Hash of game state
    parent: Optional['MCTSNode'] = None
    action_from_parent: Optional[GameAction] = None
    
    # Statistics
    visit_count: int = 0
    total_value: float = 0.0
    virtual_loss_count: int = 0
    
    # Prior probability from neural network
    prior: float = 0.0
    
    # Children
    children: Dict[str, 'MCTSNode'] = field(default_factory=dict)
    
    # Valid actions at this node
    valid_actions: List[GameAction] = field(default_factory=list)
    
    # Whether node is fully expanded
    is_expanded: bool = False
    is_terminal: bool = False
    terminal_value: float = 0.0
    
    @property
    def mean_value(self) -> float:
        """Average value of this node"""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count
    
    @property
    def adjusted_visit_count(self) -> int:
        """Visit count including virtual losses"""
        return self.visit_count + self.virtual_loss_count
    
    def ucb_score(self, c_puct: float, parent_visits: int) -> float:
        """
        Calculate UCB score using PUCT formula (AlphaZero style)
        
        Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        """
        if parent_visits == 0:
            return float('inf')
        
        exploitation = self.mean_value
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.adjusted_visit_count)
        
        return exploitation + exploration
    
    def __hash__(self):
        return self.state_hash
    
    def __eq__(self, other):
        return self.state_hash == other.state_hash


class TranspositionTableMCTS:
    """
    Transposition table for MCTS to handle graph-like game trees.
    
    Allows reusing subtrees when states can be reached via different paths.
    """
    
    def __init__(self, max_size: int = 100000):
        self.table: Dict[int, MCTSNode] = {}
        self.max_size = max_size
        self.access_order: List[int] = []
    
    def get(self, state_hash: int) -> Optional[MCTSNode]:
        """Get node by state hash"""
        return self.table.get(state_hash)
    
    def put(self, state_hash: int, node: MCTSNode):
        """Store node in table"""
        if len(self.table) >= self.max_size:
            self._evict()
        
        self.table[state_hash] = node
        self.access_order.append(state_hash)
    
    def _evict(self):
        """Remove oldest entries"""
        evict_count = len(self.table) // 4
        for _ in range(evict_count):
            if self.access_order:
                old_hash = self.access_order.pop(0)
                self.table.pop(old_hash, None)
    
    def clear(self):
        """Clear the table"""
        self.table.clear()
        self.access_order.clear()


class MCTS:
    """
    Monte Carlo Tree Search with Neural Network guidance.
    
    Implements PUCT formula from AlphaZero for combining MCTS with neural networks.
    
    Algorithm:
    1. SELECT: Traverse tree using PUCT to select promising nodes
    2. EXPAND: Expand leaf node and evaluate with neural network
    3. BACKUP: Propagate values back up the tree
    """
    
    def __init__(
        self,
        policy_value_fn,  # Function that returns (action_probs, value) for a state
        get_valid_actions_fn,  # Function that returns valid actions for a state
        simulate_action_fn,  # Function that simulates an action and returns new state
        config: Optional[MCTSConfig] = None
    ):
        self.policy_value_fn = policy_value_fn
        self.get_valid_actions_fn = get_valid_actions_fn
        self.simulate_action_fn = simulate_action_fn
        self.config = config or MCTSConfig()
        
        self.root: Optional[MCTSNode] = None
        
        if self.config.use_transposition:
            self.transposition_table = TranspositionTableMCTS()
        else:
            self.transposition_table = None
        
        # Statistics
        self.stats = {
            'simulations': 0,
            'max_depth_reached': 0,
            'nodes_created': 0,
            'transposition_hits': 0
        }
    
    def compute_state_hash(self, state) -> int:
        """Compute hash of game state for transposition table"""
        # Use state's tensor encoding for hashing
        if hasattr(state, 'to_tensor'):
            try:
                # Try with perspective argument
                tensor = state.to_tensor(PlayerRole.ATTACKER)
            except TypeError:
                # Try without perspective
                try:
                    tensor = state.to_tensor()
                except:
                    tensor = None
            
            if tensor is not None:
                # Flatten and convert to hashable
                flat = tensor.flatten()[:100]  # Take first 100 elements
                return hash(tuple(float(x) for x in flat))
        
        # Fallback: hash based on key state attributes
        hash_components = [
            getattr(state, 'turn_number', 0) or getattr(state, 'current_turn', 0),
            getattr(state, 'phase', type('', (), {'value': 0})).value,
        ]
        
        # Get attacker controlled nodes count
        attacker = getattr(state, 'attacker', None) or getattr(state, 'attacker_state', None)
        if attacker:
            controlled = getattr(attacker, 'controlled_nodes', set()) or set()
            hash_components.append(len(controlled))
            hash_components.append(getattr(attacker, 'score', 0))
        
        defender = getattr(state, 'defender', None) or getattr(state, 'defender_state', None)
        if defender:
            hash_components.append(getattr(defender, 'score', 0))
        
        # Add node status
        nodes = getattr(state, 'nodes', {})
        for node_id, node in nodes.items():
            hash_components.append(node.status.value)
            hash_components.append(int(getattr(node, 'is_compromised', False) or 
                                       getattr(node, 'compromised', False)))
        
        return hash(tuple(hash_components))
    
    def get_action_key(self, action: GameAction) -> str:
        """Get unique key for an action"""
        target = getattr(action, 'target_node_id', None) or getattr(action, 'target_node', None)
        return f"{action.action_type.name}_{target}"
    
    def create_node(self, state, parent: Optional[MCTSNode] = None, 
                    action_from_parent: Optional[GameAction] = None,
                    prior: float = 0.0) -> MCTSNode:
        """Create a new MCTS node"""
        state_hash = self.compute_state_hash(state)
        
        # Check transposition table
        if self.transposition_table:
            existing = self.transposition_table.get(state_hash)
            if existing:
                self.stats['transposition_hits'] += 1
                return existing
        
        node = MCTSNode(
            state_hash=state_hash,
            parent=parent,
            action_from_parent=action_from_parent,
            prior=prior
        )
        
        # Check if terminal
        victory_result = state.check_victory_conditions() if hasattr(state, 'check_victory_conditions') else None
        if victory_result is not None:
            winner, _ = victory_result
            node.is_terminal = True
            # Set terminal value from perspective of current player
            current = getattr(state, 'current_player', PlayerRole.ATTACKER)
            node.terminal_value = 1.0 if winner == current else -1.0
        
        # Store in transposition table
        if self.transposition_table:
            self.transposition_table.put(state_hash, node)
        
        self.stats['nodes_created'] += 1
        return node
    
    def expand_node(self, node: MCTSNode, state) -> float:
        """
        Expand a leaf node and return value estimate.
        
        Uses neural network to get action priors and state value.
        """
        if node.is_terminal:
            return node.terminal_value
        
        # Get valid actions
        valid_actions = self.get_valid_actions_fn(state)
        node.valid_actions = valid_actions
        
        if not valid_actions:
            # No valid actions - terminal state
            node.is_terminal = True
            node.terminal_value = 0.0  # Draw
            return 0.0
        
        # Get policy and value from neural network
        action_probs, value = self.policy_value_fn(state)
        
        # Create action-to-prior mapping
        action_priors = self._compute_action_priors(valid_actions, action_probs)
        
        # Add Dirichlet noise at root for exploration
        if node.parent is None:
            noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(valid_actions))
            epsilon = self.config.dirichlet_epsilon
            for i, action in enumerate(valid_actions):
                key = self.get_action_key(action)
                action_priors[key] = (1 - epsilon) * action_priors[key] + epsilon * noise[i]
        
        # Store priors for children (created lazily)
        node._action_priors = action_priors
        node.is_expanded = True
        
        return value
    
    def _compute_action_priors(self, valid_actions: List[GameAction], 
                               action_probs: np.ndarray) -> Dict[str, float]:
        """Map neural network output to action priors"""
        priors = {}
        
        # Normalize priors over valid actions
        total = 0.0
        for action in valid_actions:
            key = self.get_action_key(action)
            # Use action type index as proxy
            type_idx = action.action_type.value
            if type_idx < len(action_probs):
                prior = float(action_probs[type_idx])
            else:
                prior = 1.0 / len(valid_actions)  # Uniform prior
            priors[key] = prior
            total += prior
        
        # Normalize
        if total > 0:
            for key in priors:
                priors[key] /= total
        
        return priors
    
    def select_child(self, node: MCTSNode, state) -> Tuple[MCTSNode, GameAction, Any]:
        """
        Select best child using PUCT formula.
        
        Returns child node, action, and resulting state.
        """
        if not node.is_expanded:
            raise ValueError("Cannot select child from unexpanded node")
        
        # Progressive widening: limit children based on visit count
        if self.config.progressive_widening:
            max_children = int(
                self.config.pw_alpha * (node.visit_count ** self.config.pw_beta)
            )
            max_children = max(1, min(max_children, len(node.valid_actions)))
        else:
            max_children = len(node.valid_actions)
        
        # Get candidate actions
        candidate_actions = node.valid_actions[:max_children]
        
        best_score = float('-inf')
        best_action = None
        best_child = None
        
        for action in candidate_actions:
            key = self.get_action_key(action)
            
            # Get or create child
            if key in node.children:
                child = node.children[key]
            else:
                # Create child lazily
                new_state = self.simulate_action_fn(state, action)
                prior = node._action_priors.get(key, 1.0 / len(candidate_actions))
                child = self.create_node(new_state, parent=node, 
                                        action_from_parent=action, prior=prior)
                node.children[key] = child
            
            # Calculate UCB score
            score = child.ucb_score(self.config.c_puct, node.visit_count)
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        # Simulate action to get new state
        new_state = self.simulate_action_fn(state, best_action)
        
        return best_child, best_action, new_state
    
    def simulate(self, root_state) -> None:
        """
        Run one MCTS simulation from root.
        
        SELECT -> EXPAND -> BACKUP
        """
        node = self.root
        state = root_state
        path = [node]
        depth = 0
        
        # Apply virtual loss
        node.virtual_loss_count += int(self.config.virtual_loss)
        
        # SELECT: Traverse to leaf
        while node.is_expanded and not node.is_terminal and depth < self.config.max_depth:
            node, action, state = self.select_child(node, state)
            path.append(node)
            node.virtual_loss_count += int(self.config.virtual_loss)
            depth += 1
        
        self.stats['max_depth_reached'] = max(self.stats['max_depth_reached'], depth)
        
        # EXPAND and evaluate
        if node.is_terminal:
            value = node.terminal_value
        else:
            value = self.expand_node(node, state)
        
        # BACKUP
        for i, node in enumerate(reversed(path)):
            # Remove virtual loss
            node.virtual_loss_count -= int(self.config.virtual_loss)
            
            # Flip value for alternating players
            if i % 2 == 1:
                value = -value
            
            node.visit_count += 1
            node.total_value += value
        
        self.stats['simulations'] += 1
    
    def search(self, state, num_simulations: Optional[int] = None, 
               time_limit: Optional[float] = None) -> Dict[str, Any]:
        """
        Run MCTS search from given state.
        
        Args:
            state: Current game state
            num_simulations: Number of simulations (overrides config)
            time_limit: Time limit in seconds (overrides config)
        
        Returns:
            Dictionary with search results including action probabilities
        """
        # Reset statistics
        self.stats = {
            'simulations': 0,
            'max_depth_reached': 0,
            'nodes_created': 0,
            'transposition_hits': 0
        }
        
        # Create root node
        self.root = self.create_node(state)
        if not self.root.is_expanded:
            self.expand_node(self.root, state)
        
        # Determine stopping condition
        n_sims = num_simulations or self.config.num_simulations
        t_limit = time_limit or self.config.time_limit
        
        start_time = time.time()
        
        # Run simulations
        for i in range(n_sims):
            if t_limit and (time.time() - start_time) >= t_limit:
                break
            
            self.simulate(state)
        
        elapsed = time.time() - start_time
        
        # Compute action probabilities from visit counts
        action_visits = {}
        total_visits = 0
        
        for key, child in self.root.children.items():
            action_visits[key] = child.visit_count
            total_visits += child.visit_count
        
        # Apply temperature
        if self.config.temperature == 0:
            # Deterministic: pick most visited
            action_probs = {k: 0.0 for k in action_visits}
            if action_visits:
                best_key = max(action_visits, key=action_visits.get)
                action_probs[best_key] = 1.0
        else:
            # Apply temperature
            action_probs = {}
            temp_sum = 0.0
            for key, visits in action_visits.items():
                prob = (visits / max(1, total_visits)) ** (1.0 / self.config.temperature)
                action_probs[key] = prob
                temp_sum += prob
            
            # Normalize
            if temp_sum > 0:
                for key in action_probs:
                    action_probs[key] /= temp_sum
        
        return {
            'action_probs': action_probs,
            'root_value': self.root.mean_value,
            'simulations': self.stats['simulations'],
            'max_depth': self.stats['max_depth_reached'],
            'nodes_created': self.stats['nodes_created'],
            'transposition_hits': self.stats['transposition_hits'],
            'time_seconds': elapsed
        }
    
    def get_best_action(self, state) -> Optional[GameAction]:
        """
        Get best action based on search results.
        
        Returns action with highest visit count.
        """
        results = self.search(state)
        
        if not results['action_probs']:
            return None
        
        # Get action with highest probability
        best_key = max(results['action_probs'], key=results['action_probs'].get)
        
        # Find corresponding action
        for action in self.root.valid_actions:
            if self.get_action_key(action) == best_key:
                return action
        
        return None
    
    def get_action_distribution(self, state) -> List[Tuple[GameAction, float]]:
        """
        Get full action probability distribution.
        
        Returns list of (action, probability) tuples.
        """
        results = self.search(state)
        
        distribution = []
        for action in self.root.valid_actions:
            key = self.get_action_key(action)
            prob = results['action_probs'].get(key, 0.0)
            distribution.append((action, prob))
        
        return distribution


class MCTSParallel(MCTS):
    """
    Parallelized MCTS using virtual loss.
    
    Supports leaf parallelization and root parallelization.
    Note: Requires threading or multiprocessing for true parallelism.
    """
    
    def __init__(self, *args, num_workers: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_workers = num_workers
    
    def search_parallel(self, state, num_simulations: Optional[int] = None) -> Dict[str, Any]:
        """
        Run parallel MCTS search.
        
        For simplicity, this implementation uses sequential simulation
        with virtual loss to allow for future parallelization.
        """
        # For now, just use regular search with virtual loss
        # True parallelization would require threading/multiprocessing
        return self.search(state, num_simulations)


class AlphaZeroMCTS(MCTS):
    """
    MCTS variant optimized for AlphaZero-style training.
    
    Additional features:
    - Training data collection
    - Policy target generation
    - Self-play support
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_examples = []
    
    def self_play_step(self, state) -> Tuple[GameAction, np.ndarray, float]:
        """
        Perform one step of self-play.
        
        Returns:
            action: Selected action
            policy_target: MCTS policy (visit proportions)
            value: Root value estimate
        """
        results = self.search(state)
        
        # Create policy target from MCTS visit counts
        policy_target = np.zeros(len(self.root.valid_actions))
        for i, action in enumerate(self.root.valid_actions):
            key = self.get_action_key(action)
            policy_target[i] = results['action_probs'].get(key, 0.0)
        
        # Sample action according to distribution (during training)
        action_probs = np.array([results['action_probs'].get(self.get_action_key(a), 0.0) 
                                 for a in self.root.valid_actions])
        action_probs = action_probs / (action_probs.sum() + 1e-8)
        
        action_idx = np.random.choice(len(self.root.valid_actions), p=action_probs)
        action = self.root.valid_actions[action_idx]
        
        return action, policy_target, results['root_value']
    
    def collect_training_data(self, state, action: GameAction, 
                             policy_target: np.ndarray, value_target: float):
        """Store training example for later use"""
        if hasattr(state, 'to_tensor'):
            state_tensor = state.to_tensor()
        else:
            state_tensor = np.zeros(256)  # Placeholder
        
        self.training_examples.append({
            'state': state_tensor,
            'policy': policy_target,
            'value': value_target
        })
    
    def get_training_data(self) -> List[Dict[str, np.ndarray]]:
        """Get collected training examples"""
        return self.training_examples
    
    def clear_training_data(self):
        """Clear collected training examples"""
        self.training_examples = []
