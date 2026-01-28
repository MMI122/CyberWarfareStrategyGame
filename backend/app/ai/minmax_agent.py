# =============================================================================
# Cyber Warfare Strategy Game - MinMax AI Agent
# =============================================================================
"""
Hand-coded MinMax algorithm with Alpha-Beta pruning.
This is a pure implementation without using pre-built game AI libraries.

Key Features:
1. Alpha-Beta Pruning - Reduces search space significantly
2. Transposition Tables - Avoid re-evaluating identical states
3. Iterative Deepening - Find best move within time constraints
4. Move Ordering - Examine promising moves first
5. Quiescence Search - Avoid horizon effects
6. Killer Move Heuristic - Improve pruning efficiency
7. Aspiration Windows - Narrow search bounds
"""

import time
import hashlib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict

# Import game components
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.core import (
    GameState, GameEngine, GameAction, ActionResult,
    PlayerRole, ActionType, GamePhase, NodeStatus,
    Difficulty
)


# =============================================================================
# Transposition Table
# =============================================================================

class NodeType(Enum):
    """Type of node in transposition table"""
    EXACT = 1    # Exact evaluation
    LOWER = 2    # Beta cutoff (lower bound)
    UPPER = 3    # Alpha cutoff (upper bound)


@dataclass
class TTEntry:
    """Transposition table entry"""
    hash_key: str
    depth: int
    value: float
    node_type: NodeType
    best_move: Optional[GameAction] = None
    age: int = 0


class TranspositionTable:
    """
    Hash table storing evaluated positions.
    Uses Zobrist-like hashing for fast state lookup.
    """
    
    def __init__(self, max_size: int = 1_000_000):
        """Initialize with maximum number of entries"""
        self.max_size = max_size
        self.table: OrderedDict[str, TTEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.current_age = 0
    
    def compute_hash(self, state: GameState) -> str:
        """
        Compute a hash key for the game state.
        Uses essential state features for fast lookup.
        """
        # Build a string representation of key state features
        key_parts = [
            f"turn:{state.turn_number}",
            f"player:{state.current_player.name}",
            f"a_ap:{state.attacker.action_points}",
            f"d_ap:{state.defender.action_points}",
        ]
        
        # Add node statuses (sorted for consistency)
        for node_id in sorted(state.nodes.keys()):
            node = state.nodes[node_id]
            key_parts.append(f"n{node_id}:{node.status.value}")
        
        # Add controlled nodes
        controlled = sorted(state.attacker.controlled_nodes)
        key_parts.append(f"ctrl:{','.join(map(str, controlled))}")
        
        # Hash the combined string
        combined = "|".join(key_parts)
        return hashlib.md5(combined.encode()).hexdigest()
    
    def lookup(self, state: GameState, depth: int, alpha: float, beta: float
               ) -> Tuple[Optional[float], Optional[GameAction]]:
        """
        Look up a position in the table.
        
        Returns:
            (value, best_move) if found and usable, (None, None) otherwise
        """
        hash_key = self.compute_hash(state)
        
        if hash_key not in self.table:
            self.misses += 1
            return None, None
        
        entry = self.table[hash_key]
        
        # Check if entry is deep enough
        if entry.depth < depth:
            self.misses += 1
            return None, None
        
        self.hits += 1
        
        # Use the entry based on its type
        if entry.node_type == NodeType.EXACT:
            return entry.value, entry.best_move
        elif entry.node_type == NodeType.LOWER and entry.value >= beta:
            return entry.value, entry.best_move
        elif entry.node_type == NodeType.UPPER and entry.value <= alpha:
            return entry.value, entry.best_move
        
        # Entry not usable for pruning, but return best move for ordering
        return None, entry.best_move
    
    def store(self, state: GameState, depth: int, value: float, 
              node_type: NodeType, best_move: Optional[GameAction] = None):
        """Store an evaluation in the table"""
        hash_key = self.compute_hash(state)
        
        # Check if we should replace an existing entry
        if hash_key in self.table:
            existing = self.table[hash_key]
            # Replace if new entry is deeper or same depth
            if depth >= existing.depth:
                self.table[hash_key] = TTEntry(
                    hash_key=hash_key,
                    depth=depth,
                    value=value,
                    node_type=node_type,
                    best_move=best_move,
                    age=self.current_age
                )
            # Move to end (most recently used)
            self.table.move_to_end(hash_key)
        else:
            # Add new entry
            self.table[hash_key] = TTEntry(
                hash_key=hash_key,
                depth=depth,
                value=value,
                node_type=node_type,
                best_move=best_move,
                age=self.current_age
            )
            
            # Evict oldest entries if table is full
            while len(self.table) > self.max_size:
                self.table.popitem(last=False)
    
    def increment_age(self):
        """Increment the age counter for a new search"""
        self.current_age += 1
    
    def get_stats(self) -> dict:
        """Get table statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "size": len(self.table),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
        }
    
    def clear(self):
        """Clear the table"""
        self.table.clear()
        self.hits = 0
        self.misses = 0


# =============================================================================
# Killer Move Heuristic
# =============================================================================

class KillerMoves:
    """
    Stores "killer moves" - moves that caused beta cutoffs at each depth.
    These moves are often good choices in sibling positions.
    """
    
    def __init__(self, max_depth: int = 50, slots_per_depth: int = 2):
        """Initialize killer move storage"""
        self.max_depth = max_depth
        self.slots = slots_per_depth
        # killer_moves[depth] = [move1, move2, ...]
        self.killer_moves: List[List[Optional[GameAction]]] = [
            [None] * slots_per_depth for _ in range(max_depth)
        ]
    
    def store(self, depth: int, move: GameAction):
        """Store a killer move at the given depth"""
        if depth >= self.max_depth:
            return
        
        killers = self.killer_moves[depth]
        
        # Don't store duplicates
        for existing in killers:
            if existing and self._moves_equal(existing, move):
                return
        
        # Shift existing killers and insert new one
        for i in range(len(killers) - 1, 0, -1):
            killers[i] = killers[i - 1]
        killers[0] = move
    
    def get(self, depth: int) -> List[GameAction]:
        """Get killer moves at the given depth"""
        if depth >= self.max_depth:
            return []
        return [m for m in self.killer_moves[depth] if m is not None]
    
    def _moves_equal(self, m1: GameAction, m2: GameAction) -> bool:
        """Check if two moves are equivalent"""
        return (m1.action_type == m2.action_type and 
                m1.target_node == m2.target_node and
                m1.player == m2.player)
    
    def clear(self):
        """Clear all killer moves"""
        for depth in range(self.max_depth):
            self.killer_moves[depth] = [None] * self.slots


# =============================================================================
# History Heuristic
# =============================================================================

class HistoryTable:
    """
    Tracks which moves have historically been good.
    Used for move ordering to improve pruning.
    """
    
    def __init__(self):
        """Initialize history table"""
        # history[action_type][target_node] = score
        self.history: Dict[ActionType, Dict[int, float]] = {}
        for action_type in ActionType:
            self.history[action_type] = {}
    
    def get_score(self, move: GameAction) -> float:
        """Get history score for a move"""
        target = move.target_node if move.target_node is not None else -1
        return self.history[move.action_type].get(target, 0)
    
    def update(self, move: GameAction, depth: int, is_good: bool):
        """Update history score for a move"""
        target = move.target_node if move.target_node is not None else -1
        
        # Depth-weighted update
        delta = depth * depth
        if is_good:
            current = self.history[move.action_type].get(target, 0)
            self.history[move.action_type][target] = current + delta
        else:
            current = self.history[move.action_type].get(target, 0)
            self.history[move.action_type][target] = max(0, current - delta)
    
    def clear(self):
        """Clear all history scores"""
        for action_type in ActionType:
            self.history[action_type].clear()


# =============================================================================
# Move Ordering
# =============================================================================

class MoveOrderer:
    """
    Orders moves to maximize alpha-beta pruning efficiency.
    Better move ordering = more pruning = faster search.
    """
    
    def __init__(self, tt: TranspositionTable, killers: KillerMoves, 
                 history: HistoryTable):
        self.tt = tt
        self.killers = killers
        self.history = history
    
    def order_moves(self, state: GameState, moves: List[GameAction], 
                    depth: int, tt_move: Optional[GameAction] = None
                    ) -> List[GameAction]:
        """
        Order moves from most to least promising.
        
        Order:
        1. Transposition table move (if available)
        2. Winning captures / high-value targets
        3. Killer moves
        4. History heuristic ordering
        5. Remaining moves
        """
        if not moves:
            return moves
        
        # Calculate scores for each move
        scored_moves = []
        for move in moves:
            score = self._score_move(state, move, depth, tt_move)
            scored_moves.append((score, move))
        
        # Sort by score (descending)
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        
        return [move for score, move in scored_moves]
    
    def _score_move(self, state: GameState, move: GameAction, 
                    depth: int, tt_move: Optional[GameAction]) -> float:
        """Calculate priority score for a move"""
        score = 0.0
        
        # TT move gets highest priority
        if tt_move and self._moves_equal(move, tt_move):
            return 10000
        
        # Killer moves get high priority
        killers = self.killers.get(depth)
        for i, killer in enumerate(killers):
            if killer and self._moves_equal(move, killer):
                score += 5000 - i * 100
                break
        
        # History heuristic
        score += self.history.get_score(move)
        
        # Action-specific scoring
        if move.action_type == ActionType.EXPLOIT:
            # Exploits are usually important
            score += 1000
            if move.target_node is not None:
                target = state.nodes.get(move.target_node)
                if target:
                    # Prefer critical targets
                    if target.node_type.name == "CRITICAL":
                        score += 2000
                    # Prefer nodes with high data value
                    score += target.data_value * 10
        
        elif move.action_type == ActionType.EXFILTRATE:
            # Data exfiltration is a win condition
            score += 1500
        
        elif move.action_type == ActionType.INSTALL_BACKDOOR:
            # Backdoors are strategic
            score += 800
        
        elif move.action_type == ActionType.SCAN:
            # Scans help exploration
            score += 200
        
        elif move.action_type == ActionType.END_TURN:
            # End turn is usually last choice
            score -= 1000
        
        return score
    
    def _moves_equal(self, m1: GameAction, m2: GameAction) -> bool:
        """Check if two moves are equivalent"""
        return (m1.action_type == m2.action_type and 
                m1.target_node == m2.target_node)


# =============================================================================
# State Evaluator
# =============================================================================

class StateEvaluator:
    """
    Evaluates game states with sophisticated heuristics.
    Positive values favor the maximizing player (attacker).
    """
    
    # Weights for different evaluation factors
    WEIGHTS = {
        # Attacker factors
        "controlled_nodes": 100,
        "critical_controlled": 500,
        "data_stolen": 50,
        "backdoors": 200,
        "access_level": 50,
        "visibility": 20,
        
        # Defender factors
        "online_nodes": 30,
        "patched_vulns": 40,
        "monitoring": 25,
        "isolated_threats": 150,
        
        # Game state
        "ap_advantage": 10,
        "turn_pressure": 5,
        "detection_penalty": -30,
    }
    
    def evaluate(self, state: GameState, maximizing_player: PlayerRole) -> float:
        """
        Evaluate the state from the perspective of the maximizing player.
        
        Args:
            state: Game state to evaluate
            maximizing_player: The player we're maximizing for
            
        Returns:
            Evaluation score (positive = good for maximizer)
        """
        # Check for terminal states first
        victory_result = state.check_victory_conditions()
        if victory_result:
            winner, condition = victory_result
            if winner == maximizing_player:
                return 100000 - state.turn_number  # Prefer faster wins
            else:
                return -100000 + state.turn_number  # Prefer slower losses
        
        # Calculate component scores
        attacker_score = self._evaluate_attacker(state)
        defender_score = self._evaluate_defender(state)
        positional_score = self._evaluate_position(state)
        
        # Combine scores
        if maximizing_player == PlayerRole.ATTACKER:
            return attacker_score - defender_score + positional_score
        else:
            return defender_score - attacker_score - positional_score
    
    def _evaluate_attacker(self, state: GameState) -> float:
        """Evaluate from attacker's perspective"""
        score = 0.0
        attacker = state.attacker
        
        # Controlled nodes
        score += len(attacker.controlled_nodes) * self.WEIGHTS["controlled_nodes"]
        
        # Critical nodes controlled
        for node_id in attacker.controlled_nodes:
            node = state.nodes.get(node_id)
            if node and node.node_type.name == "CRITICAL":
                score += self.WEIGHTS["critical_controlled"]
        
        # Data stolen
        data_stolen = attacker.data_stolen if hasattr(attacker, 'data_stolen') else attacker.exfiltrated_data
        score += data_stolen * self.WEIGHTS["data_stolen"]
        
        # Backdoors for persistence
        backdoors = len(attacker.backdoors_installed) if hasattr(attacker, 'backdoors_installed') else len(attacker.entry_points)
        score += backdoors * self.WEIGHTS["backdoors"]
        
        # Access levels
        if hasattr(attacker, 'access_levels'):
            for node_id, level in attacker.access_levels.items():
                score += level.value * self.WEIGHTS["access_level"]
        
        # Visibility (more visible nodes = more options)
        visible_count = sum(1 for n in state.nodes.values() if n.visible_to_attacker)
        score += visible_count * self.WEIGHTS["visibility"]
        
        # Detection penalty
        score += attacker.detection_count * self.WEIGHTS["detection_penalty"]
        
        return score
    
    def _evaluate_defender(self, state: GameState) -> float:
        """Evaluate from defender's perspective"""
        score = 0.0
        defender = state.defender
        
        # Online nodes
        online_count = sum(1 for n in state.nodes.values() 
                          if n.status == NodeStatus.ONLINE)
        score += online_count * self.WEIGHTS["online_nodes"]
        
        # Patched vulnerabilities
        patched_count = sum(
            1 for n in state.nodes.values() 
            for v in n.vulnerabilities if v.is_patched
        )
        score += patched_count * self.WEIGHTS["patched_vulns"]
        
        # Monitoring levels
        monitoring_total = sum(n.monitoring_level for n in state.nodes.values())
        score += monitoring_total * self.WEIGHTS["monitoring"]
        
        # Successfully isolated compromised nodes
        isolated_compromised = sum(
            1 for n in state.nodes.values() 
            if n.status == NodeStatus.ISOLATED
        )
        score += isolated_compromised * self.WEIGHTS["isolated_threats"]
        
        return score
    
    def _evaluate_position(self, state: GameState) -> float:
        """Evaluate positional/strategic factors"""
        score = 0.0
        
        # Action point advantage
        ap_diff = state.attacker.action_points - state.defender.action_points
        score += ap_diff * self.WEIGHTS["ap_advantage"]
        
        # Turn pressure (attacker usually wants to win faster)
        turns_remaining = state.max_turns - state.turn_number
        if turns_remaining < 10:
            # Urgency for attacker
            score += (10 - turns_remaining) * self.WEIGHTS["turn_pressure"]
        
        return score
    
    def is_quiet(self, state: GameState) -> bool:
        """
        Check if the position is "quiet" (no immediate tactical threats).
        Used for quiescence search.
        """
        # Check for immediate attack opportunities
        for node in state.nodes.values():
            if node.status == NodeStatus.COMPROMISED:
                # Compromised node under attack - not quiet
                if node.alert_level > 3:
                    return False
        
        # Check for critical nodes under threat
        attacker = state.attacker
        for node_id in attacker.controlled_nodes:
            node = state.nodes.get(node_id)
            if node and node.node_type.name == "CRITICAL":
                return False  # Critical node compromised - tactical
        
        return True


# =============================================================================
# MinMax Agent
# =============================================================================

@dataclass
class SearchStats:
    """Statistics for a search iteration"""
    nodes_searched: int = 0
    nodes_pruned: int = 0
    tt_hits: int = 0
    depth_reached: int = 0
    time_ms: float = 0.0
    best_move: Optional[GameAction] = None
    best_score: float = 0.0


class MinMaxAgent:
    """
    MinMax AI agent with Alpha-Beta pruning.
    
    This is a hand-coded implementation suitable for PhD research,
    demonstrating deep understanding of adversarial search algorithms.
    
    Key Features:
    - Alpha-Beta Pruning: Standard two-player zero-sum game pruning
    - Transposition Tables: Avoid re-evaluating identical positions
    - Iterative Deepening: Anytime algorithm, returns best move found so far
    - Move Ordering: Improve pruning with killer moves, history heuristic
    - Quiescence Search: Avoid horizon effects in tactical positions
    - Aspiration Windows: Narrow search bounds based on previous iteration
    
    Example usage:
        agent = MinMaxAgent(role=PlayerRole.ATTACKER)
        action = agent.get_best_action(game_state, time_limit=5.0)
    """
    
    # Constants
    INF = float('inf')
    NEG_INF = float('-inf')
    
    def __init__(
        self,
        role: PlayerRole,
        max_depth: int = 10,
        tt_size: int = 500_000,
        use_quiescence: bool = True,
        use_aspiration: bool = True,
        aspiration_window: float = 50.0
    ):
        """
        Initialize the MinMax agent.
        
        Args:
            role: Which player this agent controls
            max_depth: Maximum search depth (can be increased with iterative deepening)
            tt_size: Size of transposition table
            use_quiescence: Whether to use quiescence search
            use_aspiration: Whether to use aspiration windows
            aspiration_window: Initial aspiration window size
        """
        self.role = role
        self.max_depth = max_depth
        self.use_quiescence = use_quiescence
        self.use_aspiration = use_aspiration
        self.aspiration_window = aspiration_window
        
        # Initialize components
        self.tt = TranspositionTable(max_size=tt_size)
        self.killers = KillerMoves(max_depth=max_depth + 20)
        self.history = HistoryTable()
        self.move_orderer = MoveOrderer(self.tt, self.killers, self.history)
        self.evaluator = StateEvaluator()
        
        # Search state
        self.nodes_searched = 0
        self.nodes_pruned = 0
        self.start_time = 0.0
        self.time_limit = 0.0
        self.should_stop = False
        
        # Statistics history
        self.search_history: List[SearchStats] = []
    
    def get_best_action(
        self,
        state: GameState,
        time_limit: float = 5.0,
        min_depth: int = 2
    ) -> GameAction:
        """
        Find the best action using iterative deepening MinMax.
        
        Args:
            state: Current game state
            time_limit: Maximum time to search (seconds)
            min_depth: Minimum depth to search before considering time
            
        Returns:
            Best action found
        """
        self.start_time = time.time()
        self.time_limit = time_limit
        self.should_stop = False
        self.search_history.clear()
        
        # Get valid actions
        valid_actions = self._get_valid_actions(state)
        if not valid_actions:
            return None
        if len(valid_actions) == 1:
            return valid_actions[0]
        
        # Initialize with first move
        best_action = valid_actions[0]
        best_score = self.NEG_INF if self._is_maximizing(state) else self.INF
        
        # Iterative deepening
        previous_score = 0.0
        for depth in range(1, self.max_depth + 1):
            self.nodes_searched = 0
            self.nodes_pruned = 0
            self.tt.increment_age()
            
            try:
                # Aspiration window search
                if self.use_aspiration and depth > 1:
                    score, action = self._aspiration_search(
                        state, depth, previous_score
                    )
                else:
                    # Full window search
                    score, action = self._search_root(
                        state, depth, self.NEG_INF, self.INF
                    )
                
                if action is not None:
                    best_action = action
                    best_score = score
                    previous_score = score
                
                # Record statistics
                elapsed = (time.time() - self.start_time) * 1000
                stats = SearchStats(
                    nodes_searched=self.nodes_searched,
                    nodes_pruned=self.nodes_pruned,
                    tt_hits=self.tt.hits,
                    depth_reached=depth,
                    time_ms=elapsed,
                    best_move=best_action,
                    best_score=best_score
                )
                self.search_history.append(stats)
                
                # Check time
                if depth >= min_depth and self._time_exceeded():
                    break
                    
            except TimeoutError:
                break
        
        return best_action
    
    def _aspiration_search(
        self, state: GameState, depth: int, previous_score: float
    ) -> Tuple[float, Optional[GameAction]]:
        """
        Search with aspiration windows.
        Starts with a narrow window around the expected score.
        """
        alpha = previous_score - self.aspiration_window
        beta = previous_score + self.aspiration_window
        
        score, action = self._search_root(state, depth, alpha, beta)
        
        # If score falls outside window, research with full window
        if score <= alpha:
            # Failed low - research with wider window
            score, action = self._search_root(state, depth, self.NEG_INF, score + 1)
        elif score >= beta:
            # Failed high - research with wider window
            score, action = self._search_root(state, depth, score - 1, self.INF)
        
        return score, action
    
    def _search_root(
        self, state: GameState, depth: int, alpha: float, beta: float
    ) -> Tuple[float, Optional[GameAction]]:
        """
        Search from the root position.
        Returns (score, best_action).
        """
        maximizing = self._is_maximizing(state)
        best_action = None
        
        # Get and order moves
        valid_actions = self._get_valid_actions(state)
        tt_value, tt_move = self.tt.lookup(state, depth, alpha, beta)
        ordered_actions = self.move_orderer.order_moves(
            state, valid_actions, depth, tt_move
        )
        
        if maximizing:
            best_score = self.NEG_INF
            for action in ordered_actions:
                # Simulate action
                new_state = self._simulate_action(state, action)
                
                # Recursive search
                score = self._minimax(new_state, depth - 1, alpha, beta, False)
                
                if score > best_score:
                    best_score = score
                    best_action = action
                
                alpha = max(alpha, score)
                if beta <= alpha:
                    self.nodes_pruned += 1
                    self.killers.store(depth, action)
                    self.history.update(action, depth, True)
                    break
                else:
                    self.history.update(action, depth, False)
        else:
            best_score = self.INF
            for action in ordered_actions:
                new_state = self._simulate_action(state, action)
                score = self._minimax(new_state, depth - 1, alpha, beta, True)
                
                if score < best_score:
                    best_score = score
                    best_action = action
                
                beta = min(beta, score)
                if beta <= alpha:
                    self.nodes_pruned += 1
                    self.killers.store(depth, action)
                    self.history.update(action, depth, True)
                    break
                else:
                    self.history.update(action, depth, False)
        
        # Store in transposition table
        node_type = NodeType.EXACT
        if best_score <= alpha:
            node_type = NodeType.UPPER
        elif best_score >= beta:
            node_type = NodeType.LOWER
        self.tt.store(state, depth, best_score, node_type, best_action)
        
        return best_score, best_action
    
    def _minimax(
        self, state: GameState, depth: int, 
        alpha: float, beta: float, maximizing: bool
    ) -> float:
        """
        The core MinMax algorithm with Alpha-Beta pruning.
        
        Args:
            state: Current game state
            depth: Remaining search depth
            alpha: Best score for maximizer found so far
            beta: Best score for minimizer found so far
            maximizing: True if this is a maximizing node
            
        Returns:
            Evaluation score for this position
        """
        self.nodes_searched += 1
        
        # Check time limit periodically
        if self.nodes_searched % 1000 == 0 and self._time_exceeded():
            raise TimeoutError("Search time exceeded")
        
        # Check for terminal state
        victory_result = state.check_victory_conditions()
        if victory_result:
            winner, _ = victory_result
            if winner == self.role:
                return 100000 - (self.max_depth - depth)  # Prefer faster wins
            else:
                return -100000 + (self.max_depth - depth)  # Prefer slower losses
        
        # Depth limit reached
        if depth <= 0:
            if self.use_quiescence and not self.evaluator.is_quiet(state):
                return self._quiescence_search(state, alpha, beta, maximizing, 4)
            return self.evaluator.evaluate(state, self.role)
        
        # Transposition table lookup
        tt_value, tt_move = self.tt.lookup(state, depth, alpha, beta)
        if tt_value is not None:
            return tt_value
        
        # Generate and order moves
        valid_actions = self._get_valid_actions(state)
        if not valid_actions:
            # No valid moves - evaluate current position
            return self.evaluator.evaluate(state, self.role)
        
        ordered_actions = self.move_orderer.order_moves(
            state, valid_actions, depth, tt_move
        )
        
        best_move = None
        
        if maximizing:
            best_score = self.NEG_INF
            
            for action in ordered_actions:
                # Simulate the action
                new_state = self._simulate_action(state, action)
                
                # Recursive minimax call
                score = self._minimax(new_state, depth - 1, alpha, beta, False)
                
                if score > best_score:
                    best_score = score
                    best_move = action
                
                # Alpha-Beta pruning
                alpha = max(alpha, score)
                if beta <= alpha:
                    # Beta cutoff
                    self.nodes_pruned += 1
                    self.killers.store(depth, action)
                    self.history.update(action, depth, True)
                    break
                else:
                    self.history.update(action, depth, False)
        else:
            best_score = self.INF
            
            for action in ordered_actions:
                new_state = self._simulate_action(state, action)
                score = self._minimax(new_state, depth - 1, alpha, beta, True)
                
                if score < best_score:
                    best_score = score
                    best_move = action
                
                # Alpha-Beta pruning
                beta = min(beta, score)
                if beta <= alpha:
                    # Alpha cutoff
                    self.nodes_pruned += 1
                    self.killers.store(depth, action)
                    self.history.update(action, depth, True)
                    break
                else:
                    self.history.update(action, depth, False)
        
        # Store in transposition table
        node_type = NodeType.EXACT
        if best_score <= alpha:
            node_type = NodeType.UPPER
        elif best_score >= beta:
            node_type = NodeType.LOWER
        self.tt.store(state, depth, best_score, node_type, best_move)
        
        return best_score
    
    def _quiescence_search(
        self, state: GameState, alpha: float, beta: float,
        maximizing: bool, depth_remaining: int
    ) -> float:
        """
        Quiescence search to avoid horizon effects.
        Only considers "noisy" moves (captures, high-impact actions).
        """
        self.nodes_searched += 1
        
        # Stand-pat score
        stand_pat = self.evaluator.evaluate(state, self.role)
        
        if depth_remaining <= 0:
            return stand_pat
        
        if maximizing:
            if stand_pat >= beta:
                return beta
            alpha = max(alpha, stand_pat)
        else:
            if stand_pat <= alpha:
                return alpha
            beta = min(beta, stand_pat)
        
        # Get only tactical moves
        tactical_moves = self._get_tactical_moves(state)
        
        if not tactical_moves:
            return stand_pat
        
        if maximizing:
            for action in tactical_moves:
                new_state = self._simulate_action(state, action)
                score = self._quiescence_search(
                    new_state, alpha, beta, False, depth_remaining - 1
                )
                
                if score >= beta:
                    return beta
                alpha = max(alpha, score)
            
            return alpha
        else:
            for action in tactical_moves:
                new_state = self._simulate_action(state, action)
                score = self._quiescence_search(
                    new_state, alpha, beta, True, depth_remaining - 1
                )
                
                if score <= alpha:
                    return alpha
                beta = min(beta, score)
            
            return beta
    
    def _get_tactical_moves(self, state: GameState) -> List[GameAction]:
        """Get only high-impact tactical moves for quiescence search"""
        all_moves = self._get_valid_actions(state)
        tactical = []
        
        for move in all_moves:
            # Include exploits, exfiltration, and isolation
            if move.action_type in [
                ActionType.EXPLOIT, 
                ActionType.EXFILTRATE,
                ActionType.ISOLATE,
                ActionType.DEPLOY_MALWARE
            ]:
                tactical.append(move)
        
        return tactical
    
    def _get_valid_actions(self, state: GameState) -> List[GameAction]:
        """Get valid actions for the current player"""
        # Create a temporary engine for action validation
        from app.core.actions import ActionValidator
        from app.core.data_structures import GameConfig
        
        config = state.config if state.config else GameConfig()
        validator = ActionValidator(config)
        
        current_player = state.attacker if state.current_player == PlayerRole.ATTACKER else state.defender
        
        return validator.get_valid_actions(
            player=current_player,
            nodes=state.nodes,
            edges=state.edges,
            game_phase=state.phase
        )
    
    def _simulate_action(self, state: GameState, action: GameAction) -> GameState:
        """Simulate an action and return the new state"""
        from app.core.actions import ActionExecutor
        from app.core.data_structures import GameConfig
        
        # Clone the state
        new_state = state.clone()
        
        # Get the player
        config = new_state.config if new_state.config else GameConfig()
        executor = ActionExecutor(config)
        
        player = new_state.attacker if action.player == PlayerRole.ATTACKER else new_state.defender
        
        # Execute action
        executor.execute(action, player, new_state.nodes, new_state.edges)
        
        # Handle turn switching if AP depleted or END_TURN
        if action.action_type == ActionType.END_TURN or player.action_points <= 0:
            new_state.switch_player()
        
        return new_state
    
    def _is_maximizing(self, state: GameState) -> bool:
        """Check if current player is the maximizing player"""
        return state.current_player == self.role
    
    def _time_exceeded(self) -> bool:
        """Check if time limit has been exceeded"""
        if self.time_limit <= 0:
            return False
        elapsed = time.time() - self.start_time
        return elapsed >= self.time_limit
    
    def get_search_stats(self) -> Dict:
        """Get statistics from the most recent search"""
        if not self.search_history:
            return {}
        
        latest = self.search_history[-1]
        return {
            "nodes_searched": latest.nodes_searched,
            "nodes_pruned": latest.nodes_pruned,
            "depth_reached": latest.depth_reached,
            "time_ms": latest.time_ms,
            "best_score": latest.best_score,
            "tt_stats": self.tt.get_stats(),
        }
    
    def get_search_history(self) -> List[Dict]:
        """Get statistics from all depth iterations"""
        return [
            {
                "depth": s.depth_reached,
                "nodes": s.nodes_searched,
                "pruned": s.nodes_pruned,
                "time_ms": s.time_ms,
                "score": s.best_score,
            }
            for s in self.search_history
        ]
    
    def clear_tables(self):
        """Clear all search tables (for new game)"""
        self.tt.clear()
        self.killers.clear()
        self.history.clear()


# =============================================================================
# Convenience Functions
# =============================================================================

def create_minmax_agent(
    role: PlayerRole,
    difficulty: Difficulty = Difficulty.MEDIUM
) -> MinMaxAgent:
    """
    Create a MinMax agent with difficulty-appropriate settings.
    
    Args:
        role: Which player the agent controls
        difficulty: Game difficulty (affects search depth/time)
        
    Returns:
        Configured MinMaxAgent instance
    """
    settings = {
        Difficulty.EASY: {"max_depth": 4, "tt_size": 100_000},
        Difficulty.MEDIUM: {"max_depth": 6, "tt_size": 300_000},
        Difficulty.HARD: {"max_depth": 8, "tt_size": 500_000},
        Difficulty.EXPERT: {"max_depth": 12, "tt_size": 1_000_000},
    }
    
    config = settings.get(difficulty, settings[Difficulty.MEDIUM])
    
    return MinMaxAgent(
        role=role,
        max_depth=config["max_depth"],
        tt_size=config["tt_size"]
    )
