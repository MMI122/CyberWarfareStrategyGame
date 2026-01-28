# =============================================================================
# Cyber Warfare Strategy Game - Game Engine
# =============================================================================
"""
Main game engine that orchestrates gameplay.
Manages turns, validates actions, updates state, and checks victory conditions.
"""

import time
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto

from .enums import (
    PlayerRole, GamePhase, ActionType, NodeStatus,
    VictoryCondition, Difficulty
)
from .data_structures import (
    GameAction, ActionResult, GameConfig, PlayerState, NetworkNode, NetworkEdge
)
from .game_state import GameState
from .network_topology import NetworkTopologyGenerator, generate_topology
from .actions import ActionValidator, ActionExecutor


class GameEventType(Enum):
    """Types of events that can be emitted by the game engine"""
    GAME_STARTED = auto()
    TURN_STARTED = auto()
    ACTION_PERFORMED = auto()
    ACTION_FAILED = auto()
    TURN_ENDED = auto()
    PHASE_CHANGED = auto()
    VICTORY = auto()
    ALERT_TRIGGERED = auto()
    NODE_COMPROMISED = auto()
    NODE_RESTORED = auto()


@dataclass
class GameEvent:
    """Represents a game event for logging and UI updates"""
    event_type: GameEventType
    turn: int
    player_role: Optional[PlayerRole] = None
    action: Optional[GameAction] = None
    result: Optional[ActionResult] = None
    message: str = ""
    data: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class GameEngine:
    """
    Main game engine class.
    
    Responsibilities:
    - Initialize and manage game state
    - Validate and execute player actions
    - Manage turn flow
    - Check victory conditions
    - Emit events for UI/logging
    - Support AI agent integration
    
    Example usage:
        engine = GameEngine()
        engine.initialize_game(difficulty=Difficulty.MEDIUM)
        
        while not engine.is_game_over():
            actions = engine.get_valid_actions()
            action = select_action(actions)  # Player or AI selects
            result = engine.perform_action(action)
    """
    
    def __init__(self, config: Optional[GameConfig] = None):
        """Initialize the game engine with optional custom configuration"""
        self.config = config or GameConfig()
        self.state: Optional[GameState] = None
        self.validator: Optional[ActionValidator] = None
        self.executor: Optional[ActionExecutor] = None
        
        # Event system
        self.event_listeners: Dict[GameEventType, List[Callable]] = {}
        self.event_history: List[GameEvent] = []
        
        # Game statistics
        self.stats = {
            "total_actions": 0,
            "attacker_actions": 0,
            "defender_actions": 0,
            "successful_exploits": 0,
            "failed_exploits": 0,
            "nodes_compromised": 0,
            "nodes_restored": 0,
            "data_exfiltrated": 0,
        }
    
    # =========================================================================
    # Game Initialization
    # =========================================================================
    
    def initialize_game(
        self,
        difficulty: Difficulty = Difficulty.MEDIUM,
        topology_type: str = "corporate",
        seed: Optional[int] = None,
        attacker_starts: bool = True
    ) -> GameState:
        """
        Initialize a new game.
        
        Args:
            difficulty: Game difficulty level
            topology_type: Type of network topology
            seed: Random seed for reproducibility
            attacker_starts: Whether attacker goes first
            
        Returns:
            The initialized game state
        """
        # Update config with difficulty settings
        self.config.difficulty = difficulty
        self.config.max_turns = difficulty.max_turns
        self.config.attacker_ap = 5 + (difficulty.value - 1)
        self.config.defender_ap = 5 + (4 - difficulty.value)
        
        # Generate network topology
        nodes, edges = generate_topology(
            topology_type=topology_type,
            difficulty=difficulty,
            seed=seed
        )
        
        # Create player states
        attacker = PlayerState(
            role=PlayerRole.ATTACKER,
            action_points=self.config.attacker_ap,
            max_action_points=self.config.attacker_ap,
        )
        
        defender = PlayerState(
            role=PlayerRole.DEFENDER,
            action_points=self.config.defender_ap,
            max_action_points=self.config.defender_ap,
        )
        
        # Create game state
        self.state = GameState(
            nodes=nodes,
            edges=edges,
            attacker=attacker,
            defender=defender,
            config=self.config,
            current_player=PlayerRole.ATTACKER if attacker_starts else PlayerRole.DEFENDER,
            phase=GamePhase.PLAYING,
        )
        
        # Initialize validator and executor
        self.validator = ActionValidator(self.config)
        self.executor = ActionExecutor(self.config)
        
        # Reset stats
        self._reset_stats()
        
        # Emit game started event
        self._emit_event(GameEvent(
            event_type=GameEventType.GAME_STARTED,
            turn=0,
            message=f"Game started! Difficulty: {difficulty.name}, Topology: {topology_type}",
            data={
                "node_count": len(nodes),
                "difficulty": difficulty.name,
                "topology": topology_type,
            }
        ))
        
        # Emit first turn start
        self._emit_event(GameEvent(
            event_type=GameEventType.TURN_STARTED,
            turn=self.state.turn_number,
            player_role=self.state.current_player,
            message=f"Turn {self.state.turn_number}: {self.state.current_player.name}'s turn",
        ))
        
        return self.state
    
    def load_state(self, state: GameState):
        """Load an existing game state"""
        self.state = state
        self.config = state.config
        self.validator = ActionValidator(self.config)
        self.executor = ActionExecutor(self.config)
    
    # =========================================================================
    # Action Management
    # =========================================================================
    
    def get_valid_actions(self) -> List[GameAction]:
        """Get all valid actions for the current player"""
        if self.state is None or self.validator is None:
            return []
        
        if self.state.phase != GamePhase.PLAYING:
            return []
        
        current_player = self.get_current_player()
        return self.validator.get_valid_actions(
            player=current_player,
            nodes=self.state.nodes,
            edges=self.state.edges,
            game_phase=self.state.phase
        )
    
    def validate_action(self, action: GameAction) -> Tuple[bool, str]:
        """Validate if an action can be performed"""
        if self.state is None or self.validator is None:
            return False, "Game not initialized"
        
        if action.player_role != self.state.current_player:
            return False, "Not your turn"
        
        current_player = self.get_current_player()
        return self.validator.validate(
            action=action,
            player=current_player,
            nodes=self.state.nodes,
            edges=self.state.edges,
            game_phase=self.state.phase
        )
    
    def perform_action(self, action: GameAction) -> ActionResult:
        """
        Perform an action in the game.
        
        Args:
            action: The action to perform
            
        Returns:
            ActionResult describing what happened
        """
        if self.state is None or self.executor is None:
            return ActionResult(
                action=action,
                success=False,
                message="Game not initialized"
            )
        
        # Validate action
        is_valid, error = self.validate_action(action)
        if not is_valid:
            self._emit_event(GameEvent(
                event_type=GameEventType.ACTION_FAILED,
                turn=self.state.turn_number,
                player_role=action.player_role,
                action=action,
                message=f"Action failed: {error}"
            ))
            return ActionResult(
                action=action,
                success=False,
                message=error
            )
        
        # Execute action
        current_player = self.get_current_player()
        result = self.executor.execute(
            action=action,
            player=current_player,
            nodes=self.state.nodes,
            edges=self.state.edges
        )
        
        # Update statistics
        self._update_stats(action, result)
        
        # Emit action event
        self._emit_event(GameEvent(
            event_type=GameEventType.ACTION_PERFORMED,
            turn=self.state.turn_number,
            player_role=action.player_role,
            action=action,
            result=result,
            message=result.message
        ))
        
        # Handle end turn
        if action.action_type == ActionType.END_TURN or current_player.action_points <= 0:
            self._end_turn()
        
        # Check victory conditions
        victory_result = self.state.check_victory_conditions()
        if victory_result:
            winner, victory_condition = victory_result
            self.state.phase = GamePhase.GAME_OVER
            self.state.victory_condition = victory_condition
            self.state.winner = winner
            
            self._emit_event(GameEvent(
                event_type=GameEventType.VICTORY,
                turn=self.state.turn_number,
                message=f"Game Over! {winner.name} wins by {victory_condition.name}!",
                data={"winner": winner.name, "condition": victory_condition.name}
            ))
        
        return result
    
    # =========================================================================
    # Turn Management
    # =========================================================================
    
    def _end_turn(self):
        """End the current turn and switch to the other player"""
        if self.state is None:
            return
        
        current_role = self.state.current_player
        
        # Emit turn ended event
        self._emit_event(GameEvent(
            event_type=GameEventType.TURN_ENDED,
            turn=self.state.turn_number,
            player_role=current_role,
            message=f"Turn ended for {current_role.name}"
        ))
        
        # Switch players
        if current_role == PlayerRole.ATTACKER:
            self.state.current_player = PlayerRole.DEFENDER
        else:
            self.state.current_player = PlayerRole.ATTACKER
            # Increment turn number when both players have gone
            self.state.turn_number += 1
        
        # Refresh action points
        next_player = self.get_current_player()
        next_player.action_points = next_player.max_action_points
        
        # Check turn limit
        if self.state.turn_number > self.config.max_turns:
            self.state.phase = GamePhase.GAME_OVER
            self.state.victory_condition = VictoryCondition.SURVIVED_TIME_LIMIT
            self.state.winner = PlayerRole.DEFENDER
            return
        
        # Emit turn started event
        self._emit_event(GameEvent(
            event_type=GameEventType.TURN_STARTED,
            turn=self.state.turn_number,
            player_role=self.state.current_player,
            message=f"Turn {self.state.turn_number}: {self.state.current_player.name}'s turn"
        ))
        
        # Process passive effects (malware damage, backdoor maintenance, etc.)
        self._process_passive_effects()
    
    def _process_passive_effects(self):
        """Process passive effects at the start of each turn"""
        if self.state is None:
            return
        
        # Infected nodes take damage each turn
        for node in self.state.nodes.values():
            if node.status == NodeStatus.INFECTED:
                node.health -= 5
                if node.health <= 0:
                    node.health = 0
                    node.status = NodeStatus.OFFLINE
        
        # Alert levels decay slightly
        for node in self.state.nodes.values():
            if node.alert_level > 0:
                node.alert_level = max(0, node.alert_level - 1)
    
    # =========================================================================
    # State Queries
    # =========================================================================
    
    def get_current_player(self) -> PlayerState:
        """Get the current player's state"""
        if self.state is None:
            raise RuntimeError("Game not initialized")
        
        if self.state.current_player == PlayerRole.ATTACKER:
            return self.state.attacker
        return self.state.defender
    
    def get_opponent(self) -> PlayerState:
        """Get the opponent player's state"""
        if self.state is None:
            raise RuntimeError("Game not initialized")
        
        if self.state.current_player == PlayerRole.ATTACKER:
            return self.state.defender
        return self.state.attacker
    
    def is_game_over(self) -> bool:
        """Check if the game is over"""
        return self.state is not None and self.state.phase == GamePhase.GAME_OVER
    
    def get_winner(self) -> Optional[PlayerRole]:
        """Get the winner if game is over"""
        if self.state is None or self.state.phase != GamePhase.GAME_OVER:
            return None
        return self.state.winner
    
    def get_scores(self) -> Dict[str, int]:
        """Get current scores for both players"""
        if self.state is None:
            return {"attacker": 0, "defender": 0}
        return {
            "attacker": self.state.attacker.score,
            "defender": self.state.defender.score,
        }
    
    # =========================================================================
    # AI Integration
    # =========================================================================
    
    def get_state_copy(self) -> GameState:
        """Get a copy of the current game state (for AI simulation)"""
        if self.state is None:
            raise RuntimeError("Game not initialized")
        return self.state.clone()
    
    def simulate_action(self, state: GameState, action: GameAction) -> Tuple[GameState, ActionResult]:
        """
        Simulate an action on a game state copy.
        Does not modify the actual game state.
        
        Args:
            state: Game state copy to simulate on
            action: Action to simulate
            
        Returns:
            Tuple of (new_state, action_result)
        """
        # Create a temporary executor for simulation
        executor = ActionExecutor(state.config)
        
        # Get the player
        player = state.attacker if action.player_role == PlayerRole.ATTACKER else state.defender
        
        # Validate (quick check)
        validator = ActionValidator(state.config)
        is_valid, error = validator.validate(
            action, player, state.nodes, state.edges, state.phase
        )
        
        if not is_valid:
            return state, ActionResult(
                action=action,
                success=False,
                message=error
            )
        
        # Clone state and execute
        new_state = state.clone()
        new_player = new_state.attacker if action.player_role == PlayerRole.ATTACKER else new_state.defender
        
        result = executor.execute(
            action=action,
            player=new_player,
            nodes=new_state.nodes,
            edges=new_state.edges
        )
        
        # Check if turn ends
        if action.action_type == ActionType.END_TURN or new_player.action_points <= 0:
            # Switch player in simulated state
            if new_state.current_player == PlayerRole.ATTACKER:
                new_state.current_player = PlayerRole.DEFENDER
            else:
                new_state.current_player = PlayerRole.ATTACKER
                new_state.turn_number += 1
            
            # Refresh AP
            next_player = new_state.attacker if new_state.current_player == PlayerRole.ATTACKER else new_state.defender
            next_player.action_points = next_player.max_action_points
        
        return new_state, result
    
    def evaluate_state(self, state: GameState, player_role: PlayerRole) -> float:
        """
        Evaluate the game state from a player's perspective.
        Used for AI heuristic evaluation.
        
        Args:
            state: Game state to evaluate
            player_role: Role to evaluate from
            
        Returns:
            Evaluation score (positive = good for player)
        """
        # Get player and opponent
        if player_role == PlayerRole.ATTACKER:
            player = state.attacker
            opponent = state.defender
        else:
            player = state.defender
            opponent = state.attacker
        
        # Base score from player scores
        score = player.score - opponent.score * 0.5
        
        if player_role == PlayerRole.ATTACKER:
            # Attacker wants: more compromised nodes, data stolen, backdoors
            score += len(player.controlled_nodes) * 10
            score += player.data_stolen * 5
            score += len(player.backdoors_installed) * 15
            
            # Critical nodes are high value
            for node_id in player.controlled_nodes:
                node = state.nodes.get(node_id)
                if node and node.node_type.name == "CRITICAL":
                    score += 50
            
            # Penalty for detection
            total_alerts = sum(n.alert_level for n in state.nodes.values())
            score -= total_alerts * 2
            
        else:  # Defender
            # Defender wants: online nodes, low alert levels, clean network
            online_count = sum(1 for n in state.nodes.values() if n.status == NodeStatus.ONLINE)
            score += online_count * 5
            
            # Penalty for compromised nodes
            compromised = sum(1 for n in state.nodes.values() if n.status == NodeStatus.COMPROMISED)
            score -= compromised * 20
            
            # Critical nodes still safe
            for node in state.nodes.values():
                if node.node_type.name == "CRITICAL" and node.status == NodeStatus.ONLINE:
                    score += 30
            
            # Bonus for patches
            patched_vulns = sum(
                1 for n in state.nodes.values() 
                for v in n.vulnerabilities if v.is_patched
            )
            score += patched_vulns * 5
        
        return score
    
    # =========================================================================
    # Event System
    # =========================================================================
    
    def add_event_listener(self, event_type: GameEventType, callback: Callable[[GameEvent], None]):
        """Register a callback for a specific event type"""
        if event_type not in self.event_listeners:
            self.event_listeners[event_type] = []
        self.event_listeners[event_type].append(callback)
    
    def remove_event_listener(self, event_type: GameEventType, callback: Callable):
        """Remove a registered callback"""
        if event_type in self.event_listeners:
            self.event_listeners[event_type] = [
                cb for cb in self.event_listeners[event_type] if cb != callback
            ]
    
    def _emit_event(self, event: GameEvent):
        """Emit an event to all registered listeners"""
        self.event_history.append(event)
        
        listeners = self.event_listeners.get(event.event_type, [])
        for callback in listeners:
            try:
                callback(event)
            except Exception as e:
                print(f"Event listener error: {e}")
    
    def get_event_history(self, 
                          event_type: Optional[GameEventType] = None,
                          limit: Optional[int] = None) -> List[GameEvent]:
        """Get event history, optionally filtered by type"""
        events = self.event_history
        
        if event_type is not None:
            events = [e for e in events if e.event_type == event_type]
        
        if limit is not None:
            events = events[-limit:]
        
        return events
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def _reset_stats(self):
        """Reset game statistics"""
        self.stats = {
            "total_actions": 0,
            "attacker_actions": 0,
            "defender_actions": 0,
            "successful_exploits": 0,
            "failed_exploits": 0,
            "nodes_compromised": 0,
            "nodes_restored": 0,
            "data_exfiltrated": 0,
        }
    
    def _update_stats(self, action: GameAction, result: ActionResult):
        """Update statistics after an action"""
        self.stats["total_actions"] += 1
        
        if action.player_role == PlayerRole.ATTACKER:
            self.stats["attacker_actions"] += 1
        else:
            self.stats["defender_actions"] += 1
        
        if action.action_type == ActionType.EXPLOIT:
            if result.success:
                self.stats["successful_exploits"] += 1
                self.stats["nodes_compromised"] += 1
            else:
                self.stats["failed_exploits"] += 1
        
        if action.action_type == ActionType.RESTORE and result.success:
            self.stats["nodes_restored"] += 1
        
        if action.action_type == ActionType.EXFILTRATE and result.success:
            data = result.state_changes.get("data_stolen", 0)
            self.stats["data_exfiltrated"] += data
    
    def get_stats(self) -> dict:
        """Get current game statistics"""
        return dict(self.stats)
    
    # =========================================================================
    # Victory Conditions
    # =========================================================================
    
    def _determine_winner(self, condition: VictoryCondition) -> PlayerRole:
        """Determine the winner based on victory condition"""
        attacker_wins = [
            VictoryCondition.ALL_CRITICAL_COMPROMISED,
            VictoryCondition.DATA_EXFILTRATED,
            VictoryCondition.PERSISTENT_ACCESS,
            VictoryCondition.NETWORK_DESTROYED,
        ]
        
        if condition in attacker_wins:
            return PlayerRole.ATTACKER
        return PlayerRole.DEFENDER
    
    # =========================================================================
    # Serialization
    # =========================================================================
    
    def to_dict(self) -> dict:
        """Serialize game engine state to dictionary"""
        return {
            "config": self.config.to_dict(),
            "state": self.state.to_dict() if self.state else None,
            "stats": self.stats,
            "event_count": len(self.event_history),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "GameEngine":
        """Create game engine from dictionary"""
        config = GameConfig.from_dict(data["config"])
        engine = cls(config)
        
        if data["state"]:
            engine.state = GameState.from_dict(data["state"])
            engine.validator = ActionValidator(config)
            engine.executor = ActionExecutor(config)
        
        engine.stats = data.get("stats", {})
        
        return engine


# =============================================================================
# Convenience Functions
# =============================================================================

def create_game(
    difficulty: Difficulty = Difficulty.MEDIUM,
    topology: str = "corporate",
    seed: Optional[int] = None
) -> GameEngine:
    """
    Create and initialize a new game.
    
    Args:
        difficulty: Game difficulty
        topology: Network topology type
        seed: Random seed
        
    Returns:
        Initialized GameEngine
    """
    engine = GameEngine()
    engine.initialize_game(
        difficulty=difficulty,
        topology_type=topology,
        seed=seed
    )
    return engine


def play_random_game(
    difficulty: Difficulty = Difficulty.MEDIUM,
    max_turns: int = 100
) -> Dict:
    """
    Play a random game (for testing/demonstration).
    
    Returns:
        Game results dictionary
    """
    import random
    
    engine = create_game(difficulty=difficulty)
    
    turn_count = 0
    while not engine.is_game_over() and turn_count < max_turns:
        actions = engine.get_valid_actions()
        if not actions:
            break
        
        action = random.choice(actions)
        engine.perform_action(action)
        turn_count += 1
    
    return {
        "winner": engine.get_winner().name if engine.get_winner() else "DRAW",
        "turns": engine.state.turn_number if engine.state else 0,
        "scores": engine.get_scores(),
        "stats": engine.get_stats(),
    }
