# =============================================================================
# Core Game Module
# =============================================================================
"""
Core game engine components including:
- Game state representation
- Network topology
- Actions and rules
- Victory conditions
"""

from .enums import (
    NodeType, NodeStatus, AccessLevel, PlayerRole,
    ActionType, GamePhase, VictoryCondition, Difficulty
)
from .data_structures import (
    Vulnerability, NetworkNode, NetworkEdge,
    PlayerState, GameAction, GameConfig, ActionResult
)
from .game_state import GameState
from .network_topology import NetworkTopologyGenerator, generate_topology
from .actions import ActionValidator, ActionExecutor
from .game_engine import GameEngine, GameEvent, GameEventType, create_game, play_random_game

__all__ = [
    # Enums
    "NodeType", "NodeStatus", "AccessLevel", "PlayerRole",
    "ActionType", "GamePhase", "VictoryCondition", "Difficulty",
    # Data structures
    "Vulnerability", "NetworkNode", "NetworkEdge",
    "PlayerState", "GameAction", "GameConfig", "ActionResult",
    # Core classes
    "GameState", "NetworkTopologyGenerator", "generate_topology",
    "ActionValidator", "ActionExecutor",
    "GameEngine", "GameEvent", "GameEventType",
    # Convenience functions
    "create_game", "play_random_game",
]
