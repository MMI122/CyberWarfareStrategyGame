# =============================================================================
# Cyber Warfare Strategy Game - Game State
# =============================================================================
"""
The complete game state representation.
This is the central data structure that captures everything about a game.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from copy import deepcopy
import uuid
import numpy as np

from .enums import (
    NodeType, NodeStatus, AccessLevel, PlayerRole,
    ActionType, GamePhase, VictoryCondition, Difficulty
)
from .data_structures import (
    NetworkNode, NetworkEdge, PlayerState, GameAction, 
    GameConfig, Vulnerability
)


@dataclass
class GameState:
    """
    Complete state of a game instance.
    
    This class encapsulates all information needed to:
    - Display the game
    - Determine legal moves
    - Execute actions
    - Check victory conditions
    - Save/load games
    
    The state is designed to be efficiently cloneable for AI search.
    """
    
    # ==========================================================================
    # Identifiers
    # ==========================================================================
    game_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # ==========================================================================
    # Network Topology
    # ==========================================================================
    nodes: Dict[int, NetworkNode] = field(default_factory=dict)
    edges: List[NetworkEdge] = field(default_factory=list)
    topology_type: str = "corporate"
    
    # ==========================================================================
    # Player States
    # ==========================================================================
    attacker: PlayerState = field(default_factory=lambda: PlayerState(PlayerRole.ATTACKER))
    defender: PlayerState = field(default_factory=lambda: PlayerState(PlayerRole.DEFENDER))
    current_player: PlayerRole = PlayerRole.ATTACKER
    
    # ==========================================================================
    # Game Progress
    # ==========================================================================
    turn_number: int = 0
    phase: GamePhase = GamePhase.SETUP
    actions_this_turn: List[GameAction] = field(default_factory=list)
    
    # ==========================================================================
    # Game History
    # ==========================================================================
    action_history: List[GameAction] = field(default_factory=list)
    
    # ==========================================================================
    # Victory/End State
    # ==========================================================================
    winner: Optional[PlayerRole] = None
    victory_condition: Optional[VictoryCondition] = None
    game_over: bool = False
    
    # ==========================================================================
    # Configuration
    # ==========================================================================
    config: GameConfig = field(default_factory=GameConfig)
    
    # ==========================================================================
    # Properties
    # ==========================================================================
    
    @property
    def max_turns(self) -> int:
        """Maximum number of turns"""
        return self.config.max_turns
    
    @property
    def node_count(self) -> int:
        """Total number of nodes"""
        return len(self.nodes)
    
    @property
    def critical_nodes(self) -> List[NetworkNode]:
        """Get all critical objective nodes"""
        return [n for n in self.nodes.values() if n.node_type == NodeType.CRITICAL]
    
    @property
    def compromised_nodes(self) -> List[NetworkNode]:
        """Get all compromised nodes"""
        return [n for n in self.nodes.values() if n.is_compromised]
    
    @property
    def online_nodes(self) -> List[NetworkNode]:
        """Get all online nodes"""
        return [n for n in self.nodes.values() if n.is_online]
    
    @property
    def offline_nodes(self) -> List[NetworkNode]:
        """Get all offline nodes"""
        return [n for n in self.nodes.values() if n.status == NodeStatus.OFFLINE]
    
    @property
    def compromised_critical_count(self) -> int:
        """Number of critical nodes that are compromised"""
        return sum(1 for n in self.critical_nodes if n.is_compromised)
    
    @property
    def total_critical_count(self) -> int:
        """Total number of critical nodes"""
        return len(self.critical_nodes)
    
    @property
    def network_destruction_ratio(self) -> float:
        """Ratio of offline to total nodes"""
        if self.node_count == 0:
            return 0.0
        return len(self.offline_nodes) / self.node_count
    
    # ==========================================================================
    # Player Access
    # ==========================================================================
    
    def get_current_player_state(self) -> PlayerState:
        """Get the state of the current player"""
        if self.current_player == PlayerRole.ATTACKER:
            return self.attacker
        return self.defender
    
    def get_opponent_state(self) -> PlayerState:
        """Get the state of the opponent"""
        if self.current_player == PlayerRole.ATTACKER:
            return self.defender
        return self.attacker
    
    def get_player_state(self, role: PlayerRole) -> PlayerState:
        """Get state for a specific player role"""
        if role == PlayerRole.ATTACKER:
            return self.attacker
        return self.defender
    
    # ==========================================================================
    # Node Access
    # ==========================================================================
    
    def get_node(self, node_id: int) -> Optional[NetworkNode]:
        """Get a node by ID"""
        return self.nodes.get(node_id)
    
    def get_adjacent_nodes(self, node_id: int) -> List[NetworkNode]:
        """Get all nodes adjacent to a given node"""
        node = self.get_node(node_id)
        if node is None:
            return []
        return [self.nodes[adj_id] for adj_id in node.adjacent_nodes 
                if adj_id in self.nodes]
    
    def get_edge(self, node1_id: int, node2_id: int) -> Optional[NetworkEdge]:
        """Get the edge between two nodes"""
        key = (min(node1_id, node2_id), max(node1_id, node2_id))
        for edge in self.edges:
            if edge.key == key:
                return edge
        return None
    
    def are_adjacent(self, node1_id: int, node2_id: int) -> bool:
        """Check if two nodes are adjacent"""
        node = self.get_node(node1_id)
        if node is None:
            return False
        return node2_id in node.adjacent_nodes
    
    # ==========================================================================
    # Visibility (Fog of War)
    # ==========================================================================
    
    def get_visible_nodes(self, role: PlayerRole) -> List[NetworkNode]:
        """Get nodes visible to a player"""
        if not self.config.fog_of_war:
            return list(self.nodes.values())
        
        if role == PlayerRole.DEFENDER:
            # Defender sees all nodes
            return list(self.nodes.values())
        else:
            # Attacker only sees scanned/compromised nodes and their neighbors
            visible = set()
            for node in self.nodes.values():
                if node.visible_to_attacker or node.scanned_by_attacker or node.is_compromised:
                    visible.add(node.id)
                    # Can also see adjacent nodes (but not their details)
                    for adj_id in node.adjacent_nodes:
                        visible.add(adj_id)
            return [self.nodes[nid] for nid in visible if nid in self.nodes]
    
    def is_node_visible(self, node_id: int, role: PlayerRole) -> bool:
        """Check if a node is visible to a player"""
        if not self.config.fog_of_war:
            return True
        if role == PlayerRole.DEFENDER:
            return True
        node = self.get_node(node_id)
        if node is None:
            return False
        return node.visible_to_attacker or node.scanned_by_attacker or node.is_compromised
    
    # ==========================================================================
    # State Queries for AI
    # ==========================================================================
    
    def get_attacker_reachable_nodes(self) -> List[NetworkNode]:
        """
        Get nodes the attacker can potentially reach/target.
        These are nodes adjacent to compromised nodes.
        """
        reachable = set()
        for node_id in self.attacker.controlled_nodes:
            node = self.get_node(node_id)
            if node and node.is_online:
                for adj_id in node.adjacent_nodes:
                    adj_node = self.get_node(adj_id)
                    if adj_node and adj_node.is_online and not adj_node.is_compromised:
                        reachable.add(adj_id)
        return [self.nodes[nid] for nid in reachable]
    
    def get_exploitable_nodes(self) -> List[NetworkNode]:
        """Get nodes that can be exploited (have unpatched vulnerabilities)"""
        reachable = self.get_attacker_reachable_nodes()
        return [n for n in reachable if n.unpatched_vulnerabilities]
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[NetworkNode]:
        """Get all nodes of a specific type"""
        return [n for n in self.nodes.values() if n.node_type == node_type]
    
    def get_nodes_by_status(self, status: NodeStatus) -> List[NetworkNode]:
        """Get all nodes with a specific status"""
        return [n for n in self.nodes.values() if n.status == status]
    
    # ==========================================================================
    # Victory Condition Checks
    # ==========================================================================
    
    def check_victory_conditions(self) -> Optional[Tuple[PlayerRole, VictoryCondition]]:
        """
        Check all victory conditions and return winner if any are met.
        Returns (winner, condition) or None if game continues.
        """
        # Attacker victory: All critical nodes compromised
        if self.compromised_critical_count == self.total_critical_count > 0:
            return (PlayerRole.ATTACKER, VictoryCondition.ALL_CRITICAL_COMPROMISED)
        
        # Attacker victory: Data exfiltration target reached
        if self.attacker.exfiltrated_data >= self.config.data_exfiltration_target:
            return (PlayerRole.ATTACKER, VictoryCondition.DATA_EXFILTRATED)
        
        # Attacker victory: Persistent access
        if self.attacker.turns_with_access >= self.config.persistent_access_turns:
            return (PlayerRole.ATTACKER, VictoryCondition.PERSISTENT_ACCESS)
        
        # Attacker victory: Network destruction
        if self.network_destruction_ratio >= self.config.network_destruction_threshold:
            return (PlayerRole.ATTACKER, VictoryCondition.NETWORK_DESTROYED)
        
        # Defender victory: Time limit reached
        if self.turn_number >= self.max_turns:
            return (PlayerRole.DEFENDER, VictoryCondition.SURVIVED_TIME_LIMIT)
        
        # Defender victory: Attacker eliminated
        if len(self.attacker.controlled_nodes) == 0 and self.turn_number > 0:
            # Attacker had presence but lost it all
            if len(self.attacker.entry_points) == 0:
                return (PlayerRole.DEFENDER, VictoryCondition.ATTACKER_ELIMINATED)
        
        # Defender victory: Attacker exhausted
        if (self.attacker.action_points == 0 and 
            self.attacker.compute_units == 0 and
            len(self.attacker.controlled_nodes) == 0):
            return (PlayerRole.DEFENDER, VictoryCondition.ATTACKER_EXHAUSTED)
        
        return None
    
    def is_terminal(self) -> bool:
        """Check if the game has ended"""
        return self.game_over or self.check_victory_conditions() is not None
    
    # ==========================================================================
    # State Manipulation
    # ==========================================================================
    
    def switch_player(self):
        """Switch to the other player's turn"""
        self.current_player = self.current_player.opponent
        if self.current_player == PlayerRole.ATTACKER:
            # New round
            self.turn_number += 1
            self.actions_this_turn = []
    
    def start_turn(self):
        """Initialize a new turn"""
        player = self.get_current_player_state()
        player.regenerate_resources(
            ap_regen=self.config.ap_per_turn,
            cu_regen=self.config.cu_per_turn
        )
        
        # Update attacker's persistent access counter
        if self.current_player == PlayerRole.ATTACKER:
            if len(self.attacker.controlled_nodes) > 0:
                self.attacker.turns_with_access += 1
            else:
                self.attacker.turns_with_access = 0
    
    def end_game(self, winner: PlayerRole, condition: VictoryCondition):
        """Mark the game as ended"""
        self.winner = winner
        self.victory_condition = condition
        self.game_over = True
        self.phase = GamePhase.GAME_OVER
        
        # Calculate final scores
        self._calculate_final_scores()
    
    def _calculate_final_scores(self):
        """Calculate final scores for both players"""
        # Attacker score
        self.attacker.score = (
            len(self.attacker.controlled_nodes) * 10 +
            self.attacker.exfiltrated_data * 5 +
            self.compromised_critical_count * 100 -
            self.attacker.detection_count * 20
        )
        
        # Defender score
        protected_criticals = self.total_critical_count - self.compromised_critical_count
        self.defender.score = (
            protected_criticals * 100 +
            self.defender.patches_applied * 20 +
            self.attacker.detection_count * 30 +
            (self.turn_number if self.winner == PlayerRole.DEFENDER else 0) * 2
        )
    
    # ==========================================================================
    # Cloning (for AI Search)
    # ==========================================================================
    
    def clone(self) -> 'GameState':
        """
        Create a deep copy of the game state.
        Optimized for AI search - only copies mutable state.
        """
        new_state = GameState(
            game_id=self.game_id,
            nodes={nid: node.clone() for nid, node in self.nodes.items()},
            edges=[edge.clone() for edge in self.edges],
            topology_type=self.topology_type,
            attacker=self.attacker.clone(),
            defender=self.defender.clone(),
            current_player=self.current_player,
            turn_number=self.turn_number,
            phase=self.phase,
            actions_this_turn=[a.clone() for a in self.actions_this_turn],
            action_history=[],  # Don't copy full history for search
            winner=self.winner,
            victory_condition=self.victory_condition,
            game_over=self.game_over,
            config=self.config  # Config is immutable, no need to copy
        )
        return new_state
    
    # ==========================================================================
    # Serialization
    # ==========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for JSON serialization"""
        return {
            "game_id": self.game_id,
            "nodes": {str(nid): node.to_dict() for nid, node in self.nodes.items()},
            "edges": [edge.to_dict() for edge in self.edges],
            "topology_type": self.topology_type,
            "attacker": self.attacker.to_dict(),
            "defender": self.defender.to_dict(),
            "current_player": self.current_player.name,
            "turn_number": self.turn_number,
            "phase": self.phase.name,
            "actions_this_turn": [a.to_dict() for a in self.actions_this_turn],
            "winner": self.winner.name if self.winner else None,
            "victory_condition": self.victory_condition.name if self.victory_condition else None,
            "game_over": self.game_over,
            "config": self.config.to_dict()
        }
    
    # ==========================================================================
    # State Encoding for Neural Networks
    # ==========================================================================
    
    def to_tensor(self, perspective: PlayerRole, grid_size: int = 16) -> np.ndarray:
        """
        Convert game state to a tensor representation for neural networks.
        
        Args:
            perspective: Which player's view to encode
            grid_size: Size of the grid to embed the topology into
            
        Returns:
            numpy array of shape (num_channels, grid_size, grid_size)
        """
        num_channels = 20
        tensor = np.zeros((num_channels, grid_size, grid_size), dtype=np.float32)
        
        # Map node positions to grid
        for node in self.nodes.values():
            # Normalize position to grid
            x = int((node.position[0] + 1) / 2 * (grid_size - 1))
            y = int((node.position[1] + 1) / 2 * (grid_size - 1))
            x = max(0, min(grid_size - 1, x))
            y = max(0, min(grid_size - 1, y))
            
            # Apply fog of war for attacker perspective
            if perspective == PlayerRole.ATTACKER and self.config.fog_of_war:
                if not self.is_node_visible(node.id, PlayerRole.ATTACKER):
                    continue
            
            # Channel 0-5: Node type one-hot
            tensor[node.node_type.value - 1, y, x] = 1.0
            
            # Channel 6-9: Node status one-hot
            tensor[5 + node.status.value, y, x] = 1.0
            
            # Channel 10: Health normalized
            tensor[10, y, x] = node.health / node.max_health
            
            # Channel 11: Point value normalized
            tensor[11, y, x] = node.point_value / 100.0
            
            # Channel 12: Access level normalized
            tensor[12, y, x] = node.access_level.value / 3.0
            
            # Channel 13: Vulnerability score
            if node.unpatched_vulnerabilities:
                vuln_score = max(v.exploit_success_rate for v in node.unpatched_vulnerabilities)
                tensor[13, y, x] = vuln_score
            
            # Channel 14: Detection level
            tensor[14, y, x] = node.detection_level
            
            # Channel 15: Visibility (for attacker)
            tensor[15, y, x] = 1.0 if node.scanned_by_attacker else 0.0
            
            # Channel 16: Is compromised
            tensor[16, y, x] = 1.0 if node.is_compromised else 0.0
            
            # Channel 17: Is critical
            tensor[17, y, x] = 1.0 if node.is_critical else 0.0
            
            # Channel 18: Has backdoor
            tensor[18, y, x] = 1.0 if node.has_backdoor else 0.0
            
            # Channel 19: Has backup
            tensor[19, y, x] = 1.0 if node.has_backup else 0.0
        
        return tensor
    
    def get_global_features(self, perspective: PlayerRole) -> np.ndarray:
        """
        Get global game features as a flat vector.
        
        Returns:
            numpy array of shape (num_global_features,)
        """
        player = self.get_player_state(perspective)
        opponent = self.get_player_state(perspective.opponent)
        
        features = [
            # Turn progress
            self.turn_number / self.max_turns,
            
            # Player resources
            player.action_points / player.max_action_points,
            player.compute_units / player.max_compute_units,
            player.intel_points / 100.0,
            
            # Control
            len(player.controlled_nodes) / max(1, self.node_count),
            
            # Objectives
            self.compromised_critical_count / max(1, self.total_critical_count),
            self.network_destruction_ratio,
            
            # Detection
            player.detection_count / 10.0 if perspective == PlayerRole.ATTACKER else 0.0,
            self.defender.alert_level if perspective == PlayerRole.DEFENDER else 0.0,
            
            # Exfiltration progress
            self.attacker.exfiltrated_data / self.config.data_exfiltration_target,
        ]
        
        return np.array(features, dtype=np.float32)
    
    # ==========================================================================
    # String Representation
    # ==========================================================================
    
    def __str__(self) -> str:
        """Human-readable summary of game state"""
        lines = [
            f"=== Game {self.game_id[:8]} ===",
            f"Turn: {self.turn_number}/{self.max_turns}",
            f"Current Player: {self.current_player.name}",
            f"Nodes: {self.node_count} ({len(self.compromised_nodes)} compromised, {len(self.offline_nodes)} offline)",
            f"Critical: {self.compromised_critical_count}/{self.total_critical_count} compromised",
            f"",
            f"ATTACKER: AP={self.attacker.action_points}/{self.attacker.max_action_points}, "
            f"CU={self.attacker.compute_units}, Nodes={len(self.attacker.controlled_nodes)}, "
            f"Data={self.attacker.exfiltrated_data}",
            f"",
            f"DEFENDER: AP={self.defender.action_points}/{self.defender.max_action_points}, "
            f"CU={self.defender.compute_units}, Alert={self.defender.alert_level:.1%}, "
            f"Patches={self.defender.patches_applied}",
        ]
        
        if self.game_over:
            lines.append(f"\nGAME OVER: {self.winner.name} wins via {self.victory_condition}")
        
        return "\n".join(lines)
