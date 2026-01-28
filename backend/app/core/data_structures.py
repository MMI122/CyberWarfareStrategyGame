# =============================================================================
# Cyber Warfare Strategy Game - Core Data Structures
# =============================================================================
"""
Core data structures for representing game elements.
These are the fundamental building blocks of the game state.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from copy import deepcopy
import uuid

from .enums import (
    NodeType, NodeStatus, AccessLevel, PlayerRole, 
    ActionType, GamePhase, VictoryCondition, Difficulty
)


# =============================================================================
# Vulnerability
# =============================================================================

@dataclass
class Vulnerability:
    """
    Represents a security vulnerability on a node.
    
    Attributes:
        id: Unique identifier for this vulnerability
        name: Human-readable name (e.g., "CVE-2024-1234")
        severity: How severe the vulnerability is (0.0 - 1.0)
        exploit_difficulty: How hard it is to exploit (0.0 - 1.0, lower = easier)
        is_known: Whether the defender knows about this vulnerability
        is_patched: Whether the vulnerability has been fixed
    """
    id: str
    name: str
    severity: float  # 0.0 - 1.0
    exploit_difficulty: float  # 0.0 - 1.0
    is_known: bool = False
    is_patched: bool = False
    
    def __post_init__(self):
        """Validate vulnerability parameters"""
        self.severity = max(0.0, min(1.0, self.severity))
        self.exploit_difficulty = max(0.0, min(1.0, self.exploit_difficulty))
    
    @property
    def exploit_success_rate(self) -> float:
        """Calculate base success rate for exploiting this vulnerability"""
        if self.is_patched:
            return 0.0
        # Higher severity and lower difficulty = higher success
        return self.severity * (1.0 - self.exploit_difficulty)
    
    def clone(self) -> 'Vulnerability':
        """Create a deep copy of this vulnerability"""
        return Vulnerability(
            id=self.id,
            name=self.name,
            severity=self.severity,
            exploit_difficulty=self.exploit_difficulty,
            is_known=self.is_known,
            is_patched=self.is_patched
        )


# =============================================================================
# Network Node
# =============================================================================

@dataclass
class NetworkNode:
    """
    Represents a single node in the network topology.
    
    Nodes are the primary game elements that players interact with.
    The attacker tries to compromise nodes, the defender protects them.
    """
    id: int
    node_type: NodeType
    position: Tuple[float, float, float]  # 3D coordinates for visualization
    
    # State
    status: NodeStatus = NodeStatus.ONLINE
    health: int = 100
    max_health: int = 100
    
    # Security
    access_level: AccessLevel = AccessLevel.NONE  # Attacker's access
    vulnerabilities: List[Vulnerability] = field(default_factory=list)
    has_backdoor: bool = False  # RAT deployed
    detection_level: float = 0.0  # 0.0 - 1.0
    alert_level: int = 0  # How suspicious the defender is
    monitoring_level: int = 0  # Defender monitoring intensity
    
    # Value
    point_value: int = 10
    data_value: int = 0  # Exfiltrable data
    
    # Connectivity (will be populated by topology generator)
    adjacent_nodes: List[int] = field(default_factory=list)
    
    # Visibility
    visible_to_attacker: bool = False
    scanned_by_attacker: bool = False
    scanned: bool = False  # Alias for scanned_by_attacker
    
    # Decoy/Honeypot
    is_decoy: bool = False
    
    # Metadata
    name: str = ""
    subnet: str = "default"
    
    # Backup (for defender's restore action)
    has_backup: bool = False
    backup_turn: int = -1
    
    def __post_init__(self):
        """Initialize default values based on node type"""
        if self.max_health == 100 and self.health == 100:
            self.max_health = self.node_type.default_hp
            self.health = self.max_health
        if self.point_value == 10:
            self.point_value = self.node_type.default_value
        if not self.name:
            self.name = f"{self.node_type.name}_{self.id}"
    
    @property
    def is_compromised(self) -> bool:
        """Check if node is under attacker control"""
        return self.status == NodeStatus.COMPROMISED or self.access_level >= AccessLevel.USER
    
    @property
    def is_online(self) -> bool:
        """Check if node is operational"""
        return self.status == NodeStatus.ONLINE or self.status == NodeStatus.COMPROMISED
    
    @property
    def is_critical(self) -> bool:
        """Check if this is a critical objective node"""
        return self.node_type == NodeType.CRITICAL
    
    @property
    def unpatched_vulnerabilities(self) -> List[Vulnerability]:
        """Get list of vulnerabilities that haven't been patched"""
        return [v for v in self.vulnerabilities if not v.is_patched]
    
    @property
    def best_vulnerability(self) -> Optional[Vulnerability]:
        """Get the easiest to exploit unpatched vulnerability"""
        unpatched = self.unpatched_vulnerabilities
        if not unpatched:
            return None
        return max(unpatched, key=lambda v: v.exploit_success_rate)
    
    def take_damage(self, amount: int) -> bool:
        """
        Apply damage to the node.
        Returns True if node was destroyed (went offline).
        """
        self.health = max(0, self.health - amount)
        if self.health == 0:
            self.status = NodeStatus.OFFLINE
            return True
        return False
    
    def heal(self, amount: int):
        """Restore health to the node"""
        self.health = min(self.max_health, self.health + amount)
    
    def clone(self) -> 'NetworkNode':
        """Create a deep copy of this node"""
        return NetworkNode(
            id=self.id,
            node_type=self.node_type,
            position=self.position,
            status=self.status,
            health=self.health,
            max_health=self.max_health,
            access_level=self.access_level,
            vulnerabilities=[v.clone() for v in self.vulnerabilities],
            has_backdoor=self.has_backdoor,
            detection_level=self.detection_level,
            alert_level=self.alert_level,
            monitoring_level=self.monitoring_level,
            point_value=self.point_value,
            data_value=self.data_value,
            adjacent_nodes=self.adjacent_nodes.copy(),
            visible_to_attacker=self.visible_to_attacker,
            scanned_by_attacker=self.scanned_by_attacker,
            scanned=self.scanned,
            is_decoy=self.is_decoy,
            name=self.name,
            subnet=self.subnet,
            has_backup=self.has_backup,
            backup_turn=self.backup_turn
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "node_type": self.node_type.name,
            "position": self.position,
            "status": self.status.name,
            "health": self.health,
            "max_health": self.max_health,
            "access_level": self.access_level.name,
            "vulnerabilities": [
                {
                    "id": v.id,
                    "name": v.name,
                    "severity": v.severity,
                    "exploit_difficulty": v.exploit_difficulty,
                    "is_known": v.is_known,
                    "is_patched": v.is_patched
                }
                for v in self.vulnerabilities
            ],
            "has_backdoor": self.has_backdoor,
            "detection_level": self.detection_level,
            "alert_level": self.alert_level,
            "monitoring_level": self.monitoring_level,
            "point_value": self.point_value,
            "data_value": self.data_value,
            "adjacent_nodes": self.adjacent_nodes,
            "visible_to_attacker": self.visible_to_attacker,
            "scanned_by_attacker": self.scanned_by_attacker,
            "scanned": self.scanned,
            "is_decoy": self.is_decoy,
            "name": self.name,
            "subnet": self.subnet,
            "has_backup": self.has_backup,
            "backup_turn": self.backup_turn
        }


# =============================================================================
# Network Edge
# =============================================================================

@dataclass
class NetworkEdge:
    """
    Represents a connection between two nodes in the network.
    
    Edges define the topology and how data/attacks can flow.
    """
    source_id: int
    target_id: int
    bandwidth: float = 1.0  # Affects data transfer speed
    latency: float = 1.0    # Affects action timing
    is_encrypted: bool = False
    is_active: bool = True
    requires_credentials: bool = False
    
    @property
    def key(self) -> Tuple[int, int]:
        """Return a normalized key for this edge (smaller id first)"""
        return (min(self.source_id, self.target_id), 
                max(self.source_id, self.target_id))
    
    def connects(self, node_id: int) -> bool:
        """Check if this edge connects to a given node"""
        return self.source_id == node_id or self.target_id == node_id
    
    def other_node(self, node_id: int) -> int:
        """Get the node on the other end of this edge"""
        if self.source_id == node_id:
            return self.target_id
        elif self.target_id == node_id:
            return self.source_id
        raise ValueError(f"Node {node_id} is not part of this edge")
    
    def clone(self) -> 'NetworkEdge':
        """Create a deep copy of this edge"""
        return NetworkEdge(
            source_id=self.source_id,
            target_id=self.target_id,
            bandwidth=self.bandwidth,
            latency=self.latency,
            is_encrypted=self.is_encrypted,
            is_active=self.is_active,
            requires_credentials=self.requires_credentials
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "bandwidth": self.bandwidth,
            "latency": self.latency,
            "is_encrypted": self.is_encrypted,
            "is_active": self.is_active,
            "requires_credentials": self.requires_credentials
        }


# =============================================================================
# Player State
# =============================================================================

@dataclass
class PlayerState:
    """
    Represents the state of a player (attacker or defender).
    
    Tracks resources, controlled nodes, and progress.
    """
    role: PlayerRole
    
    # Resources
    action_points: int = 50
    max_action_points: int = 50
    compute_units: int = 100
    max_compute_units: int = 100
    intel_points: int = 0
    budget: int = 100  # Defender only - for deploying new assets
    
    # Attacker-specific
    controlled_nodes: Set[int] = field(default_factory=set)
    entry_points: Set[int] = field(default_factory=set)  # Nodes with backdoors
    backdoors_installed: Set[int] = field(default_factory=set)  # Alias for entry_points
    access_levels: Dict[int, 'AccessLevel'] = field(default_factory=dict)  # Access level per node
    exfiltrated_data: int = 0
    data_stolen: int = 0  # Alias for exfiltrated_data
    detection_count: int = 0  # Times detected
    turns_with_access: int = 0  # For persistent access victory
    
    # Defender-specific
    alert_level: float = 0.0  # 0.0 - 1.0
    honeypots_deployed: int = 0
    patches_applied: int = 0
    
    # Common
    total_actions_taken: int = 0
    action_history: List['GameAction'] = field(default_factory=list)
    score: int = 0
    
    def can_afford(self, ap_cost: int, cu_cost: int = 0) -> bool:
        """Check if player can afford an action"""
        return self.action_points >= ap_cost and self.compute_units >= cu_cost
    
    def spend_resources(self, ap_cost: int, cu_cost: int = 0):
        """Deduct resources for an action"""
        self.action_points -= ap_cost
        self.compute_units -= cu_cost
    
    def regenerate_resources(self, ap_regen: int = 50, cu_regen: int = 10):
        """Regenerate resources at start of turn"""
        self.action_points = min(self.max_action_points, 
                                  self.action_points + ap_regen)
        self.compute_units = min(self.max_compute_units,
                                  self.compute_units + cu_regen)
    
    def clone(self) -> 'PlayerState':
        """Create a deep copy of this player state"""
        return PlayerState(
            role=self.role,
            action_points=self.action_points,
            max_action_points=self.max_action_points,
            compute_units=self.compute_units,
            max_compute_units=self.max_compute_units,
            intel_points=self.intel_points,
            budget=self.budget,
            controlled_nodes=self.controlled_nodes.copy(),
            entry_points=self.entry_points.copy(),
            backdoors_installed=self.backdoors_installed.copy(),
            access_levels=self.access_levels.copy(),
            exfiltrated_data=self.exfiltrated_data,
            data_stolen=self.data_stolen,
            detection_count=self.detection_count,
            turns_with_access=self.turns_with_access,
            alert_level=self.alert_level,
            honeypots_deployed=self.honeypots_deployed,
            patches_applied=self.patches_applied,
            total_actions_taken=self.total_actions_taken,
            action_history=[],  # Don't copy history for AI clones
            score=self.score
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "role": self.role.name,
            "action_points": self.action_points,
            "max_action_points": self.max_action_points,
            "compute_units": self.compute_units,
            "max_compute_units": self.max_compute_units,
            "intel_points": self.intel_points,
            "budget": self.budget,
            "controlled_nodes": list(self.controlled_nodes),
            "entry_points": list(self.entry_points),
            "backdoors_installed": list(self.backdoors_installed),
            "access_levels": {k: v.name for k, v in self.access_levels.items()},
            "exfiltrated_data": self.exfiltrated_data,
            "data_stolen": self.data_stolen,
            "detection_count": self.detection_count,
            "turns_with_access": self.turns_with_access,
            "alert_level": self.alert_level,
            "honeypots_deployed": self.honeypots_deployed,
            "patches_applied": self.patches_applied,
            "total_actions_taken": self.total_actions_taken,
            "score": self.score
        }


# =============================================================================
# Game Action
# =============================================================================

@dataclass
class GameAction:
    """
    Represents an action taken by a player.
    
    Actions are the primary way players interact with the game.
    """
    action_type: ActionType
    player_role: PlayerRole = None  # Which player is performing the action
    player: PlayerRole = None  # Alias for player_role
    source_node: Optional[int] = None  # Node from which action originates
    target_node: Optional[int] = None  # Target node of the action
    target_node_id: Optional[int] = None  # Alias for target_node
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Computed at validation time
    ap_cost: int = 0
    cu_cost: int = 0
    success_probability: float = 1.0
    
    # Filled after execution
    executed: bool = False
    success: bool = False
    detected: bool = False
    effects: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Set default costs and sync aliases"""
        if self.ap_cost == 0:
            self.ap_cost = self.action_type.ap_cost
        # Sync player/player_role aliases
        if self.player_role is None and self.player is not None:
            self.player_role = self.player
        elif self.player is None and self.player_role is not None:
            self.player = self.player_role
        # Sync target_node/target_node_id aliases
        if self.target_node_id is None and self.target_node is not None:
            self.target_node_id = self.target_node
        elif self.target_node is None and self.target_node_id is not None:
            self.target_node = self.target_node_id
    
    @property
    def is_valid_for_player(self) -> bool:
        """Check if action is valid for the player's role"""
        role = self.player_role or self.player
        if self.action_type == ActionType.PASS or self.action_type == ActionType.END_TURN:
            return True
        if role == PlayerRole.ATTACKER:
            return self.action_type.is_attacker_action
        else:
            return self.action_type.is_defender_action
    
    def clone(self) -> 'GameAction':
        """Create a deep copy of this action"""
        return GameAction(
            action_type=self.action_type,
            player_role=self.player_role,
            player=self.player,
            source_node=self.source_node,
            target_node=self.target_node,
            target_node_id=self.target_node_id,
            parameters=deepcopy(self.parameters),
            ap_cost=self.ap_cost,
            cu_cost=self.cu_cost,
            success_probability=self.success_probability,
            executed=self.executed,
            success=self.success,
            detected=self.detected,
            effects=self.effects.copy()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "action_type": self.action_type.name,
            "player_role": self.player_role.name if self.player_role else None,
            "player": self.player.name if self.player else None,
            "source_node": self.source_node,
            "target_node": self.target_node,
            "target_node_id": self.target_node_id,
            "parameters": self.parameters,
            "ap_cost": self.ap_cost,
            "cu_cost": self.cu_cost,
            "success_probability": self.success_probability,
            "executed": self.executed,
            "success": self.success,
            "detected": self.detected,
            "effects": self.effects
        }
    
    def __str__(self) -> str:
        """Human-readable representation"""
        role = self.player_role or self.player
        target = self.target_node_id or self.target_node
        parts = [f"{role.name if role else 'UNKNOWN'} {self.action_type.name}"]
        if self.source_node is not None:
            parts.append(f"from node {self.source_node}")
        if target is not None:
            parts.append(f"targeting node {target}")
        return " ".join(parts)
    
    def __repr__(self) -> str:
        return (f"GameAction({self.action_type.name}, {self.player.name}, "
                f"src={self.source_node}, tgt={self.target_node})")


# =============================================================================
# Game Configuration
# =============================================================================

@dataclass
class GameConfig:
    """
    Configuration settings for a game instance.
    """
    # Basic settings
    difficulty: Difficulty = Difficulty.MEDIUM
    max_turns: int = 50
    fog_of_war: bool = True
    
    # Victory conditions
    data_exfiltration_target: int = 100
    persistent_access_turns: int = 10
    network_destruction_threshold: float = 0.75
    
    # Resource settings
    initial_ap: int = 50
    ap_per_turn: int = 50
    initial_cu: int = 100
    cu_per_turn: int = 10
    defender_budget: int = 100
    
    # Detection settings
    base_detection_chance: float = 0.1
    detection_increase_per_action: float = 0.05
    
    # Random seed for reproducibility
    seed: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "difficulty": self.difficulty.name,
            "max_turns": self.max_turns,
            "fog_of_war": self.fog_of_war,
            "data_exfiltration_target": self.data_exfiltration_target,
            "persistent_access_turns": self.persistent_access_turns,
            "network_destruction_threshold": self.network_destruction_threshold,
            "initial_ap": self.initial_ap,
            "ap_per_turn": self.ap_per_turn,
            "initial_cu": self.initial_cu,
            "cu_per_turn": self.cu_per_turn,
            "defender_budget": self.defender_budget,
            "base_detection_chance": self.base_detection_chance,
            "detection_increase_per_action": self.detection_increase_per_action,
            "seed": self.seed
        }


# =============================================================================
# Action Result
# =============================================================================

@dataclass  
class ActionResult:
    """
    Result of executing a game action.
    """
    action: GameAction
    success: bool
    message: str = ""
    detected: bool = False
    was_detected: bool = False  # Alias for detected
    effects: List[str] = field(default_factory=list)
    state_changes: Dict[str, Any] = field(default_factory=dict)
    points_gained: int = 0
    damage_dealt: int = 0
    discovered_info: Dict[str, Any] = field(default_factory=dict)
    ends_turn: bool = False
    
    def __post_init__(self):
        """Sync aliases"""
        if self.was_detected and not self.detected:
            self.detected = self.was_detected
        elif self.detected and not self.was_detected:
            self.was_detected = self.detected
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "action": self.action.to_dict(),
            "success": self.success,
            "message": self.message,
            "detected": self.detected,
            "effects": self.effects,
            "state_changes": self.state_changes,
            "points_gained": self.points_gained,
            "damage_dealt": self.damage_dealt,
            "discovered_info": self.discovered_info,
            "ends_turn": self.ends_turn
        }
