# =============================================================================
# Cyber Warfare Strategy Game - Enumerations
# =============================================================================
"""
All enumeration types used throughout the game.
These define the discrete values for game elements.
"""

from enum import Enum, auto
from typing import List


class NodeType(Enum):
    """
    Types of network nodes in the game.
    Each type has different properties and strategic value.
    """
    WORKSTATION = auto()    # Basic endpoint, easy to compromise
    SERVER = auto()         # High-value target, more HP
    ROUTER = auto()         # Network infrastructure, controls connectivity
    FIREWALL = auto()       # Security device, hard to compromise
    CRITICAL = auto()       # Primary objective for attacker
    HONEYPOT = auto()       # Decoy/trap deployed by defender
    
    def __str__(self) -> str:
        return self.name.lower()
    
    @property
    def symbol(self) -> str:
        """Return ASCII symbol for display"""
        symbols = {
            NodeType.WORKSTATION: "◯",
            NodeType.SERVER: "◇",
            NodeType.ROUTER: "◈",
            NodeType.FIREWALL: "⬡",
            NodeType.CRITICAL: "★",
            NodeType.HONEYPOT: "◐",
        }
        return symbols.get(self, "?")
    
    @property
    def default_hp(self) -> int:
        """Default health points for this node type"""
        hp_values = {
            NodeType.WORKSTATION: 50,
            NodeType.SERVER: 100,
            NodeType.ROUTER: 75,
            NodeType.FIREWALL: 200,
            NodeType.CRITICAL: 150,
            NodeType.HONEYPOT: 30,
        }
        return hp_values.get(self, 50)
    
    @property
    def default_value(self) -> int:
        """Default point value for compromising this node"""
        values = {
            NodeType.WORKSTATION: 10,
            NodeType.SERVER: 50,
            NodeType.ROUTER: 25,
            NodeType.FIREWALL: 30,
            NodeType.CRITICAL: 100,
            NodeType.HONEYPOT: 0,
        }
        return values.get(self, 10)
    
    @property
    def vulnerability_level(self) -> float:
        """Base vulnerability level (0.0 - 1.0, higher = easier to exploit)"""
        vuln = {
            NodeType.WORKSTATION: 0.7,
            NodeType.SERVER: 0.5,
            NodeType.ROUTER: 0.3,
            NodeType.FIREWALL: 0.1,
            NodeType.CRITICAL: 0.2,
            NodeType.HONEYPOT: 0.9,
        }
        return vuln.get(self, 0.5)


class NodeStatus(Enum):
    """
    Current operational status of a node.
    """
    ONLINE = auto()         # Normal operation
    COMPROMISED = auto()    # Under attacker control
    INFECTED = auto()       # Malware deployed, taking damage
    ISOLATED = auto()       # Disconnected from network by defender
    OFFLINE = auto()        # Disabled/destroyed
    PATCHING = auto()       # Being updated (temporary vulnerability)
    
    def __str__(self) -> str:
        return self.name.lower()
    
    @property
    def symbol(self) -> str:
        symbols = {
            NodeStatus.ONLINE: "●",
            NodeStatus.COMPROMISED: "▣",
            NodeStatus.INFECTED: "⚠",
            NodeStatus.ISOLATED: "◆",
            NodeStatus.OFFLINE: "○",
            NodeStatus.PATCHING: "◉",
        }
        return symbols.get(self, "?")


class AccessLevel(Enum):
    """
    Privilege levels on a compromised node.
    Higher levels grant more capabilities.
    """
    NONE = 0        # No access
    USER = 1        # Basic user access
    ADMIN = 2       # Administrator access
    ROOT = 3        # Full system control
    
    def __str__(self) -> str:
        return self.name.lower()
    
    def __lt__(self, other: 'AccessLevel') -> bool:
        return self.value < other.value
    
    def __le__(self, other: 'AccessLevel') -> bool:
        return self.value <= other.value
    
    def __gt__(self, other: 'AccessLevel') -> bool:
        return self.value > other.value
    
    def __ge__(self, other: 'AccessLevel') -> bool:
        return self.value >= other.value


class PlayerRole(Enum):
    """
    The two opposing roles in the game.
    """
    ATTACKER = auto()   # Red team - tries to compromise network
    DEFENDER = auto()   # Blue team - protects the network
    
    def __str__(self) -> str:
        return self.name.lower()
    
    @property
    def opponent(self) -> 'PlayerRole':
        """Return the opposing role"""
        if self == PlayerRole.ATTACKER:
            return PlayerRole.DEFENDER
        return PlayerRole.ATTACKER
    
    @property
    def color(self) -> str:
        """Return display color for this role"""
        return "red" if self == PlayerRole.ATTACKER else "blue"


class ActionType(Enum):
    """
    All possible actions in the game.
    Actions are categorized by which player can use them.
    """
    # ==========================================================================
    # ATTACKER ACTIONS
    # ==========================================================================
    SCAN = auto()           # Reveal node information and vulnerabilities
    EXPLOIT = auto()        # Attempt to compromise a node
    ESCALATE = auto()       # Increase privilege level on compromised node
    PIVOT = auto()          # Move to an adjacent node
    EXFILTRATE = auto()     # Extract data from a compromised critical node
    COVER_TRACKS = auto()   # Reduce detection level on a node
    DEPLOY_RAT = auto()     # Deploy persistent backdoor (Remote Access Trojan)
    INSTALL_BACKDOOR = auto()  # Alias for DEPLOY_RAT - persistent access
    DOS_ATTACK = auto()     # Denial of Service - take node offline
    DEPLOY_MALWARE = auto() # Deploy malware to damage a node
    
    # ==========================================================================
    # DEFENDER ACTIONS
    # ==========================================================================
    MONITOR = auto()        # Increase detection capability on a node
    PATCH = auto()          # Fix a known vulnerability
    ISOLATE = auto()        # Disconnect a node from the network
    HONEYPOT = auto()       # Deploy a decoy node
    DECEIVE = auto()        # Convert a node to honeypot/decoy
    BACKUP = auto()         # Create a restore point for a node
    RESTORE = auto()        # Restore node from backup
    FIREWALL_UPGRADE = auto()  # Upgrade firewall rules
    TRACE = auto()          # Attempt to locate attacker
    RECONNECT = auto()      # Reconnect an isolated node
    
    # ==========================================================================
    # COMMON ACTIONS
    # ==========================================================================
    PASS = auto()           # Skip turn, regain extra AP
    END_TURN = auto()       # End current turn
    
    def __str__(self) -> str:
        return self.name.lower()
    
    @property
    def is_attacker_action(self) -> bool:
        """Check if this is an attacker-only action"""
        attacker_actions = {
            ActionType.SCAN, ActionType.EXPLOIT, ActionType.ESCALATE,
            ActionType.PIVOT, ActionType.EXFILTRATE, ActionType.COVER_TRACKS,
            ActionType.DEPLOY_RAT, ActionType.DOS_ATTACK
        }
        return self in attacker_actions
    
    @property
    def is_defender_action(self) -> bool:
        """Check if this is a defender-only action"""
        defender_actions = {
            ActionType.MONITOR, ActionType.PATCH, ActionType.ISOLATE,
            ActionType.HONEYPOT, ActionType.BACKUP, ActionType.RESTORE,
            ActionType.FIREWALL_UPGRADE, ActionType.TRACE, ActionType.RECONNECT
        }
        return self in defender_actions
    
    @property
    def ap_cost(self) -> int:
        """Base action point cost for this action"""
        costs = {
            # Attacker
            ActionType.SCAN: 1,
            ActionType.EXPLOIT: 2,
            ActionType.ESCALATE: 2,
            ActionType.PIVOT: 1,
            ActionType.EXFILTRATE: 3,
            ActionType.COVER_TRACKS: 1,
            ActionType.DEPLOY_RAT: 3,
            ActionType.INSTALL_BACKDOOR: 3,
            ActionType.DOS_ATTACK: 2,
            ActionType.DEPLOY_MALWARE: 2,
            # Defender
            ActionType.MONITOR: 1,
            ActionType.PATCH: 2,
            ActionType.ISOLATE: 1,
            ActionType.HONEYPOT: 2,
            ActionType.DECEIVE: 2,
            ActionType.BACKUP: 1,
            ActionType.RESTORE: 2,
            ActionType.FIREWALL_UPGRADE: 3,
            ActionType.TRACE: 2,
            ActionType.RECONNECT: 1,
            # Common
            ActionType.PASS: 0,
            ActionType.END_TURN: 0,
        }
        return costs.get(self, 1)
    
    @property
    def requires_target(self) -> bool:
        """Check if this action requires a target node"""
        no_target = {ActionType.PASS, ActionType.END_TURN}
        return self not in no_target
    
    @property
    def description(self) -> str:
        """Human-readable description of the action"""
        descriptions = {
            ActionType.SCAN: "Reveal node information and vulnerabilities",
            ActionType.EXPLOIT: "Attempt to compromise a vulnerable node",
            ActionType.ESCALATE: "Increase privilege level on compromised node",
            ActionType.PIVOT: "Move presence to an adjacent compromised node",
            ActionType.EXFILTRATE: "Extract valuable data from the node",
            ActionType.COVER_TRACKS: "Reduce detection level",
            ActionType.DEPLOY_RAT: "Install persistent backdoor",
            ActionType.INSTALL_BACKDOOR: "Install persistent backdoor",
            ActionType.DOS_ATTACK: "Take the node offline",
            ActionType.DEPLOY_MALWARE: "Deploy malware to damage node",
            ActionType.MONITOR: "Increase intrusion detection",
            ActionType.PATCH: "Fix a known vulnerability",
            ActionType.ISOLATE: "Disconnect node from network",
            ActionType.HONEYPOT: "Deploy a decoy trap",
            ActionType.DECEIVE: "Convert node to honeypot",
            ActionType.BACKUP: "Create a restore point",
            ActionType.RESTORE: "Restore from backup",
            ActionType.FIREWALL_UPGRADE: "Strengthen firewall rules",
            ActionType.TRACE: "Attempt to locate the attacker",
            ActionType.RECONNECT: "Reconnect isolated node",
            ActionType.PASS: "Skip turn and recover AP",
            ActionType.END_TURN: "End the current turn",
        }
        return descriptions.get(self, "Unknown action")


class GamePhase(Enum):
    """
    Phases within a game turn.
    """
    SETUP = auto()          # Initial game setup
    PLAYING = auto()        # Game in progress (either player's turn)
    ATTACKER_TURN = auto()  # Attacker is making moves
    DEFENDER_TURN = auto()  # Defender is making moves
    RESOLUTION = auto()     # Resolving simultaneous effects
    GAME_OVER = auto()      # Game has ended
    
    def __str__(self) -> str:
        return self.name.lower()


class VictoryCondition(Enum):
    """
    Ways the game can end.
    """
    # Attacker victories
    ALL_CRITICAL_COMPROMISED = auto()   # All critical nodes under attacker control
    DATA_EXFILTRATED = auto()           # Exfiltrated enough data
    PERSISTENT_ACCESS = auto()          # Maintained access for N turns
    NETWORK_DESTROYED = auto()          # 75%+ nodes offline
    
    # Defender victories
    SURVIVED_TIME_LIMIT = auto()        # Survived max turns
    ATTACKER_ELIMINATED = auto()        # Removed all attacker presence
    ATTACKER_EXHAUSTED = auto()         # Attacker ran out of resources
    
    # Draw
    DRAW = auto()                       # Stalemate
    
    def __str__(self) -> str:
        return self.name.replace("_", " ").title()
    
    @property
    def winner(self) -> PlayerRole | None:
        """Return the winner for this condition, or None for draw"""
        attacker_wins = {
            VictoryCondition.ALL_CRITICAL_COMPROMISED,
            VictoryCondition.DATA_EXFILTRATED,
            VictoryCondition.PERSISTENT_ACCESS,
            VictoryCondition.NETWORK_DESTROYED,
        }
        defender_wins = {
            VictoryCondition.SURVIVED_TIME_LIMIT,
            VictoryCondition.ATTACKER_ELIMINATED,
            VictoryCondition.ATTACKER_EXHAUSTED,
        }
        if self in attacker_wins:
            return PlayerRole.ATTACKER
        elif self in defender_wins:
            return PlayerRole.DEFENDER
        return None


class Difficulty(Enum):
    """
    Game difficulty levels affecting network size and complexity.
    """
    EASY = 1        # Small network, fewer vulnerabilities
    MEDIUM = 2      # Medium network, balanced
    HARD = 3        # Large network, many paths
    EXPERT = 4      # Complex network, challenging
    
    def __str__(self) -> str:
        return self.name.lower()
    
    @property
    def node_count_range(self) -> tuple[int, int]:
        """Min and max node count for this difficulty"""
        ranges = {
            Difficulty.EASY: (10, 15),
            Difficulty.MEDIUM: (20, 30),
            Difficulty.HARD: (35, 50),
            Difficulty.EXPERT: (60, 100),
        }
        return ranges.get(self, (20, 30))
    
    @property
    def critical_count(self) -> int:
        """Number of critical nodes to protect"""
        counts = {
            Difficulty.EASY: 1,
            Difficulty.MEDIUM: 2,
            Difficulty.HARD: 3,
            Difficulty.EXPERT: 4,
        }
        return counts.get(self, 2)
    
    @property
    def critical_node_count(self) -> int:
        """Alias for critical_count"""
        return self.critical_count
    
    @property
    def max_turns(self) -> int:
        """Maximum turns for the game"""
        turns = {
            Difficulty.EASY: 30,
            Difficulty.MEDIUM: 50,
            Difficulty.HARD: 75,
            Difficulty.EXPERT: 100,
        }
        return turns.get(self, 50)


# =============================================================================
# Utility Functions
# =============================================================================

def get_actions_for_role(role: PlayerRole) -> List[ActionType]:
    """Get all valid actions for a given player role"""
    if role == PlayerRole.ATTACKER:
        return [a for a in ActionType if a.is_attacker_action or a == ActionType.PASS]
    else:
        return [a for a in ActionType if a.is_defender_action or a == ActionType.PASS]
