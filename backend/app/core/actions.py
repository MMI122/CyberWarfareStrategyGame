# =============================================================================
# Cyber Warfare Strategy Game - Actions Module
# =============================================================================
"""
Handles action validation, execution, and effect resolution.
All game actions flow through ActionValidator -> ActionExecutor.
"""

import random
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

from .enums import (
    ActionType, NodeType, NodeStatus, AccessLevel,
    PlayerRole, GamePhase
)
from .data_structures import (
    GameAction, ActionResult, NetworkNode, PlayerState, GameConfig
)


# =============================================================================
# Action Validation
# =============================================================================

class ActionValidator:
    """
    Validates game actions before execution.
    
    Checks:
    - Player has sufficient action points
    - Action type is valid for current game phase
    - Target node exists and is valid
    - Player has required access level
    - Action prerequisites are met
    """
    
    def __init__(self, config: GameConfig):
        self.config = config
    
    def validate(
        self,
        action: GameAction,
        player: PlayerState,
        nodes: dict,
        edges: list,
        game_phase: GamePhase
    ) -> Tuple[bool, str]:
        """
        Validate if an action can be performed.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check game phase
        if game_phase != GamePhase.PLAYING:
            return False, f"Cannot perform actions during {game_phase.name} phase"
        
        # Check if it's the player's turn
        # (This should be checked by the game engine, but double-check)
        
        # Check action points
        ap_cost = action.action_type.ap_cost
        if player.action_points < ap_cost:
            return False, f"Insufficient AP: need {ap_cost}, have {player.action_points}"
        
        # Validate by action type
        validators = {
            ActionType.SCAN: self._validate_scan,
            ActionType.EXPLOIT: self._validate_exploit,
            ActionType.ESCALATE: self._validate_escalate,
            ActionType.PIVOT: self._validate_pivot,
            ActionType.EXFILTRATE: self._validate_exfiltrate,
            ActionType.INSTALL_BACKDOOR: self._validate_install_backdoor,
            ActionType.DEPLOY_MALWARE: self._validate_deploy_malware,
            ActionType.PATCH: self._validate_patch,
            ActionType.ISOLATE: self._validate_isolate,
            ActionType.MONITOR: self._validate_monitor,
            ActionType.TRACE: self._validate_trace,
            ActionType.RESTORE: self._validate_restore,
            ActionType.DECEIVE: self._validate_deceive,
            ActionType.END_TURN: self._validate_end_turn,
        }
        
        validator = validators.get(action.action_type)
        if validator is None:
            return False, f"Unknown action type: {action.action_type}"
        
        return validator(action, player, nodes, edges)
    
    def _validate_scan(
        self, action: GameAction, player: PlayerState,
        nodes: dict, edges: list
    ) -> Tuple[bool, str]:
        """Validate SCAN action"""
        if player.role != PlayerRole.ATTACKER:
            return False, "Only attacker can scan"
        
        # Target must be visible to attacker
        target = nodes.get(action.target_node_id)
        if target is None:
            return False, "Target node does not exist"
        
        if not target.visible_to_attacker:
            return False, "Target node is not visible"
        
        return True, ""
    
    def _validate_exploit(
        self, action: GameAction, player: PlayerState,
        nodes: dict, edges: list
    ) -> Tuple[bool, str]:
        """Validate EXPLOIT action"""
        if player.role != PlayerRole.ATTACKER:
            return False, "Only attacker can exploit"
        
        target = nodes.get(action.target_node_id)
        if target is None:
            return False, "Target node does not exist"
        
        if not target.visible_to_attacker:
            return False, "Target node is not visible"
        
        if target.status == NodeStatus.OFFLINE:
            return False, "Cannot exploit offline node"
        
        # Must have at least one known vulnerability
        if not target.scanned:
            return False, "Node must be scanned first"
        
        # Check if player can reach this node
        if not self._can_reach_node(action.target_node_id, player, nodes, edges):
            return False, "Cannot reach target node from controlled nodes"
        
        return True, ""
    
    def _validate_escalate(
        self, action: GameAction, player: PlayerState,
        nodes: dict, edges: list
    ) -> Tuple[bool, str]:
        """Validate ESCALATE (privilege escalation) action"""
        if player.role != PlayerRole.ATTACKER:
            return False, "Only attacker can escalate"
        
        target = nodes.get(action.target_node_id)
        if target is None:
            return False, "Target node does not exist"
        
        # Must already have access to the node
        if action.target_node_id not in player.controlled_nodes:
            return False, "Must have access to node to escalate"
        
        # Check current access level
        current_level = player.access_levels.get(action.target_node_id, AccessLevel.NONE)
        if current_level == AccessLevel.ROOT:
            return False, "Already have root access"
        
        return True, ""
    
    def _validate_pivot(
        self, action: GameAction, player: PlayerState,
        nodes: dict, edges: list
    ) -> Tuple[bool, str]:
        """Validate PIVOT (lateral movement) action"""
        if player.role != PlayerRole.ATTACKER:
            return False, "Only attacker can pivot"
        
        target = nodes.get(action.target_node_id)
        if target is None:
            return False, "Target node does not exist"
        
        # Must have at least one controlled node adjacent to target
        adjacent = False
        for ctrl_id in player.controlled_nodes:
            ctrl_node = nodes.get(ctrl_id)
            if ctrl_node and action.target_node_id in ctrl_node.adjacent_nodes:
                adjacent = True
                break
        
        if not adjacent:
            return False, "Target must be adjacent to a controlled node"
        
        return True, ""
    
    def _validate_exfiltrate(
        self, action: GameAction, player: PlayerState,
        nodes: dict, edges: list
    ) -> Tuple[bool, str]:
        """Validate EXFILTRATE action"""
        if player.role != PlayerRole.ATTACKER:
            return False, "Only attacker can exfiltrate"
        
        target = nodes.get(action.target_node_id)
        if target is None:
            return False, "Target node does not exist"
        
        # Must control the node with at least USER access
        if action.target_node_id not in player.controlled_nodes:
            return False, "Must control node to exfiltrate data"
        
        access = player.access_levels.get(action.target_node_id, AccessLevel.NONE)
        if access.value < AccessLevel.USER.value:
            return False, "Need at least USER access to exfiltrate"
        
        # Node must have data to exfiltrate
        if target.data_value <= 0:
            return False, "No data to exfiltrate from this node"
        
        return True, ""
    
    def _validate_install_backdoor(
        self, action: GameAction, player: PlayerState,
        nodes: dict, edges: list
    ) -> Tuple[bool, str]:
        """Validate INSTALL_BACKDOOR action"""
        if player.role != PlayerRole.ATTACKER:
            return False, "Only attacker can install backdoors"
        
        target = nodes.get(action.target_node_id)
        if target is None:
            return False, "Target node does not exist"
        
        # Must control the node with ROOT access
        if action.target_node_id not in player.controlled_nodes:
            return False, "Must control node to install backdoor"
        
        access = player.access_levels.get(action.target_node_id, AccessLevel.NONE)
        if access != AccessLevel.ROOT:
            return False, "Need ROOT access to install backdoor"
        
        # Check if backdoor already installed
        if target.has_backdoor:
            return False, "Backdoor already installed"
        
        return True, ""
    
    def _validate_deploy_malware(
        self, action: GameAction, player: PlayerState,
        nodes: dict, edges: list
    ) -> Tuple[bool, str]:
        """Validate DEPLOY_MALWARE action"""
        if player.role != PlayerRole.ATTACKER:
            return False, "Only attacker can deploy malware"
        
        target = nodes.get(action.target_node_id)
        if target is None:
            return False, "Target node does not exist"
        
        # Must control the node
        if action.target_node_id not in player.controlled_nodes:
            return False, "Must control node to deploy malware"
        
        return True, ""
    
    def _validate_patch(
        self, action: GameAction, player: PlayerState,
        nodes: dict, edges: list
    ) -> Tuple[bool, str]:
        """Validate PATCH action (defender)"""
        if player.role != PlayerRole.DEFENDER:
            return False, "Only defender can patch"
        
        target = nodes.get(action.target_node_id)
        if target is None:
            return False, "Target node does not exist"
        
        # Check if node has vulnerabilities to patch
        if not target.vulnerabilities:
            return False, "No vulnerabilities to patch"
        
        return True, ""
    
    def _validate_isolate(
        self, action: GameAction, player: PlayerState,
        nodes: dict, edges: list
    ) -> Tuple[bool, str]:
        """Validate ISOLATE action (defender)"""
        if player.role != PlayerRole.DEFENDER:
            return False, "Only defender can isolate"
        
        target = nodes.get(action.target_node_id)
        if target is None:
            return False, "Target node does not exist"
        
        if target.status == NodeStatus.ISOLATED:
            return False, "Node is already isolated"
        
        if target.status == NodeStatus.OFFLINE:
            return False, "Cannot isolate offline node"
        
        return True, ""
    
    def _validate_monitor(
        self, action: GameAction, player: PlayerState,
        nodes: dict, edges: list
    ) -> Tuple[bool, str]:
        """Validate MONITOR action (defender)"""
        if player.role != PlayerRole.DEFENDER:
            return False, "Only defender can monitor"
        
        target = nodes.get(action.target_node_id)
        if target is None:
            return False, "Target node does not exist"
        
        if target.status == NodeStatus.OFFLINE:
            return False, "Cannot monitor offline node"
        
        return True, ""
    
    def _validate_trace(
        self, action: GameAction, player: PlayerState,
        nodes: dict, edges: list
    ) -> Tuple[bool, str]:
        """Validate TRACE action (defender)"""
        if player.role != PlayerRole.DEFENDER:
            return False, "Only defender can trace"
        
        # Trace requires a starting point (a compromised node we detected)
        target = nodes.get(action.target_node_id)
        if target is None:
            return False, "Target node does not exist"
        
        if target.status not in [NodeStatus.COMPROMISED, NodeStatus.INFECTED]:
            return False, "Can only trace from compromised/infected nodes"
        
        return True, ""
    
    def _validate_restore(
        self, action: GameAction, player: PlayerState,
        nodes: dict, edges: list
    ) -> Tuple[bool, str]:
        """Validate RESTORE action (defender)"""
        if player.role != PlayerRole.DEFENDER:
            return False, "Only defender can restore"
        
        target = nodes.get(action.target_node_id)
        if target is None:
            return False, "Target node does not exist"
        
        if target.status == NodeStatus.ONLINE:
            return False, "Node is already online"
        
        return True, ""
    
    def _validate_deceive(
        self, action: GameAction, player: PlayerState,
        nodes: dict, edges: list
    ) -> Tuple[bool, str]:
        """Validate DECEIVE (honeypot) action (defender)"""
        if player.role != PlayerRole.DEFENDER:
            return False, "Only defender can set up honeypots"
        
        target = nodes.get(action.target_node_id)
        if target is None:
            return False, "Target node does not exist"
        
        if target.node_type == NodeType.HONEYPOT:
            return False, "Node is already a honeypot"
        
        return True, ""
    
    def _validate_end_turn(
        self, action: GameAction, player: PlayerState,
        nodes: dict, edges: list
    ) -> Tuple[bool, str]:
        """Validate END_TURN action"""
        return True, ""
    
    def _can_reach_node(
        self, target_id: int, player: PlayerState,
        nodes: dict, edges: list
    ) -> bool:
        """Check if player can reach a node from their controlled nodes"""
        # If player controls no nodes, check from internet (node 0)
        if not player.controlled_nodes:
            # Can only reach entry points (visible nodes)
            target = nodes.get(target_id)
            return target is not None and target.visible_to_attacker
        
        # BFS from controlled nodes
        visited = set()
        queue = list(player.controlled_nodes)
        visited.update(queue)
        
        while queue:
            current = queue.pop(0)
            if current == target_id:
                return True
            
            node = nodes.get(current)
            if node:
                for adj in node.adjacent_nodes:
                    if adj not in visited:
                        visited.add(adj)
                        queue.append(adj)
        
        return False
    
    def get_valid_actions(
        self,
        player: PlayerState,
        nodes: dict,
        edges: list,
        game_phase: GamePhase
    ) -> List[GameAction]:
        """Get all valid actions for the current player"""
        valid_actions = []
        
        if game_phase != GamePhase.PLAYING:
            return valid_actions
        
        # Get action types for this role
        if player.role == PlayerRole.ATTACKER:
            action_types = [
                ActionType.SCAN, ActionType.EXPLOIT, ActionType.ESCALATE,
                ActionType.PIVOT, ActionType.EXFILTRATE, 
                ActionType.INSTALL_BACKDOOR, ActionType.DEPLOY_MALWARE,
                ActionType.END_TURN
            ]
        else:
            action_types = [
                ActionType.PATCH, ActionType.ISOLATE, ActionType.MONITOR,
                ActionType.TRACE, ActionType.RESTORE, ActionType.DECEIVE,
                ActionType.END_TURN
            ]
        
        # Check each action type for each node
        for action_type in action_types:
            if action_type == ActionType.END_TURN:
                valid_actions.append(GameAction(
                    action_type=ActionType.END_TURN,
                    target_node_id=-1,
                    player_role=player.role
                ))
                continue
            
            # Check AP cost
            if player.action_points < action_type.ap_cost:
                continue
            
            # Try each node as target
            for node_id in nodes:
                action = GameAction(
                    action_type=action_type,
                    target_node_id=node_id,
                    player_role=player.role
                )
                is_valid, _ = self.validate(action, player, nodes, edges, game_phase)
                if is_valid:
                    valid_actions.append(action)
        
        return valid_actions


# =============================================================================
# Action Execution
# =============================================================================

class ActionExecutor:
    """
    Executes validated game actions and resolves their effects.
    
    Each action modifies the game state and returns an ActionResult
    describing what happened.
    """
    
    def __init__(self, config: GameConfig):
        self.config = config
    
    def execute(
        self,
        action: GameAction,
        player: PlayerState,
        nodes: dict,
        edges: list
    ) -> ActionResult:
        """
        Execute a validated action.
        
        Args:
            action: The action to execute
            player: The player performing the action
            nodes: Network nodes dict
            edges: Network edges list
            
        Returns:
            ActionResult describing the outcome
        """
        # Deduct action points
        ap_cost = action.action_type.ap_cost
        player.action_points -= ap_cost
        
        # Execute by action type
        executors = {
            ActionType.SCAN: self._execute_scan,
            ActionType.EXPLOIT: self._execute_exploit,
            ActionType.ESCALATE: self._execute_escalate,
            ActionType.PIVOT: self._execute_pivot,
            ActionType.EXFILTRATE: self._execute_exfiltrate,
            ActionType.INSTALL_BACKDOOR: self._execute_install_backdoor,
            ActionType.DEPLOY_MALWARE: self._execute_deploy_malware,
            ActionType.PATCH: self._execute_patch,
            ActionType.ISOLATE: self._execute_isolate,
            ActionType.MONITOR: self._execute_monitor,
            ActionType.TRACE: self._execute_trace,
            ActionType.RESTORE: self._execute_restore,
            ActionType.DECEIVE: self._execute_deceive,
            ActionType.END_TURN: self._execute_end_turn,
        }
        
        executor = executors.get(action.action_type)
        if executor is None:
            return ActionResult(
                action=action,
                success=False,
                message="Unknown action type"
            )
        
        result = executor(action, player, nodes, edges)
        
        # Track action history
        player.action_history.append(action)
        
        return result
    
    def _execute_scan(
        self, action: GameAction, player: PlayerState,
        nodes: dict, edges: list
    ) -> ActionResult:
        """Execute SCAN action - reveals node information and adjacent nodes"""
        target = nodes[action.target_node_id]
        
        # Mark as scanned
        target.scanned = True
        
        # Reveal vulnerabilities (some)
        discovered = []
        for vuln in target.vulnerabilities:
            if random.random() < 0.7:  # 70% chance to discover each vuln
                vuln.is_known = True
                discovered.append(vuln.name)
        
        # Reveal adjacent nodes
        revealed_nodes = []
        for adj_id in target.adjacent_nodes:
            adj_node = nodes.get(adj_id)
            if adj_node and not adj_node.visible_to_attacker:
                adj_node.visible_to_attacker = True
                revealed_nodes.append(adj_id)
        
        # Small chance defender detects the scan
        detected = random.random() < 0.1
        if detected:
            target.alert_level += 1
        
        return ActionResult(
            action=action,
            success=True,
            message=f"Scanned {target.name}. Found {len(discovered)} vulnerabilities.",
            discovered_info={
                "vulnerabilities": discovered,
                "revealed_nodes": revealed_nodes
            },
            was_detected=detected
        )
    
    def _execute_exploit(
        self, action: GameAction, player: PlayerState,
        nodes: dict, edges: list
    ) -> ActionResult:
        """Execute EXPLOIT action - attempts to gain access to a node"""
        target = nodes[action.target_node_id]
        
        # Calculate success chance based on vulnerabilities
        best_vuln = None
        best_chance = 0.1  # Base 10% chance
        
        for vuln in target.vulnerabilities:
            if vuln.is_known:
                # Success = severity * (1 - difficulty)
                chance = vuln.severity * (1 - vuln.exploit_difficulty)
                if chance > best_chance:
                    best_chance = chance
                    best_vuln = vuln
        
        # Roll for success
        success = random.random() < best_chance
        
        if success:
            # Gain access
            target.status = NodeStatus.COMPROMISED
            player.controlled_nodes.add(action.target_node_id)
            player.access_levels[action.target_node_id] = AccessLevel.USER
            
            # Gain points
            points = target.point_value
            player.score += points
            
            # Mark vulnerability as exploited
            if best_vuln:
                best_vuln.exploited = True
            
            # Detection chance
            detected = random.random() < 0.3
            if detected:
                target.alert_level += 2
            
            return ActionResult(
                action=action,
                success=True,
                message=f"Successfully exploited {target.name}. Gained USER access.",
                points_gained=points,
                was_detected=detected,
                state_changes={"node_status": NodeStatus.COMPROMISED.name}
            )
        else:
            # Failed exploit - higher detection chance
            detected = random.random() < 0.5
            if detected:
                target.alert_level += 3
            
            return ActionResult(
                action=action,
                success=False,
                message=f"Failed to exploit {target.name}.",
                was_detected=detected
            )
    
    def _execute_escalate(
        self, action: GameAction, player: PlayerState,
        nodes: dict, edges: list
    ) -> ActionResult:
        """Execute ESCALATE action - attempts privilege escalation"""
        target = nodes[action.target_node_id]
        current_level = player.access_levels.get(action.target_node_id, AccessLevel.NONE)
        
        # Success chance based on current level and node type
        if current_level == AccessLevel.USER:
            base_chance = 0.4
            next_level = AccessLevel.ADMIN
        elif current_level == AccessLevel.ADMIN:
            base_chance = 0.25
            next_level = AccessLevel.ROOT
        else:
            return ActionResult(
                action=action,
                success=False,
                message="Cannot escalate from current level"
            )
        
        # Harder on critical systems
        if target.node_type == NodeType.CRITICAL:
            base_chance *= 0.5
        
        success = random.random() < base_chance
        
        if success:
            player.access_levels[action.target_node_id] = next_level
            
            points = 10 if next_level == AccessLevel.ADMIN else 20
            player.score += points
            
            detected = random.random() < 0.2
            if detected:
                target.alert_level += 2
            
            return ActionResult(
                action=action,
                success=True,
                message=f"Escalated to {next_level.name} on {target.name}.",
                points_gained=points,
                was_detected=detected
            )
        else:
            detected = random.random() < 0.4
            if detected:
                target.alert_level += 2
            
            return ActionResult(
                action=action,
                success=False,
                message=f"Failed privilege escalation on {target.name}.",
                was_detected=detected
            )
    
    def _execute_pivot(
        self, action: GameAction, player: PlayerState,
        nodes: dict, edges: list
    ) -> ActionResult:
        """Execute PIVOT action - lateral movement to adjacent node"""
        target = nodes[action.target_node_id]
        
        # Find a controlled node adjacent to target
        source_node = None
        for ctrl_id in player.controlled_nodes:
            ctrl = nodes.get(ctrl_id)
            if ctrl and action.target_node_id in ctrl.adjacent_nodes:
                source_node = ctrl
                break
        
        # Success chance based on access level at source
        source_level = player.access_levels.get(source_node.id, AccessLevel.USER)
        base_chance = 0.3 + (source_level.value * 0.1)
        
        # Check edge properties
        for edge in edges:
            if edge.connects(source_node.id) and edge.other_node(source_node.id) == action.target_node_id:
                if edge.requires_credentials:
                    base_chance *= 0.5
                break
        
        success = random.random() < base_chance
        
        if success:
            target.status = NodeStatus.COMPROMISED
            target.visible_to_attacker = True
            player.controlled_nodes.add(action.target_node_id)
            player.access_levels[action.target_node_id] = AccessLevel.USER
            
            points = target.point_value // 2
            player.score += points
            
            detected = random.random() < 0.25
            if detected:
                target.alert_level += 2
            
            return ActionResult(
                action=action,
                success=True,
                message=f"Pivoted to {target.name} from {source_node.name}.",
                points_gained=points,
                was_detected=detected
            )
        else:
            detected = random.random() < 0.4
            if detected:
                target.alert_level += 1
            
            return ActionResult(
                action=action,
                success=False,
                message=f"Failed to pivot to {target.name}.",
                was_detected=detected
            )
    
    def _execute_exfiltrate(
        self, action: GameAction, player: PlayerState,
        nodes: dict, edges: list
    ) -> ActionResult:
        """Execute EXFILTRATE action - steals data from node"""
        target = nodes[action.target_node_id]
        
        # Success based on access level
        access = player.access_levels.get(action.target_node_id, AccessLevel.NONE)
        base_chance = 0.5 + (access.value * 0.15)
        
        success = random.random() < base_chance
        
        if success:
            data_stolen = target.data_value
            player.data_stolen += data_stolen
            player.score += data_stolen * 2
            target.data_value = 0  # Data exfiltrated
            
            # High detection chance
            detected = random.random() < 0.6
            if detected:
                target.alert_level += 5
            
            return ActionResult(
                action=action,
                success=True,
                message=f"Exfiltrated {data_stolen} units of data from {target.name}.",
                points_gained=data_stolen * 2,
                was_detected=detected,
                state_changes={"data_stolen": data_stolen}
            )
        else:
            detected = random.random() < 0.7
            if detected:
                target.alert_level += 3
            
            return ActionResult(
                action=action,
                success=False,
                message=f"Failed to exfiltrate data from {target.name}.",
                was_detected=detected
            )
    
    def _execute_install_backdoor(
        self, action: GameAction, player: PlayerState,
        nodes: dict, edges: list
    ) -> ActionResult:
        """Execute INSTALL_BACKDOOR action - persistent access"""
        target = nodes[action.target_node_id]
        
        # High success rate with ROOT
        success = random.random() < 0.85
        
        if success:
            target.has_backdoor = True
            player.backdoors_installed.add(action.target_node_id)
            
            points = 25
            player.score += points
            
            # Low detection (stealthy)
            detected = random.random() < 0.1
            
            return ActionResult(
                action=action,
                success=True,
                message=f"Installed backdoor on {target.name}.",
                points_gained=points,
                was_detected=detected
            )
        else:
            detected = random.random() < 0.3
            if detected:
                target.alert_level += 2
            
            return ActionResult(
                action=action,
                success=False,
                message=f"Failed to install backdoor on {target.name}.",
                was_detected=detected
            )
    
    def _execute_deploy_malware(
        self, action: GameAction, player: PlayerState,
        nodes: dict, edges: list
    ) -> ActionResult:
        """Execute DEPLOY_MALWARE action - damages/destroys node"""
        target = nodes[action.target_node_id]
        
        success = random.random() < 0.9
        
        if success:
            # Damage the node
            damage = random.randint(30, 60)
            target.health -= damage
            target.status = NodeStatus.INFECTED
            
            # If health depleted, node goes offline
            if target.health <= 0:
                target.health = 0
                target.status = NodeStatus.OFFLINE
                player.controlled_nodes.discard(action.target_node_id)
            
            points = damage // 2
            player.score += points
            
            # Medium detection chance
            detected = random.random() < 0.4
            if detected:
                target.alert_level += 4
            
            return ActionResult(
                action=action,
                success=True,
                message=f"Deployed malware on {target.name}. Dealt {damage} damage.",
                points_gained=points,
                damage_dealt=damage,
                was_detected=detected
            )
        else:
            detected = random.random() < 0.5
            
            return ActionResult(
                action=action,
                success=False,
                message=f"Failed to deploy malware on {target.name}.",
                was_detected=detected
            )
    
    # =========================================================================
    # Defender Actions
    # =========================================================================
    
    def _execute_patch(
        self, action: GameAction, player: PlayerState,
        nodes: dict, edges: list
    ) -> ActionResult:
        """Execute PATCH action - removes vulnerabilities"""
        target = nodes[action.target_node_id]
        
        # Patch the most severe known vulnerability
        patched = None
        for vuln in target.vulnerabilities:
            if vuln.is_known and not vuln.is_patched:
                vuln.is_patched = True
                patched = vuln
                break
        
        if patched:
            points = int(patched.severity * 20)
            player.score += points
            
            return ActionResult(
                action=action,
                success=True,
                message=f"Patched {patched.name} on {target.name}.",
                points_gained=points
            )
        else:
            return ActionResult(
                action=action,
                success=False,
                message="No vulnerabilities to patch."
            )
    
    def _execute_isolate(
        self, action: GameAction, player: PlayerState,
        nodes: dict, edges: list
    ) -> ActionResult:
        """Execute ISOLATE action - disconnects node from network"""
        target = nodes[action.target_node_id]
        
        target.status = NodeStatus.ISOLATED
        
        # If attacker controlled this node, they lose access
        # (but backdoors persist)
        
        points = 15
        player.score += points
        
        return ActionResult(
            action=action,
            success=True,
            message=f"Isolated {target.name} from the network.",
            points_gained=points
        )
    
    def _execute_monitor(
        self, action: GameAction, player: PlayerState,
        nodes: dict, edges: list
    ) -> ActionResult:
        """Execute MONITOR action - increases detection on node"""
        target = nodes[action.target_node_id]
        
        target.monitoring_level += 1
        
        # Chance to detect existing compromise
        detected_compromise = False
        if target.status == NodeStatus.COMPROMISED:
            if random.random() < 0.3 * target.monitoring_level:
                detected_compromise = True
                target.alert_level += 5
        
        points = 10
        player.score += points
        
        msg = f"Monitoring {target.name}."
        if detected_compromise:
            msg += " ALERT: Compromise detected!"
        
        return ActionResult(
            action=action,
            success=True,
            message=msg,
            points_gained=points,
            discovered_info={"compromise_detected": detected_compromise}
        )
    
    def _execute_trace(
        self, action: GameAction, player: PlayerState,
        nodes: dict, edges: list
    ) -> ActionResult:
        """Execute TRACE action - traces attacker from compromised node"""
        target = nodes[action.target_node_id]
        
        # Chance to reveal attacker's other controlled nodes
        revealed = []
        for edge in edges:
            if edge.connects(action.target_node_id):
                other_id = edge.other_node(action.target_node_id)
                other = nodes.get(other_id)
                if other and other.status == NodeStatus.COMPROMISED:
                    if random.random() < 0.4:
                        revealed.append(other_id)
                        other.alert_level += 3
        
        points = len(revealed) * 15
        player.score += points
        
        return ActionResult(
            action=action,
            success=len(revealed) > 0,
            message=f"Traced from {target.name}. Found {len(revealed)} additional compromised nodes.",
            points_gained=points,
            discovered_info={"compromised_nodes": revealed}
        )
    
    def _execute_restore(
        self, action: GameAction, player: PlayerState,
        nodes: dict, edges: list
    ) -> ActionResult:
        """Execute RESTORE action - restores node from isolation/offline"""
        target = nodes[action.target_node_id]
        
        old_status = target.status
        target.status = NodeStatus.ONLINE
        target.health = target.max_health
        
        # Clear backdoor if restoring from offline (reimaged)
        if old_status == NodeStatus.OFFLINE:
            target.has_backdoor = False
        
        points = 20
        player.score += points
        
        return ActionResult(
            action=action,
            success=True,
            message=f"Restored {target.name} to online status.",
            points_gained=points
        )
    
    def _execute_deceive(
        self, action: GameAction, player: PlayerState,
        nodes: dict, edges: list
    ) -> ActionResult:
        """Execute DECEIVE action - converts node to honeypot"""
        target = nodes[action.target_node_id]
        
        # Convert to honeypot
        target.node_type = NodeType.HONEYPOT
        target.is_decoy = True
        
        points = 15
        player.score += points
        
        return ActionResult(
            action=action,
            success=True,
            message=f"Converted {target.name} to a honeypot.",
            points_gained=points
        )
    
    def _execute_end_turn(
        self, action: GameAction, player: PlayerState,
        nodes: dict, edges: list
    ) -> ActionResult:
        """Execute END_TURN action"""
        return ActionResult(
            action=action,
            success=True,
            message="Turn ended.",
            ends_turn=True
        )
