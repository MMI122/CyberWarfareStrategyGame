# =============================================================================
# Cyber Warfare Strategy Game - Network Topology Generator
# =============================================================================
"""
Generates various network topologies for the game.
Different topologies represent different organizational structures.
"""

import random
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .enums import NodeType, Difficulty
from .data_structures import NetworkNode, NetworkEdge, Vulnerability, GameConfig


@dataclass
class TopologyConfig:
    """Configuration for topology generation"""
    min_nodes: int = 20
    max_nodes: int = 30
    critical_count: int = 2
    firewall_count: int = 2
    server_ratio: float = 0.2
    router_ratio: float = 0.1
    connectivity: float = 0.3  # Average connections per node ratio
    subnet_count: int = 3
    seed: Optional[int] = None


class NetworkTopologyGenerator:
    """
    Generates network topologies for the game.
    
    Supports multiple topology types:
    - corporate: Standard corporate network with DMZ
    - government: Highly segmented government network
    - financial: Dense mesh network
    - random: Random topology
    """
    
    # Vulnerability templates
    VULNERABILITY_TEMPLATES = [
        {"name": "CVE-2024-0001", "severity": 0.9, "difficulty": 0.3, "desc": "Remote Code Execution"},
        {"name": "CVE-2024-0002", "severity": 0.7, "difficulty": 0.4, "desc": "SQL Injection"},
        {"name": "CVE-2024-0003", "severity": 0.6, "difficulty": 0.5, "desc": "Cross-Site Scripting"},
        {"name": "CVE-2024-0004", "severity": 0.8, "difficulty": 0.6, "desc": "Buffer Overflow"},
        {"name": "CVE-2024-0005", "severity": 0.5, "difficulty": 0.3, "desc": "Information Disclosure"},
        {"name": "CVE-2024-0006", "severity": 0.7, "difficulty": 0.7, "desc": "Privilege Escalation"},
        {"name": "CVE-2024-0007", "severity": 0.4, "difficulty": 0.2, "desc": "Denial of Service"},
        {"name": "CVE-2024-0008", "severity": 0.9, "difficulty": 0.8, "desc": "Authentication Bypass"},
    ]
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the generator with optional seed for reproducibility"""
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        self.vulnerability_counter = 0
    
    def generate(
        self, 
        topology_type: str = "corporate",
        difficulty: Difficulty = Difficulty.MEDIUM,
        config: Optional[TopologyConfig] = None
    ) -> Tuple[Dict[int, NetworkNode], List[NetworkEdge]]:
        """
        Generate a network topology.
        
        Args:
            topology_type: Type of topology (corporate, government, financial, random)
            difficulty: Game difficulty affecting size and complexity
            config: Optional custom configuration
            
        Returns:
            Tuple of (nodes dict, edges list)
        """
        # Get config based on difficulty
        if config is None:
            config = self._get_config_for_difficulty(difficulty)
        
        # Generate based on type
        generators = {
            "corporate": self._generate_corporate,
            "government": self._generate_government,
            "financial": self._generate_financial,
            "random": self._generate_random,
        }
        
        generator = generators.get(topology_type, self._generate_corporate)
        nodes, edges = generator(config)
        
        # Add vulnerabilities to nodes
        self._add_vulnerabilities(nodes, difficulty)
        
        # Mark entry point (internet-facing node)
        self._mark_entry_point(nodes, edges)
        
        return nodes, edges
    
    def _get_config_for_difficulty(self, difficulty: Difficulty) -> TopologyConfig:
        """Get topology configuration based on difficulty"""
        min_nodes, max_nodes = difficulty.node_count_range
        critical_count = difficulty.critical_count
        
        configs = {
            Difficulty.EASY: TopologyConfig(
                min_nodes=min_nodes, max_nodes=max_nodes,
                critical_count=critical_count, firewall_count=1,
                server_ratio=0.15, router_ratio=0.1,
                connectivity=0.25, subnet_count=2
            ),
            Difficulty.MEDIUM: TopologyConfig(
                min_nodes=min_nodes, max_nodes=max_nodes,
                critical_count=critical_count, firewall_count=2,
                server_ratio=0.2, router_ratio=0.12,
                connectivity=0.3, subnet_count=3
            ),
            Difficulty.HARD: TopologyConfig(
                min_nodes=min_nodes, max_nodes=max_nodes,
                critical_count=critical_count, firewall_count=3,
                server_ratio=0.25, router_ratio=0.15,
                connectivity=0.35, subnet_count=4
            ),
            Difficulty.EXPERT: TopologyConfig(
                min_nodes=min_nodes, max_nodes=max_nodes,
                critical_count=critical_count, firewall_count=4,
                server_ratio=0.3, router_ratio=0.15,
                connectivity=0.4, subnet_count=5
            ),
        }
        return configs.get(difficulty, configs[Difficulty.MEDIUM])
    
    def _generate_corporate(
        self, 
        config: TopologyConfig
    ) -> Tuple[Dict[int, NetworkNode], List[NetworkEdge]]:
        """
        Generate a corporate network topology.
        
        Structure:
        - Internet → DMZ Firewall → DMZ (web servers)
        - DMZ → Core Router → Internal segments
        - Each segment has workstations and servers
        - Critical assets in protected segment
        """
        nodes: Dict[int, NetworkNode] = {}
        edges: List[NetworkEdge] = []
        node_id = 0
        
        # Determine node count
        total_nodes = random.randint(config.min_nodes, config.max_nodes)
        
        # Create DMZ firewall (entry point)
        dmz_firewall = self._create_node(
            node_id, NodeType.FIREWALL, 
            position=(0.0, 0.9, 0.0),
            subnet="dmz", name="DMZ_Firewall"
        )
        dmz_firewall.visible_to_attacker = True  # Entry point is visible
        nodes[node_id] = dmz_firewall
        node_id += 1
        
        # Create DMZ servers (web-facing)
        dmz_server_count = max(2, total_nodes // 10)
        dmz_servers = []
        for i in range(dmz_server_count):
            x = -0.5 + i * (1.0 / dmz_server_count)
            server = self._create_node(
                node_id, NodeType.SERVER,
                position=(x, 0.7, 0.0),
                subnet="dmz", name=f"Web_Server_{i+1}"
            )
            nodes[node_id] = server
            dmz_servers.append(node_id)
            edges.append(NetworkEdge(0, node_id))  # Connect to DMZ firewall
            node_id += 1
        
        # Create core router
        core_router = self._create_node(
            node_id, NodeType.ROUTER,
            position=(0.0, 0.5, 0.0),
            subnet="core", name="Core_Router"
        )
        nodes[node_id] = core_router
        core_router_id = node_id
        # Connect core router to DMZ firewall
        edges.append(NetworkEdge(0, core_router_id, requires_credentials=True))
        node_id += 1
        
        # Create internal firewall
        internal_firewall = self._create_node(
            node_id, NodeType.FIREWALL,
            position=(0.0, 0.3, 0.0),
            subnet="internal", name="Internal_Firewall"
        )
        nodes[node_id] = internal_firewall
        internal_fw_id = node_id
        edges.append(NetworkEdge(core_router_id, internal_fw_id))
        node_id += 1
        
        # Create subnets
        remaining_nodes = total_nodes - node_id
        nodes_per_subnet = remaining_nodes // config.subnet_count
        
        subnet_routers = []
        for subnet_idx in range(config.subnet_count):
            subnet_name = f"subnet_{subnet_idx + 1}"
            
            # Subnet router
            x_offset = -0.8 + subnet_idx * (1.6 / (config.subnet_count - 1)) if config.subnet_count > 1 else 0
            router = self._create_node(
                node_id, NodeType.ROUTER,
                position=(x_offset, 0.1, 0.0),
                subnet=subnet_name, name=f"Router_{subnet_name}"
            )
            nodes[node_id] = router
            subnet_routers.append(node_id)
            edges.append(NetworkEdge(internal_fw_id, node_id))
            node_id += 1
            
            # Add critical node to last subnet
            if subnet_idx == config.subnet_count - 1:
                for crit_idx in range(config.critical_count):
                    critical = self._create_node(
                        node_id, NodeType.CRITICAL,
                        position=(x_offset + 0.1 * crit_idx, -0.3, 0.0),
                        subnet=subnet_name, name=f"Critical_Database_{crit_idx + 1}"
                    )
                    critical.data_value = 50  # Can exfiltrate data
                    nodes[node_id] = critical
                    edges.append(NetworkEdge(subnet_routers[-1], node_id, requires_credentials=True))
                    node_id += 1
            
            # Add workstations and servers to subnet
            subnet_nodes = nodes_per_subnet - 1  # -1 for router
            server_count = max(1, int(subnet_nodes * config.server_ratio))
            workstation_count = subnet_nodes - server_count
            
            for i in range(server_count):
                server = self._create_node(
                    node_id, NodeType.SERVER,
                    position=(x_offset - 0.15 + random.uniform(-0.1, 0.1), -0.1 - i * 0.15, random.uniform(-0.1, 0.1)),
                    subnet=subnet_name, name=f"Server_{subnet_name}_{i+1}"
                )
                server.data_value = random.randint(10, 30)
                nodes[node_id] = server
                edges.append(NetworkEdge(subnet_routers[-1], node_id))
                node_id += 1
            
            for i in range(workstation_count):
                ws = self._create_node(
                    node_id, NodeType.WORKSTATION,
                    position=(x_offset + 0.15 + random.uniform(-0.1, 0.1), -0.1 - i * 0.1, random.uniform(-0.1, 0.1)),
                    subnet=subnet_name, name=f"Workstation_{subnet_name}_{i+1}"
                )
                nodes[node_id] = ws
                edges.append(NetworkEdge(subnet_routers[-1], node_id))
                node_id += 1
        
        # Add some cross-connections between subnets
        for i in range(len(subnet_routers) - 1):
            if random.random() < config.connectivity:
                edges.append(NetworkEdge(subnet_routers[i], subnet_routers[i + 1]))
        
        # Update adjacency lists
        self._update_adjacency_lists(nodes, edges)
        
        return nodes, edges
    
    def _generate_government(
        self, 
        config: TopologyConfig
    ) -> Tuple[Dict[int, NetworkNode], List[NetworkEdge]]:
        """
        Generate a government network topology.
        Highly segmented with multiple security zones.
        """
        # Similar to corporate but with more firewalls and segmentation
        nodes, edges = self._generate_corporate(config)
        
        # Add extra firewalls between segments
        # (Implementation simplified for brevity)
        
        return nodes, edges
    
    def _generate_financial(
        self, 
        config: TopologyConfig
    ) -> Tuple[Dict[int, NetworkNode], List[NetworkEdge]]:
        """
        Generate a financial network topology.
        Dense mesh with multiple redundant paths.
        """
        nodes: Dict[int, NetworkNode] = {}
        edges: List[NetworkEdge] = []
        
        total_nodes = random.randint(config.min_nodes, config.max_nodes)
        
        # Create nodes in a grid-like pattern
        grid_size = int(math.ceil(math.sqrt(total_nodes)))
        
        node_id = 0
        node_positions = []
        
        for i in range(total_nodes):
            row = i // grid_size
            col = i % grid_size
            
            x = -0.8 + (col / (grid_size - 1)) * 1.6 if grid_size > 1 else 0
            y = 0.8 - (row / (grid_size - 1)) * 1.6 if grid_size > 1 else 0
            
            # Determine node type based on position
            if i == 0:
                node_type = NodeType.FIREWALL
                name = "Gateway_Firewall"
            elif i < config.critical_count + 1:
                node_type = NodeType.CRITICAL
                name = f"Trading_System_{i}"
            elif random.random() < config.server_ratio:
                node_type = NodeType.SERVER
                name = f"Server_{i}"
            elif random.random() < config.router_ratio:
                node_type = NodeType.ROUTER
                name = f"Router_{i}"
            else:
                node_type = NodeType.WORKSTATION
                name = f"Terminal_{i}"
            
            node = self._create_node(
                node_id, node_type,
                position=(x, y, random.uniform(-0.1, 0.1)),
                subnet=f"zone_{row}", name=name
            )
            if node_type == NodeType.CRITICAL:
                node.data_value = 100
            nodes[node_id] = node
            node_positions.append((x, y))
            node_id += 1
        
        # Create mesh connections (connect to nearby nodes)
        for i in range(total_nodes):
            for j in range(i + 1, total_nodes):
                dist = math.sqrt(
                    (node_positions[i][0] - node_positions[j][0])**2 +
                    (node_positions[i][1] - node_positions[j][1])**2
                )
                # Connect if close enough
                if dist < 0.5 and random.random() < config.connectivity:
                    edges.append(NetworkEdge(i, j))
        
        # Ensure connectivity
        self._ensure_connectivity(nodes, edges)
        
        # Mark entry point
        nodes[0].visible_to_attacker = True
        
        # Update adjacency lists
        self._update_adjacency_lists(nodes, edges)
        
        return nodes, edges
    
    def _generate_random(
        self, 
        config: TopologyConfig
    ) -> Tuple[Dict[int, NetworkNode], List[NetworkEdge]]:
        """Generate a random network topology"""
        nodes: Dict[int, NetworkNode] = {}
        edges: List[NetworkEdge] = []
        
        total_nodes = random.randint(config.min_nodes, config.max_nodes)
        
        # Create nodes
        for i in range(total_nodes):
            # Random position in 3D space
            x = random.uniform(-0.9, 0.9)
            y = random.uniform(-0.9, 0.9)
            z = random.uniform(-0.2, 0.2)
            
            # Random node type (weighted)
            roll = random.random()
            if i == 0:
                node_type = NodeType.FIREWALL
            elif i <= config.critical_count:
                node_type = NodeType.CRITICAL
            elif roll < config.server_ratio:
                node_type = NodeType.SERVER
            elif roll < config.server_ratio + config.router_ratio:
                node_type = NodeType.ROUTER
            elif roll < config.server_ratio + config.router_ratio + 0.05:
                node_type = NodeType.FIREWALL
            else:
                node_type = NodeType.WORKSTATION
            
            node = self._create_node(
                i, node_type,
                position=(x, y, z),
                subnet=f"subnet_{i % config.subnet_count}",
                name=f"{node_type.name}_{i}"
            )
            if node_type == NodeType.CRITICAL:
                node.data_value = random.randint(40, 80)
            nodes[i] = node
        
        # Create random edges
        for i in range(total_nodes):
            num_connections = int(total_nodes * config.connectivity)
            for _ in range(random.randint(1, num_connections)):
                j = random.randint(0, total_nodes - 1)
                if i != j:
                    # Check if edge already exists
                    key = (min(i, j), max(i, j))
                    if not any(e.key == key for e in edges):
                        edges.append(NetworkEdge(i, j))
        
        # Ensure connectivity
        self._ensure_connectivity(nodes, edges)
        
        # Mark entry point
        nodes[0].visible_to_attacker = True
        
        # Update adjacency lists
        self._update_adjacency_lists(nodes, edges)
        
        return nodes, edges
    
    def _create_node(
        self,
        node_id: int,
        node_type: NodeType,
        position: Tuple[float, float, float],
        subnet: str,
        name: str
    ) -> NetworkNode:
        """Create a new network node with default values"""
        return NetworkNode(
            id=node_id,
            node_type=node_type,
            position=position,
            health=node_type.default_hp,
            max_health=node_type.default_hp,
            point_value=node_type.default_value,
            subnet=subnet,
            name=name
        )
    
    def _add_vulnerabilities(self, nodes: Dict[int, NetworkNode], difficulty: Difficulty):
        """Add vulnerabilities to nodes based on their type and difficulty"""
        vuln_count_by_difficulty = {
            Difficulty.EASY: (1, 3),
            Difficulty.MEDIUM: (1, 2),
            Difficulty.HARD: (0, 2),
            Difficulty.EXPERT: (0, 1),
        }
        min_vuln, max_vuln = vuln_count_by_difficulty.get(difficulty, (1, 2))
        
        for node in nodes.values():
            # Determine number of vulnerabilities based on node type
            base_vuln_chance = node.node_type.vulnerability_level
            num_vulns = random.randint(min_vuln, max_vuln)
            
            # Firewalls have fewer vulnerabilities
            if node.node_type == NodeType.FIREWALL:
                num_vulns = max(0, num_vulns - 1)
            
            for _ in range(num_vulns):
                if random.random() < base_vuln_chance:
                    vuln = self._create_vulnerability()
                    node.vulnerabilities.append(vuln)
    
    def _create_vulnerability(self) -> Vulnerability:
        """Create a random vulnerability from templates"""
        template = random.choice(self.VULNERABILITY_TEMPLATES)
        self.vulnerability_counter += 1
        
        return Vulnerability(
            id=f"VULN-{self.vulnerability_counter:04d}",
            name=template["name"],
            severity=template["severity"] * random.uniform(0.8, 1.2),
            exploit_difficulty=template["difficulty"] * random.uniform(0.8, 1.2),
            is_known=random.random() < 0.3  # 30% chance defender knows about it
        )
    
    def _mark_entry_point(self, nodes: Dict[int, NetworkNode], edges: List[NetworkEdge]):
        """Mark the entry point node(s) that attackers can initially access"""
        # Find node 0 (usually the gateway/DMZ firewall)
        if 0 in nodes:
            nodes[0].visible_to_attacker = True
        
        # Also mark any DMZ servers as potentially visible
        for node in nodes.values():
            if node.subnet == "dmz":
                node.visible_to_attacker = True
    
    def _ensure_connectivity(self, nodes: Dict[int, NetworkNode], edges: List[NetworkEdge]):
        """Ensure all nodes are connected (no isolated nodes)"""
        if not nodes:
            return
        
        # Simple BFS to find connected components
        visited = set()
        queue = [0]
        visited.add(0)
        
        while queue:
            current = queue.pop(0)
            for edge in edges:
                if edge.connects(current):
                    other = edge.other_node(current)
                    if other not in visited:
                        visited.add(other)
                        queue.append(other)
        
        # Connect any unvisited nodes
        for node_id in nodes:
            if node_id not in visited:
                # Connect to a random visited node
                target = random.choice(list(visited))
                edges.append(NetworkEdge(node_id, target))
                visited.add(node_id)
    
    def _update_adjacency_lists(self, nodes: Dict[int, NetworkNode], edges: List[NetworkEdge]):
        """Update the adjacency lists in all nodes based on edges"""
        # Clear existing adjacencies
        for node in nodes.values():
            node.adjacent_nodes = []
        
        # Build adjacency from edges
        for edge in edges:
            if edge.is_active:
                if edge.source_id in nodes:
                    nodes[edge.source_id].adjacent_nodes.append(edge.target_id)
                if edge.target_id in nodes:
                    nodes[edge.target_id].adjacent_nodes.append(edge.source_id)


# =============================================================================
# Convenience function
# =============================================================================

def generate_topology(
    topology_type: str = "corporate",
    difficulty: Difficulty = Difficulty.MEDIUM,
    seed: Optional[int] = None
) -> Tuple[Dict[int, NetworkNode], List[NetworkEdge]]:
    """
    Convenience function to generate a topology.
    
    Args:
        topology_type: Type of topology
        difficulty: Game difficulty
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (nodes dict, edges list)
    """
    generator = NetworkTopologyGenerator(seed=seed)
    return generator.generate(topology_type, difficulty)
