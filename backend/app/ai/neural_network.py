"""
Neural Network Architectures for Deep RL Agent

This module contains the neural network architectures used by the Deep RL agent:
- PolicyNetwork: Outputs action probabilities (actor)
- ValueNetwork: Outputs state value estimates (critic)
- ActorCritic: Combined architecture for PPO

Research Features:
- Graph Neural Network for network topology encoding
- Attention mechanisms for strategic focus
- Residual connections for training stability
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import math

# Try to import torch, fall back to numpy-only implementation
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class NetworkConfig:
    """Configuration for neural network architecture"""
    input_dim: int = 256  # State encoding dimension
    hidden_dim: int = 256  # Hidden layer dimension
    num_layers: int = 3  # Number of hidden layers
    num_heads: int = 4  # Attention heads
    dropout: float = 0.1  # Dropout rate
    use_attention: bool = True
    use_residual: bool = True


class ActivationFunction(Enum):
    """Supported activation functions"""
    RELU = "relu"
    TANH = "tanh"
    GELU = "gelu"
    LEAKY_RELU = "leaky_relu"


# ============================================================================
# NumPy-only Neural Network Implementation (No PyTorch dependency)
# ============================================================================

class NumpyLinear:
    """Linear layer implemented in pure NumPy"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        
        # Xavier/Glorot initialization
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.weight = np.random.uniform(-limit, limit, (out_features, in_features))
        
        if bias:
            self.bias = np.zeros(out_features)
        else:
            self.bias = None
        
        # For gradient tracking
        self.weight_grad = np.zeros_like(self.weight)
        self.bias_grad = np.zeros_like(self.bias) if bias else None
        self._input_cache = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: y = xW^T + b"""
        self._input_cache = x
        output = np.dot(x, self.weight.T)
        if self.bias is not None:
            output += self.bias
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass for gradient computation"""
        if self._input_cache is None:
            raise RuntimeError("Must call forward before backward")
        
        # Compute gradients
        if len(grad_output.shape) == 1:
            grad_output = grad_output.reshape(1, -1)
            input_2d = self._input_cache.reshape(1, -1)
        else:
            input_2d = self._input_cache
        
        self.weight_grad = np.dot(grad_output.T, input_2d)
        if self.bias is not None:
            self.bias_grad = np.sum(grad_output, axis=0)
        
        # Gradient w.r.t. input
        grad_input = np.dot(grad_output, self.weight)
        return grad_input.squeeze()
    
    def parameters(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return parameters and their gradients"""
        params = [(self.weight, self.weight_grad)]
        if self.bias is not None:
            params.append((self.bias, self.bias_grad))
        return params


class NumpyLayerNorm:
    """Layer normalization implemented in pure NumPy"""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = np.ones(normalized_shape)
        self.beta = np.zeros(normalized_shape)
        self.gamma_grad = np.zeros_like(self.gamma)
        self.beta_grad = np.zeros_like(self.beta)
        self._cache = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        self._cache = (x, x_norm, mean, var)
        return self.gamma * x_norm + self.beta
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass"""
        x, x_norm, mean, var = self._cache
        N = x.shape[-1]
        
        self.gamma_grad = np.sum(grad_output * x_norm, axis=0)
        self.beta_grad = np.sum(grad_output, axis=0)
        
        dx_norm = grad_output * self.gamma
        std_inv = 1.0 / np.sqrt(var + self.eps)
        
        dx = (1.0 / N) * std_inv * (
            N * dx_norm - np.sum(dx_norm, axis=-1, keepdims=True) -
            x_norm * np.sum(dx_norm * x_norm, axis=-1, keepdims=True)
        )
        return dx
    
    def parameters(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        return [(self.gamma, self.gamma_grad), (self.beta, self.beta_grad)]


class NumpyActivation:
    """Activation functions in pure NumPy"""
    
    def __init__(self, activation: ActivationFunction = ActivationFunction.RELU):
        self.activation = activation
        self._input_cache = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input_cache = x
        
        if self.activation == ActivationFunction.RELU:
            return np.maximum(0, x)
        elif self.activation == ActivationFunction.TANH:
            return np.tanh(x)
        elif self.activation == ActivationFunction.GELU:
            # Approximate GELU
            return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        elif self.activation == ActivationFunction.LEAKY_RELU:
            return np.where(x > 0, x, 0.01 * x)
        return x
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        x = self._input_cache
        
        if self.activation == ActivationFunction.RELU:
            return grad_output * (x > 0)
        elif self.activation == ActivationFunction.TANH:
            return grad_output * (1 - np.tanh(x)**2)
        elif self.activation == ActivationFunction.GELU:
            # Approximate GELU gradient
            cdf = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
            pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
            return grad_output * (cdf + x * pdf)
        elif self.activation == ActivationFunction.LEAKY_RELU:
            return grad_output * np.where(x > 0, 1, 0.01)
        return grad_output


class NumpyDropout:
    """Dropout layer in pure NumPy"""
    
    def __init__(self, p: float = 0.1):
        self.p = p
        self.training = True
        self._mask = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.p == 0:
            return x
        
        self._mask = (np.random.random(x.shape) > self.p) / (1 - self.p)
        return x * self._mask
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if not self.training or self.p == 0:
            return grad_output
        return grad_output * self._mask


class NumpySoftmax:
    """Softmax layer in pure NumPy"""
    
    def __init__(self, dim: int = -1):
        self.dim = dim
        self._output_cache = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Numerical stability
        x_max = np.max(x, axis=self.dim, keepdims=True)
        exp_x = np.exp(x - x_max)
        self._output_cache = exp_x / np.sum(exp_x, axis=self.dim, keepdims=True)
        return self._output_cache
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Softmax backward pass"""
        s = self._output_cache
        # Jacobian-vector product for softmax
        return s * (grad_output - np.sum(grad_output * s, axis=self.dim, keepdims=True))


class NumpyMultiHeadAttention:
    """Multi-head self-attention in pure NumPy"""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / np.sqrt(self.head_dim)
        
        # Linear projections
        self.q_proj = NumpyLinear(embed_dim, embed_dim)
        self.k_proj = NumpyLinear(embed_dim, embed_dim)
        self.v_proj = NumpyLinear(embed_dim, embed_dim)
        self.out_proj = NumpyLinear(embed_dim, embed_dim)
        
        self.dropout = NumpyDropout(dropout)
        self.softmax = NumpySoftmax(dim=-1)
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass for multi-head attention
        x: (batch_size, seq_len, embed_dim) or (seq_len, embed_dim)
        """
        # Handle 2D input
        if len(x.shape) == 2:
            x = x.reshape(1, *x.shape)
        
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        q = self.q_proj.forward(x)
        k = self.k_proj.forward(x)
        v = self.v_proj.forward(x)
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Attention scores
        attn_scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        
        if mask is not None:
            attn_scores = attn_scores + mask * -1e9
        
        attn_probs = self.softmax.forward(attn_scores)
        attn_probs = self.dropout.forward(attn_probs)
        
        # Apply attention to values
        attn_output = np.matmul(attn_probs, v)
        
        # Reshape back
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        
        # Output projection
        output = self.out_proj.forward(attn_output)
        
        return output.squeeze() if batch_size == 1 else output
    
    def parameters(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        params = []
        params.extend(self.q_proj.parameters())
        params.extend(self.k_proj.parameters())
        params.extend(self.v_proj.parameters())
        params.extend(self.out_proj.parameters())
        return params


class NumpyMLP:
    """Multi-layer perceptron in pure NumPy"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        activation: ActivationFunction = ActivationFunction.RELU,
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        self.layers = []
        self.activations = []
        self.layer_norms = [] if use_layer_norm else None
        self.dropouts = []
        
        # Build layers
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        
        for i in range(len(dims) - 1):
            self.layers.append(NumpyLinear(dims[i], dims[i + 1]))
            
            # No activation after last layer
            if i < len(dims) - 2:
                self.activations.append(NumpyActivation(activation))
                if use_layer_norm:
                    self.layer_norms.append(NumpyLayerNorm(dims[i + 1]))
                self.dropouts.append(NumpyDropout(dropout))
        
        self.training = True
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            
            if i < len(self.activations):
                if self.layer_norms is not None:
                    x = self.layer_norms[i].forward(x)
                x = self.activations[i].forward(x)
                if self.training:
                    x = self.dropouts[i].forward(x)
        
        return x
    
    def parameters(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        if self.layer_norms is not None:
            for ln in self.layer_norms:
                params.extend(ln.parameters())
        return params
    
    def train(self):
        self.training = True
        for dropout in self.dropouts:
            dropout.training = True
    
    def eval(self):
        self.training = False
        for dropout in self.dropouts:
            dropout.training = False


class NumpyResidualBlock:
    """Residual block for deeper networks"""
    
    def __init__(self, dim: int, dropout: float = 0.1):
        self.linear1 = NumpyLinear(dim, dim)
        self.linear2 = NumpyLinear(dim, dim)
        self.layer_norm1 = NumpyLayerNorm(dim)
        self.layer_norm2 = NumpyLayerNorm(dim)
        self.activation = NumpyActivation(ActivationFunction.RELU)
        self.dropout = NumpyDropout(dropout)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        residual = x
        
        x = self.layer_norm1.forward(x)
        x = self.linear1.forward(x)
        x = self.activation.forward(x)
        x = self.dropout.forward(x)
        
        x = self.layer_norm2.forward(x)
        x = self.linear2.forward(x)
        x = self.dropout.forward(x)
        
        return x + residual
    
    def parameters(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        params = []
        params.extend(self.linear1.parameters())
        params.extend(self.linear2.parameters())
        params.extend(self.layer_norm1.parameters())
        params.extend(self.layer_norm2.parameters())
        return params


# ============================================================================
# High-Level Neural Network Modules
# ============================================================================

class StateEncoder:
    """
    Encodes game state into a fixed-size vector representation.
    
    Features encoded:
    - Node features (type, status, vulnerabilities, access level)
    - Edge features (connectivity, firewalls)
    - Player features (compromised nodes, resources, progress)
    - Global game features (turn, phase, scores)
    """
    
    def __init__(self, config: NetworkConfig):
        self.config = config
        
        # Node encoder
        self.node_encoder = NumpyMLP(
            input_dim=32,  # Per-node feature dimension
            hidden_dim=config.hidden_dim,
            output_dim=config.hidden_dim // 2,
            num_layers=2
        )
        
        # Edge encoder  
        self.edge_encoder = NumpyMLP(
            input_dim=8,  # Per-edge feature dimension
            hidden_dim=config.hidden_dim // 2,
            output_dim=config.hidden_dim // 4,
            num_layers=2
        )
        
        # Global state encoder
        self.global_encoder = NumpyMLP(
            input_dim=32,  # Global game features
            hidden_dim=config.hidden_dim,
            output_dim=config.hidden_dim // 2,
            num_layers=2
        )
        
        # Attention for node aggregation
        if config.use_attention:
            self.attention = NumpyMultiHeadAttention(
                embed_dim=config.hidden_dim // 2,
                num_heads=config.num_heads,
                dropout=config.dropout
            )
        
        # Final projection
        self.output_proj = NumpyLinear(
            config.hidden_dim + config.hidden_dim // 4,
            config.hidden_dim
        )
    
    def encode_nodes(self, state) -> np.ndarray:
        """Encode all nodes in the network"""
        node_features = []
        
        # Access nodes directly from state (state.nodes, not state.network.nodes)
        nodes = state.nodes if hasattr(state, 'nodes') else {}
        
        for node in nodes.values():
            features = [
                node.node_type.value,  # Node type
                node.status.value,  # Node status
                node.access_level.value,  # Current access level
                len(node.vulnerabilities),  # Number of vulnerabilities
                sum(1 for v in node.vulnerabilities if not v.is_patched),  # Unpatched vulns
                float(getattr(node, 'is_compromised', False) or getattr(node, 'compromised', False)),
                float(getattr(node, 'scanned', False) or getattr(node, 'scanned_by_attacker', False)),  # Is scanned
                float(getattr(node, 'is_decoy', False)),  # Is decoy
                getattr(node, 'importance', 1.0),  # Importance value
                getattr(node, 'defense_level', 1.0),  # Defense level
                getattr(node, 'alert_level', 0),  # Alert level
                getattr(node, 'monitoring_level', 0),  # Monitoring level
            ]
            
            # Pad to fixed size
            while len(features) < 32:
                features.append(0.0)
            
            node_features.append(features[:32])
        
        if not node_features:
            return np.zeros((1, 32), dtype=np.float32)
        
        return np.array(node_features, dtype=np.float32)
    
    def encode_edges(self, state) -> np.ndarray:
        """Encode network edges"""
        edge_features = []
        
        # Access edges directly from state
        edges = state.edges if hasattr(state, 'edges') else []
        
        for edge in edges:
            features = [
                float(getattr(edge, 'has_firewall', False)),
                float(getattr(edge, 'is_encrypted', False)),
                getattr(edge, 'latency', 1.0),
                getattr(edge, 'bandwidth', 1.0),
            ]
            
            while len(features) < 8:
                features.append(0.0)
            
            edge_features.append(features[:8])
        
        if not edge_features:
            return np.zeros((1, 8), dtype=np.float32)
        
        return np.array(edge_features, dtype=np.float32)
    
    def encode_global(self, state) -> np.ndarray:
        """Encode global game state"""
        # Use attacker/defender attributes (state.attacker, state.defender)
        attacker = getattr(state, 'attacker', None) or getattr(state, 'attacker_state', None)
        defender = getattr(state, 'defender', None) or getattr(state, 'defender_state', None)
        
        # Get turn info
        current_turn = getattr(state, 'turn_number', 0) or getattr(state, 'current_turn', 0)
        phase = getattr(state, 'phase', None) or getattr(state, 'current_phase', None)
        phase_value = phase.value if phase else 0
        
        # Max turns from config
        config = getattr(state, 'config', None)
        max_turns = config.max_turns if config else 50
        
        # Get attacker node counts (use controlled_nodes or compromised_nodes)
        attacker_controlled = len(getattr(attacker, 'controlled_nodes', set()) or 
                                  getattr(attacker, 'compromised_nodes', set()) or set()) if attacker else 0
        attacker_known = len(getattr(attacker, 'known_nodes', set()) or set()) if attacker else 0
        
        features = [
            current_turn,
            phase_value,
            getattr(attacker, 'score', 0) if attacker else 0,
            getattr(defender, 'score', 0) if defender else 0,
            getattr(attacker, 'action_points', 50) if attacker else 50,
            getattr(defender, 'action_points', 50) if defender else 50,
            attacker_controlled,
            attacker_known,
            len(getattr(attacker, 'backdoors_installed', set()) or set()) if attacker else 0,
            getattr(attacker, 'data_stolen', 0) or getattr(attacker, 'exfiltrated_data', 0) if attacker else 0,
            float(max_turns - current_turn),  # Turns remaining
        ]
        
        # Pad to fixed size
        while len(features) < 32:
            features.append(0.0)
        
        return np.array(features[:32], dtype=np.float32)
    
    def forward(self, state) -> np.ndarray:
        """
        Encode full game state into fixed-size vector
        """
        # Encode components
        node_features = self.encode_nodes(state)
        edge_features = self.encode_edges(state)
        global_features = self.encode_global(state)
        
        # Process through networks
        node_encoded = self.node_encoder.forward(node_features)  # (num_nodes, hidden/2)
        edge_encoded = self.edge_encoder.forward(edge_features)  # (num_edges, hidden/4)
        global_encoded = self.global_encoder.forward(global_features)  # (hidden/2,)
        
        # Aggregate nodes with attention
        if hasattr(self, 'attention') and self.attention is not None:
            node_aggregated = self.attention.forward(node_encoded)
            node_aggregated = np.mean(node_aggregated, axis=0)  # (hidden/2,)
        else:
            node_aggregated = np.mean(node_encoded, axis=0)
        
        # Aggregate edges
        edge_aggregated = np.mean(edge_encoded, axis=0)  # (hidden/4,)
        
        # Combine all features
        combined = np.concatenate([
            node_aggregated,
            global_encoded,
            edge_aggregated
        ])
        
        # Final projection
        state_encoding = self.output_proj.forward(combined)
        
        return state_encoding
    
    def parameters(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        params = []
        params.extend(self.node_encoder.parameters())
        params.extend(self.edge_encoder.parameters())
        params.extend(self.global_encoder.parameters())
        if hasattr(self, 'attention') and self.attention is not None:
            params.extend(self.attention.parameters())
        params.extend(self.output_proj.parameters())
        return params


class PolicyHead:
    """
    Policy network head for action probability distribution.
    
    Outputs logits for each possible action type and target.
    """
    
    def __init__(self, config: NetworkConfig, num_action_types: int, max_targets: int):
        self.config = config
        self.num_action_types = num_action_types
        self.max_targets = max_targets
        
        # Action type head
        self.action_type_head = NumpyMLP(
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim // 2,
            output_dim=num_action_types,
            num_layers=2
        )
        
        # Target head (which node to act on)
        self.target_head = NumpyMLP(
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim // 2,
            output_dim=max_targets,
            num_layers=2
        )
        
        self.softmax = NumpySoftmax(dim=-1)
    
    def forward(self, state_encoding: np.ndarray, valid_action_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute action probabilities
        
        Returns:
            action_type_probs: (num_action_types,) probability distribution
            target_probs: (max_targets,) probability distribution
        """
        # Get logits
        action_type_logits = self.action_type_head.forward(state_encoding)
        target_logits = self.target_head.forward(state_encoding)
        
        # Apply mask (set invalid actions to very negative)
        # Assume mask has shape (num_action_types + max_targets,)
        if valid_action_mask is not None:
            action_mask = valid_action_mask[:self.num_action_types]
            target_mask = valid_action_mask[self.num_action_types:self.num_action_types + self.max_targets]
            
            action_type_logits = action_type_logits + (1 - action_mask) * -1e9
            target_logits = target_logits + (1 - target_mask) * -1e9
        
        # Convert to probabilities
        action_type_probs = self.softmax.forward(action_type_logits)
        target_probs = self.softmax.forward(target_logits)
        
        return action_type_probs, target_probs
    
    def parameters(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        params = []
        params.extend(self.action_type_head.parameters())
        params.extend(self.target_head.parameters())
        return params


class ValueHead:
    """
    Value network head for state value estimation.
    
    Outputs a scalar value representing expected returns from state.
    """
    
    def __init__(self, config: NetworkConfig):
        self.config = config
        
        self.value_net = NumpyMLP(
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim // 2,
            output_dim=1,
            num_layers=2
        )
    
    def forward(self, state_encoding: np.ndarray) -> float:
        """
        Estimate value of state
        
        Returns:
            value: Scalar value estimate
        """
        value = self.value_net.forward(state_encoding)
        return float(value.item() if hasattr(value, 'item') else value)
    
    def parameters(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        return self.value_net.parameters()


class ActorCriticNetwork:
    """
    Combined Actor-Critic network for PPO.
    
    Shares state encoding between policy (actor) and value (critic) heads.
    """
    
    def __init__(
        self,
        config: Optional[NetworkConfig] = None,
        num_action_types: int = 12,
        max_targets: int = 50
    ):
        self.config = config or NetworkConfig()
        
        # Shared encoder
        self.encoder = StateEncoder(self.config)
        
        # Residual blocks for deeper processing
        self.residual_blocks = []
        if self.config.use_residual:
            for _ in range(2):
                self.residual_blocks.append(
                    NumpyResidualBlock(self.config.hidden_dim, self.config.dropout)
                )
        
        # Separate heads
        self.policy_head = PolicyHead(self.config, num_action_types, max_targets)
        self.value_head = ValueHead(self.config)
        
        self.training = True
    
    def forward(self, state, valid_action_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Forward pass through actor-critic network
        
        Returns:
            action_type_probs: Action type probability distribution
            target_probs: Target probability distribution
            value: State value estimate
        """
        # Encode state
        state_encoding = self.encoder.forward(state)
        
        # Apply residual blocks
        for block in self.residual_blocks:
            state_encoding = block.forward(state_encoding)
        
        # Get policy outputs
        action_type_probs, target_probs = self.policy_head.forward(state_encoding, valid_action_mask)
        
        # Get value estimate
        value = self.value_head.forward(state_encoding)
        
        return action_type_probs, target_probs, value
    
    def get_action_probs(self, state, valid_action_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Get only action probabilities (for inference)"""
        action_type_probs, target_probs, _ = self.forward(state, valid_action_mask)
        return action_type_probs, target_probs
    
    def get_value(self, state) -> float:
        """Get only value estimate"""
        state_encoding = self.encoder.forward(state)
        for block in self.residual_blocks:
            state_encoding = block.forward(state_encoding)
        return self.value_head.forward(state_encoding)
    
    def parameters(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get all trainable parameters"""
        params = []
        params.extend(self.encoder.parameters())
        for block in self.residual_blocks:
            params.extend(block.parameters())
        params.extend(self.policy_head.parameters())
        params.extend(self.value_head.parameters())
        return params
    
    def train(self):
        """Set to training mode"""
        self.training = True
        self.encoder.node_encoder.train()
        self.encoder.edge_encoder.train()
        self.encoder.global_encoder.train()
    
    def eval(self):
        """Set to evaluation mode"""
        self.training = False
        self.encoder.node_encoder.eval()
        self.encoder.edge_encoder.eval()
        self.encoder.global_encoder.eval()
    
    def save(self, filepath: str):
        """Save network weights to file"""
        weights = {}
        for i, (weight, grad) in enumerate(self.parameters()):
            weights[f'param_{i}'] = weight
        np.savez(filepath, **weights)
    
    def load(self, filepath: str):
        """Load network weights from file"""
        data = np.load(filepath)
        params = self.parameters()
        for i, (weight, grad) in enumerate(params):
            if f'param_{i}' in data:
                weight[:] = data[f'param_{i}']


# ============================================================================
# PyTorch Implementation (Optional, for faster training if available)
# ============================================================================

if TORCH_AVAILABLE:
    class TorchActorCritic(nn.Module):
        """PyTorch implementation for faster training"""
        
        def __init__(
            self,
            config: Optional[NetworkConfig] = None,
            num_action_types: int = 12,
            max_targets: int = 50
        ):
            super().__init__()
            self.config = config or NetworkConfig()
            
            # Shared encoder
            self.encoder = nn.Sequential(
                nn.Linear(self.config.input_dim, self.config.hidden_dim),
                nn.LayerNorm(self.config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                nn.LayerNorm(self.config.hidden_dim),
                nn.ReLU()
            )
            
            # Policy heads
            self.action_head = nn.Sequential(
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.config.hidden_dim // 2, num_action_types)
            )
            
            self.target_head = nn.Sequential(
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.config.hidden_dim // 2, max_targets)
            )
            
            # Value head
            self.value_head = nn.Sequential(
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.config.hidden_dim // 2, 1)
            )
        
        def forward(self, state_tensor: torch.Tensor, action_mask: Optional[torch.Tensor] = None):
            """Forward pass"""
            encoded = self.encoder(state_tensor)
            
            action_logits = self.action_head(encoded)
            target_logits = self.target_head(encoded)
            value = self.value_head(encoded)
            
            if action_mask is not None:
                action_logits = action_logits.masked_fill(~action_mask[:, :action_logits.size(-1)], -1e9)
            
            action_probs = F.softmax(action_logits, dim=-1)
            target_probs = F.softmax(target_logits, dim=-1)
            
            return action_probs, target_probs, value.squeeze(-1)
