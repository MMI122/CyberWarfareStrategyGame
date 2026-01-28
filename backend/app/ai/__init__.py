# =============================================================================
# AI Module
# =============================================================================
"""
AI agents for the Cyber Warfare Strategy Game.

Contains:
- MinMaxAgent: Hand-coded MinMax with Alpha-Beta pruning
- DeepRLAgent: Deep RL with PPO and MCTS

Research-grade implementations suitable for academic publication.
"""

from .minmax_agent import (
    MinMaxAgent,
    create_minmax_agent,
    TranspositionTable,
    KillerMoves,
    HistoryTable,
    MoveOrderer,
    StateEvaluator,
    SearchStats,
)

# Deep RL components
from .neural_network import (
    ActorCriticNetwork,
    NetworkConfig,
    StateEncoder,
    PolicyHead,
    ValueHead,
    NumpyMLP,
    NumpyMultiHeadAttention,
    NumpyResidualBlock
)

from .mcts import (
    MCTS,
    MCTSConfig,
    MCTSNode,
    AlphaZeroMCTS,
    MCTSParallel
)

from .ppo import (
    PPOTrainer,
    PPOConfig,
    RolloutBuffer,
    NumpyOptimizer,
    LearningRateScheduler,
    RunningMeanStd
)

from .deep_rl_agent import (
    DeepRLAgent,
    DeepRLConfig,
    TrainingMode,
    RandomAgent,
    HeuristicAgent,
    ExperienceReplayBuffer
)

__all__ = [
    # MinMax Agent
    "MinMaxAgent",
    "create_minmax_agent",
    "TranspositionTable",
    "KillerMoves",
    "HistoryTable",
    "MoveOrderer",
    "StateEvaluator",
    "SearchStats",
    
    # Neural Networks
    'ActorCriticNetwork',
    'NetworkConfig',
    'StateEncoder',
    'PolicyHead',
    'ValueHead',
    'NumpyMLP',
    'NumpyMultiHeadAttention',
    'NumpyResidualBlock',
    
    # MCTS
    'MCTS',
    'MCTSConfig',
    'MCTSNode',
    'AlphaZeroMCTS',
    'MCTSParallel',
    
    # PPO
    'PPOTrainer',
    'PPOConfig',
    'RolloutBuffer',
    'NumpyOptimizer',
    'LearningRateScheduler',
    'RunningMeanStd',
    
    # Deep RL Agent
    'DeepRLAgent',
    'DeepRLConfig',
    'TrainingMode',
    'RandomAgent',
    'HeuristicAgent',
    'ExperienceReplayBuffer'
]
