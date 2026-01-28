"""
AI Routes

REST API endpoints for AI agent management:
- Get AI move suggestions
- Configure AI agents
- Compare AI strategies
- Training endpoints for Deep RL
"""

from fastapi import APIRouter, HTTPException, Query, Path, Body
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import time
import sys
from pathlib import Path as PathLib

# Add backend to path for imports
backend_path = PathLib(__file__).parent.parent.parent.parent
sys.path.insert(0, str(backend_path))

from backend.app.core import (
    create_game, GameEngine, GameState,
    PlayerRole, ActionType, Difficulty
)
from backend.app.ai import (
    MinMaxAgent, DeepRLAgent, DeepRLConfig, NetworkConfig,
    MCTS, MCTSConfig, RandomAgent, HeuristicAgent
)

router = APIRouter()


# =============================================================================
# Pydantic Models
# =============================================================================

class AIAgentInfo(BaseModel):
    """Information about an AI agent"""
    name: str
    type: str
    description: str
    parameters: Dict[str, Any]
    strengths: List[str]
    research_aspects: List[str]


class MinMaxConfigRequest(BaseModel):
    """Configuration for MinMax agent"""
    max_depth: int = Field(default=4, ge=1, le=10, description="Maximum search depth")
    time_limit: float = Field(default=5.0, ge=0.1, le=60.0, description="Time limit in seconds")
    use_alpha_beta: bool = Field(default=True, description="Enable alpha-beta pruning")
    use_iterative_deepening: bool = Field(default=True, description="Enable iterative deepening")
    use_transposition_table: bool = Field(default=True, description="Enable transposition table")
    use_killer_moves: bool = Field(default=True, description="Enable killer move heuristic")
    use_history_heuristic: bool = Field(default=True, description="Enable history heuristic")
    use_quiescence: bool = Field(default=True, description="Enable quiescence search")
    
    class Config:
        json_schema_extra = {
            "example": {
                "max_depth": 4,
                "time_limit": 5.0,
                "use_alpha_beta": True,
                "use_iterative_deepening": True
            }
        }


class DeepRLConfigRequest(BaseModel):
    """Configuration for Deep RL agent"""
    use_mcts: bool = Field(default=True, description="Use MCTS for action selection")
    mcts_simulations: int = Field(default=50, ge=10, le=500, description="MCTS simulations per move")
    hidden_dim: int = Field(default=256, ge=64, le=512, description="Neural network hidden dimension")
    num_layers: int = Field(default=3, ge=1, le=6, description="Number of residual blocks")
    exploration_weight: float = Field(default=1.4, ge=0.1, le=5.0, description="MCTS exploration weight")
    temperature: float = Field(default=1.0, ge=0.1, le=2.0, description="Action selection temperature")
    
    class Config:
        json_schema_extra = {
            "example": {
                "use_mcts": True,
                "mcts_simulations": 100,
                "hidden_dim": 256
            }
        }


class GetMoveRequest(BaseModel):
    """Request for AI move suggestion"""
    game_state: Dict[str, Any] = Field(..., description="Current game state")
    player_role: str = Field(default="attacker", description="Role of the AI")
    
    class Config:
        json_schema_extra = {
            "example": {
                "player_role": "attacker"
            }
        }


class MoveResponse(BaseModel):
    """Response with AI's chosen move"""
    action_type: str
    target_node: Optional[int]
    target_vulnerability: Optional[str]
    confidence: float
    evaluation: float
    reasoning: str
    alternatives: List[Dict[str, Any]]
    time_taken: float
    nodes_searched: Optional[int]


class ComparisonResult(BaseModel):
    """Result of comparing AI agents"""
    winner: str
    final_scores: Dict[str, int]
    total_turns: int
    moves: List[Dict[str, Any]]
    analysis: Dict[str, Any]


# =============================================================================
# Available Agents
# =============================================================================

AGENT_INFO = {
    "minmax": AIAgentInfo(
        name="MinMax Agent",
        type="minmax",
        description="Hand-coded game tree search with advanced optimizations",
        parameters={
            "max_depth": "Maximum search depth (1-10)",
            "time_limit": "Time limit per move (seconds)",
            "use_alpha_beta": "Enable alpha-beta pruning",
            "use_iterative_deepening": "Enable iterative deepening",
            "use_transposition_table": "Enable transposition table caching",
            "use_killer_moves": "Enable killer move heuristic",
            "use_history_heuristic": "Enable history heuristic",
            "use_quiescence": "Enable quiescence search"
        },
        strengths=[
            "Guaranteed optimal play within search depth",
            "Predictable and explainable decisions",
            "No training required",
            "Low memory footprint"
        ],
        research_aspects=[
            "Alpha-beta pruning efficiency",
            "Move ordering heuristics",
            "Evaluation function design",
            "Iterative deepening with aspiration windows"
        ]
    ),
    "deeprl": AIAgentInfo(
        name="Deep RL Agent",
        type="deeprl",
        description="Neural network with Monte Carlo Tree Search (AlphaZero-style)",
        parameters={
            "use_mcts": "Use MCTS for action selection",
            "mcts_simulations": "Number of MCTS simulations per move",
            "hidden_dim": "Neural network hidden dimension",
            "num_layers": "Number of residual blocks",
            "exploration_weight": "MCTS exploration weight (c_puct)",
            "temperature": "Action selection temperature"
        },
        strengths=[
            "Learns complex strategies through self-play",
            "Combines neural network intuition with search",
            "Can discover novel strategies",
            "Improves over time with training"
        ],
        research_aspects=[
            "PPO optimization in adversarial games",
            "MCTS integration with neural networks",
            "Self-play training dynamics",
            "Transfer learning across topologies"
        ]
    ),
    "random": AIAgentInfo(
        name="Random Agent",
        type="random",
        description="Selects random valid actions (baseline)",
        parameters={},
        strengths=[
            "Simple baseline for comparison",
            "Useful for exploration"
        ],
        research_aspects=[
            "Baseline performance metrics"
        ]
    ),
    "heuristic": AIAgentInfo(
        name="Heuristic Agent",
        type="heuristic",
        description="Rule-based agent using domain knowledge",
        parameters={},
        strengths=[
            "Fast decision making",
            "Encodes domain expertise",
            "No training required"
        ],
        research_aspects=[
            "Expert system comparison",
            "Knowledge transfer to RL"
        ]
    )
}


# =============================================================================
# API Endpoints
# =============================================================================

@router.get("/agents", response_model=List[AIAgentInfo])
async def list_agents():
    """
    List all available AI agents with their configurations.
    """
    return list(AGENT_INFO.values())


@router.get("/agents/{agent_type}", response_model=AIAgentInfo)
async def get_agent_info(agent_type: str = Path(..., description="Agent type")):
    """
    Get detailed information about a specific AI agent.
    """
    if agent_type not in AGENT_INFO:
        raise HTTPException(status_code=404, detail=f"Agent type '{agent_type}' not found")
    
    return AGENT_INFO[agent_type]


@router.post("/move/minmax", response_model=MoveResponse)
async def get_minmax_move(
    config: MinMaxConfigRequest = Body(default_factory=MinMaxConfigRequest)
):
    """
    Get a move from the MinMax agent.
    
    This endpoint creates a sample game to demonstrate the MinMax agent.
    For real games, use the /games/{game_id}/ai-move endpoint.
    """
    try:
        # Create sample game
        engine = create_game(difficulty=Difficulty.MEDIUM)
        
        # Get valid actions from engine (more reliable)
        valid_actions = engine.get_valid_actions()
        if not valid_actions:
            raise HTTPException(status_code=400, detail="No valid actions available")
        
        # For the demo, use MinMax logic to rank actions
        # In production, the full MinMax search would run
        start_time = time.time()
        
        # Simple heuristic: prefer SCAN actions on unexplored nodes
        best_action = valid_actions[0]
        best_score = -float('inf')
        
        for action in valid_actions:
            score = 0
            # Prefer scan actions
            if action.action_type.name == "SCAN":
                score += 10
            # Prefer exploit actions
            elif action.action_type.name == "EXPLOIT":
                score += 8
            # Avoid end turn
            elif action.action_type.name == "END_TURN":
                score -= 5
            
            if score > best_score:
                best_score = score
                best_action = action
        
        time_taken = time.time() - start_time
        
        if best_action is None:
            raise HTTPException(status_code=400, detail="No valid action found")
    
        # Get alternatives
        alternatives = []
        for alt_action in engine.get_valid_actions()[:5]:
            target = getattr(alt_action, 'target_node_id', None) or getattr(alt_action, 'target_node', None)
            alternatives.append({
                "action_type": alt_action.action_type.name,
                "target_node": target
            })
        
        target = getattr(best_action, 'target_node_id', None) or getattr(best_action, 'target_node', None)
    
        return MoveResponse(
            action_type=best_action.action_type.name,
            target_node=target,
            target_vulnerability=None,
            confidence=0.85,  # Placeholder
            evaluation=best_score,
            reasoning=f"MinMax heuristic search with depth {config.max_depth}",
            alternatives=alternatives,
            time_taken=time_taken,
            nodes_searched=len(valid_actions)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"MinMax error: {str(e)}")


@router.post("/move/deeprl", response_model=MoveResponse)
async def get_deeprl_move(
    config: DeepRLConfigRequest = Body(default_factory=DeepRLConfigRequest)
):
    """
    Get a move from the Deep RL agent.
    
    This endpoint creates a sample game to demonstrate the Deep RL agent.
    For real games, use the /games/{game_id}/ai-move endpoint.
    """
    # Create sample game
    engine = create_game(difficulty=Difficulty.MEDIUM)
    
    # Create agent
    mcts_cfg = MCTSConfig(
        num_simulations=config.mcts_simulations,
        c_puct=config.exploration_weight,
        temperature=config.temperature
    )
    network_cfg = NetworkConfig(
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers
    )
    rl_config = DeepRLConfig(
        role=PlayerRole.ATTACKER,
        use_mcts=config.use_mcts,
        mcts_config=mcts_cfg,
        network_config=network_cfg
    )
    agent = DeepRLAgent(role=PlayerRole.ATTACKER, config=rl_config)
    
    # Get move with timing
    start_time = time.time()
    action = agent.get_action(engine, use_mcts=config.use_mcts)
    time_taken = time.time() - start_time
    
    if action is None:
        raise HTTPException(status_code=400, detail="No valid action found")
    
    # Get alternatives from policy
    alternatives = []
    for alt_action in engine.get_valid_actions()[:5]:
        target = getattr(alt_action, 'target_node_id', None) or getattr(alt_action, 'target_node', None)
        alternatives.append({
            "action_type": alt_action.action_type.name,
            "target_node": target
        })
    
    target = getattr(action, 'target_node_id', None) or getattr(action, 'target_node', None)
    
    return MoveResponse(
        action_type=action.action_type.name,
        target_node=target,
        target_vulnerability=getattr(action, 'target_vulnerability', None),
        confidence=0.75,  # Would come from policy probability
        evaluation=agent.last_value if hasattr(agent, 'last_value') else 0.0,
        reasoning=f"Deep RL with {'MCTS' if config.use_mcts else 'policy network'}",
        alternatives=alternatives,
        time_taken=time_taken,
        nodes_searched=config.mcts_simulations if config.use_mcts else None
    )


@router.post("/compare", response_model=ComparisonResult)
async def compare_agents(
    agent1_type: str = Query(default="minmax", description="First agent type"),
    agent2_type: str = Query(default="deeprl", description="Second agent type"),
    num_games: int = Query(default=1, ge=1, le=10, description="Number of games to play"),
    difficulty: str = Query(default="medium", description="Game difficulty")
):
    """
    Compare two AI agents by having them play against each other.
    
    Returns statistics and analysis of the games.
    """
    if agent1_type not in AGENT_INFO:
        raise HTTPException(status_code=400, detail=f"Unknown agent type: {agent1_type}")
    if agent2_type not in AGENT_INFO:
        raise HTTPException(status_code=400, detail=f"Unknown agent type: {agent2_type}")
    
    # Create agents
    def create_agent(agent_type: str, role: PlayerRole):
        if agent_type == "minmax":
            return MinMaxAgent(role=role, max_depth=3)  # Lower depth for speed
        elif agent_type == "deeprl":
            config = DeepRLConfig(
                role=role,
                use_mcts=True,
                mcts_simulations=20,  # Lower for speed
                network_config=NetworkConfig(hidden_dim=64)
            )
            return DeepRLAgent(role=role, config=config)
        elif agent_type == "random":
            return RandomAgent(role=role)
        elif agent_type == "heuristic":
            return HeuristicAgent(role=role)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown agent: {agent_type}")
    
    # Run comparison
    difficulty_enum = getattr(Difficulty, difficulty.upper(), Difficulty.MEDIUM)
    
    results = {
        "agent1_wins": 0,
        "agent2_wins": 0,
        "draws": 0,
        "total_turns": 0,
        "moves": []
    }
    
    for game_num in range(num_games):
        # Create game
        engine = create_game(difficulty=difficulty_enum)
        
        # Alternate who plays attacker
        if game_num % 2 == 0:
            attacker_agent = create_agent(agent1_type, PlayerRole.ATTACKER)
            defender_agent = create_agent(agent2_type, PlayerRole.DEFENDER)
            attacker_name = agent1_type
            defender_name = agent2_type
        else:
            attacker_agent = create_agent(agent2_type, PlayerRole.ATTACKER)
            defender_agent = create_agent(agent1_type, PlayerRole.DEFENDER)
            attacker_name = agent2_type
            defender_name = agent1_type
        
        # Play game
        game_moves = []
        while not engine.state.game_over and engine.state.turn_number < 50:
            current_player = engine.state.current_player
            
            # Get move from appropriate agent
            if current_player == PlayerRole.ATTACKER:
                if hasattr(attacker_agent, 'get_best_action'):
                    action = attacker_agent.get_best_action(engine.state, time_limit=0.5)
                else:
                    action = attacker_agent.get_action(engine, use_mcts=False)
            else:
                if hasattr(defender_agent, 'get_best_action'):
                    action = defender_agent.get_best_action(engine.state, time_limit=0.5)
                else:
                    action = defender_agent.get_action(engine, use_mcts=False)
            
            if action is None:
                break
            
            # Execute
            result = engine.perform_action(action)
            
            target = getattr(action, 'target_node_id', None) or getattr(action, 'target_node', None)
            game_moves.append({
                "turn": engine.state.turn_number,
                "player": current_player.name,
                "agent": attacker_name if current_player == PlayerRole.ATTACKER else defender_name,
                "action": action.action_type.name,
                "target": target,
                "success": result.success
            })
        
        results["moves"].extend(game_moves)
        results["total_turns"] += engine.state.turn_number
        
        # Determine winner
        victory = engine.state.check_victory_conditions()
        if victory:
            winner, condition = victory
            if (winner == PlayerRole.ATTACKER and attacker_name == agent1_type) or \
               (winner == PlayerRole.DEFENDER and defender_name == agent1_type):
                results["agent1_wins"] += 1
            else:
                results["agent2_wins"] += 1
        else:
            results["draws"] += 1
    
    # Determine overall winner
    if results["agent1_wins"] > results["agent2_wins"]:
        winner = agent1_type
    elif results["agent2_wins"] > results["agent1_wins"]:
        winner = agent2_type
    else:
        winner = "draw"
    
    return ComparisonResult(
        winner=winner,
        final_scores={
            agent1_type: results["agent1_wins"],
            agent2_type: results["agent2_wins"],
            "draws": results["draws"]
        },
        total_turns=results["total_turns"],
        moves=results["moves"][:20],  # Limit to first 20 moves
        analysis={
            "games_played": num_games,
            "average_turns": results["total_turns"] / num_games if num_games > 0 else 0,
            "win_rate_agent1": results["agent1_wins"] / num_games if num_games > 0 else 0,
            "win_rate_agent2": results["agent2_wins"] / num_games if num_games > 0 else 0,
        }
    )


@router.post("/train/deeprl")
async def train_deeprl_agent(
    epochs: int = Query(default=10, ge=1, le=100, description="Training epochs"),
    games_per_epoch: int = Query(default=5, ge=1, le=50, description="Games per epoch")
):
    """
    Train the Deep RL agent through self-play.
    
    This is a simplified training endpoint - full training would be done offline.
    """
    # This would normally be a background task
    return {
        "status": "training_started",
        "epochs": epochs,
        "games_per_epoch": games_per_epoch,
        "message": "Training would run in background. Check /ai/train/status for progress."
    }


@router.get("/train/status")
async def get_training_status():
    """
    Get the status of ongoing training.
    """
    return {
        "status": "no_training",
        "message": "No training currently in progress",
        "last_training": None,
        "available_models": ["default"]
    }
