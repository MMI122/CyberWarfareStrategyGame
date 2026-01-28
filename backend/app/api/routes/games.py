"""
Game Routes

REST API endpoints for game management:
- Create/list/get games
- Execute actions
- Get game state
- Manage game flow
"""

from fastapi import APIRouter, HTTPException, Query, Path, Body
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from enum import Enum
import uuid
import sys
from pathlib import Path as PathLib

# Add backend to path for imports
backend_path = PathLib(__file__).parent.parent.parent.parent
sys.path.insert(0, str(backend_path))

from backend.app.core import (
    create_game, GameEngine, GameState,
    PlayerRole, ActionType, Difficulty, GamePhase
)
from backend.app.core.data_structures import GameAction

router = APIRouter()


# =============================================================================
# Pydantic Models for Request/Response
# =============================================================================

class DifficultyLevel(str, Enum):
    """Difficulty levels for API"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class CreateGameRequest(BaseModel):
    """Request model for creating a new game"""
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    topology_type: str = Field(default="corporate", description="Network topology type")
    max_turns: Optional[int] = Field(default=None, description="Override max turns")
    player_role: str = Field(default="attacker", description="Human player role")
    ai_type: str = Field(default="minmax", description="AI opponent type: minmax, deeprl, random")
    
    class Config:
        json_schema_extra = {
            "example": {
                "difficulty": "medium",
                "topology_type": "corporate",
                "player_role": "attacker",
                "ai_type": "minmax"
            }
        }


class ActionRequest(BaseModel):
    """Request model for executing an action"""
    action_type: str = Field(..., description="Type of action to execute")
    target_node: Optional[int] = Field(default=None, description="Target node ID")
    target_vulnerability: Optional[str] = Field(default=None, description="Target vulnerability ID")
    
    class Config:
        json_schema_extra = {
            "example": {
                "action_type": "SCAN",
                "target_node": 0
            }
        }


class NodeResponse(BaseModel):
    """Response model for a network node"""
    id: int
    name: str
    node_type: str
    status: str
    is_compromised: bool
    is_visible: bool
    access_level: str
    defense_level: float
    importance: float
    vulnerabilities: List[Dict[str, Any]]
    connected_nodes: List[int]


class GameStateResponse(BaseModel):
    """Response model for game state"""
    game_id: str
    turn_number: int
    phase: str
    current_player: str
    attacker_score: int
    defender_score: int
    nodes: List[NodeResponse]
    valid_actions: List[Dict[str, Any]]
    game_over: bool
    winner: Optional[str] = None
    victory_condition: Optional[str] = None


class ActionResultResponse(BaseModel):
    """Response model for action result"""
    success: bool
    message: str
    points_gained: int
    game_state: GameStateResponse


# =============================================================================
# Game Storage (In-Memory for now)
# =============================================================================

games_store: Dict[str, Dict] = {}


def get_difficulty_enum(level: DifficultyLevel) -> Difficulty:
    """Convert API difficulty to game enum"""
    mapping = {
        DifficultyLevel.EASY: Difficulty.EASY,
        DifficultyLevel.MEDIUM: Difficulty.MEDIUM,
        DifficultyLevel.HARD: Difficulty.HARD,
        DifficultyLevel.EXPERT: Difficulty.EXPERT
    }
    return mapping.get(level, Difficulty.MEDIUM)


def game_state_to_response(engine: GameEngine, game_id: str) -> GameStateResponse:
    """Convert game engine state to API response"""
    state = engine.state
    
    # Get visible nodes based on current player
    nodes = []
    for node_id, node in state.nodes.items():
        # Determine visibility
        is_visible = True  # For now, show all nodes
        
        # Get connected nodes
        connected = []
        for edge in state.edges:
            if edge.source_id == node_id:
                connected.append(edge.target_id)
            elif edge.target_id == node_id:
                connected.append(edge.source_id)
        
        # Convert vulnerabilities
        vulns = []
        for v in node.vulnerabilities:
            vulns.append({
                "id": v.id,
                "name": v.name,
                "severity": v.severity,
                "is_patched": v.is_patched,
                "is_known": getattr(v, 'is_known', True)
            })
        
        nodes.append(NodeResponse(
            id=node_id,
            name=node.name,
            node_type=node.node_type.name,
            status=node.status.name,
            is_compromised=getattr(node, 'is_compromised', False),
            is_visible=is_visible,
            access_level=node.access_level.name,
            defense_level=getattr(node, 'defense_level', 1.0),
            importance=getattr(node, 'importance', 1.0),
            vulnerabilities=vulns,
            connected_nodes=connected
        ))
    
    # Get valid actions
    valid_actions = []
    for action in engine.get_valid_actions():
        target = getattr(action, 'target_node_id', None) or getattr(action, 'target_node', None)
        valid_actions.append({
            "action_type": action.action_type.name,
            "target_node": target,
            "ap_cost": getattr(action, 'ap_cost', 1),
            "description": f"{action.action_type.name} on node {target}"
        })
    
    # Check victory
    victory_result = state.check_victory_conditions()
    winner = None
    victory_condition = None
    game_over = state.game_over
    
    if victory_result:
        winner, vc = victory_result
        winner = winner.name
        victory_condition = vc.name
        game_over = True
    
    return GameStateResponse(
        game_id=game_id,
        turn_number=state.turn_number,
        phase=state.phase.name,
        current_player=state.current_player.name,
        attacker_score=state.attacker.score,
        defender_score=state.defender.score,
        nodes=nodes,
        valid_actions=valid_actions,
        game_over=game_over,
        winner=winner,
        victory_condition=victory_condition
    )


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/", response_model=GameStateResponse)
async def create_new_game(request: CreateGameRequest):
    """
    Create a new game session.
    
    Returns the initial game state.
    """
    # Create game engine
    difficulty = get_difficulty_enum(request.difficulty)
    engine = create_game(
        difficulty=difficulty,
        topology=request.topology_type
    )
    
    # Generate game ID
    game_id = str(uuid.uuid4())
    
    # Store game
    games_store[game_id] = {
        "engine": engine,
        "player_role": request.player_role,
        "ai_type": request.ai_type,
        "created_at": None,  # Would use datetime in real app
    }
    
    return game_state_to_response(engine, game_id)


@router.get("/", response_model=List[Dict[str, Any]])
async def list_games(
    active_only: bool = Query(default=True, description="Only return active games")
):
    """
    List all game sessions.
    """
    games = []
    for game_id, game_data in games_store.items():
        engine = game_data["engine"]
        victory = engine.state.check_victory_conditions()
        is_active = victory is None and not engine.state.game_over
        
        if active_only and not is_active:
            continue
        
        games.append({
            "game_id": game_id,
            "turn_number": engine.state.turn_number,
            "current_player": engine.state.current_player.name,
            "player_role": game_data["player_role"],
            "ai_type": game_data["ai_type"],
            "is_active": is_active
        })
    
    return games


@router.get("/{game_id}", response_model=GameStateResponse)
async def get_game(game_id: str = Path(..., description="Game ID")):
    """
    Get the current state of a game.
    """
    if game_id not in games_store:
        raise HTTPException(status_code=404, detail="Game not found")
    
    engine = games_store[game_id]["engine"]
    return game_state_to_response(engine, game_id)


@router.post("/{game_id}/action", response_model=ActionResultResponse)
async def execute_action(
    game_id: str = Path(..., description="Game ID"),
    request: ActionRequest = Body(...)
):
    """
    Execute a player action in the game.
    
    Returns the result and updated game state.
    """
    if game_id not in games_store:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game_data = games_store[game_id]
    engine = game_data["engine"]
    
    # Check if game is over
    if engine.state.game_over:
        raise HTTPException(status_code=400, detail="Game is already over")
    
    # Parse action type
    try:
        action_type = ActionType[request.action_type.upper()]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Invalid action type: {request.action_type}")
    
    # Create action
    action = GameAction(
        action_type=action_type,
        player=engine.state.current_player,
        target_node=request.target_node
    )
    
    # Add vulnerability to parameters if provided
    if request.target_vulnerability:
        action.parameters['vulnerability_id'] = request.target_vulnerability
    
    # Validate action
    valid_actions = engine.get_valid_actions()
    is_valid = any(
        a.action_type == action.action_type and 
        (getattr(a, 'target_node_id', None) or getattr(a, 'target_node', None)) == request.target_node
        for a in valid_actions
    )
    
    if not is_valid:
        raise HTTPException(status_code=400, detail="Invalid action for current state")
    
    # Execute action
    result = engine.perform_action(action)
    
    # Get updated state
    game_state = game_state_to_response(engine, game_id)
    
    return ActionResultResponse(
        success=result.success,
        message=getattr(result, 'message', 'Action executed'),
        points_gained=getattr(result, 'points_gained', 0),
        game_state=game_state
    )


@router.post("/{game_id}/ai-move", response_model=ActionResultResponse)
async def execute_ai_move(
    game_id: str = Path(..., description="Game ID"),
    time_limit: float = Query(default=2.0, description="Time limit for AI in seconds")
):
    """
    Let the AI make a move.
    
    Returns the AI's action and updated game state.
    """
    if game_id not in games_store:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game_data = games_store[game_id]
    engine = game_data["engine"]
    ai_type = game_data["ai_type"]
    
    # Check if game is over
    if engine.state.game_over:
        raise HTTPException(status_code=400, detail="Game is already over")
    
    # Determine AI role (opposite of player)
    player_role = game_data["player_role"]
    ai_role = PlayerRole.DEFENDER if player_role == "attacker" else PlayerRole.ATTACKER
    
    # Check if it's AI's turn
    if engine.state.current_player.name.lower() != ai_role.name.lower():
        raise HTTPException(status_code=400, detail="Not AI's turn")
    
    # Get AI action based on type
    from backend.app.ai import MinMaxAgent, DeepRLAgent, DeepRLConfig, NetworkConfig
    
    if ai_type == "minmax":
        agent = MinMaxAgent(role=ai_role, max_depth=4)
        action = agent.get_best_action(engine.state, time_limit=time_limit)
    elif ai_type == "deeprl":
        config = DeepRLConfig(
            role=ai_role,
            use_mcts=True,
            network_config=NetworkConfig(hidden_dim=128)
        )
        agent = DeepRLAgent(role=ai_role, config=config)
        action = agent.get_action(engine, use_mcts=True)
    else:
        # Random agent
        import numpy as np
        valid_actions = engine.get_valid_actions()
        action = np.random.choice(valid_actions) if valid_actions else None
    
    if action is None:
        raise HTTPException(status_code=400, detail="AI could not find a valid action")
    
    # Execute action
    result = engine.perform_action(action)
    
    # Get updated state
    game_state = game_state_to_response(engine, game_id)
    
    return ActionResultResponse(
        success=result.success,
        message=f"AI ({ai_type}) played: {action.action_type.name}",
        points_gained=getattr(result, 'points_gained', 0),
        game_state=game_state
    )


@router.delete("/{game_id}")
async def delete_game(game_id: str = Path(..., description="Game ID")):
    """
    Delete a game session.
    """
    if game_id not in games_store:
        raise HTTPException(status_code=404, detail="Game not found")
    
    del games_store[game_id]
    
    return {"message": "Game deleted", "game_id": game_id}


@router.get("/{game_id}/history", response_model=List[Dict[str, Any]])
async def get_game_history(game_id: str = Path(..., description="Game ID")):
    """
    Get the action history of a game.
    """
    if game_id not in games_store:
        raise HTTPException(status_code=404, detail="Game not found")
    
    engine = games_store[game_id]["engine"]
    
    history = []
    for i, event in enumerate(engine.event_history):
        history.append({
            "index": i,
            "type": event.event_type.name,
            "turn": event.turn,
            "player": event.player_role.name if event.player_role else None,
            "action": event.action.action_type.name if event.action else None,
            "message": event.message
        })
    
    return history
