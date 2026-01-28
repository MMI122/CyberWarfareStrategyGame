"""
WebSocket Routes

Real-time communication for:
- Live game updates
- Spectator mode
- AI move notifications
- Training progress updates
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from typing import Dict, List, Set
import json
import asyncio
import sys
from pathlib import Path as PathLib

# Add backend to path for imports
backend_path = PathLib(__file__).parent.parent.parent.parent
sys.path.insert(0, str(backend_path))

from backend.app.core import PlayerRole

router = APIRouter()


# =============================================================================
# Connection Manager
# =============================================================================

class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        # Map game_id -> set of websocket connections
        self.game_connections: Dict[str, Set[WebSocket]] = {}
        # Map websocket -> (game_id, player_role)
        self.connection_info: Dict[WebSocket, tuple] = {}
        # Global spectators
        self.spectators: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket, game_id: str, role: str = "spectator"):
        """Accept a new connection"""
        await websocket.accept()
        
        if game_id not in self.game_connections:
            self.game_connections[game_id] = set()
        
        self.game_connections[game_id].add(websocket)
        self.connection_info[websocket] = (game_id, role)
        
        # Send connection confirmation
        await websocket.send_json({
            "type": "connected",
            "game_id": game_id,
            "role": role,
            "message": f"Connected to game {game_id} as {role}"
        })
    
    def disconnect(self, websocket: WebSocket):
        """Remove a connection"""
        if websocket in self.connection_info:
            game_id, _ = self.connection_info[websocket]
            if game_id in self.game_connections:
                self.game_connections[game_id].discard(websocket)
                if not self.game_connections[game_id]:
                    del self.game_connections[game_id]
            del self.connection_info[websocket]
        
        self.spectators.discard(websocket)
    
    async def broadcast_to_game(self, game_id: str, message: dict):
        """Send message to all connections in a game"""
        if game_id in self.game_connections:
            dead_connections = []
            # Create a copy of the set to avoid modification during iteration
            connections = list(self.game_connections[game_id])
            for connection in connections:
                try:
                    await connection.send_json(message)
                except Exception:
                    dead_connections.append(connection)
            
            # Clean up dead connections
            for conn in dead_connections:
                self.disconnect(conn)
    
    async def send_to_player(self, game_id: str, role: str, message: dict):
        """Send message to specific player in a game"""
        if game_id in self.game_connections:
            for connection in self.game_connections[game_id]:
                info = self.connection_info.get(connection)
                if info and info[1] == role:
                    try:
                        await connection.send_json(message)
                    except Exception:
                        self.disconnect(connection)
    
    async def broadcast_global(self, message: dict):
        """Send message to all spectators"""
        dead_connections = []
        for connection in self.spectators:
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.append(connection)
        
        for conn in dead_connections:
            self.spectators.discard(conn)
    
    def get_game_connections(self, game_id: str) -> int:
        """Get number of connections for a game"""
        return len(self.game_connections.get(game_id, []))


# Global connection manager
manager = ConnectionManager()


# =============================================================================
# WebSocket Endpoints
# =============================================================================

@router.websocket("/game/{game_id}")
async def game_websocket(
    websocket: WebSocket,
    game_id: str,
    role: str = Query(default="spectator", description="Player role or spectator")
):
    """
    WebSocket connection for game updates.
    
    Clients receive:
    - Game state updates
    - Action notifications
    - Turn changes
    - Victory announcements
    """
    await manager.connect(websocket, game_id, role)
    
    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                await handle_game_message(websocket, game_id, role, message)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON"
                })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        
        # Notify other players
        await manager.broadcast_to_game(game_id, {
            "type": "player_disconnected",
            "role": role,
            "message": f"{role} disconnected"
        })


async def handle_game_message(websocket: WebSocket, game_id: str, role: str, message: dict):
    """Handle incoming WebSocket messages"""
    msg_type = message.get("type", "")
    
    if msg_type == "ping":
        # Keep-alive
        await websocket.send_json({"type": "pong"})
    
    elif msg_type == "chat":
        # Broadcast chat message
        await manager.broadcast_to_game(game_id, {
            "type": "chat",
            "from": role,
            "message": message.get("message", "")
        })
    
    elif msg_type == "request_state":
        # Client requesting current state
        from backend.app.api.routes.games import games_store, game_state_to_response
        
        if game_id in games_store:
            engine = games_store[game_id]["engine"]
            state = game_state_to_response(engine, game_id)
            await websocket.send_json({
                "type": "game_state",
                "state": state.model_dump()
            })
        else:
            await websocket.send_json({
                "type": "error",
                "message": "Game not found"
            })
    
    elif msg_type == "action":
        # Player action - validate and execute
        if role not in ["attacker", "defender"]:
            await websocket.send_json({
                "type": "error",
                "message": "Spectators cannot make moves"
            })
            return
        
        from backend.app.api.routes.games import games_store
        from backend.app.core import ActionType
        from backend.app.core.data_structures import GameAction
        
        if game_id not in games_store:
            await websocket.send_json({
                "type": "error",
                "message": "Game not found"
            })
            return
        
        engine = games_store[game_id]["engine"]
        
        # Check if it's this player's turn
        current_player = engine.state.current_player.name.lower()
        if current_player != role:
            await websocket.send_json({
                "type": "error",
                "message": "Not your turn"
            })
            return
        
        # Parse and execute action
        try:
            action_type = ActionType[message["action_type"].upper()]
            action = GameAction(
                action_type=action_type,
                player=engine.state.current_player,
                target_node=message.get("target_node")
            )
            
            result = engine.perform_action(action)
            
            # Broadcast result to all players
            await manager.broadcast_to_game(game_id, {
                "type": "action_executed",
                "player": role,
                "action": message["action_type"],
                "target": message.get("target_node"),
                "success": result.success,
                "message": getattr(result, 'message', '')
            })
            
            # Send updated state
            from backend.app.api.routes.games import game_state_to_response
            state = game_state_to_response(engine, game_id)
            await manager.broadcast_to_game(game_id, {
                "type": "game_state",
                "state": state.model_dump()
            })
            
            # Check for game over
            victory = engine.state.check_victory_conditions()
            if victory:
                winner, condition = victory
                await manager.broadcast_to_game(game_id, {
                    "type": "game_over",
                    "winner": winner.name,
                    "condition": condition.name,
                    "message": f"Game Over! {winner.name} wins by {condition.name}!"
                })
        
        except KeyError:
            await websocket.send_json({
                "type": "error",
                "message": f"Invalid action type: {message.get('action_type')}"
            })
        except Exception as e:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
    
    elif msg_type == "request_ai_move":
        # Request AI to make a move
        from backend.app.api.routes.games import games_store
        
        if game_id not in games_store:
            await websocket.send_json({
                "type": "error",
                "message": "Game not found"
            })
            return
        
        game_data = games_store[game_id]
        engine = game_data["engine"]
        ai_type = game_data["ai_type"]
        
        # Notify that AI is thinking
        await manager.broadcast_to_game(game_id, {
            "type": "ai_thinking",
            "ai_type": ai_type,
            "message": f"{ai_type} AI is thinking..."
        })
        
        # Get AI move (would be async in production)
        try:
            from backend.app.ai import MinMaxAgent, DeepRLAgent, DeepRLConfig, NetworkConfig
            
            ai_role = engine.state.current_player
            
            if ai_type == "minmax":
                agent = MinMaxAgent(role=ai_role, max_depth=4)
                action = agent.get_best_action(engine.state, time_limit=2.0)
            elif ai_type == "deeprl":
                config = DeepRLConfig(
                    role=ai_role,
                    use_mcts=True,
                    network_config=NetworkConfig(hidden_dim=128)
                )
                agent = DeepRLAgent(role=ai_role, config=config)
                action = agent.get_action(engine, use_mcts=True)
            else:
                import numpy as np
                valid_actions = engine.get_valid_actions()
                action = np.random.choice(valid_actions) if valid_actions else None
            
            if action:
                result = engine.perform_action(action)
                target = getattr(action, 'target_node_id', None) or getattr(action, 'target_node', None)
                
                await manager.broadcast_to_game(game_id, {
                    "type": "ai_move",
                    "ai_type": ai_type,
                    "action": action.action_type.name,
                    "target": target,
                    "success": result.success
                })
                
                # Send updated state
                from backend.app.api.routes.games import game_state_to_response
                state = game_state_to_response(engine, game_id)
                await manager.broadcast_to_game(game_id, {
                    "type": "game_state",
                    "state": state.model_dump()
                })
        
        except Exception as e:
            await websocket.send_json({
                "type": "error",
                "message": f"AI error: {str(e)}"
            })
    
    else:
        await websocket.send_json({
            "type": "error",
            "message": f"Unknown message type: {msg_type}"
        })


@router.websocket("/spectate")
async def spectate_websocket(websocket: WebSocket):
    """
    Global spectator WebSocket.
    
    Receives updates about all active games.
    """
    await websocket.accept()
    manager.spectators.add(websocket)
    
    try:
        # Send list of active games
        from backend.app.api.routes.games import games_store
        
        games = []
        for game_id, game_data in games_store.items():
            engine = game_data["engine"]
            games.append({
                "game_id": game_id,
                "turn": engine.state.turn_number,
                "current_player": engine.state.current_player.name,
                "spectators": manager.get_game_connections(game_id)
            })
        
        await websocket.send_json({
            "type": "active_games",
            "games": games
        })
        
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            elif message.get("type") == "join_game":
                # Switch to specific game
                game_id = message.get("game_id")
                if game_id:
                    manager.spectators.discard(websocket)
                    await manager.connect(websocket, game_id, "spectator")
    
    except WebSocketDisconnect:
        manager.spectators.discard(websocket)


@router.websocket("/training")
async def training_websocket(websocket: WebSocket):
    """
    WebSocket for training progress updates.
    
    Receives:
    - Training epoch progress
    - Loss values
    - Evaluation metrics
    """
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            elif message.get("type") == "start_training":
                # Would start training in background
                await websocket.send_json({
                    "type": "training_started",
                    "message": "Training started (demo mode)"
                })
                
                # Simulate training progress
                for epoch in range(5):
                    await asyncio.sleep(1)
                    await websocket.send_json({
                        "type": "training_progress",
                        "epoch": epoch + 1,
                        "total_epochs": 5,
                        "loss": 0.5 / (epoch + 1),
                        "policy_loss": 0.3 / (epoch + 1),
                        "value_loss": 0.2 / (epoch + 1)
                    })
                
                await websocket.send_json({
                    "type": "training_complete",
                    "message": "Training complete (demo mode)"
                })
    
    except WebSocketDisconnect:
        pass


# =============================================================================
# Helper Functions for Broadcasting
# =============================================================================

async def broadcast_action(game_id: str, player: str, action_type: str, 
                          target: int, success: bool):
    """Broadcast an action to all game participants"""
    await manager.broadcast_to_game(game_id, {
        "type": "action",
        "player": player,
        "action_type": action_type,
        "target": target,
        "success": success
    })


async def broadcast_turn_change(game_id: str, new_player: str, turn_number: int):
    """Broadcast turn change"""
    await manager.broadcast_to_game(game_id, {
        "type": "turn_change",
        "current_player": new_player,
        "turn_number": turn_number
    })


async def broadcast_game_over(game_id: str, winner: str, condition: str):
    """Broadcast game over"""
    await manager.broadcast_to_game(game_id, {
        "type": "game_over",
        "winner": winner,
        "condition": condition
    })
