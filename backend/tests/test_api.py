"""
API Tests

Tests for the FastAPI endpoints:
- Games CRUD
- AI endpoints  
- Game actions
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi.testclient import TestClient
from backend.app.api.main import app


# =============================================================================
# Test Client
# =============================================================================

client = TestClient(app)


# =============================================================================
# Root Endpoint Tests
# =============================================================================

def test_root_endpoint():
    """Test root endpoint returns welcome message"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "Cyber Warfare" in data["message"]
    print("✓ Root endpoint works")


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    print("✓ Health check works")


# =============================================================================
# Game Endpoint Tests
# =============================================================================

def test_create_game():
    """Test creating a new game"""
    response = client.post("/api/games/", json={
        "difficulty": "medium",
        "topology_type": "corporate",
        "player_role": "attacker",
        "ai_type": "minmax"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "game_id" in data
    assert data["turn_number"] == 0
    assert "nodes" in data
    assert len(data["nodes"]) > 0
    print(f"✓ Game created with ID: {data['game_id'][:8]}...")
    
    return data["game_id"]


def test_list_games():
    """Test listing games"""
    # Create a game first
    client.post("/api/games/", json={"difficulty": "easy"})
    
    response = client.get("/api/games/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    print(f"✓ Listed {len(data)} games")


def test_get_game():
    """Test getting a specific game"""
    # Create game
    create_response = client.post("/api/games/", json={"difficulty": "easy"})
    game_id = create_response.json()["game_id"]
    
    # Get game
    response = client.get(f"/api/games/{game_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["game_id"] == game_id
    print("✓ Get game works")


def test_get_nonexistent_game():
    """Test getting a game that doesn't exist"""
    response = client.get("/api/games/nonexistent-id")
    assert response.status_code == 404
    print("✓ Nonexistent game returns 404")


def test_delete_game():
    """Test deleting a game"""
    # Create game
    create_response = client.post("/api/games/", json={"difficulty": "easy"})
    game_id = create_response.json()["game_id"]
    
    # Delete game
    response = client.delete(f"/api/games/{game_id}")
    assert response.status_code == 200
    
    # Verify deleted
    get_response = client.get(f"/api/games/{game_id}")
    assert get_response.status_code == 404
    print("✓ Delete game works")


def test_get_game_history():
    """Test getting game history"""
    # Create game
    create_response = client.post("/api/games/", json={"difficulty": "easy"})
    game_id = create_response.json()["game_id"]
    
    # Get history
    response = client.get(f"/api/games/{game_id}/history")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    print("✓ Get history works")


# =============================================================================
# AI Endpoint Tests
# =============================================================================

def test_list_agents():
    """Test listing available AI agents"""
    response = client.get("/api/ai/agents")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 2  # At least minmax and deeprl
    
    agent_types = [a["type"] for a in data]
    assert "minmax" in agent_types
    assert "deeprl" in agent_types
    print(f"✓ Listed {len(data)} AI agents")


def test_get_agent_info():
    """Test getting info about specific agent"""
    response = client.get("/api/ai/agents/minmax")
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "minmax"
    assert "parameters" in data
    assert "strengths" in data
    print("✓ Get agent info works")


def test_get_minmax_move():
    """Test getting a move from MinMax agent"""
    response = client.post("/api/ai/move/minmax", json={
        "max_depth": 2,
        "time_limit": 1.0
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "action_type" in data
    assert "confidence" in data
    assert "time_taken" in data  # Time can be 0 for fast operations
    print(f"✓ MinMax suggested: {data['action_type']}")


def test_get_deeprl_move():
    """Test getting a move from Deep RL agent"""
    response = client.post("/api/ai/move/deeprl", json={
        "use_mcts": True,
        "mcts_simulations": 10
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "action_type" in data
    print(f"✓ Deep RL suggested: {data['action_type']}")


def test_compare_agents():
    """Test comparing two AI agents"""
    response = client.post("/api/ai/compare", params={
        "agent1_type": "minmax",
        "agent2_type": "random",
        "num_games": 1,
        "difficulty": "easy"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "winner" in data
    assert "final_scores" in data
    assert "total_turns" in data
    print(f"✓ Agent comparison: winner = {data['winner']}")


# =============================================================================
# Action Tests
# =============================================================================

def test_execute_action():
    """Test executing an action in a game"""
    # Create game
    create_response = client.post("/api/games/", json={
        "difficulty": "easy",
        "player_role": "attacker"
    })
    game_id = create_response.json()["game_id"]
    
    # Get valid actions from state
    state = create_response.json()
    valid_actions = state["valid_actions"]
    
    if valid_actions:
        # Try to execute first valid action
        action = valid_actions[0]
        response = client.post(f"/api/games/{game_id}/action", json={
            "action_type": action["action_type"],
            "target_node": action.get("target_node")
        })
        
        # Should either succeed or fail gracefully
        assert response.status_code in [200, 400]
        print(f"✓ Action execution attempted: {action['action_type']}")
    else:
        print("✓ No valid actions to test (game might be over)")


def test_ai_move_in_game():
    """Test AI making a move in a game"""
    # Create game with AI as defender
    create_response = client.post("/api/games/", json={
        "difficulty": "easy",
        "player_role": "attacker",  # Human is attacker
        "ai_type": "minmax"
    })
    game_id = create_response.json()["game_id"]
    
    # First make a human move (attacker goes first typically)
    state = create_response.json()
    if state["current_player"] == "ATTACKER":
        # Make a valid action as attacker
        valid_actions = state["valid_actions"]
        if valid_actions:
            action = valid_actions[0]
            client.post(f"/api/games/{game_id}/action", json={
                "action_type": action["action_type"],
                "target_node": action.get("target_node")
            })
    
    # Now try AI move (should be defender's turn)
    response = client.post(f"/api/games/{game_id}/ai-move", params={
        "time_limit": 1.0
    })
    
    # May succeed or fail depending on turn
    assert response.status_code in [200, 400]
    print("✓ AI move endpoint works")


# =============================================================================
# Run All Tests
# =============================================================================

def run_all_tests():
    """Run all API tests"""
    print("\n" + "=" * 60)
    print("CYBER WARFARE STRATEGY GAME - API TESTS")
    print("=" * 60 + "\n")
    
    tests = [
        # Root tests
        ("Root endpoint", test_root_endpoint),
        ("Health check", test_health_check),
        
        # Game tests
        ("Create game", test_create_game),
        ("List games", test_list_games),
        ("Get game", test_get_game),
        ("Get nonexistent game", test_get_nonexistent_game),
        ("Delete game", test_delete_game),
        ("Get game history", test_get_game_history),
        
        # AI tests
        ("List agents", test_list_agents),
        ("Get agent info", test_get_agent_info),
        ("MinMax move", test_get_minmax_move),
        ("Deep RL move", test_get_deeprl_move),
        ("Compare agents", test_compare_agents),
        
        # Action tests
        ("Execute action", test_execute_action),
        ("AI move in game", test_ai_move_in_game),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            print(f"\nTest: {name}")
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {str(e)}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
