#!/usr/bin/env python
"""
Quick test script for the core game engine.
"""
import sys
sys.path.insert(0, '.')

from app.core import (
    create_game, play_random_game, 
    Difficulty, GameState, PlayerRole
)

def test_game_creation():
    """Test game creation"""
    print("Test 1: Game Creation...")
    engine = create_game(difficulty=Difficulty.MEDIUM)
    assert engine.state is not None
    assert len(engine.state.nodes) > 0
    print(f"  ✓ Created game with {len(engine.state.nodes)} nodes")

def test_get_actions():
    """Test getting valid actions"""
    print("Test 2: Valid Actions...")
    engine = create_game(difficulty=Difficulty.MEDIUM)
    actions = engine.get_valid_actions()
    assert len(actions) > 0
    print(f"  ✓ Found {len(actions)} valid actions")

def test_perform_action():
    """Test performing an action"""
    print("Test 3: Perform Action...")
    engine = create_game(difficulty=Difficulty.MEDIUM)
    actions = engine.get_valid_actions()
    result = engine.perform_action(actions[0])
    assert result is not None
    print(f"  ✓ Action performed: {result.message}")

def test_random_game():
    """Test running a complete random game"""
    print("Test 4: Random Game...")
    result = play_random_game(difficulty=Difficulty.EASY)
    assert result['winner'] in ['ATTACKER', 'DEFENDER', 'DRAW']
    print(f"  ✓ Winner: {result['winner']}, Turns: {result['turns']}")

def test_state_cloning():
    """Test state cloning for AI"""
    print("Test 5: State Cloning...")
    engine = create_game(difficulty=Difficulty.MEDIUM)
    state_copy = engine.get_state_copy()
    assert len(state_copy.nodes) == len(engine.state.nodes)
    print(f"  ✓ State cloned successfully")

def test_state_evaluation():
    """Test state evaluation function"""
    print("Test 6: State Evaluation...")
    engine = create_game(difficulty=Difficulty.MEDIUM)
    attacker_score = engine.evaluate_state(engine.state, PlayerRole.ATTACKER)
    defender_score = engine.evaluate_state(engine.state, PlayerRole.DEFENDER)
    print(f"  ✓ Attacker eval: {attacker_score:.2f}, Defender eval: {defender_score:.2f}")

def test_multiple_games():
    """Test running multiple games"""
    print("Test 7: Multiple Games...")
    for i in range(5):
        result = play_random_game(difficulty=Difficulty.EASY)
        print(f"  Game {i+1}: {result['winner']} wins in {result['turns']} turns")
    print("  ✓ All games completed")

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Cyber Warfare Strategy Game - Core Engine Tests")
    print("=" * 50 + "\n")
    
    try:
        test_game_creation()
        test_get_actions()
        test_perform_action()
        test_random_game()
        test_state_cloning()
        test_state_evaluation()
        test_multiple_games()
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED! ✓")
        print("=" * 50 + "\n")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
