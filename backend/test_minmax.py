#!/usr/bin/env python
"""
Test script for the MinMax AI agent.
"""
import sys
import time
sys.path.insert(0, '.')

from app.core import (
    create_game, GameState, PlayerRole, Difficulty
)
from app.ai import MinMaxAgent, create_minmax_agent


def test_minmax_basic():
    """Test basic MinMax functionality"""
    print("Test 1: Basic MinMax Search...")
    engine = create_game(difficulty=Difficulty.EASY)
    agent = MinMaxAgent(role=PlayerRole.ATTACKER, max_depth=4)
    
    action = agent.get_best_action(engine.state, time_limit=2.0)
    assert action is not None
    print(f"  ✓ Found action: {action}")
    print(f"  Stats: {agent.get_search_stats()}")


def test_minmax_iterative_deepening():
    """Test iterative deepening"""
    print("\nTest 2: Iterative Deepening...")
    engine = create_game(difficulty=Difficulty.MEDIUM)
    agent = MinMaxAgent(role=PlayerRole.ATTACKER, max_depth=8)
    
    action = agent.get_best_action(engine.state, time_limit=3.0)
    
    history = agent.get_search_history()
    print(f"  ✓ Searched {len(history)} depth levels")
    for h in history:
        print(f"    Depth {h['depth']}: {h['nodes']} nodes, "
              f"{h['pruned']} pruned, {h['time_ms']:.1f}ms")


def test_transposition_table():
    """Test transposition table hit rate"""
    print("\nTest 3: Transposition Table...")
    engine = create_game(difficulty=Difficulty.MEDIUM)
    agent = MinMaxAgent(role=PlayerRole.ATTACKER, max_depth=6)
    
    # Run search twice on same state
    agent.get_best_action(engine.state, time_limit=1.0)
    agent.get_best_action(engine.state, time_limit=1.0)
    
    tt_stats = agent.tt.get_stats()
    print(f"  ✓ TT size: {tt_stats['size']}")
    print(f"  ✓ Hit rate: {tt_stats['hit_rate']:.2%}")


def test_minmax_vs_minmax():
    """Test MinMax playing against itself"""
    print("\nTest 4: MinMax vs MinMax Game...")
    engine = create_game(difficulty=Difficulty.EASY)
    
    attacker_agent = MinMaxAgent(role=PlayerRole.ATTACKER, max_depth=3)
    defender_agent = MinMaxAgent(role=PlayerRole.DEFENDER, max_depth=3)
    
    moves = 0
    max_moves = 30
    
    while not engine.is_game_over() and moves < max_moves:
        if engine.state.current_player == PlayerRole.ATTACKER:
            action = attacker_agent.get_best_action(engine.state, time_limit=0.5)
        else:
            action = defender_agent.get_best_action(engine.state, time_limit=0.5)
        
        if action is None:
            break
            
        result = engine.perform_action(action)
        moves += 1
        print(f"  Turn {engine.state.turn_number}: {action.action_type.name} -> {result.message[:40]}")
    
    if engine.is_game_over():
        winner = engine.get_winner()
        print(f"  ✓ Game ended: {winner.name if winner else 'DRAW'} wins!")
    else:
        print(f"  ✓ Game in progress after {moves} moves")


def test_different_difficulties():
    """Test agent at different difficulty levels"""
    print("\nTest 5: Different Difficulty Levels...")
    for diff in [Difficulty.EASY, Difficulty.MEDIUM]:
        engine = create_game(difficulty=diff)
        agent = create_minmax_agent(PlayerRole.ATTACKER, difficulty=diff)
        
        start = time.time()
        action = agent.get_best_action(engine.state, time_limit=1.5)
        elapsed = time.time() - start
        
        stats = agent.get_search_stats()
        print(f"  {diff.name}: depth={stats['depth_reached']}, "
              f"nodes={stats['nodes_searched']}, time={elapsed:.2f}s")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("MinMax AI Agent Tests")
    print("=" * 50 + "\n")
    
    try:
        test_minmax_basic()
        test_minmax_iterative_deepening()
        test_transposition_table()
        test_minmax_vs_minmax()
        test_different_difficulties()
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED! ✓")
        print("=" * 50 + "\n")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
