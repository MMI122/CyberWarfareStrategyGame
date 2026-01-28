#!/usr/bin/env python3
"""
Test Suite for Deep RL Agent

Tests:
1. Neural network forward pass
2. MCTS search functionality
3. PPO rollout buffer
4. Deep RL agent action selection
5. Self-play training simulation
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core import create_game, PlayerRole
from app.ai import (
    DeepRLAgent, DeepRLConfig, TrainingMode,
    ActorCriticNetwork, NetworkConfig,
    MCTS, MCTSConfig,
    PPOTrainer, PPOConfig, RolloutBuffer,
    RandomAgent, HeuristicAgent
)


def test_neural_network():
    """Test neural network forward pass"""
    print("\nTest 1: Neural Network Forward Pass...")
    
    # Create game and network
    engine = create_game()
    config = NetworkConfig(hidden_dim=128, num_layers=2)
    network = ActorCriticNetwork(config=config)
    
    # Forward pass
    action_probs, target_probs, value = network.forward(engine.state)
    
    # Validate outputs
    assert action_probs.shape[0] > 0, "Action probs should not be empty"
    assert target_probs.shape[0] > 0, "Target probs should not be empty"
    assert isinstance(value, float), "Value should be a float"
    
    # Check probabilities sum to ~1
    action_sum = np.sum(action_probs)
    target_sum = np.sum(target_probs)
    assert 0.99 <= action_sum <= 1.01, f"Action probs should sum to 1, got {action_sum}"
    assert 0.99 <= target_sum <= 1.01, f"Target probs should sum to 1, got {target_sum}"
    
    print(f"  ✓ Action probs shape: {action_probs.shape}")
    print(f"  ✓ Target probs shape: {target_probs.shape}")
    print(f"  ✓ Value estimate: {value:.4f}")


def test_mcts_search():
    """Test MCTS search functionality"""
    print("\nTest 2: MCTS Search...")
    
    engine = create_game()
    network = ActorCriticNetwork()
    
    def policy_value_fn(state):
        action_probs, _, value = network.forward(state)
        return action_probs, value
    
    def get_valid_actions_fn(state):
        # Use engine's method which uses current_player from state
        return engine.get_valid_actions()
    
    def simulate_action_fn(state, action):
        # simulate_action takes (state, action) and returns (new_state, result)
        new_state, result = engine.simulate_action(state, action)
        return new_state
    
    config = MCTSConfig(num_simulations=50, temperature=1.0)
    mcts = MCTS(
        policy_value_fn=policy_value_fn,
        get_valid_actions_fn=get_valid_actions_fn,
        simulate_action_fn=simulate_action_fn,
        config=config
    )
    
    # Run search
    results = mcts.search(engine.state)
    
    # Validate results
    assert 'action_probs' in results, "Results should have action_probs"
    assert 'simulations' in results, "Results should have simulations"
    assert results['simulations'] > 0, "Should have run simulations"
    
    print(f"  ✓ Simulations: {results['simulations']}")
    print(f"  ✓ Max depth: {results['max_depth']}")
    print(f"  ✓ Nodes created: {results['nodes_created']}")
    print(f"  ✓ Time: {results['time_seconds']:.3f}s")


def test_ppo_rollout_buffer():
    """Test PPO rollout buffer"""
    print("\nTest 3: PPO Rollout Buffer...")
    
    buffer = RolloutBuffer()
    
    # Add some transitions
    for i in range(10):
        state = np.random.randn(64)
        action = f"action_{i}"
        log_prob = np.random.randn()
        reward = np.random.randn()
        value = np.random.randn()
        done = i == 9
        
        buffer.add(state, action, log_prob, reward, value, done)
    
    # Compute returns
    buffer.compute_returns_and_advantages(last_value=0.0, gamma=0.99, gae_lambda=0.95)
    
    # Validate
    assert len(buffer) == 10, f"Buffer should have 10 transitions, got {len(buffer)}"
    assert buffer.advantages is not None, "Advantages should be computed"
    assert buffer.returns is not None, "Returns should be computed"
    assert len(buffer.advantages) == 10, "Advantages should have 10 values"
    
    # Test batching
    batches = buffer.get_batches(batch_size=4)
    assert len(batches) > 0, "Should create batches"
    
    print(f"  ✓ Buffer size: {len(buffer)}")
    print(f"  ✓ Advantages shape: {buffer.advantages.shape}")
    print(f"  ✓ Returns shape: {buffer.returns.shape}")
    print(f"  ✓ Num batches (batch_size=4): {len(batches)}")


def test_deep_rl_agent_action():
    """Test Deep RL agent action selection"""
    print("\nTest 4: Deep RL Agent Action Selection...")
    
    engine = create_game()
    
    # Create agent
    config = DeepRLConfig(
        role=PlayerRole.ATTACKER,
        use_mcts=False,  # Direct network inference
        network_config=NetworkConfig(hidden_dim=128)
    )
    agent = DeepRLAgent(role=PlayerRole.ATTACKER, config=config)
    
    # Get action without MCTS
    action = agent.get_action(engine, use_mcts=False)
    
    assert action is not None, "Agent should return an action"
    print(f"  ✓ Action (no MCTS): {action.action_type.name} targeting node {getattr(action, 'target_node_id', None) or getattr(action, 'target_node', None)}")
    
    # Get action with MCTS (fewer simulations for speed)
    config.mcts_simulations_eval = 20
    action_mcts = agent.get_action(engine, use_mcts=True, temperature=1.0)
    
    assert action_mcts is not None, "Agent should return an action with MCTS"
    print(f"  ✓ Action (with MCTS): {action_mcts.action_type.name} targeting node {getattr(action_mcts, 'target_node_id', None) or getattr(action_mcts, 'target_node', None)}")


def test_deep_rl_vs_random():
    """Test Deep RL agent against random opponent"""
    print("\nTest 5: Deep RL vs Random Game...")
    
    engine = create_game()
    
    # Create agents
    deep_rl_agent = DeepRLAgent(
        role=PlayerRole.ATTACKER,
        config=DeepRLConfig(
            role=PlayerRole.ATTACKER,
            use_mcts=False,  # Faster for testing
            network_config=NetworkConfig(hidden_dim=64)
        )
    )
    
    random_agent = RandomAgent(role=PlayerRole.DEFENDER)
    
    # Play game
    moves = 0
    max_moves = 30
    
    while moves < max_moves:
        victory_result = engine.state.check_victory_conditions()
        if victory_result is not None:
            winner, _ = victory_result
            break
        winner = None
        
        current = engine.state.current_player
        
        if current == PlayerRole.ATTACKER:
            action = deep_rl_agent.get_action(engine)
        else:
            valid_actions = engine.get_valid_actions()
            action = np.random.choice(valid_actions) if valid_actions else None
        
        if action:
            result = engine.perform_action(action)
            moves += 1
            if moves <= 5:
                print(f"  Move {moves}: {action.action_type.name} -> {'Success' if result.success else 'Failed'}")
    
    if moves <= 5:
        print(f"  ...")
    print(f"  ✓ Game completed after {moves} moves")
    print(f"  ✓ Winner: {winner.name if winner else 'None (in progress)'}")


def test_heuristic_agent():
    """Test heuristic agent"""
    print("\nTest 6: Heuristic Agent...")
    
    engine = create_game()
    
    agent = HeuristicAgent(role=PlayerRole.ATTACKER)
    
    action = agent.get_action(engine)
    assert action is not None, "Heuristic agent should return an action"
    
    print(f"  ✓ Action: {action.action_type.name}")
    
    # Play a few turns
    for i in range(5):
        victory_result = engine.state.check_victory_conditions()
        if victory_result is not None:
            break
        
        action = agent.get_action(engine)
        if action:
            result = engine.perform_action(action)
            print(f"  Move {i+1}: {action.action_type.name} -> {'✓' if result.success else '✗'}")


def test_network_save_load():
    """Test network save/load functionality"""
    print("\nTest 7: Network Save/Load...")
    
    import tempfile
    import os
    
    # Create network
    network = ActorCriticNetwork(config=NetworkConfig(hidden_dim=64))
    
    # Save to temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'test_network')
        network.save(filepath)
        
        # Verify file exists
        assert os.path.exists(filepath + '.npz'), "Save file should exist"
        
        # Load - should not throw
        network.load(filepath + '.npz')
        
        # Forward pass should work after load
        engine = create_game()
        action_probs, target_probs, value = network.forward(engine.state)
        
        # Basic validation
        assert action_probs.shape[0] > 0, "Should have action probs"
        assert target_probs.shape[0] > 0, "Should have target probs"
    
    print(f"  ✓ Network saved and loaded successfully")
    print(f"  ✓ Forward pass works after load")


def main():
    """Run all tests"""
    print("=" * 50)
    print("Deep RL Agent Tests")
    print("=" * 50)
    
    try:
        test_neural_network()
        test_mcts_search()
        test_ppo_rollout_buffer()
        test_deep_rl_agent_action()
        test_deep_rl_vs_random()
        test_heuristic_agent()
        test_network_save_load()
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED! ✓")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
