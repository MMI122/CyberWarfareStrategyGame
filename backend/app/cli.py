# =============================================================================
# Cyber Warfare Strategy Game - Command Line Interface
# =============================================================================
"""
Simple CLI for testing and playing the game.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.core import (
    GameEngine, GameState, create_game, play_random_game,
    Difficulty, ActionType, PlayerRole, NodeStatus,
    GameAction, GameEventType
)


def print_header():
    """Print game header"""
    print("\n" + "=" * 60)
    print("   CYBER WARFARE STRATEGY GAME")
    print("   Turn-based Attacker vs Defender")
    print("=" * 60 + "\n")


def print_state(engine: GameEngine):
    """Print current game state"""
    state = engine.state
    if not state:
        print("No game in progress")
        return
    
    print(f"\n{'='*50}")
    print(f"Turn {state.turn_number} | {state.current_player.name}'s Turn")
    print(f"{'='*50}")
    
    # Print scores
    print(f"\nScores: Attacker: {state.attacker.score} | Defender: {state.defender.score}")
    
    # Print current player info
    player = engine.get_current_player()
    print(f"\nAction Points: {player.action_points}/{player.max_action_points}")
    
    if player.role == PlayerRole.ATTACKER:
        print(f"Controlled Nodes: {len(player.controlled_nodes)}")
        print(f"Data Stolen: {player.data_stolen}")
        print(f"Backdoors: {len(player.backdoors_installed)}")
    
    # Print network summary
    online = sum(1 for n in state.nodes.values() if n.status == NodeStatus.ONLINE)
    compromised = sum(1 for n in state.nodes.values() if n.status == NodeStatus.COMPROMISED)
    offline = sum(1 for n in state.nodes.values() if n.status == NodeStatus.OFFLINE)
    
    print(f"\nNetwork Status:")
    print(f"  Online: {online} | Compromised: {compromised} | Offline: {offline}")
    print(f"  Total Nodes: {len(state.nodes)}")


def print_visible_nodes(engine: GameEngine):
    """Print nodes visible to the current player"""
    state = engine.state
    if not state:
        return
    
    player = engine.get_current_player()
    print(f"\n{'='*50}")
    print("Visible Nodes:")
    print(f"{'='*50}")
    
    for node_id, node in state.nodes.items():
        # Defender sees all, attacker only sees visible nodes
        if player.role == PlayerRole.DEFENDER or node.visible_to_attacker:
            status_icon = {
                NodeStatus.ONLINE: "🟢",
                NodeStatus.COMPROMISED: "🔴",
                NodeStatus.INFECTED: "🟠",
                NodeStatus.OFFLINE: "⚫",
                NodeStatus.ISOLATED: "🔵",
            }.get(node.status, "⚪")
            
            controlled = "👤" if node_id in player.controlled_nodes else ""
            scanned = "🔍" if node.scanned else ""
            
            print(f"  [{node_id:2d}] {status_icon} {node.name[:25]:<25} "
                  f"HP:{node.health:3d} {controlled}{scanned}")


def print_actions(actions: list):
    """Print available actions"""
    print(f"\n{'='*50}")
    print("Available Actions:")
    print(f"{'='*50}")
    
    # Group by action type
    by_type = {}
    for i, action in enumerate(actions):
        action_type = action.action_type.name
        if action_type not in by_type:
            by_type[action_type] = []
        by_type[action_type].append((i, action))
    
    for action_type, action_list in by_type.items():
        print(f"\n  {action_type}:")
        for idx, action in action_list[:5]:  # Limit display
            if action.target_node_id >= 0:
                print(f"    [{idx}] Target node {action.target_node_id}")
            else:
                print(f"    [{idx}] {action_type}")
        if len(action_list) > 5:
            print(f"    ... and {len(action_list) - 5} more")


def interactive_game():
    """Run an interactive game session"""
    print_header()
    
    # Select difficulty
    print("Select Difficulty:")
    print("  1. Easy")
    print("  2. Medium")
    print("  3. Hard")
    print("  4. Expert")
    
    choice = input("\nChoice [1-4]: ").strip()
    difficulty_map = {"1": Difficulty.EASY, "2": Difficulty.MEDIUM, 
                      "3": Difficulty.HARD, "4": Difficulty.EXPERT}
    difficulty = difficulty_map.get(choice, Difficulty.MEDIUM)
    
    # Create game
    engine = create_game(difficulty=difficulty, topology="corporate")
    
    print(f"\nGame created with {len(engine.state.nodes)} nodes")
    
    # Game loop
    while not engine.is_game_over():
        print_state(engine)
        print_visible_nodes(engine)
        
        actions = engine.get_valid_actions()
        if not actions:
            print("\nNo valid actions available!")
            break
        
        print_actions(actions)
        
        # Get player choice
        choice = input("\nEnter action number (or 'q' to quit): ").strip()
        
        if choice.lower() == 'q':
            print("Game ended by player.")
            break
        
        try:
            action_idx = int(choice)
            if 0 <= action_idx < len(actions):
                action = actions[action_idx]
                result = engine.perform_action(action)
                print(f"\n→ {result.message}")
                if result.points_gained:
                    print(f"  Points: +{result.points_gained}")
                if result.was_detected:
                    print("  ⚠️ Action was detected!")
            else:
                print("Invalid action number")
        except ValueError:
            print("Please enter a valid number")
    
    # Game over
    if engine.is_game_over():
        print(f"\n{'='*60}")
        print("GAME OVER!")
        print(f"{'='*60}")
        winner = engine.get_winner()
        print(f"Winner: {winner.name if winner else 'DRAW'}")
        print(f"Victory: {engine.state.victory_condition.name if engine.state else 'N/A'}")
        scores = engine.get_scores()
        print(f"Final Scores: Attacker: {scores['attacker']} | Defender: {scores['defender']}")
        
        stats = engine.get_stats()
        print(f"\nGame Statistics:")
        print(f"  Total Actions: {stats['total_actions']}")
        print(f"  Successful Exploits: {stats['successful_exploits']}")
        print(f"  Data Exfiltrated: {stats['data_exfiltrated']}")


def demo_game():
    """Run a quick demo with random actions"""
    print_header()
    print("Running random game demo...")
    
    result = play_random_game(difficulty=Difficulty.EASY)
    
    print(f"\n{'='*50}")
    print("Demo Game Results:")
    print(f"{'='*50}")
    print(f"Winner: {result['winner']}")
    print(f"Turns: {result['turns']}")
    print(f"Scores: {result['scores']}")
    print(f"\nStatistics:")
    for key, value in result['stats'].items():
        print(f"  {key}: {value}")


def test_core():
    """Test core components"""
    print_header()
    print("Testing core components...\n")
    
    # Test 1: Create game
    print("Test 1: Creating game...")
    try:
        engine = create_game(difficulty=Difficulty.MEDIUM)
        print(f"  ✓ Game created with {len(engine.state.nodes)} nodes")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return
    
    # Test 2: Get valid actions
    print("\nTest 2: Getting valid actions...")
    try:
        actions = engine.get_valid_actions()
        print(f"  ✓ Found {len(actions)} valid actions")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return
    
    # Test 3: Perform action
    print("\nTest 3: Performing action...")
    try:
        if actions:
            action = actions[0]
            result = engine.perform_action(action)
            print(f"  ✓ Action performed: {result.message}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return
    
    # Test 4: Clone state
    print("\nTest 4: Cloning state...")
    try:
        state_copy = engine.get_state_copy()
        print(f"  ✓ State cloned successfully")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return
    
    # Test 5: State serialization
    print("\nTest 5: Serializing state...")
    try:
        state_dict = engine.state.to_dict()
        restored = GameState.from_dict(state_dict)
        print(f"  ✓ State serialized and restored")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return
    
    # Test 6: Evaluation function
    print("\nTest 6: State evaluation...")
    try:
        score = engine.evaluate_state(engine.state, PlayerRole.ATTACKER)
        print(f"  ✓ Attacker evaluation: {score:.2f}")
        score = engine.evaluate_state(engine.state, PlayerRole.DEFENDER)
        print(f"  ✓ Defender evaluation: {score:.2f}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)


def main():
    """Main entry point"""
    print_header()
    
    print("Options:")
    print("  1. Play Interactive Game")
    print("  2. Run Demo (Random Game)")
    print("  3. Test Core Components")
    print("  4. Exit")
    
    choice = input("\nChoice [1-4]: ").strip()
    
    if choice == "1":
        interactive_game()
    elif choice == "2":
        demo_game()
    elif choice == "3":
        test_core()
    elif choice == "4":
        print("Goodbye!")
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
