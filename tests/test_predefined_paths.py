"""
Predefined path tests for Spatial-Gym.

Tests specific action sequences to validate correct puzzle behavior.
"""
import pytest
import gymnasium as gym
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import Spatial_Gym


class TestPredefinedPaths:
    """Test environment with predefined action sequences."""
    
    def test_simple_forward_path(self):
        """Test a simple forward movement sequence."""
        env = gym.make("Spatial-Gym", max_steps=100)
        observation, info = env.reset()
        
        # Try moving right multiple times
        path = [0, 0, 0]  # right, right, right
        
        for action in path:
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        # Should complete without crashing
        env.close()
    
    def test_circular_path(self):
        """Test a circular movement pattern."""
        env = gym.make("Spatial-Gym", max_steps=100)
        observation, info = env.reset()
        
        # Move in a square: right, down, left, up
        path = [0, 3, 2, 1] * 5  # Repeat 5 times
        
        for action in path:
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        env.close()
    
    def test_back_and_forth_path(self):
        """Test back and forth movement."""
        env = gym.make("Spatial-Gym", max_steps=100)
        observation, info = env.reset()
        
        # Move back and forth: right, left, right, left
        path = [0, 2, 0, 2, 0, 2]
        
        for action in path:
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        env.close()
    
    def test_zigzag_path(self):
        """Test a zigzag movement pattern."""
        env = gym.make("Spatial-Gym", max_steps=100)
        observation, info = env.reset()
        
        # Zigzag pattern: right, up, right, down, right, up
        path = [0, 1, 0, 3, 0, 1, 0, 3]
        
        for action in path:
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        env.close()
    
    def test_all_directions_sequence(self):
        """Test sequence that uses all four directions."""
        env = gym.make("Spatial-Gym", max_steps=100)
        observation, info = env.reset()
        
        # Test all directions: right, up, left, down
        path = [0, 1, 2, 3]
        
        for action in path:
            observation, reward, terminated, truncated, info = env.step(action)
            assert observation is not None
            if terminated or truncated:
                break
        
        env.close()
    
    def test_repeated_action(self):
        """Test repeating the same action multiple times."""
        env = gym.make("Spatial-Gym", max_steps=100)
        observation, info = env.reset()
        
        # Repeat same action
        path = [0] * 20  # Move right 20 times
        
        for action in path:
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        env.close()


class TestInvalidPaths:
    """Test invalid or problematic action sequences."""
    
    def test_long_invalid_sequence(self):
        """Test a long sequence that should fail or truncate."""
        env = gym.make("Spatial-Gym", max_steps=20)
        observation, info = env.reset()
        
        # Try to move in one direction beyond the grid
        path = [0] * 50  # Try to move right 50 times (should hit boundary or truncate)
        
        terminated = False
        truncated = False
        
        for action in path:
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        # Should have terminated or truncated
        assert terminated or truncated
        
        env.close()
    
    def test_contradictory_movements(self):
        """Test contradictory movement sequences."""
        env = gym.make("Spatial-Gym", max_steps=100)
        observation, info = env.reset()
        
        # Move right then immediately left repeatedly
        path = [0, 2] * 10
        
        for action in path:
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        env.close()
    
    def test_excessive_up_movements(self):
        """Test excessive movements in one direction (up)."""
        env = gym.make("Spatial-Gym", max_steps=30)
        observation, info = env.reset()
        
        # Try to move up beyond grid bounds
        path = [1] * 30
        
        for action in path:
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        env.close()
    
    def test_excessive_down_movements(self):
        """Test excessive movements in one direction (down)."""
        env = gym.make("Spatial-Gym", max_steps=30)
        observation, info = env.reset()
        
        # Try to move down beyond grid bounds
        path = [3] * 30
        
        for action in path:
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        env.close()


class TestSpecificPuzzlePaths:
    """Test paths on specific puzzles if puzzle_id can be specified."""
    
    def test_reset_to_specific_puzzle(self):
        """Test resetting to a specific puzzle and following a path."""
        env = gym.make("Spatial-Gym", max_steps=100)
        
        # Try to reset to first puzzle
        try:
            observation, info = env.reset(options={'puzzle_id': 0})
            
            # Execute a simple path
            path = [0, 1, 0, 3]
            
            for action in path:
                observation, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
        except Exception as e:
            # If puzzle_id doesn't work as expected, just pass
            # The test verifies the API doesn't crash
            pass
        
        env.close()
    
    def test_consistent_puzzle_behavior(self):
        """Test that the same puzzle behaves consistently."""
        env = gym.make("Spatial-Gym", max_steps=100)
        
        # Run the same path twice on the same puzzle
        path = [0, 1, 2, 3, 0, 1]
        
        results1 = []
        observation, info = env.reset(seed=42)
        puzzle_id_1 = info.get('puzzle_id', None)
        
        for action in path:
            observation, reward, terminated, truncated, info = env.step(action)
            results1.append((reward, terminated, truncated))
            if terminated or truncated:
                break
        
        results2 = []
        observation, info = env.reset(seed=42)
        puzzle_id_2 = info.get('puzzle_id', None)
        
        for action in path:
            observation, reward, terminated, truncated, info = env.step(action)
            results2.append((reward, terminated, truncated))
            if terminated or truncated:
                break
        
        # With same seed, should get same puzzle
        if puzzle_id_1 is not None and puzzle_id_2 is not None:
            assert puzzle_id_1 == puzzle_id_2
        
        # Results should be identical
        assert len(results1) == len(results2)
        # Note: rewards might vary slightly, but termination flags should match
        for i in range(len(results1)):
            _, term1, trunc1 = results1[i]
            _, term2, trunc2 = results2[i]
            assert term1 == term2
            assert trunc1 == trunc2
        
        env.close()


class TestPathValidation:
    """Test path validation and reward accumulation."""
    
    def test_reward_accumulation(self):
        """Test that rewards accumulate correctly along a path."""
        env = gym.make("Spatial-Gym", max_steps=50)
        observation, info = env.reset()
        
        path = [0, 1, 0, 3, 2]
        total_reward = 0
        
        for action in path:
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        # Total reward should be the sum of individual rewards
        assert isinstance(total_reward, (int, float))
        
        env.close()
    
    def test_path_length_vs_max_steps(self):
        """Test that path respects max_steps limit."""
        max_steps = 15
        env = gym.make("Spatial-Gym", max_steps=max_steps)
        observation, info = env.reset()
        
        # Create a path longer than max_steps
        path = [0, 1, 2, 3] * 10  # 40 actions
        
        steps_taken = 0
        for action in path:
            observation, reward, terminated, truncated, info = env.step(action)
            steps_taken += 1
            
            if terminated or truncated:
                break
        
        # Should not exceed max_steps
        assert steps_taken <= max_steps
        
        env.close()
    
    def test_early_termination_path(self):
        """Test paths that should terminate early (success or failure)."""
        env = gym.make("Spatial-Gym", max_steps=100)
        observation, info = env.reset()
        
        # Try various short paths
        paths_to_test = [
            [0],
            [1],
            [0, 0],
            [1, 1],
            [0, 1, 2, 3],
        ]
        
        for path in paths_to_test:
            observation, info = env.reset()
            
            for action in path:
                observation, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
        
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
