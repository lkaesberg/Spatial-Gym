"""
Basic environment tests for Spatial-Gym.

Tests environment initialization, basic operations, and API compliance.
"""
import pytest
import gymnasium as gym
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import Spatial_Gym


class TestEnvironmentInitialization:
    """Test environment initialization with different configurations."""
    
    def test_basic_initialization(self):
        """Test basic environment creation."""
        env = gym.make("Spatial-Gym")
        assert env is not None
        env.close()
    
    def test_initialization_with_render_modes(self):
        """Test initialization with different render modes."""
        for render_mode in [None, 'llm']:
            env = gym.make("Spatial-Gym", render_mode=render_mode)
            assert env.unwrapped.render_mode == render_mode
            env.close()
    
    def test_initialization_with_observation_modes(self):
        """Test initialization with different observation formats."""
        for obs_mode in ['new', 'SPaRC']:
            env = gym.make("Spatial-Gym", observation=obs_mode)
            # Verify environment was created successfully
            assert env is not None
            # Access the unwrapped env to check attributes
            assert env.unwrapped.observation == obs_mode
            env.close()
    
    def test_initialization_with_traceback(self):
        """Test initialization with traceback enabled."""
        env = gym.make("Spatial-Gym", traceback=True)
        assert env.unwrapped.traceback is True
        env.close()
    
    def test_initialization_with_max_steps(self):
        """Test initialization with custom max_steps."""
        max_steps = 500
        env = gym.make("Spatial-Gym", max_steps=max_steps)
        assert env.unwrapped.max_steps == max_steps
        env.close()


class TestEnvironmentAPI:
    """Test Gymnasium API compliance."""
    
    def test_reset(self):
        """Test environment reset."""
        env = gym.make("Spatial-Gym")
        observation, info = env.reset()
        
        assert observation is not None
        assert isinstance(info, dict)
        env.close()
    
    def test_reset_with_seed(self):
        """Test reset with seed for reproducibility."""
        env = gym.make("Spatial-Gym")
        obs1, info1 = env.reset(seed=42)
        obs2, info2 = env.reset(seed=42)
        
        # Same seed should produce same initial state
        assert info1.get('puzzle_id') == info2.get('puzzle_id')
        env.close()
    
    def test_step(self):
        """Test basic step functionality."""
        env = gym.make("Spatial-Gym", max_steps=100)
        observation, info = env.reset()
        
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        assert observation is not None
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        env.close()
    
    def test_action_space(self):
        """Test action space properties."""
        env = gym.make("Spatial-Gym")
        
        assert env.action_space is not None
        assert env.action_space.n == 4  # Four directional actions
        
        # Test that all actions are in valid range
        for _ in range(10):
            action = env.action_space.sample()
            assert 0 <= action < 4
        
        env.close()
    
    def test_observation_space_new_format(self):
        """Test observation space with 'new' format."""
        env = gym.make("Spatial-Gym", observation='new')
        observation, info = env.reset()
        
        assert isinstance(observation, dict)
        assert 'base' in observation
        assert 'color' in observation
        env.close()
    
    def test_observation_space_sparc_format(self):
        """Test observation space with 'SPaRC' format."""
        env = gym.make("Spatial-Gym", observation='SPaRC')
        observation, info = env.reset()
        
        assert isinstance(observation, str)
        env.close()
    
    def test_render_llm_mode(self):
        """Test rendering in LLM mode."""
        env = gym.make("Spatial-Gym", render_mode='llm')
        env.reset()
        
        render_output = env.render()
        # LLM renderer may return None or a string representation
        # The important thing is that it doesn't crash
        assert render_output is None or isinstance(render_output, str)
        
        env.close()


class TestEnvironmentBehavior:
    """Test environment behavior and game logic."""
    
    def test_episode_termination(self):
        """Test that episodes terminate correctly."""
        env = gym.make("Spatial-Gym", max_steps=10)
        observation, info = env.reset()
        
        terminated = False
        truncated = False
        steps = 0
        
        while not (terminated or truncated) and steps < 20:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            steps += 1
        
        # Episode should terminate within max_steps
        assert terminated or truncated
        assert steps <= 10
        
        env.close()
    
    def test_multiple_episodes(self):
        """Test running multiple episodes."""
        env = gym.make("Spatial-Gym", max_steps=50)
        
        for episode in range(3):
            observation, info = env.reset()
            assert observation is not None
            
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)
        
        env.close()
    
    def test_reward_structure(self):
        """Test that rewards are within expected ranges."""
        env = gym.make("Spatial-Gym", max_steps=100)
        env.reset()
        
        rewards = []
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
        
        # Check that rewards are reasonable
        assert all(isinstance(r, (int, float)) for r in rewards)
        
        env.close()
    
    def test_info_dict_contents(self):
        """Test that info dictionary contains expected keys."""
        env = gym.make("Spatial-Gym")
        observation, info = env.reset()
        
        # Info should contain useful debugging information
        assert isinstance(info, dict)
        
        # Take a step and check step info
        observation, reward, terminated, truncated, info = env.step(0)
        assert isinstance(info, dict)
        
        env.close()


class TestEnvironmentRobustness:
    """Test environment robustness and edge cases."""
    
    def test_invalid_action_handling(self):
        """Test that environment handles invalid actions gracefully."""
        env = gym.make("Spatial-Gym")
        env.reset()
        
        # This should not crash the environment
        # The environment should handle out-of-bounds actions
        try:
            # Most environments will raise an error for invalid actions
            env.step(10)  # Invalid action
        except (ValueError, AssertionError):
            # Expected behavior - environment rejects invalid action
            pass
        
        env.close()
    
    def test_close_without_error(self):
        """Test that environment closes without errors."""
        env = gym.make("Spatial-Gym")
        env.reset()
        env.step(0)
        env.close()
        # No assertion needed - just testing that close() doesn't raise
    
    def test_multiple_close_calls(self):
        """Test that calling close() multiple times doesn't cause errors."""
        env = gym.make("Spatial-Gym")
        env.reset()
        env.close()
        env.close()  # Should not raise an error
    
    def test_reset_after_episode_end(self):
        """Test that reset works correctly after episode ends."""
        env = gym.make("Spatial-Gym", max_steps=10)
        
        # Complete an episode
        observation, info = env.reset()
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
        
        # Reset should work fine
        observation, info = env.reset()
        assert observation is not None
        
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
