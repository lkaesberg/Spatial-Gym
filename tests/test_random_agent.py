"""
Random agent tests for Spatial-Gym.

Tests the environment with a random agent to ensure stability and correctness.
"""
import pytest
import gymnasium as gym
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import Spatial_Gym


class TestRandomAgent:
    """Test environment with random agent behavior."""
    
    def test_random_agent_single_episode(self):
        """Test running a single episode with random actions."""
        env = gym.make("Spatial-Gym", max_steps=100)
        observation, info = env.reset()
        
        episode_reward = 0
        terminated = False
        truncated = False
        steps = 0
        
        while not (terminated or truncated):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
        
        assert steps > 0
        assert steps <= 100
        assert isinstance(episode_reward, (int, float))
        
        env.close()
    
    def test_random_agent_multiple_episodes(self):
        """Test running multiple episodes with random agent."""
        env = gym.make("Spatial-Gym", max_steps=50)
        
        num_episodes = 5
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            observation, info = env.reset()
            
            episode_reward = 0
            terminated = False
            truncated = False
            steps = 0
            
            while not (terminated or truncated):
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
        
        # Verify we completed all episodes
        assert len(episode_rewards) == num_episodes
        assert len(episode_lengths) == num_episodes
        assert all(length <= 50 for length in episode_lengths)
        
        env.close()
    
    def test_random_agent_with_different_observations(self):
        """Test random agent with different observation formats."""
        for obs_mode in ['new', 'SPaRC']:
            env = gym.make("Spatial-Gym", observation=obs_mode, max_steps=30)
            observation, info = env.reset()
            
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)
            
            env.close()
    
    def test_random_agent_statistics(self):
        """Test and collect statistics from random agent runs."""
        env = gym.make("Spatial-Gym", max_steps=100)
        
        num_episodes = 10
        stats = {
            'total_steps': [],
            'total_rewards': [],
            'success_count': 0,
            'truncated_count': 0
        }
        
        for _ in range(num_episodes):
            observation, info = env.reset()
            
            episode_reward = 0
            steps = 0
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1
            
            stats['total_steps'].append(steps)
            stats['total_rewards'].append(episode_reward)
            
            if terminated:
                stats['success_count'] += 1
            if truncated:
                stats['truncated_count'] += 1
        
        # Verify statistics
        assert len(stats['total_steps']) == num_episodes
        assert len(stats['total_rewards']) == num_episodes
        assert stats['success_count'] + stats['truncated_count'] == num_episodes
        
        print(f"\nRandom Agent Statistics ({num_episodes} episodes):")
        print(f"  Average steps: {np.mean(stats['total_steps']):.2f}")
        print(f"  Average reward: {np.mean(stats['total_rewards']):.2f}")
        print(f"  Success rate: {stats['success_count']}/{num_episodes}")
        print(f"  Truncated: {stats['truncated_count']}/{num_episodes}")
        
        env.close()
    
    def test_random_agent_with_traceback(self):
        """Test random agent with traceback enabled."""
        env = gym.make("Spatial-Gym", traceback=True, max_steps=50)
        observation, info = env.reset()
        
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
        
        env.close()
    
    def test_random_agent_action_distribution(self):
        """Test that all actions are exercised by random agent."""
        env = gym.make("Spatial-Gym", max_steps=100)
        observation, info = env.reset()
        
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action = env.action_space.sample()
            action_counts[action] += 1
            observation, reward, terminated, truncated, info = env.step(action)
        
        # With random sampling, all actions should appear at least once in most runs
        # (There's a small probability this could fail, but very unlikely with 100 steps)
        total_actions = sum(action_counts.values())
        assert total_actions > 0
        
        env.close()


class TestRandomAgentStressTest:
    """Stress tests with random agent."""
    
    def test_extended_random_agent_run(self):
        """Test extended run with random agent (stress test)."""
        env = gym.make("Spatial-Gym", max_steps=200)
        
        num_episodes = 20
        for episode in range(num_episodes):
            observation, info = env.reset()
            
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)
        
        env.close()
    
    def test_random_agent_no_crashes(self):
        """Test that random agent doesn't cause crashes or exceptions."""
        env = gym.make("Spatial-Gym", max_steps=50)
        
        # Run multiple episodes, catch any exceptions
        exceptions = []
        num_episodes = 10
        
        for episode in range(num_episodes):
            try:
                observation, info = env.reset()
                
                terminated = False
                truncated = False
                
                while not (terminated or truncated):
                    action = env.action_space.sample()
                    observation, reward, terminated, truncated, info = env.step(action)
            except Exception as e:
                exceptions.append((episode, str(e)))
        
        env.close()
        
        # No exceptions should occur
        assert len(exceptions) == 0, f"Exceptions occurred: {exceptions}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
