from gymnasium.envs.registration import register
'''
This script registers the custom environment "Spatial-Gym" with Gymnasium.
'''
register(
    id="Spatial-Gym",  # Unique identifier for your environment
    entry_point="Spatial_Gym.Spatial_Gym:Spatial_Gym",  # Path to your environment class
)