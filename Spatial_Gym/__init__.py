"""
This module initializes the Spatial-Gym environment package.

It imports the `Spatial_Gym` class and registers the environment for use with Gymnasium.

Modules:
    - Spatial_Gym: Contains the implementation of the Spatial-Gym environment.
    - register_env: Handles the registration of the Spatial-Gym environment with Gymnasium.
"""

from .Spatial_Gym import Spatial_Gym
from .register_env import *