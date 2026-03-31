import pygame

def play_human(env, id=None):
    """
    Allows a human player to interact with the environment using keyboard inputs.

    Parameters:
        env (gym.Env): The environment to interact with.
        i (int): The number of times to reset the environment before starting (default is 1).

    Returns:
        obs (dict): The final observation of the environment after the episode ends.
        reward (int): The reward obtained at the end of the episode.
        info (dict): Additional information about the environment at the end of the episode.

    Description:
        - The player uses arrow keys to control the agent:
            - Right arrow: Move right
            - Up arrow: Move up
            - Left arrow: Move left
            - Down arrow: Move down
        - The game ends when the episode is terminated or truncated.
        - The environment is rendered after each step to visualize the current state.
    """
    
    if id is not None:
        options = {'puzzle_id': id}
        obs, info = env.reset(options=options)
    else:
        obs, info = env.reset()

    running = True

    while running:
        action = None
        for event in pygame.event.get():
            # Handle quitting the game
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                return obs, reward, info
            
            # Check for key presses to determine the action
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    action = 0
                elif event.key == pygame.K_UP:
                    action = 1
                elif event.key == pygame.K_LEFT:
                    action = 2
                elif event.key == pygame.K_DOWN:
                    action = 3
        
        # If a valid action is selected, step the environment
        # and render the new state
        if action is not None and action in info["legal_actions"]:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # If the episode is done, exit the loop and return the final observation, reward, and info
            if done:
                running = False
                print("Episode finished")
                return obs, reward, info
