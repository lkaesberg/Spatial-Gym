import pygame
import math


class LLMRenderer:
    """
    Renderer for LLM visualization using pygame (the current implementation).
    """
    
    def __init__(self):
        self.screen = None
        self.clock = None
        self.initialized = False
    
    def initialize(self, x_size, y_size):
        """Initialize pygame and create the display window."""
        if not self.initialized:
            pygame.init()
            self.screen = pygame.display.set_mode((x_size * 40, y_size * 40))
            pygame.display.set_caption("WitnessEnv Visualization")
            self.clock = pygame.time.Clock()
            self.initialized = True
    
    def close(self):
        """Close the pygame display."""
        if self.initialized:
            pygame.display.quit()
            self.initialized = False
    
    def render(self, obs_array, color_array, additional_info, polyshapes, x_size, y_size):
        """
        Visualizes the current state of the environment using pygame.
        """
        if not self.initialized:
            self.initialize(x_size, y_size)
            
        cell_size = 40
        margin = 2

        self.screen.fill((255, 255, 255))  # White background

        # Draw grid and elements
        for y in range(y_size):
            for x in range(x_size):
                rect = pygame.Rect(x * cell_size, y * cell_size, cell_size - margin, cell_size - margin)
                color = (200, 200, 200)  # Default: light gray

                # Draw visited cells
                if obs_array["visited"][x, y]:
                    color = (180, 255, 180) # Light green for visited cells, since they can not be visited again but are not gaps
                # Draw gaps
                if obs_array["gaps"][x, y]:
                    color = (0, 128, 0) # dark Green for gaps
                # Draw agent
                if obs_array["agent_location"][x, y]:
                    color = (0, 0, 255) # Blue for agent
                # Draw target
                if obs_array["target_location"][x, y]:
                    color = (255, 0, 0) # Red for target

                pygame.draw.rect(self.screen, color, rect)

         # Draw other properties
        for prop, array in obs_array.items():
            if prop in ["visited", "gaps", "agent_location", "target_location"]:
                continue  # Skip already visualized properties

            for y in range(y_size):
                for x in range(x_size):
                    if array[x, y]:  # If the property exists at this cell
                        center = (x * cell_size + cell_size // 2, y * cell_size + cell_size // 2)

                        prop_type = prop  # e.g., "star", "poly", "triangle", "dot"
                        color = color_array[x, y]  # Get the color from the color array
                        color = self._get_color_from_name(color)  # Extract color

                        # Draw the property based on its type
                        # Draw a star
                        if prop_type == "star":
                            self._draw_star(self.screen, color, center, cell_size // 4)
                        
                        # Draw a polyshape
                        elif prop_type == "poly":
                            shape = f'{additional_info[x, y] }' # Get the polyshape name from additional_info
                            shape_array = polyshapes[shape]
                            top_left = (x * cell_size, y * cell_size)
                            self._draw_polyshape(self.screen, shape_array, top_left, cell_size, color)

                            # Add text "poly" for polyshape
                            font = pygame.font.Font(None, 18)
                            text = font.render("poly", True, (255, 255, 255))
                            shadow = font.render("poly", True, (0, 0, 0))
                            text_rect = text.get_rect(center=(x * cell_size + cell_size // 2, y * cell_size + cell_size // 2 + 8))
                            shadow_rect = text_rect.copy()
                            shadow_rect.x += 1
                            shadow_rect.y += 1
                            self.screen.blit(shadow, shadow_rect)
                            self.screen.blit(text, text_rect)
                        
                        # Draw a polyshape with "ylop" type
                        elif prop_type == "ylop":
                            shape = f'{additional_info[x, y]}'  # Get the polyshape name from additional_info
                            shape_array = polyshapes[shape]
                            top_left = (x * cell_size, y * cell_size)
                            self._draw_polyshape(self.screen, shape_array, top_left, cell_size, color)
                            
                            # Add text "ylop"
                            font = pygame.font.Font(None, 18)
                            text = font.render("ylop", True, (255, 255, 255))
                            shadow = font.render("ylop", True, (0, 0, 0))
                            text_rect = text.get_rect(center=(x * cell_size + cell_size // 2, y * cell_size + cell_size // 2 + 8))
                            shadow_rect = text_rect.copy()
                            shadow_rect.x += 1
                            shadow_rect.y += 1
                            self.screen.blit(shadow, shadow_rect)
                            self.screen.blit(text, text_rect)
                        
                        # Draw a triangle with a count    
                        elif prop_type == "triangle":
                            pygame.draw.polygon(self.screen, color, [
                                (center[0], center[1] - cell_size // 4),  # Top
                                (center[0] - cell_size // 4, center[1] + cell_size // 4),  # Bottom-left
                                (center[0] + cell_size // 4, center[1] + cell_size // 4)   # Bottom-right
                            ])
                            
                            # Add text for triangle count
                            count = f'{additional_info[x, y]}'  # Get the count from additional_info 
                            font = pygame.font.Font(None, 28) 
                            text = font.render(count, True, (255, 255, 255)) 
                            shadow = font.render(count, True, (0, 0, 0))
                            shadow_pos = (center[0] - 7 + 1, center[1] - 20 + 1)
                            self.screen.blit(shadow, shadow_pos)
                            text_pos = (center[0] - 7, center[1] - 20)
                            self.screen.blit(text, text_pos)
                        
                        # Draw a square
                        elif prop_type == "square":
                            square_size = cell_size // 2
                            square_rect = pygame.Rect(
                                center[0] - square_size // 2,
                                center[1] - square_size // 2,
                                square_size,
                                square_size
                            )
                            pygame.draw.rect(self.screen, color, square_rect)
                        
                        # Draw a dot    
                        elif prop_type == "dot":
                            pygame.draw.circle(self.screen, (0, 0, 0), center, cell_size // 8)

        
        pygame.display.flip()
        self.clock.tick(30)  # Limit to 30 FPS

        # Note: Event handling removed from render method to avoid conflicts
        # Event handling should be done by the caller (e.g., play_human function)

    def _draw_polyshape(self, surface, shape_array, top_left, cell_size, color):
        """
        Draws a polyshape inside a given cell.

        Parameters:
            surface (pygame.Surface): The Pygame surface to draw on.
            shape_array (list of lists): A 2D array representing the polyshape, where 1 indicates a filled cell and 0 indicates an empty cell.
            top_left (tuple): The top-left corner of the cell
            cell_size (int): The size of the cell in pixels.
            color (tuple): The RGB color to use for the polyshape.
        """
        
        shape_height = len(shape_array)
        shape_width = len(shape_array[0])

        padding = cell_size // 6

        mini_block_width = (cell_size - 2 * padding) // shape_width
        mini_block_height = (cell_size - 2 * padding) // shape_height

        for y, row in enumerate(shape_array):
            for x, val in enumerate(row):
                if val:
                    rect = pygame.Rect(
                        top_left[0] + padding + x * mini_block_width,
                        top_left[1] + padding + y * mini_block_height,
                        mini_block_width,
                        mini_block_height
                    )
                    pygame.draw.rect(surface, color, rect)
    
    def _draw_star(self, surface, color, center, radius):
        """
        Draws a star shape at a specified location.

        Parameters:
            surface (pygame.Surface): The Pygame surface to draw on.
            color (tuple): The RGB color to use for the star.
            center (tuple): The (x, y) coordinates of the center of the star.
            radius (int): The radius of the star, determining its size.
        """
        points = []
        for i in range(10):
            angle = math.pi / 2 + i * math.pi / 5
            r = radius if i % 2 == 0 else radius // 2
            x = center[0] + int(math.cos(angle) * r)
            y = center[1] - int(math.sin(angle) * r)
            points.append((x, y))
        pygame.draw.polygon(surface, color, points)
        
    def _get_color_from_name(self, color):
        """
        Helper function to extract color from property name parts.
        """
        if color == 1:
            return (255, 0, 0)  # Red
        elif color == 2:
            return (0, 0, 255)  # Blue
        elif color == 3:
            return (255, 255, 0)  # Yellow
        elif color == 4:
            return (0, 255, 0)  # Green
        elif color == 5:
            return (0, 0, 0)  # Black
        elif color == 6:
            return (128, 0, 128)  # Purple
        elif color == 7:
            return (255, 165, 0)  # Orange
        elif color == 8:
            return (255, 255, 255)  # White
        else:
            return (128, 128, 128)  # Default: Gray 