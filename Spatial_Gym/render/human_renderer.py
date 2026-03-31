import pygame
import math


def draw_line_round_corners_polygon(surf, p1, p2, c, w):
    p1v = pygame.math.Vector2(p1)
    p2v = pygame.math.Vector2(p2)
    lv = (p2v - p1v).normalize()
    lnv = pygame.math.Vector2(-lv.y, lv.x) * w // 2
    pts = [p1v + lnv, p2v + lnv, p2v - lnv, p1v - lnv]
    pygame.draw.polygon(surf, c, pts)
    pygame.draw.circle(surf, c, p1, round(w / 2))
    pygame.draw.circle(surf, c, p2, round(w / 2))


class HumanRenderer:
    """
    Simple renderer for human-friendly visualization with custom colors.
    """
    
    def __init__(self, scale_factor=1.0):
        self.screen = None
        self.clock = None
        self.initialized = False
        self.current_cells_x = None
        self.current_cells_y = None
        self.scale_factor = scale_factor
        
        # Custom color scheme
        self.GRID_BACKGROUND = (0, 170, 136)    # #00AA88 - Teal for grid cells
        self.OVERALL_BACKGROUND = (17, 56, 51)  # #113833 - Dark green for overall background
        self.LINE_COLOR = (51, 68, 68)          # #334444 - Dark gray-blue for lines
        
    def initialize(self, cells_x, cells_y):
        """Initialize pygame and create the display window."""
        if not self.initialized:
            pygame.init()
            # Calculate window size: cells + 20px padding on each side (scaled)
            cell_size = int(40 * self.scale_factor)
            padding = int(20 * self.scale_factor)
            window_width = cells_x * cell_size + 2 * padding  # cells + padding
            window_height = cells_y * cell_size + 2 * padding  # cells + padding
            self.screen = pygame.display.set_mode((window_width, window_height))
            pygame.display.set_caption("SPaRC Environment - Human View")
            self.clock = pygame.time.Clock()
            self.initialized = True
    
    def close(self):
        """Close the pygame display."""
        if self.initialized:
            pygame.display.quit()
            self.initialized = False
    
    def render(self, obs_array, color_array, additional_info, polyshapes, x_size, y_size, start_location, target_location, path):
        """
        Renders a simple grid with the specified colors.
        
        Parameters:
            obs_array (dict): Dictionary of observation arrays
            color_array (numpy.ndarray): Array containing color information
            additional_info (numpy.ndarray): Array containing additional property info
            polyshapes (dict): Dictionary of polyshape definitions
            x_size (int): Width of the full grid
            y_size (int): Height of the full grid
            
        Returns:
            None
        """
        # Calculate actual puzzle cells (excluding gaps)
        cells_x = (x_size - 1) // 2
        cells_y = (y_size - 1) // 2
        
        # Check if grid size has changed and reinitialize if needed
        if not self.initialized or self.current_cells_x != cells_x or self.current_cells_y != cells_y:
            if self.initialized:
                # Close existing window before creating new one
                pygame.display.quit()
                self.initialized = False
            self.initialize(cells_x, cells_y)
            self.current_cells_x = cells_x
            self.current_cells_y = cells_y
            
        cell_size = int(40 * self.scale_factor)
        padding = int(20 * self.scale_factor)

        # Fill with overall background color
        self.screen.fill(self.OVERALL_BACKGROUND)

        # Draw cells with no gaps (adjacent to each other)
        for grid_y in range(cells_y):
            for grid_x in range(cells_x):
                # Calculate position with padding only
                x_pos = padding + grid_x * cell_size
                y_pos = padding + grid_y * cell_size
                
                # Draw the cell
                rect = pygame.Rect(x_pos, y_pos, cell_size, cell_size)
                pygame.draw.rect(self.screen, self.GRID_BACKGROUND, rect)
        
        # Draw complete grid using the round corner line function
        line_width = int(12 * self.scale_factor)
        gap_size = int(line_width * 1.5)  # Size of the gap
        
        # Calculate grid boundaries with simple integer coordinates
        grid_left = padding
        grid_right = padding + cells_x * cell_size
        grid_top = padding
        grid_bottom = padding + cells_y * cell_size
        
        # Draw all vertical lines with gaps where needed
        for line_x in range(cells_x + 1):
            x_pos = padding + line_x * cell_size
            self._draw_line_with_gaps(
                (x_pos, grid_top), (x_pos, grid_bottom),
                line_x * 2, True, obs_array, line_width, gap_size
            )
        
        # Draw all horizontal lines with gaps where needed
        for line_y in range(cells_y + 1):
            y_pos = padding + line_y * cell_size
            self._draw_line_with_gaps(
                (grid_left, y_pos), (grid_right, y_pos),
                line_y * 2, False, obs_array, line_width, gap_size
            )

        # Use the provided start and target locations
        start_line_x, start_line_y = start_location[0], start_location[1]
        end_line_x, end_line_y = target_location[0], target_location[1]
        
        # Find agent location from environment data
        agent_line_x, agent_line_y = None, None
        for y in range(y_size):
            for x in range(x_size):
                if "agent_location" in obs_array and obs_array["agent_location"][x, y] == 1:
                    agent_line_x, agent_line_y = x, y
                    break
        
        # Draw a large circle at the start position
        start_pixel_x = padding + start_line_x * cell_size//2
        start_pixel_y = padding + start_line_y * cell_size//2
        start_circle_radius = int(cell_size // 4)
        
        # Check if start position has been visited (has white path through it) and choose circle color accordingly
        start_is_visited = False
        if "visited" in obs_array and start_line_x is not None and start_line_y is not None:
            if (0 <= start_line_y < y_size and 0 <= start_line_x < x_size and 
                obs_array["visited"][start_line_x, start_line_y] == 1):
                start_is_visited = True
        
        if start_is_visited:
            # Start position has been visited (white path goes through it), draw in white
            start_circle_color = (255, 255, 255)
        else:
            # Start position hasn't been visited, draw in line color
            start_circle_color = self.LINE_COLOR
            
        pygame.draw.circle(self.screen, start_circle_color, (start_pixel_x, start_pixel_y), start_circle_radius)

        # Draw half-line at end location facing outwards
        end_pixel_x = padding + end_line_x * cell_size//2
        end_pixel_y = padding + end_line_y * cell_size//2
        
        # Determine which edge the end position is on and face outward
        line_length = int(cell_size // 4)
        
        # Check which edge we're on and set line direction accordingly
        if end_line_x == 0:  # Left edge
            line_start = (end_pixel_x, end_pixel_y)
            line_end = (end_pixel_x - line_length, end_pixel_y)  # Face left
        elif end_line_x == cells_x * 2:  # Right edge
            line_start = (end_pixel_x, end_pixel_y)
            line_end = (end_pixel_x + line_length, end_pixel_y)  # Face right
        elif end_line_y == 0:  # Top edge
            line_start = (end_pixel_x, end_pixel_y)
            line_end = (end_pixel_x, end_pixel_y - line_length)  # Face up
        elif end_line_y == cells_y * 2:  # Bottom edge
            line_start = (end_pixel_x, end_pixel_y)
            line_end = (end_pixel_x, end_pixel_y + line_length)  # Face down
        else:  # Default case (shouldn't happen if positioned on edge)
            line_start = (end_pixel_x, end_pixel_y)
            line_end = (end_pixel_x + line_length, end_pixel_y)
        
        # Check if end position has been visited (reached by agent) and choose line color accordingly
        end_is_visited = False
        if "visited" in obs_array and end_line_x is not None and end_line_y is not None:
            if (0 <= end_line_y < y_size and 0 <= end_line_x < x_size and 
                obs_array["visited"][end_line_x, end_line_y] == 1):
                end_is_visited = True
        
        if end_is_visited:
            # End position has been visited (agent reached it), draw in white (agent color)
            end_line_color = (255, 255, 255)
        else:
            # End position hasn't been visited, draw in default line color
            end_line_color = self.LINE_COLOR
        
        draw_line_round_corners_polygon(self.screen, line_start, line_end, end_line_color, line_width)

        # Draw agent path in white
        self._draw_agent_path(path, y_size, x_size, padding, cell_size, line_width)

        # Draw other properties (symbols)
        for prop, array in obs_array.items():
            if prop in ["visited", "gaps", "agent_location", "target_location"]:
                continue  # Skip already visualized properties

            for y in range(y_size):
                for x in range(x_size):
                    if array[x, y]:  # If the property exists at this cell
                        pixel_x = padding + x * cell_size//2
                        pixel_y = padding + y * cell_size//2
                        center = (pixel_x, pixel_y)

                        prop_type = prop  # e.g., "star", "poly", "triangle", "dot"
                        color = color_array[x, y]  # Get the color from the color array
                        color = self._get_color_from_name(color)  # Extract color

                        # Draw the property based on its type
                        # Draw a star
                        if prop_type == "star":
                            self._draw_star(self.screen, color, center, int(cell_size // 6))
                        
                        # Draw a polyshape
                        elif prop_type == "poly":
                            shape = f'{additional_info[x, y] }' # Get the polyshape name from additional_info
                            shape_array = polyshapes[shape]
                            top_left = (pixel_x - cell_size//4, pixel_y - cell_size//4)
                            self._draw_polyshape(self.screen, shape_array, top_left, cell_size//2, color)
                        
                        # Draw a polyshape with "ylop" type
                        elif prop_type == "ylop":
                            shape = f'{additional_info[x, y]}'  # Get the polyshape name from additional_info
                            shape_array = polyshapes[shape]
                            top_left = (pixel_x - cell_size//4, pixel_y - cell_size//4)
                            self._draw_polyshape(self.screen, shape_array, top_left, cell_size//2, color)
                        
                        # Draw a triangle with a count    
                        elif prop_type == "triangle":
                            triangle_size = int(cell_size // 6)
                            pygame.draw.polygon(self.screen, color, [
                                (center[0], center[1] - triangle_size),  # Top
                                (center[0] - triangle_size, center[1] + triangle_size),  # Bottom-left
                                (center[0] + triangle_size, center[1] + triangle_size)   # Bottom-right
                            ])
                            
                            # Add text for triangle count
                            count = f'{additional_info[x, y]}'  # Get the count from additional_info 
                            font = pygame.font.Font(None, int(16 * self.scale_factor)) 
                            text = font.render(count, True, (255, 255, 255)) 
                            shadow = font.render(count, True, (0, 0, 0))
                            shadow_pos = (center[0] - int(4 * self.scale_factor) + 1, center[1] - int(12 * self.scale_factor) + 1)
                            self.screen.blit(shadow, shadow_pos)
                            text_pos = (center[0] - int(4 * self.scale_factor), center[1] - int(12 * self.scale_factor))
                            self.screen.blit(text, text_pos)
                        
                        # Draw a square
                        elif prop_type == "square":
                            square_size = int(cell_size // 4)
                            square_rect = pygame.Rect(
                                center[0] - square_size // 2,
                                center[1] - square_size // 2,
                                square_size,
                                square_size
                            )
                            pygame.draw.rect(self.screen, color, square_rect)
                        
                        # Draw a dot as black hexagon
                        elif prop_type == "dot":
                            self._draw_hexagon(pixel_x, pixel_y, int(line_width // 3), (0, 0, 0))

        pygame.display.flip()
        self.clock.tick(30)  # Limit to 30 FPS

        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
    
    def _draw_hexagon(self, center_x, center_y, radius, color):
        """Draw a hexagon at the specified center position."""
        points = []
        for i in range(6):
            angle = i * 60 * math.pi / 180  # Convert degrees to radians
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            points.append((x, y))
        pygame.draw.polygon(self.screen, color, points)
    
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
    
    def _draw_agent_path(self, path, y_size, x_size, padding, cell_size, line_width):
        """Draw the agent's path as white lines connecting positions in order."""
        if not path or len(path) < 2:
            return
        
        # Draw white line segments connecting consecutive positions in the path
        white_line_width = max(1, int(line_width))
        
        for i in range(len(path) - 1):
            current_pos = path[i]  # [x, y] format
            next_pos = path[i + 1]  # [x, y] format
            
            # Convert to pixel coordinates (note the coordinate swap: path uses [x,y] but pixel calc needs [y,x])
            current_pixel = (
                padding + current_pos[0] * cell_size // 2,
                padding + current_pos[1] * cell_size // 2
            )
            next_pixel = (
                padding + next_pos[0] * cell_size // 2,
                padding + next_pos[1] * cell_size // 2
            )
            
            # Draw white line segment between consecutive path positions
            draw_line_round_corners_polygon(
                self.screen, current_pixel, next_pixel, 
                (255, 255, 255), white_line_width
            )
    
    def _draw_line_with_gaps(self, start_pos, end_pos, line_coord, is_vertical, obs_array, line_width, gap_size):
        """Draw a line with gaps where specified in the obs_array."""
        cell_size = int(40 * self.scale_factor)
        padding = int(20 * self.scale_factor)
        
        # Find all gap positions along this line
        gap_positions = []
        if "gaps" in obs_array:
            if is_vertical:
                # For vertical lines, check all y positions at this x coordinate
                for y in range(obs_array["gaps"].shape[1]):
                    if obs_array["gaps"][line_coord, y] == 1:
                        gap_pixel_y = padding + y * cell_size // 2
                        gap_positions.append(gap_pixel_y)
            else:
                # For horizontal lines, check all x positions at this y coordinate
                for x in range(obs_array["gaps"].shape[0]):
                    if obs_array["gaps"][x, line_coord] == 1:
                        gap_pixel_x = padding + x * cell_size // 2
                        gap_positions.append(gap_pixel_x)
        
        # If no gaps, draw the complete line
        if not gap_positions:
            draw_line_round_corners_polygon(self.screen, start_pos, end_pos, self.LINE_COLOR, line_width)
            return
        
        # Draw line segments around gaps using polygon line function
        if is_vertical:
            current_y = start_pos[1]
            for gap_y in sorted(gap_positions):
                # Draw segment before gap
                if current_y < gap_y - gap_size // 2:
                    segment_start = (start_pos[0], current_y)
                    segment_end = (start_pos[0], gap_y - gap_size // 2)
                    draw_line_round_corners_polygon(self.screen, segment_start, segment_end, self.LINE_COLOR, line_width)
                
                # Skip over the gap
                current_y = gap_y + gap_size // 2
            
            # Draw final segment after last gap
            if current_y < end_pos[1]:
                segment_start = (start_pos[0], current_y)
                segment_end = end_pos
                draw_line_round_corners_polygon(self.screen, segment_start, segment_end, self.LINE_COLOR, line_width)
        else:
            current_x = start_pos[0]
            for gap_x in sorted(gap_positions):
                # Draw segment before gap
                if current_x < gap_x - gap_size // 2:
                    segment_start = (current_x, start_pos[1])
                    segment_end = (gap_x - gap_size // 2, start_pos[1])
                    draw_line_round_corners_polygon(self.screen, segment_start, segment_end, self.LINE_COLOR, line_width)
                
                # Skip over the gap
                current_x = gap_x + gap_size // 2
            
            # Draw final segment after last gap
            if current_x < end_pos[0]:
                segment_start = (current_x, start_pos[1])
                segment_end = end_pos
                draw_line_round_corners_polygon(self.screen, segment_start, segment_end, self.LINE_COLOR, line_width) 