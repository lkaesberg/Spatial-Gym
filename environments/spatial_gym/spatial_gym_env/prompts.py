"""Prompts and answer parsers for the Spatial-Gym verifiers environment.

Copied from the SPaRC reference code, so the environment asks the same
questions the papers did (https://github.com/lkaesberg/SPaRC/tree/9f2a3d6):

* sparc/prompts/gym_step.py            -> step_prompt
* sparc/prompts/gym_step_traceback.py  -> step_backtrack_prompt
* sparc/prompts/single_shot.py         -> one_shot_prompt
* sparc/process_puzzle.py              -> STEP_USER_TEMPLATE, parse_action
* sparc/validation.py                  -> extract_solution_path

The text is unchanged.
"""

import json
import re
from typing import Dict, List, Optional

def _get_polyshapes_str(puzzle_data: Dict) -> str:
    """Extract polyshapes string from puzzle data."""
    polyshapes_str = ""
    if "polyshapes" in puzzle_data and puzzle_data["polyshapes"]:
        polyshapes_str = "POLYSHAPES DEFINITIONS:\n"
        polyshapes_json = json.loads(puzzle_data["polyshapes"])
        for shape_id, shape_def in polyshapes_json.items():
            shape_def_str = '\n'.join(map(str, shape_def))
            polyshapes_str += f"Shape {shape_id}:\n{shape_def_str}\n\n"
    return polyshapes_str


def step_prompt(puzzle_data: Dict) -> Dict:
    """
    Generate the complete prompt dict for step-by-step gym mode (no traceback).
    
    Args:
        puzzle_data: The puzzle data dictionary
        
    Returns:
        Dict with 'system' and 'user' message content
    """
    polyshapes_str = _get_polyshapes_str(puzzle_data)
    
    system_content = f"""
    You are an autonomous agent controlling a path‐finding puzzle solver.
    Your goal is to find a valid path (a continuous line) from the specified Start Node to the End Node on the provided grid, adhering to all puzzle rules.

    Core Concepts & Grid Basics:
    Grid Dimensions: You can find the puzzle grid size in the info 
    Path: The solution is a single, continuous line connecting adjacent nodes either horizontally or vertically.
    Revisiting: You can not traceback your path. you can not visit a cell twice.
    Rule Cells: Cells containing rule symbols (squares, stars, etc.) have coordinates where both x and y are odd. 
    The path goes around these rule cells, never on them. They are also marked as gaps.
    Regions: The drawn path divides the grid cells into one or more distinct enclosed areas (regions). Many rules apply based on the contents of these regions.
    Valid Path Cells: The path travels along the grid lines (edges between nodes). It can only occupy positions marked `+` or `.` in the grid layout (these correspond to positions with at least one even coordinate).
    Rule Cells: Cells containing rule symbols (squares, stars, etc.) have coordinates where both `x` and `y` are odd. The path goes *around* these rule cells, never *on* them.
    Regions: The drawn path divides the grid cells into one or more distinct enclosed areas (regions). Many rules apply based on the contents of these regions.


    Symbol Legend (Grid Notation)
    *   `S`: **Start Node** (Path begins here)
    *   `E`: **End Node** (Path ends here)
    *   `V`: **Visited Node** (Path has passed through this cell)
    *   `L`: **Current Node** (Path is currently on this cell)
    *   `+`: Valid cell for the path to occupy
    *   `N`: Empty rule cell (no rule)
    *   `G`: **Gap** (Path **CANNOT** cross this cell)
    *   `.`: **Dot** (Path **MUST** pass through this cell)
    *   `o-X`: **Square** of color X
    *   `*-X`: **Star** of color X
    *   `A-X`: **Triangle** (touch 1 edge)
    *   `B-X`: **Triangle** (touch 2 edges)
    *   `C-X`: **Triangle** (touch 3 edges)
    *   `D-X`: **Triangle** (touch 4 edges)
    *   `P-X-Y`: **Polyshape** (positive) of color X and shape ID Y
    *   `Y-X-Y`: **Negative Polyshape** (ylop) of color X and shape ID Y
        
    **Color Codes:** R=Red, B=Blue, G=Green, Y=Yellow, W=White, O=Orange, P=Purple, K=Black
    
        
    Detailed Solving Rules:
    The drawn path must satisfy ALL applicable constraints:

    1.  Path Constraints:
        Path connects adjacent nodes (horizontal/vertical moves only).
        Nodes CAN NOT be revisited. You cannot visit a cell twice.
        Path MUST pass through all Dot cells.
        Path CANNOT pass through any Gap cells.

    2.  Region-Based Rules (Apply to areas enclosed by the path):
        Squares: All squares within a single region MUST be the same color. Squares of different colors MUST be separated into different regions by the path.
        Stars: Within a single region, each star symbol MUST be paired with exactly ONE other element of the same color. Other colors within the region are irrelevant to this specific star's rule.
        
        Polyshapes(poly): The region containing this symbol MUST be able to contain the specified shape (defined in Polyshape Definitions). The shape must fit entirely within the region's boundaries.
        If multiple positive polyshapes are in one region, the region must accommodate their combined, non-overlapping forms. Rotation of polyshapes is NOT allowed. They must fit within the provided space in their given orientation.
        
        Negative Polyshapes(ylop): These subtract shape requirements, typically within the same region as corresponding positive polyshapes.
        A negative polyshape cancels out a positive polyshape of the exact same shape and color within that region.
        If all positive shapes are canceled, the region has no shape constraint. A negative shape is only considered 'used' if it cancels a positive one.
        Negative shapes can sometimes rationalize apparent overlaps or boundary violations of positive shapes if interpreted as cancellations.

    3.  Path-Based Rules (Edge Touching):
        Triangles: The path MUST touch a specific number of edges of the cell containing the triangle symbol.
            (1): Path touches EXACTLY 1 edge of the triangle's cell.
            (2): Path touches EXACTLY 2 edges of the triangle's cell.
            (3): Path touches EXACTLY 3 edges of the triangle's cell.
            (4): Path touches EXACTLY 4 edges (fully surrounds) the triangle's cell.

    Polyshape Definitions: Shapes are defined by 2D arrays where 1 indicates an occupied cell and 0 indicates an empty cell. 
    {polyshapes_str}

    At each turn you'll receive the current state:
    - Step: The current step number
    - Current Position: Your current (x, y) location
    - Legal Actions: Available moves with format [digit=DIRECTION, ...]
    - Grid State: The current grid showing your path progress
    """

    return {
        "system": system_content,
        "user": None  # User content is dynamic (observation/info/reward JSON)
    }


def step_backtrack_prompt(puzzle_data: Dict) -> Dict:
    """
    Generate the complete prompt dict for step-by-step gym mode with traceback.
    
    Args:
        puzzle_data: The puzzle data dictionary
        
    Returns:
        Dict with 'system' and 'user' message content
    """
    polyshapes_str = _get_polyshapes_str(puzzle_data)
    
    system_content = f"""
    You are an autonomous agent controlling a path‐finding puzzle solver.
    Your goal is to find a valid path (a continuous line) from the specified Start Node to the End Node on the provided grid, adhering to all puzzle rules.

    Core Concepts & Grid Basics:
    Grid Dimensions: You can find the puzzle grid size in the info 
    Path: The solution is a single, continuous line connecting adjacent nodes either horizontally or vertically.
    Revisiting: You can traceback your path, but you MUST do so in the same way you came, without crossing over your own path. 
    When tracing back, you can only move to the last cell you occupied, and then continue from there. Also when you trace back, the nodes you no longer use in your path are free to be used again.
    Rule Cells: Cells containing rule symbols (squares, stars, etc.) have coordinates where both x and y are odd. 
    The path goes around these rule cells, never on them. They are also marked as gaps.
    Regions: The drawn path divides the grid cells into one or more distinct enclosed areas (regions). Many rules apply based on the contents of these regions.
    Valid Path Cells: The path travels along the grid lines (edges between nodes). It can only occupy positions marked `+` or `.` in the grid layout (these correspond to positions with at least one even coordinate).
    Rule Cells: Cells containing rule symbols (squares, stars, etc.) have coordinates where both `x` and `y` are odd. The path goes *around* these rule cells, never *on* them.


    Symbol Legend (Grid Notation)
    *   `S`: **Start Node** (Path begins here)
    *   `E`: **End Node** (Path ends here)
    *   `V`: **Visited Node** (Path has passed through this cell)
    *   `L`: **Current Node** (Path is currently on this cell)
    *   `+`: Valid cell for the path to occupy
    *   `N`: Empty rule cell (no rule)
    *   `G`: **Gap** (Path **CANNOT** cross this cell)
    *   `.`: **Dot** (Path **MUST** pass through this cell)
    *   `o-X`: **Square** of color X
    *   `*-X`: **Star** of color X
    *   `A-X`: **Triangle** (touch 1 edge)
    *   `B-X`: **Triangle** (touch 2 edges)
    *   `C-X`: **Triangle** (touch 3 edges)
    *   `D-X`: **Triangle** (touch 4 edges)
    *   `P-X-Y`: **Polyshape** (positive) of color X and shape ID Y
    *   `Y-X-Y`: **Negative Polyshape** (ylop) of color X and shape ID Y
        
    **Color Codes:** R=Red, B=Blue, G=Green, Y=Yellow, W=White, O=Orange, P=Purple, K=Black


    Detailed Solving Rules:
    The drawn path must satisfy ALL applicable constraints:

    1.  Path Constraints:
        Path connects adjacent nodes (horizontal/vertical moves only).
        Nodes CAN be revisited. But only if you trace back to the last cell you occupied (and from there again and again ...).
        Otherwise you CANNOT cross your own path.
        Path MUST pass through all Dot cells.
        Path CANNOT pass through any Gap cells.

    2.  Region-Based Rules (Apply to areas enclosed by the path):
        Squares: All squares within a single region MUST be the same color. Squares of different colors MUST be separated into different regions by the path.
        Stars: Within a single region, each star symbol MUST be paired with exactly ONE other element of the same color. Other colors within the region are irrelevant to this specific star's rule.
        
        Polyshapes(poly): The region containing this symbol MUST be able to contain the specified shape (defined in Polyshape Definitions). The shape must fit entirely within the region's boundaries.
        If multiple positive polyshapes are in one region, the region must accommodate their combined, non-overlapping forms. Rotation of polyshapes is generally allowed unless context implies otherwise.
        
        Negative Polyshapes(ylop): These subtract shape requirements, typically within the same region as corresponding positive polyshapes.
        A negative polyshape cancels out a positive polyshape of the exact same shape and color within that region.
        If all positive shapes are canceled, the region has no shape constraint. A negative shape is only considered 'used' if it cancels a positive one.
        Negative shapes can sometimes rationalize apparent overlaps or boundary violations of positive shapes if interpreted as cancellations.

    3.  Path-Based Rules (Edge Touching):
        Triangles: The path MUST touch a specific number of edges of the cell containing the triangle symbol.
            (1): Path touches EXACTLY 1 edge of the triangle's cell.
            (2): Path touches EXACTLY 2 edges of the triangle's cell.
            (3): Path touches EXACTLY 3 edges of the triangle's cell.
            (4): Path touches EXACTLY 4 edges (fully surrounds) the triangle's cell.

    Polyshape Definitions: Shapes are defined by 2D arrays where 1 indicates an occupied cell and 0 indicates an empty cell. 
    {polyshapes_str}

    At each turn you'll receive the current state:
    - Step: The current step number
    - Current Position: Your current (x, y) location
    - Legal Actions: Available moves with format [digit=DIRECTION, ...]
    - Grid State: The current grid showing your path progress
    """

    return {
        "system": system_content,
        "user": None  # User content is dynamic (observation/info/reward JSON)
    }


def one_shot_prompt(puzzle_data: Dict) -> Dict:
    """
    Generate the complete prompt dict for single-shot puzzle solving.
    
    Args:
        puzzle_data: The puzzle data dictionary
        
    Returns:
        Dict with 'system' and 'user' message content
    """
    grid_size = puzzle_data.get("grid_size", {"width": 0, "height": 0})
    puzzle_array = puzzle_data.get("puzzle_array", [])
    grid_str = "\n".join(map(str, puzzle_array))
    start_pos = None
    end_pos = None
    for y, row in enumerate(puzzle_array):
        for x, cell in enumerate(row):
            if cell == "S":
                start_pos = f"({x}, {y})"
            elif cell == "E":
                end_pos = f"({x}, {y})"

    polyshapes_str = _get_polyshapes_str(puzzle_data)

    user_content = f"""
    ## Objective
    You are a specialized AI proficient in spatial reasoning and solving puzzles from the game 'The Witness'. Your goal is to find a valid path (a continuous line) from the specified Start Node to the End Node on the provided grid, adhering to all puzzle rules.

    ## Core Concepts & Grid Basics
    *   **Grid Dimensions:** The puzzle grid has {grid_size['width']} columns and {grid_size['height']} rows.
    *   **Coordinate System:** Nodes are identified by `(x, y)` coordinates. `(0,0)` is the top-left node. `x` increases to the right, `y` increases downwards.
    *   **Path:** The solution is a single, continuous line connecting adjacent nodes either horizontally or vertically.
    *   **No Revisits:** The path **CANNOT** visit the same node more than once.
    *   **Valid Path Cells:** The path travels along the grid lines (edges between nodes). It can only occupy positions marked `+` or `.` in the grid layout (these correspond to positions with at least one even coordinate).
    *   **Rule Cells:** Cells containing rule symbols (squares, stars, etc.) have coordinates where both `x` and `y` are odd. The path goes *around* these rule cells, never *on* them.
    *   **Regions:** The drawn path divides the grid cells into one or more distinct enclosed areas (regions). Many rules apply based on the contents of these regions.

    ## Symbol Legend (Grid Notation)
    *   `S`: **Start Node** (Path begins here)
    *   `E`: **End Node** (Path ends here)
    *   `+`: Valid cell for the path to occupy
    *   `N`: Empty rule cell (no rule)
    *   `G`: **Gap** (Path **CANNOT** cross this cell)
    *   `.`: **Dot** (Path **MUST** pass through this cell)
    *   `o-X`: **Square** of color X
    *   `*-X`: **Star** of color X
    *   `A-X`: **Triangle** (touch 1 edge)
    *   `B-X`: **Triangle** (touch 2 edges)
    *   `C-X`: **Triangle** (touch 3 edges)
    *   `D-X`: **Triangle** (touch 4 edges)
    *   `P-X-Y`: **Polyshape** (positive) of color X and shape ID Y
    *   `Y-X-Y`: **Negative Polyshape** (ylop) of color X and shape ID Y

    **Color Codes:** R=Red, B=Blue, G=Green, Y=Yellow, W=White, O=Orange, P=Purple, K=Black

    ## Detailed Solving Rules
    The drawn path must satisfy **ALL** applicable constraints:

    1.  **Path Constraints:**
        *   Path **MUST** start at `S` and end at `E`.
        *   Path connects adjacent nodes (horizontal/vertical moves only).
        *   Nodes **CANNOT** be revisited.
        *   Path **MUST** pass through all Dot (`.`) cells.
        *   Path **CANNOT** pass through any Gap (`G`) cells.

    2.  **Region-Based Rules** (Apply to areas enclosed by the path):
        *   **Squares (`o-X`):** All squares within a single region **MUST** be the same color. Squares of different colors **MUST** be separated into different regions by the path.
        *   **Stars (`*-X`):** Within a single region, each star symbol **MUST** be paired with exactly **ONE** other element (star or square) *of the same color*. Other colors within the region are irrelevant to this specific star's rule.
        *   **Polyshapes (`P-X-Y`):** The region containing this symbol **MUST** be able to contain the specified shape (defined in Polyshape Definitions). The shape must fit entirely within the region's boundaries. If multiple positive polyshapes are in one region, the region must accommodate their combined, non-overlapping forms. Rotation of polyshapes is NOT allowed. They must fit within the provided space in their given orientation.
        *   **Negative Polyshapes (`Y-X-Y`):** These "subtract" shape requirements, typically within the same region as corresponding positive polyshapes. A negative polyshape cancels out a positive polyshape of the exact same shape and color within that region. If all positive shapes are canceled, the region has no shape constraint. A negative shape is only considered 'used' if it cancels a positive one. Negative shapes can sometimes rationalize apparent overlaps or boundary violations of positive shapes if interpreted as cancellations.

    3.  **Path-Based Rules (Edge Touching):**
        *   **Triangles (`A-X`, `B-X`, `C-X`, `D-X`):** The path **MUST** touch a specific number of edges of the cell containing the triangle symbol.
            *   `A-X` (1): Path touches **EXACTLY 1** edge of the triangle's cell.
            *   `B-X` (2): Path touches **EXACTLY 2** edges of the triangle's cell.
            *   `C-X` (3): Path touches **EXACTLY 3** edges of the triangle's cell.
            *   `D-X` (4): Path touches **EXACTLY 4** edges (fully surrounds) the triangle's cell.

    ## EXAMPLE PUZZLE GRID:

    ["+",".","+","+","+","E","+"]
    ["+","C-R","+","o-K","+","o-K","+"]
    ["S","+","+","+","+","+","+"]
    ["+","P-G-112","+","*-G","+","P-B-624","+"]
    ["+","+","+","+","+","+","+"]
    ["+","*-G","+","*-G","+","o-K","+"]
    ["+","+","+",".","+","+","+"]

    EXAMPLE POLYSHAPE DEFINITIONS:
    Shape 112:
    [0,1,0,0]
    [0,1,0,0]
    [0,1,0,0]
    [0,0,0,0]

    Shape 624:
    [0,1,0,0]
    [0,1,1,0]
    [0,1,0,0]
    [0,0,0,0]

    EXAMPLE SOLUTION:

    We start at (0,2) and draw a line to (0,0).
    We then draw a line to (2,0) to reach the dot at (1,0) and surround the 3 count triangle.
    We then draw a line to (2,2) here we go down to touch the third side of the triangle cell and therefore validate the 3 count triangle.
    We continue down to (2,6) to validate the polyshape 112 and also the green star with the green polyshape
    After this we draw a line to (4,6) to start validating the polyshape 624 by surrounding it.
    Therefore we have to draw a line to (6,4) over (4,4) which creates a region for the stone at (5,5) which validates the stone.
    We continue up to (6,2) for the polyshape 624 and then go to (4,2) and after this to (4,0) to finaly validate the polyshape 624.
    This also validates the two green stars at (3,3) and (3,5) with each other and the black stone at (3,1) because its the only stone in its region.
    This line also creates a region for the black stone at (5,1) because its the only stone in its region.
    Now we can draw a line to (5,0) to reach the end node.

    #### (0,2),(0,1),(0,0),(1,0),(2,0),(2,1),(2,2),(2,3),(2,4),(2,5),(2,6),(3,6),(4,6),(4,5),(4,4),(5,4),(6,4),(6,3),(6,2),(5,2),(4,2),(4,1),(4,0),(5,0)

    ## Puzzle Input Data
    *   **Start Node:** {start_pos}
    *   **End Node:** {end_pos}
    *   **Grid Layout:**
        ```
        {grid_str}
        ```
    *   **Polyshape Definitions (if applicable):**
        *   Shapes are defined by 2D arrays where '1' indicates an occupied cell and '0' indicates an empty cell.
        ```
        {polyshapes_str}
        ```

    ## Task & Output Format
    1.  **Solve the Puzzle:** Determine the valid path from the Start Node to the End Node that satisfies all rules.
    2.  **Explain Reasoning:** Provide a step-by-step explanation of your thought process. Detail key deductions, how constraints were applied, and any backtracking or choices made.
    3.  **Provide Solution Path:** After the reasoning, output the exact marker string `####` followed immediately by the solution path as a list of node coordinates `(x, y)`. Include all intermediate nodes from start to end.

    **Example Solution Path Format:**
    ####
    (0,0), (1,0), (2,0), (2,1), ...
    """

    return {
        "system": "You are an expert at solving puzzles games.",
        "user": user_content
    }


# Per-step user message (sparc/process_puzzle.py, process_puzzle_step_by_step).
# Only change: the f-string expression {step + 1} became the placeholder {step}.
STEP_USER_TEMPLATE = 'Step: {step}\nCurrent Position: {agent_loc}\nLegal Actions: [{legal_str}]\n\nGrid State:\n{obs_str}\n\nYou MAY think step-by-step, but you MUST end your response with:\nFinal: <action>\nWhere <action> is one of 0=RIGHT, 1=UP, 2=LEFT, 3=DOWN (you may also write the direction name, e.g. Final: right).'

DIRECTION_WORD_TO_ACTION = {'right': 0, 'up': 1, 'left': 2, 'down': 3}


def parse_action(reply: str) -> int:
    """Parse an action from the model's reply.

    Accepted formats (case-insensitive):
      Final: 2          (digit 0-3)
      Final: left        (direction word)
      Final: LEFT        (direction word, any case)
    Falls back to a bare digit or direction word at the end of the reply.
    """
    # Primary: "Final: <digit>" or "Final: <word>"
    m = re.search(r'Final:\s*([0-3])\b', reply, re.IGNORECASE)
    if m:
        return int(m.group(1))

    m = re.search(r'Final:\s*(right|up|left|down)\b', reply, re.IGNORECASE)
    if m:
        return DIRECTION_WORD_TO_ACTION[m.group(1).lower()]

    # Fallback: standalone digit 0-3 at the end
    m = re.search(r'\b([0-3])\s*$', reply)
    if m:
        return int(m.group(1))

    # Fallback: standalone direction word at the end
    m = re.search(r'\b(right|up|left|down)\s*$', reply, re.IGNORECASE)
    if m:
        return DIRECTION_WORD_TO_ACTION[m.group(1).lower()]

    raise ValueError(f"Invalid model output, no 'Final: <0-3|right|up|left|down>' found in response: {reply}")


def extract_solution_path(
    solution_text: str, puzzle_data: Dict = None
) -> Optional[List[Dict[str, int]]]:
    """Extract solution path from LLM's response

    Args:
        solution_text: Text response from the LLM
        puzzle_data: Optional puzzle data dict to extract end point from

    Returns:
        List of coordinate dicts or None if no path found
    """
    # First, check if "Solution" appears in the text
    solution_marker = "####"
    if solution_marker in solution_text:
        # Only process text after "####"
        solution_part = solution_text.split(solution_marker)[-1]
    else:
        # If no solution marker, use the full text
        solution_part = solution_text

    # Look for coordinate patterns like (0,0) -> (0,1) or similar
    # Pattern for (x,y) or (x, y) coordinates
    coord_pattern = r"\((\d+),\s*(\d+)\)"
    coords = re.findall(coord_pattern, solution_part)

    if coords:
        # Extract end point from puzzle data if provided
        end_point = None
        if puzzle_data:
            puzzle_array = puzzle_data.get("puzzle_array", [])
            for y, row in enumerate(puzzle_array):
                for x, cell in enumerate(row):
                    if cell == "E":
                        end_point = {"x": x, "y": y}
                        break
                if end_point:
                    break

        # Convert string coordinates to integer dicts
        path = []
        for x, y in coords:
            point = {"x": int(x), "y": int(y)}
            path.append(point)

            # Stop extracting if we've reached the end point
            if (
                end_point
                and point["x"] == end_point["x"]
                and point["y"] == end_point["y"]
            ):
                break

        return path

    # If no coordinates found, return None
    return None
