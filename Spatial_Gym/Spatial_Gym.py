from enum import Enum
import json
from collections import Counter, deque
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml
from dataclasses import dataclass
from .render import HumanRenderer, LLMRenderer
from datasets import load_dataset

class Actions(Enum):
    """
    Enum class representing the possible actions the agent can take in the environment.

    Actions:
        right (int): Move the agent one step to the right.
        up (int): Move the agent one step upward.
        left (int): Move the agent one step to the left.
        down (int): Move the agent one step downward.
    """
    right = 0
    up = 1
    left = 2
    down = 3
    

@dataclass
class RegionData:
    id: int
    cells: list    
    area: int
    symbols: dict     
    colors: dict         

    def to_summary(self):
        return {
            "id": self.id,
            "area": self.area,
            "symbol_counts": {k: len(v) for k, v in self.symbols.items()},
            "colors": self.colors
        }

class Spatial_Gym(gym.Env):
    metadata = {"render_modes": ["human", "llm"], "render_fps": 30}
    def __init__(self, df_name='lkaesberg/SPaRC', df_split='all', df_set='test', render_mode=None, observation='new', traceback=False, max_steps=2000):
        """
        Initialize the Spatial_Gym environment.

        Loads the puzzle dataset, sets up rendering modes, and prepares the environment for agent interaction.

        Parameters:
            df_name (str): Name of the HuggingFace dataset to load (default: 'lkaesberg/SPaRC').
            df_split (str): Dataset split to use (default: 'all').
            df_set (str): Subset of the dataset for testing (default: 'test').
            render_mode (str or None): Rendering mode, either 'human', 'llm', or None.
            observation (str): Observation format, either 'new' (multi-array) or 'SPaRC' (JSON/text).
            traceback (bool): If True, enables traceback.
            max_steps (int): Maximum steps per episode before truncation.

        Loads the puzzles, initializes renderers, sets up observation and action spaces, and loads the first puzzle.
        """
        self.render_mode = render_mode
        self.observation = observation
        self.traceback = traceback
        self.max_steps = max_steps

        # Initialize renderers
        self.human_renderer = None
        self.llm_renderer = None
        if render_mode == "human":
            self.human_renderer = HumanRenderer(scale_factor=3.0)
        elif render_mode == "llm":
            self.llm_renderer = LLMRenderer()

        # Load the puzzles
        ds = load_dataset(df_name, df_split, split=df_set)
        df = ds.to_pandas()
        self.puzzles = df
        self.current_puzzle_index = 0
        self.current_step = 0
        
        self.rule_status = {}
        
        # Process the puzzles to extract relevant information
        if self.puzzles is None or isinstance(self.puzzles, Exception):
            raise ValueError("No valid dataframe provided")
        self.puzzles = self._process_puzzles(self.puzzles)
        # Load the first puzzle
        self._load_puzzle(self.current_puzzle_index) 


    # ---------- Puzzle Processing/Loading Functions ----------

    def _load_puzzle(self, index):
        '''
        Function to load a puzzle from the dataset
        and initialize the environment variables
        aswell as the observation and action spaces
        
        Parameters:
        index : int
            The index of the puzzle to load from the dataset.
            
        --------------
        puzzle variables:
        
        difficulty : int
            The difficulty of the puzzle
        
        polyshapes : dict of 2d arrays
            The polyshapes of the puzzle
        
        x_size : int
            The x size of the puzzle
            
        y_size : int
            The y size of the puzzle
            
        obs_array : dict; Dictionary of 2D arrays
            The observation array of the puzzle
        or if observation == 'SPaRC':
            The observation array in SPaRC format
        
        unique_properties : int
            The number of unique properties(star,polyshape, Co.) in the puzzle
        
        start_location : tuple
            The starting location of the agent in the puzzle
            
        target_location : tuple
            The target location of the agent in the puzzle
        
        solution_paths : list
            The solution paths of the puzzle
        
        solution_count : int
            The number of solution paths of the puzzle
        
        '''
        puzzle = self.puzzles[index]
        
        self.difficulty = puzzle['difficulty']
        self.polyshapes = puzzle['polyshapes']
        
        self.x_size = puzzle['x_size']
        self.y_size = puzzle['y_size']
        
        self.obs_array = puzzle['obs_array']
        self.color_array = puzzle['color_array']
        self.additional_info = puzzle['additional_info']
        
        if self.observation == 'SPaRC':
            raw = puzzle['observ']
            if isinstance(raw, np.ndarray) and raw.dtype == object and raw.ndim == 1:
                grid_rows = [r.astype(str).tolist() for r in raw]
            elif isinstance(raw, np.ndarray) and raw.ndim == 2:
                grid_rows = raw.astype(str).tolist()
            else:
                grid_rows = [[str(c) for c in row] for row in raw]
            w = len(grid_rows[0])
            if any(len(r) != w for r in grid_rows):
                raise ValueError("Non-rectangular SPaRC grid")
            self.observ = grid_rows

        self.start_location = puzzle['start_location']
        self.target_location = puzzle['target_location']
        
        self.solution_paths = puzzle['solution_paths']
        self.solution_count = puzzle['solution_count']
        
        # Initialize the agent's path with the starting location
        self.path = [[self.start_location[0], self.start_location[1]]]
        self.normal_reward = 0
        self.outcome_reward = 0
        
        self.rule_status = {}
        
        self._agent_location = np.array([self.start_location[0], self.start_location[1]], dtype=np.int32)
        self._target_location = np.array([self.target_location[0], self.target_location[1]], dtype=np.int32)

        self._validate_rules(terminated=False, truncated=False)
        
        # Mark the starting location as visited and set the agent's and target's positions in the observation array
        self.obs_array['visited'][self._agent_location[0], self._agent_location[1]] = 1
        self.obs_array['agent_location'][self._agent_location[0], self._agent_location[1]] = 1
        self.obs_array['target_location'][self._target_location[0], self._target_location[1]] = 1
        
        # Define the observation space for the environment
        if self.observation == 'new':
            keys = list(self.obs_array.keys())
            self.observation_space = spaces.Dict({
                'base': spaces.Dict({key: spaces.Box(low=0, high=1, shape=(self.x_size, self.y_size), dtype=np.int32) for key in keys}),
                'color': spaces.Box(low=0, high=8, shape=(self.x_size, self.y_size), dtype=np.int32),
                'additional_info': spaces.Box(low=0, high=143632, shape=(self.x_size, self.y_size), dtype=np.int64)
            })
        
        elif self.observation == 'SPaRC':
            init_json = self._build_json_obs()
            overlay_chars = set("LV.")
            charset = "".join(sorted(set(init_json) | overlay_chars))
            max_length = int(len(init_json) * 2)
            self._json_charset = charset
            self.observation_space = spaces.Text(max_length=max_length, charset=charset)

        else:
            raise ValueError("Invalid observation type. Choose 'new' or 'SPaRC'.")

        # Define the action space (4 discrete actions: right, up, left, down)
        self.action_space = gym.spaces.Discrete(4)
        # Map actions to directions 
        self._action_to_direction = {
            Actions.right.value: np.array([1, 0]),
            Actions.up.value: np.array([0, -1]),
            Actions.left.value: np.array([-1, 0]),
            Actions.down.value: np.array([0, 1]),
        }
    
    def _process_puzzles(self, df):
        """
        Processes a DataFrame of puzzles and returns a list of puzzle dictionaries.

        Parameters:
            df (pd.DataFrame): The DataFrame containing puzzle data.
            
        --------
        
        Returns:
            list: A list of dictionaries, each representing a processed puzzle.
        """
            
        puzzles = []

        for i in range(len(df)):

            puzzle = {}
            
            # Extract difficulty
            difficulty = df['difficulty_level'][i]
            puzzle.update({'difficulty': difficulty})
            
            # Extract grid size
            grid_size = df['grid_size'][i]
            x_size = grid_size['width']
            y_size = grid_size['height']
            x_size = x_size + x_size + 1
            y_size = y_size + y_size + 1
            puzzle.update({'x_size': x_size, 'y_size': y_size})
            
            # Extract solution paths
            solution_count = df['solution_count'][i]
            solutions = df['solutions'][i]
            solution_paths = []
            for item in solutions:
                path = [[point["x"], point["y"]] for point in item["path"]]
                solution_paths.append(path)
            puzzle.update({'solution_count': solution_count, 'solution_paths': solution_paths})
            
            # Extract the polyshapes (eg. an L shape)
            polyshapes = df['polyshapes'][i]
            polyshapes_yaml = yaml.safe_load(polyshapes)
            puzzle.update({'polyshapes': polyshapes_yaml})
            
            # Extract start and target locations
            text_visualization = df['text_visualization'][i]
            text_yaml = yaml.safe_load(text_visualization)
            start_location = (text_yaml["puzzle"]["start"]["x"], text_yaml["puzzle"]["start"]["y"])
            target_location = (text_yaml["puzzle"]["end"]["x"], text_yaml["puzzle"]["end"]["y"])
            puzzle.update({'start_location': start_location, 'target_location': target_location})
            
            # Initialize observation arrays
            obs_array = {
                'visited': np.zeros((x_size, y_size), dtype=np.int32),
                'gaps': np.zeros((x_size, y_size), dtype=np.int32),
                'agent_location': np.zeros((x_size, y_size), dtype=np.int32),
                'target_location': np.zeros((x_size, y_size), dtype=np.int32)
            }
            
            color_array = np.zeros((x_size, y_size), dtype=np.int32)
            additional_info = np.zeros((x_size, y_size), dtype=np.int64)
            
            # Extract symbols, colors and additional info 
            for cell in text_yaml["puzzle"]["cells"]:
                properties = cell.get("properties", {})
                count = None
                shape = None
                color = None
                for key, value in properties.items():
                    if key == 'type':
                        if value == 'star' or value == 'square':
                            symbol = f"{value}"
                            color = properties.get('color', '')
                        elif value == 'triangle':
                            symbol = f"{value}"
                            color = properties.get('color', '')
                            count = properties.get('count', '')
                        else:
                            symbol = f"{value}"
                            color = properties.get('color', '')
                            shape = properties.get('polyshape', '')
                        
                    elif key == 'dot':
                        symbol = 'dot'
                    # Add new property to obs_array if not already present
                    if symbol not in obs_array:
                        obs_array.update({symbol: np.zeros((x_size, y_size), dtype=np.int32)})
                        
                    # Update the colors
                    if color:
                        color_to_number = {"red": 1, "blue": 2, "yellow": 3, "green": 4, "black": 5, "purple": 6, "orange": 7, "white": 8} 
                        position = cell.get("position", {}) 
                        x, y = position.get("x"), position.get("y")                  
                        for color_ in color_to_number:
                            if color_ == color:
                                color_array[x][y] = color_to_number[color_]
                            
                    # update additional information
                    if count:
                        position = cell.get("position", {}) 
                        x, y = position.get("x"), position.get("y")  
                        additional_info[x][y] = count
                    elif shape:
                        position = cell.get("position", {}) 
                        x, y = position.get("x"), position.get("y")  
                        additional_info[x][y] = shape
                    
            
            # Populate observation arrays
            for cell in text_yaml["puzzle"]["cells"]:
                position = cell.get("position", {})
                properties = cell.get("properties", {})
                x, y = position.get("x"), position.get("y")

                for key, value in properties.items():
                    if key == 'type':
                        symbol = f"{value}"
                    elif key == 'dot':
                        symbol = 'dot'
                    elif key == 'gap':
                        symbol = 'gaps'
                    # Update the corresponding observation array
                    if symbol in obs_array:
                        obs_array[symbol][x, y] = 1

            x_size = x_size - 1
            y_size = y_size - 1
            # Mark all the green cells as gaps
            for k in range(x_size):
                for j in range(y_size):
                    if k % 2 == 1 and j % 2 == 1:
                        obs_array['gaps'][k, j] = 1

            puzzle.update({'obs_array': obs_array})
            puzzle.update({'color_array': color_array})
            puzzle.update({'additional_info': additional_info})

            # If using the SPaRC observation format
            if self.observation == 'SPaRC':
                observ = df['puzzle_array'][i]
                puzzle.update({'observ': observ})

            id = df['id'][i]
            puzzle.update({'id': id})

            # Add the processed puzzle to the list
            puzzles.append(puzzle)
        
        return puzzles

    # ---------- End Puzzle Processing/Loading Functions ----------

    # ---------- Compute Regions Functions ----------
    
    def _cell_index_grid(self):
        '''
        Helper Function for _compute_regions
        Generate a mask for Rule cells.
        '''
        
        visited = self.obs_array['visited']
        h, w = visited.shape
        mask = np.zeros((h, w), dtype=bool)
        for x in range(h):
            for y in range(w):
                if x % 2 == 1 and y % 2 == 1:
                    mask[x, y] = True
                    
        return mask
    
    def _cell_index_grid2(self):
        '''
        Helper Function for _compute_regions
        Generate a mask for visited cells and gaps.
        '''
        
        visited = self.obs_array['visited']
        gaps = self.obs_array['gaps']
        h, w = visited.shape
        mask = np.zeros((h, w), dtype=bool)
        path_nodes = {tuple(p) for p in self.path}
        for x in range(h):
            for y in range(w):
                if gaps[x, y] == 1:
                    mask[x, y] = True
                    
        for x, y in path_nodes:
            mask[x, y] = True
            
        return mask
    
    def _neighbours_cell(self, x, y, h, w):
        """
        Helper Function for _compute_regions
        Generate neighboring cell coordinates (1 unit away) in the grid.
        """
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w:
                yield nx, ny
    

    def _compute_regions(self):
        """
        Compute regions in the grid based on the observation layers.
        """
        mask = self._cell_index_grid()
        mask2 = self._cell_index_grid2()
        h, w = mask.shape
        region_map = -1 * np.ones((h, w), dtype=np.int32)
        regions = []
        rid = 0

        for x in range(h):
            for y in range(w):
                if mask[x, y] and region_map[x, y] == -1:
                    enqueued_non_cells = np.zeros((h, w), dtype=bool)
                    q = deque([(x, y)])
                    region_map[x, y] = rid
                    cells = []
                    while q:
                        cx, cy = q.popleft()
                        if mask[cx, cy]:
                            cells.append((cx, cy))
                        for nx, ny in self._neighbours_cell(cx, cy, h, w):
                            if mask[nx, ny] and region_map[nx, ny] == -1:
                                region_map[nx, ny] = rid
                                q.append((nx, ny))
                            if not mask2[nx, ny] and not enqueued_non_cells[nx, ny]:
                                enqueued_non_cells[nx, ny] = True
                                q.append((nx, ny))
                    regions.append(RegionData(id=rid, cells=cells, area=len(cells), symbols={}, colors={}))
                    rid += 1
                
        return regions, region_map
    
    def _collect_region_symbols(self, regions, region_map):
        """
        Collect symbols for each region based on the observation layers.
        """

        if not regions:
            return

        # Build fast lookup
        regions_by_id = {r.id: r for r in regions}

        skip_layers = {'visited', 'gaps', 'agent_location', 'target_location'}
        for layer, arr in self.obs_array.items():
            if layer in skip_layers:
                continue
            # Collect all coordinates where symbol appears
            xs, ys = np.where(arr == 1)
            for x, y in zip(xs, ys):
                rid = region_map[x, y]
                if rid == -1:
                    continue
                reg = regions_by_id[rid]
                reg.symbols.setdefault(layer, []).append((x, y))
                # Color info (if available)
                color_val = self.color_array[x, y]
                if color_val:
                    reg.colors[color_val] = reg.colors.get(color_val, 0) + 1

    # ---------- End Compute Regions Functions ----------

    # ---------- Rule Check Functions ----------
    
    def _rule_reached_target(self):
        '''
        Checks:
        The agent must reach the target location.
        '''
        return bool(np.array_equal(self._agent_location, self._target_location)), {
            "agent_loc": self._agent_location.tolist(),
            "target_loc": self._target_location.tolist()
        }

    def _rule_path_not_crossing(self):
        '''
        Checks:
        No path nodes should be crossed more than once.
        '''
        path_nodes = [tuple(p) for p in self.path]
        counts = Counter(path_nodes)
        dup = {k: v for k, v in counts.items() if v > 1}
        return len(dup) == 0, {"duplicates": dup}

    def _rule_no_gap_violations(self):
        '''
        Checks:
        No gaps should be crossed by the agent's path.
        '''
        gaps = self.obs_array['gaps']
        violations = []
        for (x, y) in self.path:
            gx, gy = x, y
            if gaps[gx, gy] == 1:
                violations.append((gx, gy))
        return len(violations) == 0, {"violations": violations}

    def _rule_all_dots_collected(self):
        """
        Checks:
        All dots must be collected (visited) by the agent.
        """
        if 'dot' not in self.obs_array:
            return True, {"total": 0, "collected": 0}
        dot_mask = self.obs_array['dot'] == 1
        visited = self.obs_array['visited'] == 1
        total = int(dot_mask.sum())
        collected = int((dot_mask & visited).sum())
        return (total == 0) or (collected == total), {"total": total, "collected": collected}

    def _rule_color_square_separation(self, regions):
            """
            Checks:
            All squares inside any single region must have same color;
            different colors must be separated by path -> no region with >1 square color.
            """
            if 'square' not in self.obs_array:
                return True, {"regions": []}
            bad = []
            details = []
            for r in regions:
                squares = r.symbols.get('square', [])
                if not squares:
                    continue
                colors = set(self.color_array[x, y] for (x, y) in squares if self.color_array[x, y] != 0)
                if len(colors) > 1:
                    bad.append(r.id)
                details.append({"region": r.id, "square_count": len(squares), "colors": list(colors)})
            return len(bad) == 0, {"violating_regions": bad, "region_square_details": details}

    def _rule_star_pairing_exact(self, regions):
        """
        Checks:
        Each star must share region with exactly one other same-color symbol.
        => For each region: for each star color, count must be 0 or 2 (not 1, not >2).
        """
        if 'star' not in self.obs_array:
            return True, {"regions": []}
        violations = []
        per_region = []
        for r in regions:
            star_coords = r.symbols.get('star', [])
            if not star_coords:
                continue
            
            color_counts_all = {}
            for layer, coords in r.symbols.items():
                for (x, y) in coords:
                    c = self.color_array[x, y]
                    if c == 0:
                        continue
                    color_counts_all[c] = color_counts_all.get(c, 0) + 1

            star_colors = {}
            for (x, y) in star_coords:
                c = self.color_array[x, y]
                if c == 0:
                    violations.append({"region": r.id, "color": 0, "found_total": 1})
                    continue
                star_colors[c] = star_colors.get(c, 0) + 1

            region_ok = True
            region_star_details = []
            for c, star_count in star_colors.items():
                total_c = color_counts_all.get(c, 0)
                ok = (total_c == 2)
                if not ok:
                    region_ok = False
                    violations.append({
                        "region": r.id,
                        "color": c,
                        "found_total": total_c,
                        "star_cells": star_count
                    })
                region_star_details.append({
                    "color": c,
                    "total_symbols_of_color": total_c,
                    "star_cells": star_count,
                    "ok": ok
                })

            per_region.append({
                "region": r.id,
                "details": region_star_details,
                "all_ok": region_ok
            })

        return len(violations) == 0, {
            "violations": violations,
            "per_region": per_region
        }


    def _rule_triangles_edges(self):
        """
        Checks:
        For each triangle cell: required count == number of touched edges.
        """
        if 'triangle' not in self.obs_array:
            return True, {"mismatches": []}
        tri = self.obs_array['triangle']
        h, w = tri.shape
        mismatches = []
        for x in range(1, h-1):
            for y in range(1, w-1):
                if tri[x, y] == 1:
                    required = int(self.additional_info[x, y])
                    if required <= 0:
                        continue
                    touches = self._triangle_touches(x, y)
                    if touches != required:
                        mismatches.append({"x": x, "y": y, "required": required, "touches": touches})
        return len(mismatches) == 0, {"mismatches": mismatches}
    
    def _triangle_touches(self, tri_x, tri_y):
        path_nodes = {(p[0], p[1]) for p in self.path}
        neighbor_nodes = [
            (tri_x + 1, tri_y),
            (tri_x - 1, tri_y),
            (tri_x, tri_y - 1),
            (tri_x, tri_y + 1)
        ]
        return sum(1 for n in neighbor_nodes if n in path_nodes)

    def _rule_poly_ylop_balance(self, regions):
        """
        Combined check:
         - area_ok: sum(poly) - sum(ylop) == region.area
         - exact_ok: shapes can be placed exactly (no rotation/mirror) inside region
        Rule passes only if both are True for all regions that contain any poly/ylop.
        """
        instances = self._extract_poly_instances()
        if not instances:
            return True, {"regions": []}

        # Map instances to regions
        _, region_map = self._compute_regions()
        by_region = {}
        for inst in instances:
            x, y = inst["x"], inst["y"]
            if 0 <= x < region_map.shape[0] and 0 <= y < region_map.shape[1]:
                rid = region_map[x, y]
                if rid != -1:
                    by_region.setdefault(rid, []).append(inst)

        regions_by_id = {r.id: r for r in regions}
        violations = []
        region_details = []

        for rid, lst in by_region.items():
            region = regions_by_id.get(rid)
            if region is None:
                continue

            # area check
            poly_area = sum(inst["area"] for inst in lst if inst["kind"] == "poly")
            ylop_area = sum(inst["area"] for inst in lst if inst["kind"] == "ylop")
            net = poly_area - ylop_area
            area_ok = (region.area == net)

            detail = {
                "region": rid,
                "area_check": {
                    "region_area": region.area,
                    "poly_area": poly_area,
                    "ylop_area": ylop_area,
                    "net": net,
                    "ok": area_ok
                }
            }

            # exact-fit (only attempt if area looks right)
            if area_ok:
                exact_ok, exact_det = self._polyfit_region_exact(region, lst)
            else:
                exact_ok, exact_det = False, {"skipped": True}

            detail["exact_fit"] = {"ok": exact_ok, **exact_det}
            detail["ok"] = area_ok and exact_ok
            region_details.append(detail)

        violations = [d["region"] for d in region_details if not d["ok"]]

        return len(violations) == 0, {
            "violations": violations,
            "region_details": region_details
        }
    
    # ---------- End Rule Check Functions ----------
        
    # ---------- Poly/Ylop helpers ----------
        
    def _extract_poly_instances(self):
        """
        Helper Function for _rule_poly_ylop_balance
        Extracts instances of polyshapes and ylop shapes from the additional_info array.
        """
        instances = []
        if not isinstance(self.polyshapes, dict):
            return instances
        h, w = self.additional_info.shape
        for x in range(h):
            for y in range(w):
                val = self.additional_info[x, y]
                if val != 0:
                    name = f'{val}'
                    if name not in self.polyshapes:
                        continue
                    shape_arr = np.array(self.polyshapes[name])
                    area = int(shape_arr.sum())
                    kind = 'poly' if (self.obs_array['poly'][x, y] == 1) else 'ylop'
                    instances.append({"name": name, "x": x, "y": y, "area": area, "kind": kind})
        return instances
    
    def _polyfit_region_exact(self, region, instances):
        '''
        Helper function to check exact fit of polygons in a region.
        '''
        H, W = self.x_size, self.y_size

        region_center_mask = np.zeros((H, W), dtype=bool)
        for (rx, ry) in region.cells:
            region_center_mask[rx, ry] = True
        region_size = int(region_center_mask[1::2, 1::2].sum())

        polys, ylops = [], []
        poly_area = 0
        ylop_area = 0
        for inst in instances:
            name = inst["name"]
            arr = np.array(self.polyshapes[name], dtype=np.int32)
            area = int(arr.sum())
            if inst["kind"] == "poly":
                polys.append({"name": name, "array": arr})
                poly_area += area
            else:
                ylops.append({"name": name, "array": arr})
                ylop_area += area

        net = poly_area - ylop_area

        # If ylops cancel polys exactly, geometry poses no constraint
        if net == 0: 
            poly_names = Counter(p["name"] for p in polys)
            ylop_names = Counter(y["name"] for y in ylops)
            if poly_names == ylop_names:
                return True, {
                    "region_id": region.id,
                    "region_area": region_size,
                    "poly_area": poly_area,
                    "ylop_area": ylop_area,
                    "net": net
                }
        
        grid = np.zeros((H, W), dtype=np.int32)
        if net > 0:
            grid[region_center_mask] = -1

        anchors_all = [(x, y) for x in range(1, H, 2) for y in range(1, W, 2)]

        ok = self._polyfit_place_ylops(ylops, 0, polys, grid, anchors_all)

        return ok, {
            "region_id": region.id,
            "region_area": region_size,
            "poly_area": poly_area,
            "ylop_area": ylop_area,
            "net": net
        }
        
    def _polyfit_place_ylops(self, ylops, idx, polys, grid, anchors):
        '''
        Helper function to place ylop shapes in the grid.
        '''

        if idx == len(ylops):
            # hand over to poly placement (to be implemented next)
            return self._polyfit_place_polys(polys, grid)

        ylop = ylops[idx]
        arr_o = ylop["array"]

        offsets = self._get_offsets(arr_o)
        for (ax, ay) in anchors:
            if not self._try_place_polys(grid, offsets, ax, ay, sign=-1):
                continue

            if self._polyfit_place_ylops(ylops, idx + 1, polys, grid, anchors):
                return True

            self._unplace_offsets(grid, offsets, ax, ay, sign=-1)

        return False

    def _polyfit_place_polys(self, polys, grid):
        '''
        Helper function to place poly shapes in the grid.
        '''
        if np.any(grid > 0):
            return False

        if not polys:
            return not np.any(grid < 0)

        negs = np.argwhere((grid < 0))
        if negs.size == 0:
            return True
        nx, ny = negs[np.lexsort((negs[:,1], negs[:,0]))][0]
        target = [(int(nx), int(ny))]

        for (ax, ay) in target:
            tried_names = set()
            for i, poly in enumerate(polys):
                name = poly["name"]
                if name in tried_names:
                    continue
                tried_names.add(name)

                arr_o = poly["array"]
                offsets = self._get_offsets(arr_o)
                if not self._try_place_polys(grid, offsets, ax, ay, sign=+1):
                    continue

                rem = polys[:i] + polys[i+1:]
                if self._polyfit_place_polys(rem, grid):
                    return True

                self._unplace_offsets(grid, offsets, ax, ay, sign=+1)

        return False


    def _get_offsets(self, shape_arr):
        '''
        Helper function to get the offsets of a shape array.
        '''
        shape = np.array(shape_arr, dtype=np.int32)
        xs, ys = np.where(shape == 1)
        if len(xs) == 0:
            return []
        ax = xs.min()
        ay = ys[np.where(xs == ax)[0]].min()
        offsets = []
        for x, y in zip(xs, ys):
            cx = 2 * (x - ax)
            cy = 2 * (y - ay)
            offsets.append((cx, cy))
        return offsets


    def _try_place_polys(self, grid, offsets, anchor_x, anchor_y, sign):
        '''
        Helper function to try placing polygons in the grid.
        '''
        H, W = grid.shape
        targets = []
        for dx, dy in offsets:
            tx, ty = anchor_x + dx, anchor_y + dy
            if tx < 0 or tx >= H or ty < 0 or ty >= W:
                return False
            targets.append((tx, ty))
        for tx, ty in targets:
            grid[tx, ty] += sign
        return True
    
    def _unplace_offsets(self, grid, offsets, anchor_x, anchor_y, sign):
        '''
        Helper function to unplace shapes from the grid.
        '''
        for dx, dy in offsets:
            tx, ty = anchor_x + dx, anchor_y + dy
            grid[tx, ty] -= sign
    
    # ---------- End Poly/Ylop helpers ----------
    
    # ---------- Validate Rules Functions ----------

    def _run_rule_validators(self, regions, terminated, truncated):
        '''
        Run all rule validators on the given regions.
        '''

        rule_results = {}

        def add(name, passed, detail):
            rule_results[name] = {"passed": passed, "detail": detail}

        # Global / path related
        p, d = self._rule_reached_target()
        add("reached_target", p, d)
        p, d = self._rule_path_not_crossing()
        add("path_not_crossing", p, d)
        p, d = self._rule_no_gap_violations()
        add("no_gap_violations", p, d)
        p, d = self._rule_all_dots_collected()
        add("all_dots_collected", p, d)

        # Region & symbol based
        p, d = self._rule_color_square_separation(regions)
        add("square_color_separation", p, d)
        p, d = self._rule_star_pairing_exact(regions)
        add("star_pairing_exact", p, d)
        p, d = self._rule_triangles_edges()
        add("triangles_edge_count", p, d)
        p, d = self._rule_poly_ylop_balance(regions)
        add("poly_ylop_area", p, d)

        # Aggregate success (only core puzzle rules, not termination flags)
        core = [k for k in rule_results.keys()
                if k not in ("_terminated", "_truncated", "all_rules_satisfied")]
        all_pass = all(rule_results[k]["passed"] for k in core)
        add("all_rules_satisfied", all_pass, {"rules_checked": core})

        rule_results["_terminated"] = {"passed": True, "detail": terminated}
        rule_results["_truncated"] = {"passed": True, "detail": truncated}
        return rule_results

    def _validate_rules(self, terminated=False, truncated=False):
        '''
        Validate the rules for the current puzzle state.
        '''
        regions, region_map = self._compute_regions()
        self._collect_region_symbols(regions, region_map)
        self.rule_status = self._run_rule_validators(regions, terminated, truncated)
        # Attach region summaries
        self.rule_status["_regions"] = {r.id: r.to_summary() for r in regions}
        return self.rule_status

    # ---------- End Validate Rules Functions ----------

    # ---------- Get Observation/Info Functions ----------

    def _get_obs(self):
        '''
        Function to return the current observation of the puzzle
        if observation == 'new':
        
        Returns: dictionary of:
        dictionary of 2D arrays,
        2D array,
        2D array
        ----------
        {
        obs : dict; dictionary of 2D arrays
            A dictionary containing the current locations of the properties of the puzzle
        color_array : 2D array
            A 2D array containing the colors of the properties in the puzzle
        additional_info : 2D array
            A 2D array containing additional information about the properties in the puzzle
        }

        if observation == 'SPaRC':
        Returns a json string representation of the puzzle in the SPaRC format
        '''
        if self.observation == 'new':
            return {'base': self.obs_array, 'color': self.color_array, 'additional_info': self.additional_info}
        
        elif self.observation == 'SPaRC':
            observ = self._build_json_obs()
            return observ

        else:
            raise ValueError("Invalid observation type. Choose 'new' or 'SPaRC'.")

    def _build_json_obs(self):
        '''
        Helper Function to turn the SPaRC observation into JSON format
        '''
        return json.dumps(self.observ, separators=(',', ':'))
    
    def _get_info(self):
        '''
        Function to return extra information of the current puzzle
        
        Returns:
        info : dict
            A dictionary containing the following information:
            - solution_count: The number of solutions for the current puzzle
            - difficulty: The difficulty of the current puzzle
            - grid_y_size: The y size of the current puzzle
            - grid_x_size: The x size of the current puzzle
            - legal_actions: The legal actions for the current state of the agent
            - current_step: The current step of the agent
            - agent_location: The current location of the agent
            - rule_status: The current status of the rules
            - Rewards: The current rewards
        '''
        self._validate_rules(terminated=False, truncated=False)
        info = {"solution_count": self.solution_count,
        "difficulty": self.difficulty,
        "grid_x_size": self.x_size,
        "grid_y_size": self.y_size,
        "legal_actions": self._get_legal_actions(),
        "current_step": self.current_step,
        "agent_location": self._agent_location,
        "rule_status": self.rule_status,
        "Rewards": {"normal_reward": self.normal_reward, "outcome_reward": self.outcome_reward}
        }
        return info
    
    def _get_legal_actions(self):
        '''
        Function to get the legal actions for the current state of the agent
        
        Returns:
        legal : list
            A list of legal actions for the current state of the agent
        '''
        legal = []
        
        for action, direction in self._action_to_direction.items():
            next_loc = self._agent_location + direction
            # np.clip to make sure we don't go out of bounds
            agent_location_temp = np.clip(next_loc, [0, 0], [self.x_size - 1, self.y_size - 1])
            # Check if the next location is not a gap
            if self.obs_array['gaps'][agent_location_temp[0], agent_location_temp[1]] == 0:
                if self.obs_array['visited'][agent_location_temp[0], agent_location_temp[1]] == 1:
                    if self.traceback:
                        if len(self.path) >= 2:
                            last_loc = np.array([self.path[-2][0], self.path[-2][1]], dtype=np.int32)
                            if np.array_equal(last_loc, agent_location_temp):
                                if np.array_equal(next_loc, agent_location_temp): 
                                    legal.append(action)
                else:
                    if np.array_equal(next_loc, agent_location_temp):
                        legal.append(action)
            
        return legal

    # ---------- End Get Observation/Info Functions ----------

    # ---------- Core Gym Env Functions ----------

    def reset(self, seed=None, options=None):
        '''
        Function to reset the environment and load the next puzzle
        Parameters:
        seed : int
            The seed for the random number generator
        options : dict
            Additional options for resetting the environment
            Only used option is 'puzzle_id' to load a specific puzzle by its ID
        ----------
        Returns:
        obs : depends on obs type given in gym.make()
            The Observation of the current puzzle state
        info : dict
            A dictionary containing the extra information of the current puzzle
        '''
        super().reset(seed=seed)
        
        puzzle_id = None
        if options is not None:
            # If a specific puzzle ID is given, load that puzzle
            puzzle_id = options.get('puzzle_id', None)
            for idx, puzzle in enumerate(self.puzzles):
                if puzzle['id'] == puzzle_id:
                    self.current_puzzle_index = idx
                    break
        else:
            if seed is not None:
                self.current_puzzle_index = int(self.np_random.integers(len(self.puzzles)))
            else:
                self.current_puzzle_index = (self.current_puzzle_index + 1) % len(self.puzzles)
        
        # Also possible to randomly select a puzzle
        # self.current_puzzle_index = np.random.randint(0, len(self.puzzles))
        
        self.current_step = 0
        
        # Load the next puzzle
        self._load_puzzle(self.current_puzzle_index)
        
        # Visualize the initial state of the environment
        if self.render_mode == "human":
            if self.human_renderer is None:
                self.human_renderer = HumanRenderer(scale_factor=3.0)
            self.render()
        elif self.render_mode == "llm":
            if self.llm_renderer is None:
                self.llm_renderer = LLMRenderer()
            self.render()
        
        # Return the initial observation for the new puzzle
        return self._get_obs(), self._get_info()
    
    
    def step(self, action):
        '''
        Function to take a step in the environment
        Parameters:
        action : int
            The action to take in the environment
        ----------
        Returns:
        obs : depends on obs type given in gym.make()
            The Observation of the current puzzle state
        reward : int
            The reward for taking the action
        terminated : bool
            Whether the episode has terminated
        truncated : bool
            Whether the episode has been truncated
        info : dict
            A dictionary containing the extra information of the current puzzle
        '''
        
        orig_loc = self._agent_location.copy()
        self.current_step += 1
        self.normal_reward = 0
        truncated = self.current_step >= self.max_steps
        
        # If the action is not in the legal actions, we do not move
        if action in self._get_legal_actions():
            direction = self._action_to_direction[action]
            agent_location_temp = self._agent_location + direction
            
            if self.obs_array['visited'][agent_location_temp[0], agent_location_temp[1]] == 1:
                if self.traceback:
                    last_loc = np.array([self.path[-2][0], self.path[-2][1]], dtype=np.int32)
                    if np.array_equal(last_loc, agent_location_temp): 
                        # If the next location is already visited and is the last location, we are allowed to move back
                        self.obs_array['agent_location'][self._agent_location[0]][self._agent_location[1]] = 0
                        self.obs_array['visited'][self._agent_location[0]][self._agent_location[1]] = 0
                        
                        # Update the SPaRC observation if it is active
                        if self.observation == 'SPaRC':
                            r, c = self._agent_location[0], self._agent_location[1]
                            self.observ[c][r] = '.' if self.obs_array['gaps'][r, c] == 1 else '+'                        
                        
                        self._agent_location = agent_location_temp

                        # Update the agent's location in the observation
                        self.obs_array['visited'][self._agent_location[0]][self._agent_location[1]] = 1
                        self.obs_array['agent_location'][self._agent_location[0]][self._agent_location[1]] = 1

                        # Also update the SPaRC observation if it is active
                        if self.observation == 'SPaRC':
                            r, c = self._agent_location[0], self._agent_location[1]
                            self.observ[c][r] = 'L'

                        # Update the path
                        del self.path[-1]
            else:
                self.obs_array['agent_location'][self._agent_location[0]][self._agent_location[1]] = 0

                # Update the SPaRC observation if it is active
                if self.observation == 'SPaRC':
                    r, c = self._agent_location[0], self._agent_location[1]
                    self.observ[c][r] = 'V'

                self._agent_location = agent_location_temp
                
                # Update the agent's location in the observation
                self.obs_array['visited'][self._agent_location[0]][self._agent_location[1]] = 1
                self.obs_array['agent_location'][self._agent_location[0]][self._agent_location[1]] = 1

                # Also update the SPaRC observation if it is active
                if self.observation == 'SPaRC':
                    r, c = self._agent_location[0], self._agent_location[1]
                    self.observ[c][r] = 'L'

                # Update the path
                path = [self._agent_location[0], self._agent_location[1]]
                self.path.append(path)
          
        
        # An episode is done if the agent has reached the target, does not mean success
        terminated = np.array_equal(self._agent_location, self._target_location)
        
        # If there are no legal actions left (for the next step), the episode is truncated
        if self._get_legal_actions() == []:
            truncated = True
        
        if terminated:
            truncated = False  # If terminated, not truncated
        
        # Reward logic:
        # Have an outcome reward and a normal reward
        # The normal reward is updated during the episode, the outcome reward is only updated at the end of the episode
        if terminated or truncated:
            for i in range(self.solution_count):
                if np.array_equal(self.path, self.solution_paths[i]):
                    self.outcome_reward = 1
                    self.normal_reward = 1
                    break
                
            if self.outcome_reward != 1:
                self.outcome_reward = -1
                self.normal_reward = -1
        else:
            self.outcome_reward = 0
            if not np.array_equal(orig_loc, self._agent_location):
                for i in range(self.solution_count):
                    current_solution_path = self.solution_paths[i]
                    if self._is_on_solution_path(self.path, current_solution_path):
                        self.normal_reward = 0.01
                        break
                    else:
                        self.normal_reward = -0.01

        
        # Update the observation
        self._validate_rules(terminated=terminated, truncated=truncated)
        observation = self._get_obs()
        info = self._get_info()

        # Unfortunately, I have to return the normal reward here, since gymnasium expects a reward to be a scalar value, not a dictionary of scalar values
        reward = self.normal_reward
        
        # Visualize the current state of the environment
        if self.render_mode == "human" or self.render_mode == "llm":
            self.render()
        
        return observation, reward, terminated, truncated, info


    # ---------- End Core Gym Env Functions ----------


    def _is_on_solution_path(self, current_path, solution_path):
        """
        Helper function for step function.
        Checks if the current path is still on the solution path.

        Args:
            current_path (list): The path taken so far.
            solution_path (list): The correct solution path.

        Returns:
            bool: True if the current path is on the solution path, False otherwise.
        """
        # If the current path is longer than the solution path, return False
        if len(current_path) > len(solution_path):
            return False

        # Compare each step in the current path with the solution path
        for i in range(len(current_path)):
            if current_path[i] != solution_path[i]:
                return False

        return True

    # ---------- Visualization Functions ----------

    def render(self):
        """
        Renders the environment using the appropriate renderer based on render_mode.
        
        Returns:
            str or None: For LLM mode, returns a text representation. For human mode, returns None.
        """
        if self.render_mode == "human" and self.human_renderer is not None:
            return self.human_renderer.render(
                self.obs_array, 
                self.color_array, 
                self.additional_info, 
                self.polyshapes, 
                self.x_size, 
                self.y_size,
                self.start_location,
                self.target_location,
                self.path
            )
        elif self.render_mode == "llm" and self.llm_renderer is not None:
            return self.llm_renderer.render(
                self.obs_array, 
                self.color_array, 
                self.additional_info, 
                self.polyshapes, 
                self.x_size, 
                self.y_size
            )
        else:
                         # No rendering or unsupported mode
             return None
    
    def close(self):
        """
        Close the environment and cleanup any resources.
        """
        if self.human_renderer is not None:
            self.human_renderer.close()
            self.human_renderer = None
            
        if self.llm_renderer is not None:
            self.llm_renderer.close()
            self.llm_renderer = None


    # ---------- End Visualization Functions ----------
