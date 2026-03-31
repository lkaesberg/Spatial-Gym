# Spatial-Gym: A Gymnasium Environment for Spatial Reasoning Benchmarking

[![PyPI](https://img.shields.io/pypi/v/Spatial-Gym)](https://pypi.org/project/Spatial-Gym/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Abstract

Spatial-Gym is a Gymnasium-compatible environment designed for evaluating spatial reasoning capabilities of Large Language Models (LLMs) and other AI agents. Built upon the spatial puzzle dataset introduced in SPaRC ([Kaesberg et al.](https://sparc.gipplab.org/)), this environment provides a standardized interface for benchmarking agent performance on grid-based spatial reasoning tasks. The environment supports multiple observation formats, customizable rendering modes for human and LLM interaction, and comprehensive evaluation metrics for systematic analysis of spatial reasoning abilities.

## Key Features

- **Standardized RL Interface**: Full Gymnasium API compliance for seamless integration with existing RL frameworks
- **Dual Observation Modes**: Structured tensor representation or JSON-based symbolic encoding
- **Multi-Modal Rendering**: Human-readable visualizations and LLM-optimized text representations
- **Flexible Dataset Support**: Compatible with HuggingFace datasets following the SPaRC format
- **Comprehensive Metrics**: Episode-level tracking of success rate, path efficiency, and reasoning patterns
- **Backtracking Support**: Optional state reversibility for exploring different solution strategies

## Installation

### From PyPI

```bash
pip install Spatial-Gym
```

### From Source

```bash
git clone https://github.com/lkaesberg/Spatial-Gym.git
cd Spatial-Gym
pip install -e .
```

## Quick Start

```python
import gymnasium as gym
import Spatial_Gym

# Initialize environment with default configuration
env = gym.make(
    "Spatial-Gym",
    df_name='lkaesberg/SPaRC',
    df_split='all',
    df_set='test',
    render_mode='human',
    observation='new',
    traceback=True,
    max_steps=1000
)

# Standard RL loop
observation, info = env.reset()
terminated = False

while not terminated:
    action = env.action_space.sample()  # Replace with your agent
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()

env.close()
```

## Environment Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df_name` | str | `'lkaesberg/SPaRC'` | HuggingFace dataset identifier |
| `df_split` | str | `'all'` | Dataset split to use |
| `df_set` | str | `'test'` | Subset of data (train/val/test) |
| `render_mode` | str | `None` | Visualization mode: `'human'`, `'llm'`, or `None` |
| `observation` | str | `'new'` | Observation format: `'new'` (tensor) or `'SPaRC'` (JSON) |
| `traceback` | bool | `False` | Enable state reversibility |
| `max_steps` | int | `2000` | Maximum steps per episode |

## Environment Specification

### Action Space

**Discrete(4)**: Four directional moves in the grid environment.

| Action | Value | Description |
|--------|-------|-------------|
| RIGHT | 0 | Move agent one cell to the right |
| UP | 1 | Move agent one cell upward |
| LEFT | 2 | Move agent one cell to the left |
| DOWN | 3 | Move agent one cell downward |

### Observation Space

#### Tensor Format (`observation='new'`)

A dictionary containing:
- **`base`** (Dict[str, np.ndarray]): One-hot encoded spatial features
  - `visited`: Binary grid marking visited cells
  - `gaps`: Binary grid indicating traversable/non-traversable cells
  - `agent_location`: One-hot encoding of agent position
  - `target_location`: One-hot encoding of goal position
  - Additional puzzle-specific properties (e.g., `stars`, `triangles`)
- **`color`** (np.ndarray): Integer grid (1-8) representing color properties
- **`additional_info`** (np.ndarray): Puzzle-specific metadata (polyshape IDs, counts)

#### JSON Format (`observation='SPaRC'`)

String-encoded JSON representing the grid state with symbolic notation, following the original [SPaRC specification](https://github.com/lkaesberg/SPaRC).

### Reward Structure

- **+1.0**: Successfully solving the puzzle
- **-1.0**: Invalid termination or failure state
- **+0.01**: Incremental reward for remaining on valid solution path (encourages exploration while maintaining progress)

### Episode Termination

- **Success**: Agent reaches target location satisfying all puzzle constraints
- **Failure**: Agent enters invalid state or violates puzzle rules
- **Truncation**: Maximum step limit reached

## API Reference

### Core Methods

```python
env.reset(options: Optional[Dict] = None) -> Tuple[Observation, Dict]
```
Initializes or resets the environment to a new puzzle state.
- **Parameters**: 
  - `options`: Optional dictionary with `'puzzle_id'` key to load specific puzzle
- **Returns**: Initial observation and info dictionary

```python
env.step(action: int) -> Tuple[Observation, float, bool, bool, Dict]
```
Executes one environment step given an action.
- **Parameters**: 
  - `action`: Integer in range [0, 3] representing directional move
- **Returns**: Observation, reward, terminated flag, truncated flag, info dictionary

```python
env.render() -> Optional[np.ndarray]
```
Generates visual or textual representation of current state based on `render_mode`.

```python
env.close()
```
Releases environment resources and closes rendering windows.

## Experimental Setup

### Dataset

The environment uses puzzles from the SPaRC dataset, which contains spatial reasoning challenges of varying complexity. Each puzzle is defined by:
- Grid dimensions (variable size)
- Initial agent position
- Target position
- Spatial constraints (gaps, regions, colored elements)
- Solution paths of varying lengths

### Evaluation Metrics

The `info` dictionary returned by `step()` and `reset()` contains:
- **Success Rate**: Binary indicator of puzzle completion
- **Path Length**: Number of steps taken
- **Optimality**: Ratio of actual path length to shortest possible path
- **Invalid Actions**: Count of rule violations
- **Puzzle Metadata**: Difficulty rating, constraint types, grid size

## Use Cases

### Benchmarking LLM Spatial Reasoning

```python
import gymnasium as gym
import Spatial_Gym
from your_llm_wrapper import LLMAgent

env = gym.make("Spatial-Gym", render_mode='llm', observation='SPaRC')
agent = LLMAgent(model="gpt-4")

observation, info = env.reset()
for _ in range(100):  # Evaluate on 100 puzzles
    done = False
    while not done:
        action = agent.predict(env.render())  # LLM sees text representation
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    
    # Log metrics
    print(f"Puzzle {info['puzzle_id']}: Success={info['success']}, Steps={info['steps']}")
    observation, info = env.reset()
```

### Reinforcement Learning Training

```python
from stable_baselines3 import PPO

env = gym.make("Spatial-Gym", observation='new', max_steps=500)
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
model.save("spatial_reasoning_agent")
```

## Repository Structure

```
Spatial-Gym/
├── Spatial_Gym/           # Core environment implementation
│   ├── __init__.py        # Package initialization
│   ├── Spatial_Gym.py     # Main environment class
│   ├── register_env.py    # Gymnasium registration
│   └── render/            # Rendering modules
│       ├── human_renderer.py
│       └── llm_renderer.py
├── llm_testing/           # LLM evaluation utilities
│   ├── llm_host.py        # LLM interaction wrapper
│   └── parse_logs.py      # Result analysis tools
├── Final_Product.py       # Interactive demo script
├── human_play.py          # Human player interface
├── pyproject.toml         # Package configuration
└── README.md              # This file
```

## Citation

If you use Spatial-Gym in your research, please cite:

```bibtex
@software{spatial_gym2024,
  title={Spatial-Gym: A Gymnasium Environment for Spatial Reasoning Benchmarking},
  author={Kaesberg, Lars Benedikt and Mark, Tobias},
  year={2024},
  url={https://github.com/lkaesberg/Spatial-Gym}
}
```

For the underlying SPaRC dataset and puzzles:

```bibtex
@inproceedings{kaesberg2024sparc,
  title={SPaRC: Spatial Reasoning Challenges for Large Language Models},
  author={Kaesberg, Lars Benedikt and others},
  booktitle={Proceedings of ACL},
  year={2024},
  url={https://sparc.gipplab.org/}
}
```

## Contributing

We welcome contributions! Please follow these guidelines:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes with descriptive messages
4. Add tests for new functionality
5. Submit a pull request

For bug reports and feature requests, please use the [GitHub issue tracker](https://github.com/lkaesberg/Spatial-Gym/issues).

## License

This project is licensed under the MIT License - see the [LICENCE](LICENCE) file for details.

## Acknowledgments

- **Lars Benedikt Kaesberg** (l.kaesberg@uni-goettingen.de) - Project conception and supervision
- **Jan Philip Wahle** - Project supervision
- **Tobias Mark** - Initial implementation and environment design
- **SPaRC Team** - Original puzzle dataset and framework ([sparc.gipplab.org](https://sparc.gipplab.org/))

## Contact

For questions, suggestions, or collaboration inquiries, please contact:
- Lars Benedikt Kaesberg: l.kaesberg@uni-goettingen.de
- GitHub Issues: https://github.com/lkaesberg/Spatial-Gym/issues