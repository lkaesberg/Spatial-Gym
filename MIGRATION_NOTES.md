# Migration from SPaRC-Gym to Spatial-Gym

## Overview
This document summarizes the changes made when migrating from SPaRC-Gym to Spatial-Gym.

## Repository Changes
- **Old Repository**: `tobiTKM/SPaRC-Gym`
- **New Repository**: `lkaesberg/Spatial-Gym`
- **URL**: https://github.com/lkaesberg/Spatial-Gym

## Package Changes
- **Package Name**: `SPaRC-Gym` → `Spatial-Gym`
- **Module Name**: `SPaRC_Gym` → `Spatial_Gym`
- **Version**: `0.2.0` → `1.0.0`

## File Structure Changes
```
SPaRC_Gym/           →  Spatial_Gym/
├── SPaRC_Gym.py     →  Spatial_Gym.py
├── __init__.py          (updated imports)
└── register_env.py      (updated registration)
```

## Code Changes
All occurrences of `SPaRC` have been replaced with `Spatial` in:
- Python files (`.py`)
- Configuration files (`pyproject.toml`, `.yml`)
- Documentation (`README.md`)
- Comments and docstrings

## Updated Components

### 1. README.md
- Rewritten for ACL-style conference submission
- Added abstract and key features
- Comprehensive environment specification
- API reference with examples
- Citation information (BibTeX format)
- Enhanced documentation structure

### 2. pyproject.toml
- Updated package metadata
- Added keywords and classifiers
- Enhanced project URLs
- Updated authors list
- Bumped version to 1.0.0

### 3. GitHub Actions Pipeline
- Modernized to latest action versions (v4)
- Added manual workflow trigger
- Updated PyPI publishing action
- Added hatchling to build dependencies

## Usage Changes

### Old Usage
```python
import gymnasium as gym
import SPaRC_Gym
env = gym.make("SPaRC-Gym", ...)
```

### New Usage
```python
import gymnasium as gym
import Spatial_Gym
env = gym.make("Spatial-Gym", ...)
```

## Installation Changes

### Old
```bash
pip install SPaRC-Gym
```

### New
```bash
pip install Spatial-Gym
```

## Notes
- Git history was reset (clean start, not a fork)
- All functionality preserved
- API remains backward compatible
- Dataset references (`lkaesberg/SPaRC`) remain unchanged
