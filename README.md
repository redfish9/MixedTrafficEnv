# MixedTrafficEnv

A modified version of the highway-env package that supports mixed traffic simulation with heterogeneous vehicles.

## Acknowledgments

This project is based on the original [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv) repository by Farama Foundation. We extend our gratitude to the original authors for their excellent work.

## Installation

Follow these steps to install and set up the environment:

### 1. Clone the Repository

```bash
git clone https://github.com/redfish9/MixedTrafficEnv.git highway_env
```

### 2. Install the Original Highway-Env Package

Install the original highway-env package (version 1.7 specifically - newer versions are not compatible):

```bash
pip install highway-env==1.7
```

### 3. Replace the Original Package with This Repository

First, find the installation path of the original highway-env package:

```bash
pip show highway_env
```

Then, copy all files from this repository to replace the original package:

```bash
cp -r highway_env/* path_to_your_highway_env
```

Replace `path_to_your_highway_env` with the actual installation path shown by the `pip show` command.

## Requirements

This project has the same dependencies as the original HighwayEnv repository. The main requirements are:

- Python >= 3.7
- NumPy
- Matplotlib
- Gymnasium
- Pygame
- Pandas
- Scipy

See the original [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv) repository for detailed requirements.

## Features

This modified version of highway-env provides:

- Support for heterogeneous traffic with different vehicle types
- Enhanced simulation capabilities for mixed traffic scenarios
- Custom environments for intention prediction and interaction modeling

## Usage

After installation, you can import and use the environment just like the original highway-env package:

```python
import highway_env
import gym

# Register custom environments
from highway_env.create_env import register_highway_envs
register_highway_envs()

# Create an environment
env = gym.make("intention-v0")

# Configure as needed
env.configure({
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy"]
    }
})

# Use the environment
observation = env.reset()
```

## License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2023 redfish9

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
