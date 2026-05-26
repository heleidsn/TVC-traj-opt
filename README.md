# TVC Rocket Trajectory Optimization

TVC (Thrust Vector Control) Rocket Trajectory Optimization using Crocoddyl and Acados optimal control libraries.

## Features

- **Trajectory Optimization**: Crocoddyl FDDP algorithm and Acados NLP solver
- **Multi-Waypoint Planning**: Supports multiple waypoints with specified arrival times and yaw angles
- **Real-time GUI**: Interactive PyQt5/PySide2 GUI for parameter adjustment and visualization
- **Comprehensive Constraints**: 
  - Control constraints (TVC pitch/roll angles, thrust, yaw torque)
  - State constraints (velocity, angles, angular velocity)
  - Separate horizontal and vertical velocity limits
- **Physical Parameters**: Configurable mass, moment of inertia, and thrust position

## Requirements

### Python Version
- Python 3.7+

### Main Dependencies
- `crocoddyl`: Optimal control library (Crocoddyl-based scripts)
- `numpy`: Numerical computing
- `matplotlib`: Visualization
- `PyQt5` or `PySide2`: GUI framework
- `pyyaml`: YAML configuration file parsing

### Optional (for Acados)
- `casadi`: Symbolic framework
- `acados`: Nonlinear MPC/NLP solver (requires build from source)

## Installation

### 1. Base Installation (Crocoddyl + GUI)

#### Using conda (Recommended)

```bash
conda create -n tvc-opt python=3.10
conda activate tvc-opt
conda install -c conda-forge crocoddyl
pip install -r requirements.txt
```

#### Using pip + virtualenv

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
# or venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

**Note**: `crocoddyl` may need to be installed via conda-forge on some systems.

### 2. Acados Installation (Optional, for `tvc_traj_opt_acados.py`)

Acados provides faster trajectory optimization with native constraint handling. Build from source:

#### Prerequisites
- `git`, `make`, `cmake`
- `casadi`: `pip install casadi`

#### Build Acados (Linux/Mac)

```bash
git clone https://github.com/acados/acados.git
cd acados
git submodule update --recursive --init

mkdir -p build
cd build
cmake -DACADOS_WITH_QPOASES=ON -DBUILD_SHARED_LIBS=ON ..
make install -j4
```

#### Install Python Interface

```bash
pip install -e <acados_root>/interfaces/acados_template
```

#### Set Environment Variables

Add to your shell config (`~/.bashrc` or `~/.zshrc`):

```bash
export ACADOS_SOURCE_DIR=/path/to/acados
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ACADOS_SOURCE_DIR/lib
```

On macOS, use `DYLD_LIBRARY_PATH` instead of `LD_LIBRARY_PATH`.

See [acados installation docs](https://docs.acados.org/installation/) for more details.

## Project Structure

```
TVC-traj-opt/
├── run_tvc_traj_opt.py         # **Main entry** — launches trajectory optimization GUI
├── scripts/                    # Implementation (called by GUI / CLI)
│   ├── tvc_traj_opt_gui.py     # GUI implementation (Methods 1–7)
│   ├── tvc_traj_opt.py         # Method 1: Crocoddyl custom calcDiff
│   ├── tvc_traj_opt_pinocchio.py
│   ├── tvc_traj_opt_acados.py  # Methods 4–5–7 (Acados)
│   ├── tvc_traj_opt_acados_min_time.py  # Method 6 (Spannagl FFTF)
│   ├── tvc_attitude_ctrl_gui.py # Attitude control simulation (separate app)
│   └── tvc_common.py           # Shared utilities
├── config/                     # Configuration files
│   └── tvc_params.json         # Default params for attitude GUI
├── results/                    # Output directory
├── models/                     # Model files
└── assets/                     # Project resources
```

## Usage

### GUI Application (main entry)

```bash
python run_tvc_traj_opt.py
```

All optimization methods (Crocoddyl Methods 1–3, Acados Methods 4–7) are selected inside the GUI; solvers live under `scripts/`.

The GUI allows you to:
- Set waypoints with positions, yaw angles, and arrival times
- Adjust cost weights and constraints
- Configure physical parameters
- Visualize optimization results in real-time

### Command-line Script (Crocoddyl)

```bash
python -u scripts/tvc_traj_opt.py
```

Use `-u` flag for unbuffered output to see real-time iteration information.

### Acados Trajectory Optimization

Acados-based optimization with native constraint handling and faster convergence:

```bash
# Ensure environment variables are set (add to ~/.bashrc for persistence)
export ACADOS_SOURCE_DIR=/path/to/acados
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ACADOS_SOURCE_DIR/lib

python -u scripts/tvc_traj_opt_acados.py
```

The script auto-detects `acados` if not found in `PATH`. It supports:
- Control bounds (TVC angles, thrust, yaw torque)
- State path constraints (velocity, angles)
- Optional actuator dynamics (first-order lag)
- Optional control rate penalty

### TVC Attitude Control Simulation

Interactive P+PID attitude control simulation with GUI:

```bash
python scripts/tvc_attitude_ctrl_gui.py
```

Parameters are saved/loaded from `config/tvc_params.json` by default.

## Waypoint Format

Waypoints are specified as: `[x, y, z, yaw_deg, arrival_time]`
- `x, y, z`: Position in meters
- `yaw_deg`: Yaw angle in degrees
- `arrival_time`: Arrival time in seconds

## Default Waypoints

- **Start**: [0.0, 0.0, 0.0, 0°, 0s]
- **Waypoint 1**: [0.0, 0.0, 10.0, 0°, 5s]
- **Waypoint 2**: [5.0, 0.0, 10.0, 0°, 10s]

## Configuration

### Physical Parameters
- Mass (kg)
- Moment of Inertia (Ixx, Iyy, Izz in kg·m²)
- Thrust Position (r_thrust_x, r_thrust_y, r_thrust_z in m)

### Constraints
- **Control Constraints**: TVC pitch/roll angles, thrust, yaw torque
- **State Constraints**: 
  - Horizontal velocity (m/s)
  - Vertical velocity (m/s)
  - Euler angles (roll, pitch, yaw in degrees)
  - Angular velocity (rad/s)

### Cost Weights
- Position (p)
- Velocity (v)
- Attitude (R)
- Angular Velocity (w)
- Control (u)
- Control Change (du)
- Constraint Penalties (k_bound, k_state_bound)

## Output

After optimization, the following are generated:
- **Visualization Plots**: Complete plots of all states, controls, and cost
- **Trajectory Data**: State and control trajectories

## Author

Lei He
