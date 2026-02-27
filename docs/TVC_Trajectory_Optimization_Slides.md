# TVC Rocket Trajectory Optimization
## Technical Presentation

---

# Outline

1. **Overview** — Project goals and features
2. **Problem Formulation** — State, control, dynamics, cost
3. **Optimization Methods** — Method 1, 2, 3 comparison
4. **FDDP Algorithm** — Backward/forward pass, Jacobians
5. **Implementation Details** — Finite difference, constraints
6. **GUI & Usage** — How to run and configure

---

# 1. Overview

---

## Project Overview

**TVC (Thrust Vector Control) Rocket Trajectory Optimization**

- Uses **Crocoddyl** optimal control library
- Multi-waypoint planning with arrival times and yaw
- Real-time PyQt5 GUI for parameter tuning
- Three optimization methods (speed vs flexibility trade-off)

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Trajectory Optimization** | FDDP algorithm from Crocoddyl |
| **Multi-Waypoint** | Arbitrary waypoints with time and yaw |
| **Constraints** | Control bounds, state bounds (velocity, angles) |
| **Cost Weights** | Separate Running Cost and Terminal Cost |
| **Physical Parameters** | Mass, inertia, thrust position configurable |

---

## Project Structure

```
TVC-traj-opt/
├── scripts/
│   ├── tvc_traj_opt.py        # Method 1: Custom calcDiff (numerical diff)
│   ├── tvc_traj_opt_pinocchio.py  # Method 2/3: Pinocchio + FDDP/BoxFDDP
│   └── tvc_traj_opt_gui.py    # GUI application
├── models/tvc/                # URDF model
├── config/                    # YAML config
└── results/                   # Trajectories, plots, videos
```

---

# 2. Problem Formulation

---

## State and Control

**State** \(x \in \mathbb{R}^{17}\):
- \(p \in \mathbb{R}^3\) — position
- \(v \in \mathbb{R}^3\) — velocity
- \(q \in \mathbb{R}^4\) — quaternion (attitude)
- \(\omega \in \mathbb{R}^3\) — angular velocity
- \(u_{\text{prev}} \in \mathbb{R}^4\) — previous control (for \(\Delta u\) cost)

**Control** \(u \in \mathbb{R}^4\):
- \(\theta_p\) — TVC pitch angle
- \(\theta_r\) — TVC roll angle
- \(T\) — thrust magnitude
- \(\tau_{\text{yaw}}\) — yaw torque

---

## Dynamics (Discrete-time)

Semi-implicit Euler + quaternion integration:

\[
\begin{aligned}
a &= \frac{1}{m} R_{wb} R_{tvc}(\theta_p, \theta_r) \begin{bmatrix} 0 \\ 0 \\ T \end{bmatrix} + \begin{bmatrix} 0 \\ 0 \\ -g \end{bmatrix} \\
v_{k+1} &= v_k + \Delta t \cdot a \\
p_{k+1} &= p_k + \Delta t \cdot v_{k+1} \\
\tau &= r \times F_b + [0, 0, \tau_{\text{yaw}}]^T \\
\omega_{k+1} &= \omega_k + \Delta t \cdot I^{-1}(\tau - \omega \times I\omega) \\
q_{k+1} &= \text{quat\_mul}(\text{quat\_exp}(\omega_{k+1} \Delta t), q_k)
\end{aligned}
\]

---

## Cost Function

**Running cost** (per step):
\[
\ell = w_p \|p - p_g\|^2 + w_v \|v - v_g\|^2 + w_R \|e_R\|^2 + w_\omega \|\omega - \omega_g\|^2 + w_u \|u - u_{ref}\|^2 + w_{du} \|\Delta u\|^2 + \text{penalties}
\]

**Terminal cost** (at waypoint): Same structure with higher weights (e.g. \(w_p^{\text{term}}=200\)).

**Penalties**: Quadratic penalty for control/state bound violations.

---

## Constraints (Penalty Form)

| Type | Variables | Example |
|------|-----------|---------|
| **Control** | \(\theta_p, \theta_r, T, \tau_{\text{yaw}}\) | \(\theta_p \in [-10°, 10°]\) |
| **State** | \(v_{xy}, v_z\), roll, pitch, yaw, \(\|\omega\|\) | \(v_{xy} \leq 1\) m/s |

All enforced via \(k \cdot (x - \text{bound})^2\) when violated.

---

# 3. Optimization Methods

---

## Three Methods Overview

| Method | Solver | Dynamics/Jacobian | Speed |
|--------|--------|-------------------|------|
| **Method 1** | FDDP | Custom model + **numerical diff** | Slow |
| **Method 2** | FDDP | Pinocchio + **analytical** | Fast |
| **Method 3** | BoxFDDP | Pinocchio + **native control bounds** | Fastest |

---

## Method 1: Custom calcDiff

- **File**: `tvc_traj_opt.py`
- **Action model**: `TVCRocketActionModel` (inherits `ActionModelAbstract`)
- **Jacobians**: **Forward finite difference** with \(\epsilon = 10^{-6}\)
- **Pros**: No Pinocchio dependency, full control
- **Cons**: Slower (nx+nu extra evaluations per step)

---

## Method 2: Pinocchio + FDDP

- **File**: `tvc_traj_opt_pinocchio.py`
- **Dynamics**: Pinocchio URDF model + custom TVC actuation
- **Jacobians**: **Analytical** (Pinocchio auto-diff)
- **Constraints**: Penalty (same as Method 1)
- **Pros**: Much faster than Method 1

---

## Method 3: Pinocchio + BoxFDDP

- Same as Method 2 but uses **SolverBoxFDDP**
- **Native control bounds** (box constraints) instead of penalty
- Better convergence when control is near bounds
- **Unified mode**: Single problem over all waypoints (global optimization)

---

# 4. FDDP Algorithm

---

## DDP / FDDP Overview

**Differential Dynamic Programming (DDP)**:
- Iteratively linearize dynamics and quadraticize cost
- Backward pass: Riccati recursion → feedback gains \(K_k\), feedforward \(k_k\)
- Forward pass: Apply \(u_k^{\text{new}} = u_k + k_k + K_k(x_k^{\text{sim}} - x_k)\)

**FDDP (Feasibility-driven)**:
- Accepts infeasible trajectories during early iterations
- Better globalization than classical DDP

---

## Jacobians Used by FDDP

| Symbol | Size | Meaning |
|--------|------|---------|
| **Fx** | nx×nx | \(\frac{\partial x_{k+1}}{\partial x_k}\) |
| **Fu** | nx×nu | \(\frac{\partial x_{k+1}}{\partial u_k}\) |
| **Lx** | nx | \(\frac{\partial \ell}{\partial x}\) |
| **Lu** | nu | \(\frac{\partial \ell}{\partial u}\) |
| Lxx, Luu, Lxu | — | Cost Hessians (second order) |

Stored in `data` by `calcDiff()`, consumed by solver in backward pass.

---

## Finite Difference (Method 1)

**Forward difference**:
\[
\frac{\partial f}{\partial v_i} \approx \frac{f(v + \epsilon e_i) - f(v)}{\epsilon}, \quad \epsilon = 10^{-6}
\]

- Fx: perturb each \(x_i\), compute \(\Delta x_{\text{next}} / \epsilon\)
- Fu: perturb each \(u_i\), compute \(\Delta x_{\text{next}} / \epsilon\)
- Lx, Lu: perturb, compute \(\Delta \text{cost} / \epsilon\)

---

# 5. Implementation Details

---

## Waypoint Format

Each waypoint: `[x, y, z, yaw_deg, arrival_time]`

- Positions in meters
- Yaw in degrees
- Arrival time in seconds (must be strictly increasing)

**Example**: Start at (0,0,0), reach (0,0,10) at 5s, then (5,0,10) at 10s.

---

## Cost Weights (GUI)

**Running Cost**: p, v, R, w, u, du, k_bound, k_state_bound

**Terminal Cost**: Separate weights for waypoint precision (default: p=200, v=50, R=200, w=20, u=0, du=0)

---

## Running the Code

**GUI** (recommended):
```bash
python scripts/tvc_traj_opt_gui.py
```

**CLI**:
```bash
python -u scripts/tvc_traj_opt.py   # Method 1
python -u scripts/tvc_traj_opt_pinocchio.py  # Method 2/3
```

---

# 6. Summary

---

## Summary

- **Problem**: TVC rocket trajectory optimization with multi-waypoints
- **Formulation**: Shooting problem, quadratic cost, penalty constraints
- **Solver**: FDDP (Crocoddyl)
- **Methods**: 1 (numerical diff) vs 2/3 (Pinocchio analytical)
- **Output**: Trajectories, plots, videos

---

## References

- Crocoddyl: https://github.com/loco-3d/crocoddyl
- Pinocchio: https://github.com/stack-of-tasks/pinocchio
- FDDP: Mastalli et al., "Crocoddyl: An Efficient and Versatile Framework for Trajectory Optimization" (ICRA 2020)

---

# Thank You

**Questions?**

---

# Appendix: How to Present

This Markdown can be rendered as slides using:

- **Marp**: `npx @marp-team/marp-cli docs/TVC_Trajectory_Optimization_Slides.md -o slides.pdf`
- **reveal.js**: Use Pandoc or similar to convert to HTML
- **VS Code**: Install "Marp for VS Code" extension, then preview
