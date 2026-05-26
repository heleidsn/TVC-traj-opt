# TVC Rocket Dynamics Model (Acados / `export_tvc_ode_model`)

**Math rendering:** This file uses `$...$` (inline) and `$$...$$` (display). [GitHub](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions) renders these automatically. In **VS Code / Cursor**, the built-in Markdown preview often does **not** typeset math until you enable it (e.g. install the **Markdown+Math** extension, or turn on a *markdown.math* option if your build provides one). Earlier drafts used LaTeX-style `\(` `\)` and `\[` `\]`; many Markdown engines leave those as **literal text**, which is why formulas looked “broken.”

This document describes the continuous-time dynamics implemented in

`scripts/tvc_traj_opt_acados.py` → `export_tvc_ode_model`.

The model is a **6-DoF rigid body** with **thrust vector control (TVC)** and an optional **yaw body torque**, formulated in **world frame** for position/velocity and **ZYX Euler angles** for attitude. **CasADi** builds the ODE; **Acados** discretizes it for trajectory optimization.

---

## 1. State and control

### 1.1 Physical state (always 12 dimensions)

$$
x_{\mathrm{phys}} =
\bigl[
p^\top,\ \boldsymbol{\eta}^\top,\ v^\top,\ \boldsymbol{\omega}^\top
\bigr]^\top \in \mathbb{R}^{12}.
$$

| Symbol | Components | Meaning |
|--------|------------|---------|
| $p$ | $x,y,z$ | Position in the **world** frame |
| $\boldsymbol{\eta}$ | $\phi,\theta,\psi$ | Roll, pitch, yaw (**ZYX** convention; see §3) |
| $v$ | $v_x,v_y,v_z$ | Linear velocity in the **world** frame |
| $\boldsymbol{\omega}$ | $\omega_x,\omega_y,\omega_z$ | Angular velocity expressed in the **body** frame |

### 1.2 Control input (always 4 dimensions)

The optimization variable $u$ is the **commanded** control:

$$
u =
\begin{bmatrix}
\theta_{p,\,\mathrm{cmd}} \\
\theta_{r,\,\mathrm{cmd}} \\
T_{\mathrm{cmd}} \\
\tau_{\mathrm{yaw},\,\mathrm{cmd}}
\end{bmatrix}.
$$

- $\theta_p,\theta_r$: TVC gimbal angles. The code applies **pitch about body $y$**, then **roll about body $x$** (`Ry(th_p) @ Rx(th_r)`).
- $T$: Thrust magnitude (along the deflected nozzle axis before rotation to world frame; see §4).
- $\tau_{\mathrm{yaw}}$: Additional torque about **body $z$**.

### 1.3 Augmented state (optional, 16 dimensions total)

Two modes extend $x_{\mathrm{phys}}$ by four states. **They are mutually exclusive** in code.

| Mode | Augmented part | Role |
|------|------------------|------|
| **Actuator dynamics** | $u_{\mathrm{act}} \in \mathbb{R}^4$ | First-order lag: $\dot u_{\mathrm{act}} = (u - u_{\mathrm{act}}) \oslash \boldsymbol{\tau}$. The **plant** uses $u_{\mathrm{act}}$ for thrust and torques. |
| **Control-rate auxiliary** | $u_{\mathrm{prev}} \in \mathbb{R}^4$ | Dynamics $\dot u_{\mathrm{prev}} = (u - u_{\mathrm{prev}})/\Delta t$ with $\Delta t$ from solver parameter `p`; supports **control-rate penalty** in the cost. |

When neither mode is active, **`nx = 12`**.

---

## 2. World-to-body rotation

Body rotation from world to body uses **ZYX**:

$$
R(\phi,\theta,\psi) = R_z(\psi)\, R_y(\theta)\, R_x(\phi).
$$

A body-fixed vector $\mathbf{v}_b$ relates to a world vector $\mathbf{v}_w$ via $\mathbf{v}_w = R\,\mathbf{v}_b$ (same construction as in the source).

---

## 3. Kinematics

### 3.1 Position and linear velocity

$$
\dot p = v.
$$

### 3.2 Euler rates vs. body angular velocity

With the same ZYX convention, the code uses

$$
\dot{\boldsymbol{\eta}} = G(\phi,\theta)\,\boldsymbol{\omega},
$$

where $G$ is the standard mapping from body rates to Euler rates (singular when $\cos\theta \to 0$, i.e. gimbal lock).

---

## 4. Forces and translational dynamics

Thrust in the **body** frame:

$$
R_{\mathrm{tvc}}(\theta_p,\theta_r) = R_y(\theta_p)\, R_x(\theta_r), \qquad
\mathbf{F}_b = R_{\mathrm{tvc}}
\begin{bmatrix} 0 \\ 0 \\ T \end{bmatrix}.
$$

Here $(\theta_p,\theta_r,T,\tau_{\mathrm{yaw}})$ are taken from **`u_{\mathrm{cmd}}`** or **`u_{\mathrm{act}}`** depending on whether actuator dynamics is enabled.

Thrust in the **world** frame and gravity:

$$
\mathbf{F}_w = R\,\mathbf{F}_b, \qquad
\mathbf{F}_g = \begin{bmatrix} 0 \\ 0 \\ -m g \end{bmatrix}.
$$

World-frame linear acceleration:

$$
\dot v = \frac{1}{m}\left(\mathbf{F}_w + \mathbf{F}_g\right).
$$

---

## 5. Moments and rotational dynamics

Thrust moment about the center of mass (body frame), with constant lever arm **`r_thrust`**:

$$
\boldsymbol{\tau}_{\mathrm{thrust}} = \mathbf{r}_{\mathrm{thrust}} \times \mathbf{F}_b, \qquad
\boldsymbol{\tau} = \boldsymbol{\tau}_{\mathrm{thrust}} +
\begin{bmatrix} 0 \\ 0 \\ \tau_{\mathrm{yaw}} \end{bmatrix}.
$$

Rigid-body Euler equations in the body frame (constant inertia $I$):

$$
\dot{\boldsymbol{\omega}} = I^{-1}\bigl(\boldsymbol{\tau} - \boldsymbol{\omega} \times (I\boldsymbol{\omega})\bigr).
$$

---

## 6. Augmented dynamics (explicit)

Let $u_{\mathrm{plant}}$ denote the four channels used in §4–§5 ($u_{\mathrm{cmd}}$ or $u_{\mathrm{act}}$).

**Baseline (12D):**

$$
\dot x_{\mathrm{phys}} = f_{\mathrm{phys}}(x_{\mathrm{phys}},\, u_{\mathrm{plant}}).
$$

**Actuator dynamics (16D):** model parameter **`p = [1/\tau_1,\ldots,1/\tau_4]^\top`** per shooting node,

$$
\dot u_{\mathrm{act}} = (u - u_{\mathrm{act}}) \odot p,
$$

with $\odot$ element-wise product, and $u_{\mathrm{plant}} = u_{\mathrm{act}}$ inside $f_{\mathrm{phys}}$.

**Control-rate state (16D):** parameter **`p \in \mathbb{R}`** set to **`1/\Delta t`** (implementation uses horizon-based $N/T_f$ where applicable),

$$
\dot u_{\mathrm{prev}} = p\,(u - u_{\mathrm{prev}}),
$$

and $u_{\mathrm{plant}} = u_{\mathrm{cmd}}$ (no lag on the plant in this mode).

---

## 7. Modeling assumptions and neglected effects

This section states what the ODE **does** assume and what it **does not** include, so results are interpreted correctly (e.g. trajectory planning vs. high-fidelity propeller–airframe coupling).

### 7.1 What is modeled

- **Rigid body** with constant body inertia $I$ and fixed thrust lever arm $\mathbf{r}_{\mathrm{thrust}}$.
- **Thrust direction** is changed only through TVC gimbal angles $(\theta_p,\theta_r)$ applied to a nominal thrust axis (no separate nozzle / gimbal mechanism states beyond optional first-order **actuator lag** on the four control channels).
- **Moments on the body** are (i) $\mathbf{r}_{\mathrm{thrust}} \times \mathbf{F}_b$ from thrust misalignment and (ii) an explicit body-$z$ torque $\tau_{\mathrm{yaw}}$. Rotational dynamics use the usual **Euler equation** with gyroscopic *rigid-body* term $\boldsymbol{\omega} \times (I\boldsymbol{\omega})$ only—there is **no** additional spin momentum from a propeller or turbine treated as a separate rotating part.

### 7.2 Gyroscopic / propeller–disk effects (not modeled)

For a **high-speed rotating propeller, rotor, or jet spool**, changing thrust direction in reality requires **torques** that alter the **angular momentum** of that rotating mass. Reactions include:

- **Gyroscopic coupling**: body rate and thrust-axis tilt interact with rotor angular momentum (precession, extra moments on the airframe).
- **Reaction torques** when **slewing** the thrust line against large rotor angular momentum (sometimes described as the need to “torque” the disk orientation; the reaction appears on the airframe / gimbal structure).

The present implementation **does not** add:

- A rotor spin state (e.g. $\Omega$) or stored angular momentum vector $\mathbf{h}_{\mathrm{rotor}}$;
- Extra moments such as $\boldsymbol{\omega} \times \mathbf{h}_{\mathrm{rotor}}$ or terms tied to **gimbal rate** $\dot{\theta}_p,\dot{\theta}_r$ from gyroscopic reaction;
- Propeller **hinge moments** or aerodynamic torque on the disk beyond what you might crudely fold into $\tau_{\mathrm{yaw}}$ or tuning of $I$.

**Interpretation:** The model is appropriate for **coarse trajectory and TVC allocation** when gyroscopic torques are small compared to thrust–lever moments, or when time scales are slow. For **fast thrust-vector slewing** with **heavy rotor angular momentum**, predictions may be **optimistic** unless those effects are added (extended state or equivalent disturbance / margin).

### 7.3 Other standard simplifications

- **Constant $m$** (no mass depletion); **constant $g$**; no aerodynamic drag or added mass.
- **Stiction, backlash, and flexible structures** in the TVC mount are not modeled beyond optional scalar **first-order lag** on commanded inputs.
- **Euler (ZYX) attitude** implies a **kinematic singularity** when $\cos\theta \to 0$ (gimbal lock); the optimizer does not add quaternion regularization in this Acados build.

---

## 8. Acados model interface

- **Implicit ODE:** `f_impl = xdot - f_expl` with `f_expl` stacking $\dot p,\ \dot{\boldsymbol{\eta}},\ \dot v,\ \dot{\boldsymbol{\omega}}$ and, if applicable, $\dot u_{\mathrm{act}}$ or $\dot u_{\mathrm{prev}}$.
- **Symbolic variables:** `model.x`, `model.xdot`, `model.u`, and optional `model.p` for $\boldsymbol{\tau}^{-1}$ or $1/\Delta t$.

---

## 9. Post-processing and GUI

- **`acados_state_to_method1`:** Maps the **first 12 states** (Euler formulation) to a legacy 17-dimensional layout $[p, v, q_{\mathrm{wxyz}}, \boldsymbol{\omega}]$ for shared plotting with Method 1. The augmented 4 states are **not** folded into that `u_prev` slot.
- **Actual control trajectory** (when actuators are on): extracted from **`x[12:16]`** at each shooting node for visualization vs. commanded **`u`**.

---

## 10. Relation to other solvers in this repository

- **Pinocchio / Crocoddyl** paths use a **quaternion** attitude state on $\mathrm{SE}(3)$ with the same **TVC thrust direction** and **$\mathbf{r}\times\mathbf{F}_b$** torque structure; see `tvc_traj_opt_pinocchio.py`.
- **Older Method 1** (`tvc_traj_opt.py`) uses **quaternion + previous control in state** ($n_x=17$) for discrete-time cost on control increments.

Sections **1–9** focus on the **continuous-time ODE**. **§11** summarizes how the **GUI** maps **Method 4, Method 5, and Method 7** to Acados **OCP costs** in `scripts/tvc_traj_opt_acados.py` (fixed horizon, pseudo-time with $T_{\mathrm{seg}}$, and free-$t_f$ EXTERNAL).

---

## 11. GUI Acados costs: Method 4, Method 5, and Method 7

In the **TVC Trajectory Optimization GUI** (`scripts/tvc_traj_opt_gui.py`), **Method 4**, **Method 5**, and **Method 7** are **combo indices 3, 4, and 6** respectively. All three use the same **plant** as §1–§6; they differ in **time parametrization** and **cost**.

| GUI label | Combo index | Acados driver | Horizon in solver time |
|-----------|-------------|---------------|-------------------------|
| Method 4: Acados (native constraints) | 3 | `build_acados_ocp` | Physical $T_f$ = segment duration; `ocp.solver_options.tf = T_f` |
| Method 5: Acados min-time | 4 | `build_acados_ocp_min_time` + `export_tvc_ode_model_pseudotime` | Normalized $\tau \in [0,1]$; `tf = 1`; extra state $T_{\mathrm{seg}}$ = physical segment length |
| Method 7: Acados free $t_f$ + EXTERNAL | 6 | `build_acados_ocp_free_tf_external` + same pseudotime model as Method 5 | Same as Method 5 ($\tau$, state $t_f$ at end of $x$) |

**Path constraints** (velocity / Euler / $\|\boldsymbol{\omega}\|$ box via `con_h`) and **control bounds** are shared in spirit across these builders; exact matrices follow `bounds` in code.

---

### 11.1 Method 4 — fixed-time nonlinear least squares (`build_acados_ocp`)

**Cost type:** `NONLINEAR_LS` on each shooting stage and at the terminal stage. **Hessian:** Gauss–Newton (`hessian_approx = GAUSS_NEWTON`).

Let $Q = \mathrm{diag}(w_p\mathbb{I}_3,\, w_\phi,w_\theta,w_\psi,\, w_v\mathbb{I}_3,\, w_\omega\mathbb{I}_3)$ with GUI weights `p`, `v`, `roll`/`pitch`/`yaw` (defaults tied to `R`), `w`. Let $R_u = \mathrm{diag}(w_{u,1},\ldots,w_{u,4})$ be built from weight `u` (scalar or 4-vector) and characteristic scales $\sigma_i$ from control bounds (`_per_channel_ls_weights` / `_u_sigma_from_weights_or_bounds`). Segment goal state (first 12 components) is $x_g$; hover-like reference for controls is $u_{\mathrm{ref}}$ (typically $[0,0,m g,0]^\top$).

**Baseline ($n_x = 12$):** running output $y_k = [x_k^\top,\, u_k^\top]^\top$, reference $y^{\mathrm{ref}} = [x_g^\top,\, u_{\mathrm{ref}}^\top]^\top$,

$$
J_{\mathrm{run}} = \frac{1}{2} \sum_{k=0}^{N-1} \bigl\| y_k - y^{\mathrm{ref}} \bigr\|_{W}^2, \qquad
W = \begin{bmatrix} Q & 0 \\ 0 & R_u \end{bmatrix}.
$$

**Terminal:** $y^e = x_{N}^{(\mathrm{phys})}$, $y^{\mathrm{ref},e} = x_g^{(\mathrm{phys})}$,

$$
J_e = \frac{1}{2} \bigl\| x_N^{(\mathrm{phys})} - x_g^{(\mathrm{phys})} \bigr\|_{Q_e}^2, \qquad Q_e = k_{\mathrm{term}}\, Q,
$$

with $k_{\mathrm{term}} =$ `terminal_cost_multiplier` (GUI).

**Actuator dynamics ($n_x = 16$):** $x = [x_{\mathrm{phys}}^\top,\, u_{\mathrm{act}}^\top]^\top$. Running $y_k = [x_k^\top,\, u_k^\top]^\top$ with block weights on $(x_{\mathrm{phys}}, u_{\mathrm{act}})$ from $Q$ and $w_u$ on the $u_{\mathrm{act}}$ block, plus $R_u$ on $u_k$ (command). **Control-rate mode** (mutually exclusive with actuators): $y_k$ stacks $x$, $u_k$, and $(u_k - u_{\mathrm{prev}})$ with diagonal weight `du` on the increment block.

Optional **`terminal_constraint`:** hard equality on terminal **position** $p_N = p_g$ (`con_h_expr_e`).

---

### 11.2 Method 5 — minimum time with pseudo-time state $T_{\mathrm{seg}}$ (`build_acados_ocp_min_time`)

**Dynamics:** `export_tvc_ode_model_pseudotime`: state $\tilde{x} = [x_{\mathrm{phys}}^\top,\, (\mathrm{aug})^\top,\, T_{\mathrm{seg}}]^\top$ (aug = $u_{\mathrm{act}}$ or $u_{\mathrm{prev}}$ or none); $\mathrm{d}x_{\mathrm{phys}}/\mathrm{d}\tau = T_{\mathrm{seg}}\, f_{\mathrm{phys}}$, $\mathrm{d}T_{\mathrm{seg}}/\mathrm{d}\tau = 0$. Solver uses **`ocp.solver_options.tf = 1`** (normalized horizon).

**Running cost:** same **NONLINEAR_LS** structure as Method 4, but built on **$\tilde{x}$ without the last component** in the output expression (i.e. tracking on physical + aug channels and $u$ as in §11.1). References $y^{\mathrm{ref}}$ are aligned with the segment endpoint $x_g$ (and optional `schedule_ref` time interpolation in the solver loop).

**Terminal cost (extra Mayer on duration):** output

$$
y^e = \begin{bmatrix} x_N^{(\mathrm{phys})} \\ T_{\mathrm{seg},N} \end{bmatrix}, \qquad
y^{\mathrm{ref},e} = \begin{bmatrix} x_g^{(\mathrm{phys})} \\ 0 \end{bmatrix},
$$

$$
W_e = \begin{bmatrix} Q_e & 0 \\ 0 & w_{\mathrm{time}} \end{bmatrix}, \qquad
Q_e = k_{\mathrm{term}}\, Q,
$$

with $w_{\mathrm{time}} =$ `min_time_weight`. Thus the LS term penalizes $T_{\mathrm{seg}}^2$ toward **zero**, while **box constraints** enforce $T_{\min} \le T_{\mathrm{seg}} \le T_{\max}$ per segment ($T_{\max}$ from waypoint time gap $\times$ `min_time_T_max_scale`, $T_{\min}$ from GUI `min_time_T_min` and feasibility caps in code). The optimizer typically drives $T_{\mathrm{seg}}$ to the **lower bound** when the terminal state can be reached that fast.

**Hessian:** Gauss–Newton.

---

### 11.3 Method 7 — free $t_f$ with EXTERNAL cost (`build_acados_ocp_free_tf_external`)

**Dynamics:** identical pseudo-time model to Method 5 (state $t_f$ as last component, $\mathrm{d}t_f/\mathrm{d}\tau = 0$, $\mathrm{d}x/\mathrm{d}\tau = t_f\, f_{\mathrm{phys}}$).

**Cost type:** `EXTERNAL` on stages and terminal. **Hessian:** `EXACT` (appropriate for general nonlinear external costs).

Let $t_f$ denote the duration state, $N$ the horizon length, and $(\theta_{p,c},\theta_{r,c},T_c,\tau_{y,c})$ the **commanded** controls. Weights: `free_tf_w_T`, `free_tf_w_tvc`, `free_tf_w_tau_yaw`, `free_tf_w_terminal_time`, optional Mayer via `terminal_cost_multiplier` and running-state weights as in code.

**Running (stage $k$):** with thrust penalty on **actual** thrust state $T_{\mathrm{act}}$ when actuators are on (`x_{14}$`), else on $T_c$,

$$
\ell_{\mathrm{run}} =
\frac{t_f}{N}\Bigl(
w_T\, T_{\mathrm{mag}}^2
+ w_{\mathrm{tvc}}\bigl(\theta_{p,c}^2 + \theta_{r,c}^2\bigr)
+ w_{\tau}\, \tau_{y,c}^2
\Bigr),
$$

matching a discrete approximation to $\int_0^{t_f} \cdots\,\mathrm{d}t$ over $\tau \in [0,1]$.

**Terminal:**

$$
J_e = w_{\mathrm{tf,term}}\, t_f
\;+\;
\sum_{i=1}^{12} (Q_e)_{ii}\,\bigl(x^{(\mathrm{phys})}_{i,N} - (x_g)_i\bigr)^2
$$

if `free_tf_include_state_terminal` is true; otherwise only $w_{\mathrm{tf,term}}\, t_f$. Here $(Q_e)_{ii}$ uses the same diagonal pattern as $Q$ scaled by $k_{\mathrm{term}}$ (§11.1).

**Duration bounds:** same style as Method 5 (`min_time_T_min`, `min_time_T_max_scale` from the GUI).

---

### 11.4 Implementation pointers

| Item | Location |
|------|----------|
| Fixed-time OCP | `build_acados_ocp` |
| Min-time OCP | `build_acados_ocp_min_time` |
| Free-$t_f$ EXTERNAL OCP | `build_acados_ocp_free_tf_external` |
| Pseudo-time ODE | `export_tvc_ode_model_pseudotime` |
| Segment loop / meta (`plot_dt`, `optimal_segment_times`) | `solve_with_acados_waypoints_min_time`, `solve_with_acados_waypoints_free_tf` |

**Method 6** (Spannagl-style FFTF) in the GUI uses a **separate** CasADi/Acados path in `scripts/tvc_traj_opt_acados_min_time.py`; it is **not** covered by the three builders above.
