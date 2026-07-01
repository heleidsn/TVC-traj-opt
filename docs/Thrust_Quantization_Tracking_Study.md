# Numerical Simulation Study: Thrust Quantization vs. Tracking Performance

**Date:** June 2025  
**Platform:** Real TVC rocket (20 kg, thrust range 150–250 N)  
**Controller:** PX4-style cascaded position / velocity / attitude / rate tracker  
**Simulation:** `dt = 1 ms`, control rate `20 Hz`

---

## 1. Motivation

For the real rocket platform, thrust actuation may be **coarse** (discrete thrust levels) and **inaccurate** (commanded thrust ≠ actual thrust). Before flight tests, we used the numerical closed-loop tracking simulator in `TVC-traj-opt` to assess how large a thrust quantization step can be before the controller fails.

The initial concern was that discrete thrust commands (e.g. only 150 N, 200 N, 250 N) might destabilize closed-loop tracking, especially for altitude control.

---

## 2. Test Setup

| Parameter | Value |
|-----------|-------|
| Plant model | Nonlinear TVC rigid body (same as trajectory optimization) |
| Controller | `PX4CascadeTracker` |
| Simulation step | 1 ms |
| Control update rate | 20 Hz (50 ms hold between updates) |
| Thrust quantization | Enabled; step size up to **50 N** (matching the three real thrust levels: 150 / 200 / 250 N) |
| Vehicle mass | 20 kg → hover thrust ≈ 196 N |
| Test trajectories | Traj 1: (0, 0, 0) → (1, 0, 0); Traj 2: (0, 0, 0) → (1, 0, 1) |

Actuator model: nearest-neighbor rounding of absolute thrust (`quantize_thrust` in `controllers/actuator_dynamics.py`).

### 2.1 Test matrix (Cases 1–4)

| Case | Trajectory | Thrust quantization | Description |
|------|------------|---------------------|-------------|
| **Case 1** | Traj 1: (0,0,0) → (1,0,0) | None (continuous) | Baseline — no thrust discretization |
| **Case 2** | Traj 1: (0,0,0) → (1,0,0) | 10 N step | Moderate quantization |
| **Case 3** | Traj 1: (0,0,0) → (1,0,0) | 50 N step | Matches real platform (150 / 200 / 250 N only) |
| **Case 4** | Traj 2: (0,0,0) → (1,0,1) | 50 N step | Horizontal + 1 m climb under worst-case thrust resolution |

**Observed trend across Cases 1 → 3 (Traj 1):** tracking remains stable; position error increases slightly as step size grows; thrust command plots show clear staircase behaviour at 10 N (Case 2) and 50 N (Case 3), while state trajectories stay close to the reference.

**Case 4 (Traj 2, 50 N):** altitude tracking remains acceptable; x-direction lag is visible and is attributed to non-minimum-phase tilt dynamics rather than thrust quantization.

---

## 3. Key Findings

### 3.1 Large thrust quantization steps do **not** cause instability

Even with a **50 N** quantization step—so that the only available thrust commands are 150 N, 200 N, and 250 N—the closed-loop system **remains stable**. Tracking performance degrades slightly (small oscillations / limit cycling near thrust level boundaries), but the controller does not fail.

This holds for:

- **Horizontal-only move:** (0, 0, 0) → (1, 0, 0)
- **Combined horizontal + altitude move:** (0, 0, 0) → (1, 0, 1)

Altitude tracking remains acceptable in both cases.

### 3.2 Quantization vs. thrust **inaccuracy** are fundamentally different

| Effect | Nature | Controller compensation | Impact |
|--------|--------|-------------------------|--------|
| **Thrust quantization (steps)** | Bounded, approximately zero-mean nonlinearity | Time-averaging, limit cycling between adjacent levels | Slight RMSE increase, small oscillations |
| **Thrust inaccuracy (bias / scale error)** | Persistent DC disturbance | Requires calibration or integral action | Steady-state altitude/position error, broken hover assumption |

**Conclusion:** The dominant flight risk is **thrust calibration** (commanded vs. actual thrust), not the absence of intermediate thrust levels. A 10% thrust scale error is far more damaging than a 50 N quantization step.

### 3.3 Altitude control is inherently **low bandwidth**

Vertical dynamics are approximately:

\[
m \ddot{z} = T \cos\theta \cos\phi - mg
\]

For gentle trajectories with small tilt angles, altitude behaves like a **slow, heavily filtered** channel. A 50 N step corresponds to roughly ±25 N worst-case force error, or ~±1.25 m/s² (~0.13g) acceleration error. At 20 Hz, the cascaded PID can absorb this over several control cycles.

The PX4 cascade (position P → velocity PID → thrust delta) further low-pass-filters thrust commands. Moderate gains are required for stability; overly aggressive gains interact with attitude coupling and are not recommended.

### 3.4 Horizontal (x) tracking lag is **not** caused by thrust quantization

For **Case 4** (Traj 2, 50 N), altitude remains well controlled, but **noticeable lag appears in the x direction**.

This is expected from **non-minimum-phase (NMP)** behavior of thrust-vectoring rockets:

- To accelerate in +x, the vehicle must pitch forward (+pitch).
- As pitch increases from zero, the **vertical thrust component decreases before** the horizontal component becomes significant (\(a_x \propto T\sin\theta\), \(a_z \propto T\cos\theta - g\)).
- The initial response can temporarily work against the desired horizontal motion; position tracking therefore **lags** the reference.

Additional delay comes from:

- Cascaded control structure (position → velocity → tilt → attitude → rate → gimbal)
- 20 Hz control update rate
- Coupling during climb: pitching for x reduces \(T\cos\theta\), forcing the vertical loop to increase thrust before aggressive pitch is “affordable”

Horizontal motion is actuated primarily through **gimbal / tilt** (continuous), not through thrust quantization. Improving x tracking should focus on feedforward, trajectory shaping, and gain tuning—not thrust step size.

---

## 4. Physical Interpretation

**Why quantization is tolerable**

1. The real platform already has only three thrust levels; the simulation confirms this is viable for gentle trajectories.
2. Quantization acts like **bounded noise**; time-averaged thrust can approximate a continuous command (similar to PWM).
3. Height control bandwidth is low; quantization energy sits in a frequency range the vertical loop can reject.

**Why thrust inaccuracy matters more**

1. Wrong hover thrust breaks the gravity-compensation assumption (\(T_\mathrm{hover} \approx mg\)).
2. Feedforward and trajectory planning assume a known thrust–throttle mapping.
3. Bias errors are DC disturbances that filtering cannot remove without calibration or integral control.

**Why x lags (NMP)**

1. Tilt must build up before horizontal velocity accumulates.
2. Climb trajectories increase vertical–horizontal coupling.
3. Gain increases cannot eliminate NMP phase lag without risking oscillation.

---

## 5. Scenarios Where Quantization May Still Matter

The current tests are **favorable** (gentle trajectories, small tilt). Quantization could become limiting for:

- Aggressive vertical maneuvers (large \(v_z\), fast descent)
- Precision landing (final centimeters of altitude)
- Large-tilt horizontal maneuvers (strong \(T\cos\theta\) loss)
- Combined high gains + quantization (limit-cycle oscillations)

These should be tested separately before relying on the current conclusions for all flight phases.

---

## 6. Recommendations for the Real Platform

### High priority

1. **Thrust calibration** — bench or tethered tests; map commanded level → actual thrust (include hysteresis).
2. **Online hover thrust estimate** — do not assume \(T_\mathrm{hover} = mg\) exactly.
3. **Phase-specific validation** — vertical step, landing, and fast horizontal moves in addition to gentle point-to-point tracks.

### Lower priority (quantization)

1. Thrust step size is **not** the primary bottleneck for the tested trajectories.
2. Optional: small integral gain on vertical velocity to handle calibration bias (tune carefully with quantization).
3. Optional: hysteresis on thrust level switching to reduce chattering between 150 ↔ 200 ↔ 250 N.

### Horizontal tracking improvement

1. Use acceleration / velocity feedforward from the planned trajectory.
2. Shape trajectories to avoid simultaneous aggressive x and z commands.
3. Tune horizontal velocity loop (including optional \(K_i\)) with awareness of NMP limits.

---

## 7. Summary

> **Discrete thrust levels (50 N steps, three levels total) do not destabilize closed-loop tracking for gentle point-to-point trajectories on the 20 kg real platform model (Cases 1–4). Altitude control tolerates coarse thrust because it is low bandwidth. The main flight risk is thrust calibration error, not quantization. Horizontal tracking lag in Case 4 is dominated by non-minimum-phase tilt dynamics and cascade delay, not by thrust discretization.**

---

## 8. Related Code & Configuration

- Controller: `TVC-traj-opt/controllers/px4_cascade.py`
- Actuator quantization: `TVC-traj-opt/controllers/actuator_dynamics.py`
- Platform thrust limits: `TVC-traj-opt/scripts/tvc_rocket_platforms.py` (`REAL_THRUST_MIN_N = 150`, `REAL_THRUST_MAX_N = 250`)
- Example tracking params: `TVC-traj-opt/controllers/tracking_params.json` (`thrust_resolution_N: 50.0`)

---

## 9. Supporting Attachments (Cases 1–4)

Each case includes a **3D tracking animation** (GIF) and a **time-series figure** (PNG) with state, tracking error, and actuator signals.

| Case | Trajectory | Quantization | Animation (GIF) | Data plots (PNG) |
|------|------------|--------------|-----------------|------------------|
| **Case 1** | Traj 1 | Continuous | `tracking_anim-continue.gif` | `tvc_traj_opt_figure-continue.png` |
| **Case 2** | Traj 1 | 10 N | `tracking_anim-10N.gif` | `tvc_traj_opt_figure-10N.png` |
| **Case 3** | Traj 1 | 50 N | `tracking_anim-50N.gif` | `tvc_traj_opt_figure-50N.png` |
| **Case 4** | Traj 2 | 50 N | `tracking_anim-50N-traj2.gif` | `tvc_traj_opt_figure-50N-traj2.png` |

**How to read the figures**

- **GIFs:** 3D view of the executed trajectory vs. reference in the simulation environment.
- **PNGs:** Multi-panel time histories — position/velocity tracking, attitude, and control inputs. The thrust channel shows a **staircase profile** in Cases 2–4; Case 1 (continuous) shows a smooth thrust trace. State trajectories remain close to the reference even in Case 3 and Case 4.

Raw run outputs are stored under folder `26032026/` (date-stamped export from the GUI).

---

## Appendix: Suggested Follow-Up Simulations

1. Thrust **scale error** (e.g. ±10%) vs. quantization only — compare altitude and position RMSE.
2. Pure vertical 1 m step with 50 N quantization.
3. Quantization + first-order thrust lag (`tau_thrust`) combined.
4. With / without trajectory acceleration feedforward — quantify x-direction lag reduction.
