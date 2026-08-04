# Monthly Work Summary — Actuator Dynamics & Online Replanning

**Purpose:** Slide-ready notes (English) for the monthly report.  
**Scope:** (1) Numerical simulation with actuator dynamics; (2) Online trajectory replanning on Jetson.

---

## Suggested slide outline

1. Motivation & simulation setup  
2. Thrust uncertainty / quantization (error tolerance)  
3. Thrust channel: first-order model (local closed loop) & bandwidth sweep (1 → 0.2 Hz)  
4. Actuator bandwidth vs. stability margins — **rate / gimbal example** (1 / 3 / 5 Hz)  
5. Retuning under slow actuators (1 Hz) — what you gain and what you lose  
6. Takeaway: faster actuators enable higher control bandwidth  
7. Why thrust BW can be lower than TVC/attitude BW (plant order + cascade)  
8. Online trajectory replanning on Jetson  
9. Summary & next steps  

---

# Part 1 — Simulation with Actuator Dynamics

## 1.1 Motivation

Closed-loop tracking simulations previously assumed **ideal actuators** (command = plant input).  
For the real TVC rocket we must account for:

1. **Thrust uncertainty** — discrete thrust levels and/or command ≠ actual thrust  
2. **Finite actuator bandwidth** — first-order lag on gimbal / thrust / yaw torque  

Both are now available in the numerical simulator (`ActuatorDynamics`) and in the **Stability margins** Bode tool (open-loop \(L(j\omega)\), phase margin with vs. without lag).

---

## 1.2 Thrust uncertainty — quantization & error tolerance

### What we modeled

- **Thrust quantization:** round commanded thrust to the nearest step (e.g. 0.5 N proxy, 10 N / 50 N real-platform style).  
- Separate from **calibration bias / scale error** (persistent DC mismatch).

### Main findings (from quantization study)

| Topic | Result |
|-------|--------|
| Large step size (up to **50 N**) | Closed-loop tracking remains **stable** on gentle trajectories |
| Effect of quantization | Bounded, roughly zero-mean nonlinearity → slight RMSE ↑, small limit cycling |
| Thrust **inaccuracy** (bias/scale) | Much more dangerous: breaks \(T_\mathrm{hover}\approx mg\), steady altitude error |
| Practical priority | **Calibrate thrust** first; quantization alone is not the primary flight risk for tested cases |

**Slide message:**  
> Discrete thrust levels are tolerable for the tested trajectories; **thrust calibration error** is the real uncertainty to worry about.

*(Insert: thrust staircase plot + position tracking overlay with / without quantization.)*

### 1.2.1 Thrust scale/bias mismatch → altitude static error; I fixes it

In the simulator we can model commanded-vs-plant thrust mismatch as a persistent calibration error:

\[
T_\mathrm{plant} = \mathrm{scale}\cdot T_\mathrm{lag} + \mathrm{bias}.
\]

For small tilt near hover, vertical dynamics are approximately:

\[
m\ddot{z} \approx T_\mathrm{plant} - mg,
\]
so a nonzero bias (and/or a scale error that breaks the assumed hover-thrust mapping) behaves like a **constant DC disturbance** in the vertical channel.

In the current controller (PX4-style cascade implemented in `PX4CascadeTracker`), the Z path is:

* Position outer loop: **P only** (`Kp_pos_z`)
* Velocity Z loop: **PID** with an integral term (`Ki_vel_z`)

Therefore:

* If the velocity-loop integral is weak/disabled (`Ki_vel_z` small or 0), the controller cannot fully reject a constant thrust disturbance, and the system settles at a shifted equilibrium → **static altitude/position error** appears.
* Increasing `Ki_vel_z` adds integral action (vertical loop becomes effectively “type-1” for the DC disturbance), allowing the integrator to accumulate the velocity error until the commanded thrust is adjusted to the mismatch-compensated value → **steady-state error is driven back to ~0** (matching your observation).

**Takeaway:** thrust scale/bias mismatch primarily shows up as a **steady-state vertical equilibrium error**; an integrator in the velocity (height) loop is the correct mechanism to remove it (at the cost of slower convergence and potential windup if saturations occur).

---

## 1.3 Thrust dynamics — first-order simplification & bandwidth sweep

### Why a first-order model is used here

A small liquid rocket engine typically has its **own local closed-loop control** (valve / chamber-pressure / thrust regulation).  
From the **vehicle-level** cascade (position → velocity → thrust command), that inner loop compresses much of the fast valve–plumbing–combustion physics into an **equivalent low-pass response**.

Therefore, in this study we **do not** model the full engine physics. We approximate the thrust channel seen by the flight controller as a **first-order lag**:

\[
G_T(s)=\frac{1}{\tau_T s+1},\qquad
f_{c,T}=\frac{1}{2\pi\tau_T}.
\]

This is a standard control-oriented reduction: adequate when the vehicle vertical loops are much slower than the engine’s local loop, and when step responses show little overshoot.

### Closed-loop tracking sweep (thrust bandwidth)

With the vehicle controller gains fixed, we enabled thrust lag only and swept bandwidth:

| Thrust \(f_{c,T}\) | \(\tau_T\) (approx.) | Closed-loop tracking observation |
|--------------------|----------------------|----------------------------------|
| **1.0 Hz** | ≈ 0.159 s | Acceptable — no clear oscillation |
| **0.5 Hz** | ≈ 0.318 s | Still acceptable |
| **0.3 Hz** | ≈ 0.531 s | Degraded / sluggish, not yet strongly oscillatory |
| **0.2 Hz** | ≈ 0.796 s | **Clear control oscillation** appears |

**Slide message:**  
> Because the engine already has a local closed loop, vehicle-level thrust is modeled as **first-order**.  
> Under our cascade gains, vertical tracking stays usable down through ~0.3 Hz; **obvious oscillation shows up at ~0.2 Hz**.  
> This supports the claim that **thrust can tolerate much lower bandwidth than the TVC/attitude path** (cf. §1.8).

*(Insert: altitude / \(v_z\) / thrust cmd-vs-act time histories for 1.0, 0.5, 0.3, 0.2 Hz.)*

---

## 1.4 Actuator bandwidth — first-order lag model (all channels)

Each channel (gimbal, thrust, yaw torque) uses the same first-order form in simulation:

\[
G_\mathrm{act}(s)=\frac{1}{\tau s+1},\qquad
f_c=\frac{1}{2\pi\tau}\quad\text{(−3 dB bandwidth)}
\]

| \(f_c\) | \(\tau\) | Typical use in this work |
|---------|----------|---------------------------|
| 5 Hz | ≈ 0.032 s | Fast gimbal reference |
| 3 Hz | ≈ 0.053 s | Mid gimbal |
| 1 Hz | ≈ 0.159 s | Slow gimbal / nominal thrust start of sweep |
| 0.5–0.2 Hz | ≈ 0.32–0.80 s | **Thrust-only stress test** (see §1.3) |

Lag adds **extra phase lag** near the loop gain crossover → **phase margin (PM) drops** (especially critical on the **rate / gimbal** channel).

---

## 1.5 Rate loop example — PM loss vs. actuator bandwidth

### Ideal rate open-loop (after plant / allocation cancellation)

For P-only rate control with matched \(\omega_z^2\) allocation:

\[
L_\mathrm{rate}(s)=\frac{K_p}{s}
\quad\Rightarrow\quad
\omega_{\mathrm{gc}}=K_p,\quad
\mathrm{PM}_0\approx 90^\circ
\]

With actuator lag:

\[
L_\mathrm{rate}(s)=\frac{K_p}{s(\tau s+1)}
\]

### Numerical example (\(K_p=10\) → ideal \(f_{\mathrm{gc}}\approx 1.59\,\mathrm{Hz}\))

| Gimbal \(f_c\) | \(\tau\) | PM without lag | PM with lag | \(\Delta\)PM | \(f_{\mathrm{gc}}\) (no lag → with lag) |
|----------------|----------|----------------|-------------|--------------|----------------------------------------|
| **1 Hz** | 0.159 s | 90° | **≈ 43°** | **−47°** | 1.59 → 1.08 Hz |
| **3 Hz** | 0.053 s | 90° | **≈ 64°** | −26° | 1.59 → 1.44 Hz |
| **5 Hz** | 0.032 s | 90° | **≈ 73°** | −17° | 1.59 → 1.52 Hz |

*(Insert: Bode mag/phase, solid = no lag, dashed = with \(\tau\); mark PM at gain crossover.)*

**Slide message:**  
> Same controller gains, slower actuator → much smaller phase margin.  
> At **1 Hz**, PM falls from ~90° to ~43° — barely acceptable / fragile.

---

## 1.6 If the actuator is only ~1 Hz — retune gains to recover PM

To restore adequate PM under \(f_c=1\,\mathrm{Hz}\), **reduce rate \(K_p\)** (move \(\omega_{\mathrm{gc}}\) down, away from the lag corner).

| Rate \(K_p\) | PM (no lag → with 1 Hz lag) | \(f_{\mathrm{gc}}\) (no lag → with lag) |
|--------------|-----------------------------|----------------------------------------|
| 10 | 90° → 43° | 1.59 → 1.08 Hz |
| 6 | 90° → 53° | 0.95 → 0.76 Hz |
| 4 | 90° → 61° | 0.64 → 0.56 Hz |
| 3 | 90° → 66° | 0.48 → 0.44 Hz |

### Consequences of lowering gains

| Upside | Downside |
|--------|----------|
| Recover PM / robustness | Slower rate response |
| Less oscillation with lag | Outer loops (att → vel → pos) must also slow down (cascade separation) |
| Safer under slow hardware | Weaker disturbance rejection; larger tracking lag / overshoot on aggressive refs |

**Slide message:**  
> Slow actuators force **conservative gains**. The whole cascade must run slower — system “speed” is limited by the actuator, not only by the PID numbers.

---

## 1.7 Cascade implication — we want actuators as fast as possible

Design rule of thumb for nested loops:

\[
\omega_{\mathrm{actuator}}
\;\gg\;
\omega_{\mathrm{rate}}
\;\gg\;
\omega_{\mathrm{att}}
\;\gg\;
\omega_{\mathrm{vel}}
\;\gg\;
\omega_{\mathrm{pos}}
\]

Typical adjacent-loop separation: **\(3\times\)–\(5\times\)** in \(\omega_{\mathrm{gc}}\).

If gimbal \(f_c\) is low:

1. Rate PM collapses (unless \(K_p\) is reduced)  
2. Attitude / velocity / position bandwidths must drop accordingly  
3. Horizontal tracking becomes sluggish; easy to get overshoot if outer \(K_p\) stays aggressive while inner loops are slow  

**One-line takeaway for slides:**  
> **Faster actuator dynamics → higher allowable control bandwidth → better tracking agility.**  
> Low \(f_c\) does not only add lag; it **caps the entire cascade**.

---

## 1.8 Why thrust can run at lower bandwidth than TVC / attitude

### Plant order (actuator → position)

| Channel | Open-loop plant (small-signal) | Order |
|---------|--------------------------------|-------|
| **Thrust → altitude \(z\)** | \(\dfrac{Z}{\Delta T}\approx\dfrac{1}{m s^2}\) | **2nd order** (vel + pos) |
| **TVC angle \(\delta\) → horizontal \(x\)** | \(\dfrac{X}{\delta}\approx\dfrac{g\,\omega_z^2}{s^4}\) | **4th order** (rate + att + vel + pos) |

### Controller structure

| | Horizontal (via TVC) | Vertical (via thrust) |
|--|----------------------|------------------------|
| Cascade | Rate → Att → Vel\(_{xy}\) → Pos\(_{xy}\) | Vel\(_z\) → Pos\(_z\) (no attitude inner loop) |
| Actuator role | Inside **innermost** rate loop | Near **outer** vertical loops |
| Typical outer \(\omega_{\mathrm{gc}}\) | Needs fast tilt response | Altitude loops are intentionally slow |
| Bandwidth demand on hardware | **Gimbal \(f_c\) must be high** | **Thrust \(f_c\) can be lower** |

### Intuition for slides

- Each integrator costs ~90° phase; a 4th-order TVC path needs **fast inner loops** (and a fast gimbal) so outer loops still see \(T_\mathrm{inner}\approx 1\).  
- Vertical path is only 2nd order and usually low-bandwidth → a slower thrust valve / engine is often acceptable.  
- This matches practice: e.g. gimbal \(f_c\sim 2\)–\(5\,\mathrm{Hz}\) vs. thrust \(f_c\sim 1\,\mathrm{Hz}\) can still work **if** vertical gains stay modest.

**Caveat:** Raising Vel\(_z\) / Pos\(_z\) aggressively will also demand higher thrust bandwidth — the structural relief is not unlimited.

**Link to §1.3:** The closed-loop sweep (usable to ~0.3 Hz, oscillatory at **0.2 Hz**) is consistent with a low-bandwidth vertical path: thrust \(f_c\) can sit far below gimbal \(f_c\) before the vehicle-level controller breaks.

---

## 1.9 Part 1 — summary bullets (copy onto closing slide)

1. Simulate with **thrust quantization + first-order actuator lag**.  
2. Quantization: tolerable on tested trajectories; **calibration bias** is the critical uncertainty.  
3. **Thrust:** engine local closed loop → model as **first-order**; sweep \(f_{c,T}=1 / 0.5 / 0.3 / 0.2\,\mathrm{Hz}\); **clear oscillation at 0.2 Hz**.  
4. Rate / gimbal: slower \(f_c\) (1 vs 3 vs 5 Hz) → large **PM loss** at fixed gains.  
5. At slow gimbal BW, recovering PM requires **lower \(K_p\)** → whole cascade slows down.  
6. Prefer **faster actuators** to unlock higher cascade bandwidth.  
7. Thrust BW requirement is **structurally lower** than TVC/attitude BW (2nd-order vertical vs 4th-order horizontal cascade).

---

# Part 2 — Online Trajectory Replanning on Jetson

## 2.1 Goal

Move from **offline** trajectory optimization (desktop GUI) to **onboard / edge** replanning:

- Replan from the **current state** toward upcoming waypoint(s)  
- Run on **Jetson** (Orin-class) alongside PX4 / SITL  
- Support safety / mission updates without a full ground-station solve each time  

## 2.2 System picture (slide diagram)

```
[Sensors / PX4 EKF] → current state
        ↓
[Online planner node]  Acados (or selected solver), warm solver
        ↓  planned path @ replan_rate_hz
[Tracker / PX4] ← reference
        ↓
[Actuators]  (+ optional lag / thrust quantization in sim)
```

GUI / SITL integration highlights:

- Checkbox: **Online safety planner**  
- Replan rate setting (e.g. 0.2–50 Hz target; report **actual** Hz on device)  
- RViz visualization: `/online_planner/planned_path`  
- Same physics / bounds / cost weights as offline planning where applicable  

## 2.3 Why Jetson matters

| Regime | What runs | Requirement |
|--------|-----------|-------------|
| **Offline / boot** | Build or first instantiate Acados solver | Seconds OK (warm-up once) |
| **Online replan** | `set references` + `solve` only | Must fit desired replan rate |

Desktop vs Jetson:

- Acados **solve** is far cheaper than full NLP setup each time  
- On Orin: report solve time / ms per iteration and achievable replan Hz (fill measured numbers)  
- Critical engineering point: **keep a long-lived solver** — do not rebuild every replan  

*(Insert: Ultra 7 vs Jetson Orin bar chart — setup time vs solve time; achieved replan Hz.)*

## 2.4 What to show in the talk

1. **Architecture** — state in → replan → path out → tracker  
2. **Timing** — warm-up once + cyclic solve (timeline cartoon)  
3. **Demo** — SITL or hardware-in-loop: waypoint change / recovery, RViz path updates  
4. **Numbers** — target vs actual replan rate; solve time on Jetson; tracking RMSE during replans  

## 2.5 Part 2 — summary bullets

1. Online replanning closes the loop between **current state** and **local optimal path**.  
2. Jetson is feasible for **warm Acados** online rates; cold setup must stay off the critical path.  
3. Integration path: ROS online_planner + GUI/SITL controls + RViz.  
4. Next: harden timing under load, flight demo with optimization-mode references, disturbance cases.

---

# Combined closing slide

| Theme | Key result |
|-------|------------|
| Thrust uncertainty | Quantization OK for gentle trajs; **calibrate** thrust |
| Thrust dynamics model | Engine local CL → **1st-order** \(G_T=1/(\tau_T s+1)\) |
| Thrust BW sweep | 1 / 0.5 / 0.3 Hz OK; **oscillation at 0.2 Hz** |
| Gimbal / rate BW | 1 Hz gimbal destroys rate PM unless gains drop |
| Retuning | Lower gains recover PM but **slow the whole cascade** |
| Design preference | **Faster actuators → higher control bandwidth** |
| Thrust vs TVC BW | Vertical 2nd-order vs horizontal 4th-order cascade |
| Jetson replan | Warm solver online; measure Orin Hz and integrate |

---

# Appendix — formulas for speaker notes

**Rate loop with lag**

\[
L(s)=\frac{K_p}{s(\tau s+1)},\quad
f_c=\frac{1}{2\pi\tau},\quad
\mathrm{PM}=180^\circ+\angle L(j\omega_{\mathrm{gc}})
\]

**Horizontal plant**

\[
\frac{X}{\delta}\sim\frac{g\,\omega_z^2}{s^4},\quad
\omega_z^2=\frac{\ell\,mg}{I}
\]

**Vertical plant**

\[
\frac{Z}{\Delta T}\sim\frac{1}{m s^2}
\]

**Cascade separation (rule of thumb)**

\[
\omega_{\mathrm{gc,inner}}\approx (3\text{–}5)\,\omega_{\mathrm{gc,outer}}
\]

---

*Document generated for monthly slides. Numerical PM table uses representative \(K_p=10\) rate P-control and continuous Bode margins from `controllers/stability_margins.py` (same model as the GUI Stability margins tab). Replace Jetson timing placeholders with measured Orin logs before the talk.*
