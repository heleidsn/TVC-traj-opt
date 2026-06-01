#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TVC Rocket Trajectory Optimization using Acados (Method 4 / Method 5)

Acados handles constraints natively (control bounds, state path constraints),
often with better convergence than penalty-based approaches.

Implementation is split into ``tvc_traj_opt_acados_dynamics.py`` (plant),
``tvc_traj_opt_acados_cost.py`` (OCP costs), and ``tvc_traj_opt_acados_state.py`` (state maps).

**Method 5 (minimum time):** set ``weights['acados_objective'] = 'min_time'`` (or use GUI Method 5).
Each leg minimizes physical segment duration with an upper bound from waypoint times
(see ``min_time_T_max_scale``, ``min_time_T_min``). Solvers return a 5-tuple
``(xs, us, loggers, us_actual, meta)``; fixed-time Mode 4 returns ``meta == {}``.

Usage:
    python -u tvc_traj_opt_acados.py
    python -u tvc_traj_opt_acados.py --method 5   # min-time (Method 5)
    python -u tvc_traj_opt_acados.py --method 7   # free-tf EXTERNAL (Method 7)

Requires: acados, casadi (pip install casadi; acados from source)

Before running, ensure Acados libs are in LD_LIBRARY_PATH:
    export ACADOS_SOURCE_DIR=/path/to/acados
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ACADOS_SOURCE_DIR/lib
"""

import os
import sys
import gc
import time
import numpy as np
from pathlib import Path

# Setup Acados environment before import (fixes libqpOASES_e.so / libhpipm.so not found).
#
# IMPORTANT: setting ``LD_LIBRARY_PATH`` via ``os.environ`` from Python is too
# late: the dynamic linker caches the search path at process startup, so a
# later ``dlopen("libqpOASES_e.so")`` will still fail. We therefore preload
# the acados libraries with ``ctypes.CDLL(<absolute path>, RTLD_GLOBAL)`` —
# their symbols become globally available and basename ``dlopen`` calls
# succeed without needing ``LD_LIBRARY_PATH``.
def _setup_acados_env():
    """Find acados, set env vars and preload its shared libraries."""
    import ctypes

    acados_root = os.environ.get("ACADOS_SOURCE_DIR")
    if not acados_root:
        try:
            import acados_template
            pkg_path = Path(acados_template.__file__).resolve().parent
            # acados_template is in interfaces/acados_template, go up to acados root
            for _ in range(4):
                pkg_path = pkg_path.parent
                if (pkg_path / "lib" / "libacados.so").exists():
                    acados_root = str(pkg_path)
                    break
        except Exception:
            pass
    if not acados_root:
        for fallback in (
            os.path.expanduser("~/Documents/GitHub/acados"),
            os.path.expanduser("~/acados"),
            "/opt/acados",
        ):
            if os.path.isfile(os.path.join(fallback, "lib", "libacados.so")):
                acados_root = fallback
                break
    if not acados_root:
        return

    lib_path = os.path.join(acados_root, "lib")
    if not os.path.isdir(lib_path):
        return
    os.environ.setdefault("ACADOS_SOURCE_DIR", acados_root)
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_path not in ld_path.split(os.pathsep):
        os.environ["LD_LIBRARY_PATH"] = lib_path + (os.pathsep + ld_path if ld_path else "")

    # Preload acados' dependencies in dependency order with ``RTLD_GLOBAL``
    # so their symbols are visible to later dlopen calls. ``os.environ``
    # alone is not enough because the dynamic linker caches LD_LIBRARY_PATH
    # at process startup. acados_template will load libacados.so itself.
    #
    # ``libosqp`` is required because libacados has an undefined reference
    # to ``LINSYS_SOLVER_NAME`` that libosqp provides.
    mode = ctypes.RTLD_GLOBAL
    for libname in ("libblasfeo.so", "libhpipm.so", "libqpOASES_e.so", "libosqp.so"):
        full = os.path.join(lib_path, libname)
        if os.path.isfile(full):
            try:
                ctypes.CDLL(full, mode=mode)
            except OSError:
                pass  # Best-effort; downstream import will report a clearer error.

_setup_acados_env()

# Optional: Acados and CasADi
try:
    from acados_template import AcadosOcpSolver
    ACADOS_AVAILABLE = True
except ImportError:
    ACADOS_AVAILABLE = False


# Dynamics (CasADi plant), state maps, and OCP cost construction live in sibling modules.
from tvc_traj_opt_acados_dynamics import export_tvc_ode_model, export_tvc_ode_model_pseudotime
from tvc_traj_opt_acados_state import (
    _acados_x_for_method1,
    acados_state_to_method1,
    method1_to_acados_state,
    waypoint_to_acados_state,
)
from tvc_traj_opt_acados_cost import (
    build_acados_ocp,
    build_acados_ocp_min_time,
    build_acados_ocp_free_tf_external,
    _sync_acados_terminal_ls_from_ocp,
    _unified_segment_idx_for_stage,
    _unified_W_reg_no_state_tracking,
    _unified_waypoint_running_W_from_terminal,
    _unified_yref_waypoint,
)

def _clear_acados_module_cache(code_export_dir):
    """Clear Acados solver module cache to avoid segment 2 loading wrong solver from segment 1"""
    # Only remove c_generated_code modules, not acados_template
    to_remove = [k for k in list(sys.modules.keys())
                 if ('c_generated_code' in k or 'tvc_seg' in k) and 'acados_template' not in k]
    for k in to_remove:
        del sys.modules[k]
    gc.collect()


def _extract_sqp_cost_history(solver, final_cost, verbose_solve):
    """
    Build per-iterate NLP cost curve for plotting (matches acados_template.get_iterates:
    iterate indices are 0 .. nlp_iter inclusive — use nlp_iter, not sqp_iter).
    """
    if not verbose_solve:
        return [float(final_cost)] if final_cost is not None else [0.0]
    costs_list = []
    try:
        nlp_iter = int(solver.get_stats("nlp_iter"))
    except Exception:
        try:
            nlp_iter = int(solver.get_stats("sqp_iter"))
        except Exception:
            nlp_iter = 0
    try:
        for k in range(nlp_iter + 1):
            it = solver.get_iterate(k)
            solver.set_iterate(it)
            costs_list.append(float(solver.get_cost()))
        try:
            solver.set_iterate(solver.get_iterate(-1))
        except Exception:
            pass
    except Exception:
        costs_list = []
    if not costs_list:
        return [float(final_cost)] if final_cost is not None else [0.0]
    if final_cost is not None and len(costs_list) >= 2:
        fc = float(final_cost)
        if costs_list[-1] > 0 and abs(costs_list[-1] - fc) / max(costs_list[-1], 1e-12) > 0.01:
            costs_list[-1] = fc
        if costs_list[0] < costs_list[-1] * 0.1 and len(costs_list) > 1:
            costs_list[0] = costs_list[1]
    return costs_list


def solve_with_acados_waypoints_min_time(
    dt,
    waypoints,
    m,
    I,
    r_thrust,
    weights,
    bounds,
    max_iter=100,
    use_box_solver=False,
    callback=None,
    running_flag=None,
    terminal_weights=None,
    iteration_callback=None,
    verbose_solve=False,
):
    """
    Acados **minimum-time** variant: per segment, optimize physical duration ``T_seg`` (pseudo-time OCP
    with ``tf=1``). Waypoint time differences give **upper bounds** on ``T_seg``; lower bound from
    ``weights['min_time_T_min']``.

    Returns the same 4-tuple as ``solve_with_acados_waypoints`` plus a 5th dict with keys
    ``plot_dt``, ``segment_boundary_indices``, ``optimal_segment_times`` for GUI plotting.
    """
    if not ACADOS_AVAILABLE:
        raise ImportError("Acados not available. Install: pip install casadi; build acados from source.")

    m = float(m)
    I = np.array(I, dtype=float).reshape(3, 3)
    r_thrust = np.array(r_thrust, dtype=float).reshape(3,)

    durations = []
    for i in range(len(waypoints) - 1):
        d = waypoints[i + 1][4] - waypoints[i][4]
        if d <= 0:
            raise ValueError(f"Waypoint {i+1} time must be greater than waypoint {i} time")
        durations.append(float(d))

    T_scale = float(weights.get("min_time_T_max_scale", 1.0))
    T_min_def = float(weights.get("min_time_T_min", 0.15))
    uref = np.array([0.0, 0.0, m * 9.81, 0.0])

    all_xs = []
    all_us = []
    all_u_actual = []
    all_loggers = []
    optimal_T_list = []

    x0 = waypoint_to_acados_state(waypoints[0])
    base_export = os.path.join(os.path.dirname(os.path.abspath(__file__)), "c_generated_code")
    total_solve_time = 0.0
    total_sqp_iters = 0

    for seg_idx in range(len(durations)):
        if running_flag is not None and not running_flag():
            break

        duration = durations[seg_idx]
        T_max = max(duration * T_scale, T_min_def + 1e-3)
        T_min = min(T_min_def, T_max * 0.5)
        end_wp = waypoints[seg_idx + 1]
        xg = waypoint_to_acados_state(end_wp)

        N = max(50, int(duration / dt))
        T_guess = float(np.clip(0.55 * (T_min + T_max), T_min, T_max))

        use_actuator_dynamics = weights.get("actuator_dynamics", False)
        actuator_tau = weights.get("actuator_tau", [0.05, 0.05, 0.05, 0.05])
        use_control_rate = (not use_actuator_dynamics) and weights.get("du", 0.0) > 0

        if seg_idx > 0:
            _clear_acados_module_cache(None)
            nlp_max_iter = (
                max(300, max_iter * 3) if use_actuator_dynamics else max(200, max_iter * 2)
            )
            qp_solver = None
        else:
            nlp_max_iter = max_iter
            qp_solver = None

        model_suffix = "_mt_act" if use_actuator_dynamics else ("_mt_du" if use_control_rate else "_mt")
        code_export_dir = os.path.join(base_export, f"tvc_seg{seg_idx}{model_suffix}_N{N}")
        json_file = os.path.join(code_export_dir, f"tvc_rocket_seg{seg_idx}{model_suffix}.json")
        model_name = f"tvc_rocket_seg{seg_idx}{model_suffix}"
        model = export_tvc_ode_model_pseudotime(
            m,
            I,
            r_thrust,
            model_name=model_name,
            use_control_rate=use_control_rate,
            use_actuator_dynamics=use_actuator_dynamics,
            actuator_tau=actuator_tau,
            N_shoot=N,
        )
        ocp = build_acados_ocp_min_time(
            model,
            N,
            x0,
            xg,
            uref,
            weights,
            bounds,
            dt,
            T_min,
            T_max,
            T_guess,
            terminal_weights,
            code_export_dir=code_export_dir,
            json_file=json_file,
            nlp_solver_max_iter=nlp_max_iter,
            qp_solver=qp_solver,
            verbose_solve=verbose_solve,
        )

        x0_arr = np.asarray(x0, dtype=float).flatten()
        xg_arr = np.asarray(xg, dtype=float).flatten()
        use_16 = use_control_rate or use_actuator_dynamics
        if use_16 and len(x0_arr) == 12:
            x0_seg = np.concatenate([x0_arr[:12], uref])
        elif use_16 and len(x0_arr) >= 16:
            x0_seg = x0_arr[:16]
        else:
            x0_seg = x0_arr
        if use_16 and len(xg_arr) == 12:
            xg_seg = np.concatenate([xg_arr[:12], uref])
        elif use_16 and len(xg_arr) >= 16:
            xg_seg = xg_arr[:16]
        else:
            xg_seg = xg_arr

        x0_full = np.concatenate([x0_seg, [T_guess]])

        try:
            try:
                solver = AcadosOcpSolver(ocp, verbose=False, check_reuse_possible=False)
            except TypeError:
                solver = AcadosOcpSolver(ocp, verbose=False)
        except OSError as e:
            if "cannot open shared object file" in str(e) or "libqpOASES" in str(e) or "libhpipm" in str(e):
                raise RuntimeError(
                    f"Acados solver failed: {e}\n\n"
                    "Fix: set ACADOS_SOURCE_DIR (so the libraries can be preloaded), e.g.:\n"
                    "  export ACADOS_SOURCE_DIR=/path/to/acados\n"
                    "Or launch via ``python run_tvc_traj_opt.py`` which preloads the libs automatically."
                ) from e
            raise
        except Exception as e:
            raise RuntimeError(f"Acados solver creation failed: {e}") from e

        nx_full = int(x0_full.size)
        solver.set(0, "x", x0_full)
        if seg_idx > 0:
            try:
                lbx0 = np.full(nx_full, -1e10)
                ubx0 = np.full(nx_full, 1e10)
                lbx0[:-1] = ubx0[:-1] = np.asarray(x0_full[:-1], dtype=float)
                solver.constraints_set(0, "lbx", lbx0)
                solver.constraints_set(0, "ubx", ubx0)
            except Exception as e:
                if verbose_solve:
                    print(f"  [Note] constraints_set lbx/ubx (min-time): {e}")

        if seg_idx > 0 and use_actuator_dynamics:
            try:
                solver.options_set("qp_tau_min", 1e-8)
                solver.options_set("globalization", "FIXED_STEP")
                solver.options_set("globalization_fixed_step_length", 0.7)
            except Exception:
                pass

        if use_actuator_dynamics:
            actuator_tau_arr = np.asarray(actuator_tau).flatten()
            if len(actuator_tau_arr) < 4:
                actuator_tau_arr = np.resize(actuator_tau_arr, 4)
            p_val = np.asarray(1.0 / np.maximum(actuator_tau_arr.astype(float), 1e-6), dtype=np.float64)
            for i in range(N):
                solver.set(i, "p", p_val)

        use_schedule_ref = weights.get("schedule_ref", True)
        uref_arr_yref = np.array(uref)
        if use_schedule_ref:
            x0_12 = np.asarray(x0_seg).flatten()[:12]
            xg_12 = np.asarray(xg_seg).flatten()[:12]
            u_actual_ref = uref_arr_yref
            if seg_idx > 0 and use_actuator_dynamics:
                u_a0_ref = np.asarray(x0_seg[12:16], dtype=float)
            for i in range(N):
                alpha = float(i) / N
                x_ref = (1 - alpha) * x0_12 + alpha * xg_12
                if use_control_rate:
                    yref = np.concatenate([x_ref, uref_arr_yref, uref_arr_yref, np.zeros(4)])
                elif use_actuator_dynamics:
                    if seg_idx > 0:
                        t_frac = float(i + 1) / max(N, 1)
                        decay = 1.0 - np.exp(-4.0 * t_frac)
                        u_actual_ref = u_a0_ref + (uref_arr_yref - u_a0_ref) * decay
                    yref = np.concatenate([x_ref, u_actual_ref, uref_arr_yref])
                else:
                    yref = np.concatenate([x_ref, uref_arr_yref])
                solver.set(i, "yref", yref)

        uref_arr = np.array(uref)
        if seg_idx > 0 and use_actuator_dynamics:
            u_a0 = np.asarray(x0_seg[12:16], dtype=float)
            for i in range(1, N + 1):
                alpha = float(i) / N
                x_phys_guess = (1 - alpha) * x0_seg[:12] + alpha * xg_seg[:12]
                t_frac = float(i) / max(N, 1)
                decay = 1.0 - np.exp(-5.0 * t_frac)
                u_actual_guess = u_a0 + (uref_arr - u_a0) * decay
                x_guess = np.concatenate([x_phys_guess, u_actual_guess, [T_guess]])
                solver.set(i, "x", x_guess)
        else:
            for i in range(1, N + 1):
                alpha = float(i) / N
                x_phys = (1 - alpha) * x0_seg + alpha * xg_seg
                x_guess = np.concatenate([x_phys, [T_guess]])
                solver.set(i, "x", x_guess)
        for i in range(N):
            solver.set(i, "u", uref_arr)

        if iteration_callback is not None:
            iteration_callback(0, 0.0, 0.0, seg_idx)

        t0 = time.perf_counter()
        status = solver.solve()
        elapsed = time.perf_counter() - t0
        total_solve_time += elapsed
        if status != 0 and seg_idx > 0:
            print(
                f"  [Note] Segment {seg_idx+1} min-time Acados status={status}, "
                "solution may be partial (ignore if trajectory OK)"
            )

        cost_val = solver.get_cost()
        try:
            sqp_iter = solver.get_stats("sqp_iter")
        except Exception:
            sqp_iter = 1
        total_sqp_iters += int(sqp_iter)
        try:
            T_opt = float(np.asarray(solver.get(N, "x"), dtype=float).flatten()[-1])
        except Exception:
            T_opt = T_guess
        optimal_T_list.append(T_opt)

        if verbose_solve:
            solver.print_statistics()
            print(
                f"  Segment {seg_idx+1} min-time: cost={cost_val:.6e}, T*={T_opt:.4f}s, "
                f"SQP iter={sqp_iter}, time={elapsed:.3f}s"
            )
            sys.stdout.flush()
        if iteration_callback is not None:
            iteration_callback(int(sqp_iter), float(cost_val), 0.0, seg_idx)

        seg_u_act = None
        try:
            seg_xs = [
                acados_state_to_method1(_acados_x_for_method1(np.array(solver.get(i, "x"), copy=True)))
                for i in range(N + 1)
            ]
            seg_us = [np.array(solver.get(i, "u"), copy=True) for i in range(N)]
            x0 = _acados_x_for_method1(np.asarray(solver.get(N, "x"), dtype=float).flatten())
            if use_actuator_dynamics:
                rows = []
                for i in range(N + 1):
                    xa = np.asarray(solver.get(i, "x"), dtype=float).flatten()
                    if xa.size < 17:
                        rows = None
                        break
                    rows.append(np.array(xa[12:16], dtype=float, copy=True))
                if rows is not None:
                    seg_u_act = np.stack(rows, axis=0)
        except Exception as e:
            if status != 0:
                seg_xs = []
                for i in range(N + 1):
                    alpha = float(i) / N
                    x_core = (1 - alpha) * x0_seg + alpha * xg_seg
                    x_ac = np.concatenate([x_core, [T_guess]])
                    seg_xs.append(acados_state_to_method1(_acados_x_for_method1(x_ac)))
                seg_us = [np.array(uref_arr, copy=True) for _ in range(N)]
                x0 = np.array(xg_seg, copy=True)
                seg_u_act = None
                print(f"  [Fallback] Segment {seg_idx+1} min-time using guess ({e})")
            else:
                raise

        costs_list = _extract_sqp_cost_history(solver, cost_val, verbose_solve)

        class SimpleLogger:
            def __init__(self, costs):
                self.costs = (
                    costs
                    if isinstance(costs, (list, tuple))
                    else [costs]
                    if costs is not None
                    else [0.0]
                )

        all_loggers.append(SimpleLogger(costs_list))

        if callback is not None:
            callback(None, seg_idx, seg_xs, seg_us, all_xs, all_us)

        all_xs.append(seg_xs)
        all_us.append(seg_us)
        all_u_actual.append(seg_u_act)

        del solver
        gc.collect()

    if verbose_solve and total_sqp_iters > 0:
        print(
            f"  [Acados min-time] Total: SQP iter={total_sqp_iters}, wall={total_solve_time:.3f}s, "
            f"optimal T per seg [s]={[f'{t:.3f}' for t in optimal_T_list]}"
        )
        sys.stdout.flush()

    combined_xs = []
    combined_us = []
    u_blocks = []
    has_u_actual = bool(all_u_actual) and all(a is not None for a in all_u_actual)
    boundary_acc = []
    acc_idx = -1
    for si, (seg_xs, seg_us) in enumerate(zip(all_xs, all_us)):
        nseg = len(seg_xs) - 1
        if si == 0:
            combined_xs.extend(seg_xs)
            combined_us.extend(seg_us)
            acc_idx = nseg
            boundary_acc.append(acc_idx)
            if has_u_actual:
                u_blocks.append(np.asarray(all_u_actual[si], dtype=float))
        else:
            combined_xs.extend(seg_xs[1:])
            combined_us.extend(seg_us)
            acc_idx += nseg
            boundary_acc.append(acc_idx)
            if has_u_actual:
                u_blocks.append(np.asarray(all_u_actual[si][1:], dtype=float))
    u_actual_out = np.vstack(u_blocks) if has_u_actual and u_blocks else None

    total_T = float(sum(optimal_T_list)) if optimal_T_list else 0.0
    n_states = len(combined_xs)
    plot_dt = (
        total_T / max(n_states - 1, 1)
        if n_states > 1 and total_T > 1e-9
        else float(dt)
    )

    meta = {
        "min_time": True,
        "plot_dt": plot_dt,
        "segment_boundary_indices": boundary_acc,
        "optimal_segment_times": optimal_T_list,
        "total_sqp_iters": int(total_sqp_iters),
        "total_solve_time": float(total_solve_time),
    }
    return combined_xs, combined_us, all_loggers, u_actual_out, meta


def solve_with_acados_waypoints_free_tf(
    dt,
    waypoints,
    m,
    I,
    r_thrust,
    weights,
    bounds,
    max_iter=100,
    use_box_solver=False,
    callback=None,
    running_flag=None,
    terminal_weights=None,
    iteration_callback=None,
    verbose_solve=False,
):
    """
    Per-segment **free final time** with pseudo-time τ∈[0,1] and physical duration ``tf`` as state.
    **EXTERNAL** running cost ``(tf/N)*(w_T T^2 + …)`` and terminal ``w_tf*tf`` (+ optional state Mayer),
    matching the normalized-time / ``tf``-as-state pattern from the user's reference snippet.
    """
    if not ACADOS_AVAILABLE:
        raise ImportError("Acados not available. Install: pip install casadi; build acados from source.")

    m = float(m)
    I = np.array(I, dtype=float).reshape(3, 3)
    r_thrust = np.array(r_thrust, dtype=float).reshape(3,)

    durations = []
    for i in range(len(waypoints) - 1):
        d = waypoints[i + 1][4] - waypoints[i][4]
        if d <= 0:
            raise ValueError(f"Waypoint {i+1} time must be greater than waypoint {i} time")
        durations.append(float(d))

    T_scale = float(weights.get("min_time_T_max_scale", 1.0))
    T_min_def = float(weights.get("min_time_T_min", 0.15))
    uref = np.array([0.0, 0.0, m * 9.81, 0.0])

    all_xs = []
    all_us = []
    all_u_actual = []
    all_loggers = []
    optimal_T_list = []

    x0 = waypoint_to_acados_state(waypoints[0])
    base_export = os.path.join(os.path.dirname(os.path.abspath(__file__)), "c_generated_code")
    total_solve_time = 0.0
    total_sqp_iters = 0

    for seg_idx in range(len(durations)):
        if running_flag is not None and not running_flag():
            break

        duration = durations[seg_idx]
        T_max = max(duration * T_scale, T_min_def + 1e-3)
        T_min = min(T_min_def, T_max * 0.5)
        end_wp = waypoints[seg_idx + 1]
        xg = waypoint_to_acados_state(end_wp)

        N = max(10, int(duration / dt))
        T_guess = float(np.clip(0.55 * (T_min + T_max), T_min, T_max))

        use_actuator_dynamics = weights.get("actuator_dynamics", False)
        actuator_tau = weights.get("actuator_tau", [0.05, 0.05, 0.05, 0.05])
        use_control_rate = (not use_actuator_dynamics) and weights.get("du", 0.0) > 0

        if seg_idx > 0:
            _clear_acados_module_cache(None)
            nlp_max_iter = (
                max(300, max_iter * 3) if use_actuator_dynamics else max(200, max_iter * 2)
            )
            qp_solver = None
        else:
            nlp_max_iter = max_iter
            qp_solver = None

        model_suffix = "_ft_act" if use_actuator_dynamics else ("_ft_du" if use_control_rate else "_ft")
        code_export_dir = os.path.join(base_export, f"tvc_seg{seg_idx}{model_suffix}_N{N}")
        json_file = os.path.join(code_export_dir, f"tvc_rocket_seg{seg_idx}{model_suffix}.json")
        model_name = f"tvc_rocket_seg{seg_idx}{model_suffix}"
        model = export_tvc_ode_model_pseudotime(
            m,
            I,
            r_thrust,
            model_name=model_name,
            use_control_rate=use_control_rate,
            use_actuator_dynamics=use_actuator_dynamics,
            actuator_tau=actuator_tau,
            N_shoot=N,
        )
        ocp = build_acados_ocp_free_tf_external(
            model,
            N,
            x0,
            xg,
            uref,
            weights,
            bounds,
            dt,
            T_min,
            T_max,
            T_guess,
            terminal_weights,
            code_export_dir=code_export_dir,
            json_file=json_file,
            nlp_solver_max_iter=nlp_max_iter,
            qp_solver=qp_solver,
            verbose_solve=verbose_solve,
        )

        x0_arr = np.asarray(x0, dtype=float).flatten()
        xg_arr = np.asarray(xg, dtype=float).flatten()
        use_16 = use_control_rate or use_actuator_dynamics
        if use_16 and len(x0_arr) == 12:
            x0_seg = np.concatenate([x0_arr[:12], uref])
        elif use_16 and len(x0_arr) >= 16:
            x0_seg = x0_arr[:16]
        else:
            x0_seg = x0_arr
        if use_16 and len(xg_arr) == 12:
            xg_seg = np.concatenate([xg_arr[:12], uref])
        elif use_16 and len(xg_arr) >= 16:
            xg_seg = xg_arr[:16]
        else:
            xg_seg = xg_arr

        x0_full = np.concatenate([x0_seg, [T_guess]])

        try:
            try:
                solver = AcadosOcpSolver(ocp, verbose=False, check_reuse_possible=False)
            except TypeError:
                solver = AcadosOcpSolver(ocp, verbose=False)
        except OSError as e:
            if "cannot open shared object file" in str(e) or "libqpOASES" in str(e) or "libhpipm" in str(e):
                raise RuntimeError(
                    f"Acados solver failed: {e}\n\n"
                    "Fix: set ACADOS_SOURCE_DIR (so the libraries can be preloaded), e.g.:\n"
                    "  export ACADOS_SOURCE_DIR=/path/to/acados\n"
                    "Or launch via ``python run_tvc_traj_opt.py`` which preloads the libs automatically."
                ) from e
            raise
        except Exception as e:
            raise RuntimeError(f"Acados solver creation failed: {e}") from e

        nx_full = int(x0_full.size)
        solver.set(0, "x", x0_full)
        if seg_idx > 0:
            try:
                lbx0 = np.full(nx_full, -1e10)
                ubx0 = np.full(nx_full, 1e10)
                lbx0[:-1] = ubx0[:-1] = np.asarray(x0_full[:-1], dtype=float)
                solver.constraints_set(0, "lbx", lbx0)
                solver.constraints_set(0, "ubx", ubx0)
            except Exception as e:
                if verbose_solve:
                    print(f"  [Note] constraints_set lbx/ubx (free-tf): {e}")

        if seg_idx > 0 and use_actuator_dynamics:
            try:
                solver.options_set("qp_tau_min", 1e-8)
                solver.options_set("globalization", "FIXED_STEP")
                solver.options_set("globalization_fixed_step_length", 0.7)
            except Exception:
                pass

        if use_actuator_dynamics:
            actuator_tau_arr = np.asarray(actuator_tau).flatten()
            if len(actuator_tau_arr) < 4:
                actuator_tau_arr = np.resize(actuator_tau_arr, 4)
            p_val = np.asarray(1.0 / np.maximum(actuator_tau_arr.astype(float), 1e-6), dtype=np.float64)
            for i in range(N):
                solver.set(i, "p", p_val)

        uref_arr = np.array(uref)
        if seg_idx > 0 and use_actuator_dynamics:
            u_a0 = np.asarray(x0_seg[12:16], dtype=float)
            for i in range(1, N + 1):
                alpha = float(i) / N
                x_phys_guess = (1 - alpha) * x0_seg[:12] + alpha * xg_seg[:12]
                t_frac = float(i) / max(N, 1)
                decay = 1.0 - np.exp(-5.0 * t_frac)
                u_actual_guess = u_a0 + (uref_arr - u_a0) * decay
                x_guess = np.concatenate([x_phys_guess, u_actual_guess, [T_guess]])
                solver.set(i, "x", x_guess)
        else:
            for i in range(1, N + 1):
                alpha = float(i) / N
                x_phys = (1 - alpha) * x0_seg + alpha * xg_seg
                x_guess = np.concatenate([x_phys, [T_guess]])
                solver.set(i, "x", x_guess)
        for i in range(N):
            solver.set(i, "u", uref_arr)

        if iteration_callback is not None:
            iteration_callback(0, 0.0, 0.0, seg_idx)

        t0 = time.perf_counter()
        status = solver.solve()
        elapsed = time.perf_counter() - t0
        total_solve_time += elapsed
        if status != 0 and seg_idx > 0:
            print(
                f"  [Note] Segment {seg_idx+1} free-tf Acados status={status}, "
                "solution may be partial (ignore if trajectory OK)"
            )

        cost_val = solver.get_cost()
        try:
            sqp_iter = solver.get_stats("sqp_iter")
        except Exception:
            sqp_iter = 1
        total_sqp_iters += int(sqp_iter)
        try:
            T_opt = float(np.asarray(solver.get(N, "x"), dtype=float).flatten()[-1])
        except Exception:
            T_opt = T_guess
        optimal_T_list.append(T_opt)

        if verbose_solve:
            solver.print_statistics()
            print(
                f"  Segment {seg_idx+1} free-tf (EXTERNAL): cost={cost_val:.6e}, T*={T_opt:.4f}s, "
                f"SQP iter={sqp_iter}, time={elapsed:.3f}s"
            )
            sys.stdout.flush()
        if iteration_callback is not None:
            iteration_callback(int(sqp_iter), float(cost_val), 0.0, seg_idx)

        seg_u_act = None
        try:
            seg_xs = [
                acados_state_to_method1(_acados_x_for_method1(np.array(solver.get(i, "x"), copy=True)))
                for i in range(N + 1)
            ]
            seg_us = [np.array(solver.get(i, "u"), copy=True) for i in range(N)]
            x0 = _acados_x_for_method1(np.asarray(solver.get(N, "x"), dtype=float).flatten())
            if use_actuator_dynamics:
                rows = []
                for i in range(N + 1):
                    xa = np.asarray(solver.get(i, "x"), dtype=float).flatten()
                    if xa.size < 17:
                        rows = None
                        break
                    rows.append(np.array(xa[12:16], dtype=float, copy=True))
                if rows is not None:
                    seg_u_act = np.stack(rows, axis=0)
        except Exception as e:
            if status != 0:
                seg_xs = []
                for i in range(N + 1):
                    alpha = float(i) / N
                    x_core = (1 - alpha) * x0_seg + alpha * xg_seg
                    x_ac = np.concatenate([x_core, [T_guess]])
                    seg_xs.append(acados_state_to_method1(_acados_x_for_method1(x_ac)))
                seg_us = [np.array(uref_arr, copy=True) for _ in range(N)]
                x0 = np.array(xg_seg, copy=True)
                seg_u_act = None
                print(f"  [Fallback] Segment {seg_idx+1} free-tf using guess ({e})")
            else:
                raise

        costs_list = _extract_sqp_cost_history(solver, cost_val, verbose_solve)

        class SimpleLogger:
            def __init__(self, costs):
                self.costs = (
                    costs
                    if isinstance(costs, (list, tuple))
                    else [costs]
                    if costs is not None
                    else [0.0]
                )

        all_loggers.append(SimpleLogger(costs_list))

        if callback is not None:
            callback(None, seg_idx, seg_xs, seg_us, all_xs, all_us)

        all_xs.append(seg_xs)
        all_us.append(seg_us)
        all_u_actual.append(seg_u_act)

        del solver
        gc.collect()

    if verbose_solve and total_sqp_iters > 0:
        print(
            f"  [Acados free-tf EXTERNAL] Total: SQP iter={total_sqp_iters}, wall={total_solve_time:.3f}s, "
            f"optimal T per seg [s]={[f'{t:.3f}' for t in optimal_T_list]}"
        )
        sys.stdout.flush()

    combined_xs = []
    combined_us = []
    u_blocks = []
    has_u_actual = bool(all_u_actual) and all(a is not None for a in all_u_actual)
    boundary_acc = []
    acc_idx = -1
    for si, (seg_xs, seg_us) in enumerate(zip(all_xs, all_us)):
        nseg = len(seg_xs) - 1
        if si == 0:
            combined_xs.extend(seg_xs)
            combined_us.extend(seg_us)
            acc_idx = nseg
            boundary_acc.append(acc_idx)
            if has_u_actual:
                u_blocks.append(np.asarray(all_u_actual[si], dtype=float))
        else:
            combined_xs.extend(seg_xs[1:])
            combined_us.extend(seg_us)
            acc_idx += nseg
            boundary_acc.append(acc_idx)
            if has_u_actual:
                u_blocks.append(np.asarray(all_u_actual[si][1:], dtype=float))
    u_actual_out = np.vstack(u_blocks) if has_u_actual and u_blocks else None

    total_T = float(sum(optimal_T_list)) if optimal_T_list else 0.0
    n_states = len(combined_xs)
    plot_dt = (
        total_T / max(n_states - 1, 1)
        if n_states > 1 and total_T > 1e-9
        else float(dt)
    )

    meta = {
        "free_tf_acados": True,
        "plot_dt": plot_dt,
        "segment_boundary_indices": boundary_acc,
        "optimal_segment_times": optimal_T_list,
        "total_sqp_iters": int(total_sqp_iters),
        "total_solve_time": float(total_solve_time),
    }
    return combined_xs, combined_us, all_loggers, u_actual_out, meta


def solve_with_acados_waypoints(dt, waypoints, m, I, r_thrust, weights, bounds, max_iter=100,
                                use_box_solver=False, callback=None, running_flag=None,
                                terminal_weights=None, iteration_callback=None, verbose_solve=False):
    """
    Solve trajectory optimization with waypoints using Acados.
    
    Interface compatible with solve_with_pinocchio_waypoints for GUI.
    
    Returns:
        combined_xs: List of states in Method 1 format (17-dim)
        combined_us: List of controls
        all_loggers: List of logger-like objects (with .costs for plotting)
        meta: dict (empty for fixed-time Acados; use ``acados_objective=='min_time'`` for timing keys)
    """
    if not ACADOS_AVAILABLE:
        raise ImportError("Acados not available. Install: pip install casadi; build acados from source.")

    obj = str(weights.get("acados_objective", "tracking")).lower()
    if obj == "min_time":
        return solve_with_acados_waypoints_min_time(
            dt,
            waypoints,
            m,
            I,
            r_thrust,
            weights,
            bounds,
            max_iter=max_iter,
            use_box_solver=use_box_solver,
            callback=callback,
            running_flag=running_flag,
            terminal_weights=terminal_weights,
            iteration_callback=iteration_callback,
            verbose_solve=verbose_solve,
        )
    if obj in ("free_tf", "free_tf_external"):
        return solve_with_acados_waypoints_free_tf(
            dt,
            waypoints,
            m,
            I,
            r_thrust,
            weights,
            bounds,
            max_iter=max_iter,
            use_box_solver=use_box_solver,
            callback=callback,
            running_flag=running_flag,
            terminal_weights=terminal_weights,
            iteration_callback=iteration_callback,
            verbose_solve=verbose_solve,
        )

    m = float(m)
    I = np.array(I, dtype=float).reshape(3, 3)
    r_thrust = np.array(r_thrust, dtype=float).reshape(3,)
    
    # Segment durations
    durations = []
    for i in range(len(waypoints) - 1):
        d = waypoints[i+1][4] - waypoints[i][4]
        if d <= 0:
            raise ValueError(f"Waypoint {i+1} time must be greater than waypoint {i} time")
        durations.append(d)
    
    uref = np.array([0.0, 0.0, m*9.81, 0.0])
    
    all_xs = []
    all_us = []
    all_u_actual = []  # per segment: (N+1,4) or None if no actuator dynamics
    all_loggers = []
    
    # Initial state
    x0 = waypoint_to_acados_state(waypoints[0])
    
    # Base dir for code export (avoid segment overwrite / module cache reuse)
    base_export = os.path.join(os.path.dirname(os.path.abspath(__file__)), "c_generated_code")
    total_solve_time = 0.0
    total_sqp_iters = 0
    
    for seg_idx in range(len(durations)):
        if running_flag is not None and not running_flag():
            break
        
        duration = durations[seg_idx]
        end_wp = waypoints[seg_idx + 1]
        xg = waypoint_to_acados_state(end_wp)
        
        N = max(10, int(duration / dt))
        Tf = duration
        
        # Segment 2+: clear module cache, increase SQP iterations (more when actuator dynamics)
        use_actuator_dynamics = weights.get("actuator_dynamics", False)
        actuator_tau = weights.get("actuator_tau", [0.05, 0.05, 0.05, 0.05])
        use_control_rate = (not use_actuator_dynamics) and weights.get("du", 0.0) > 0
        if seg_idx > 0:
            _clear_acados_module_cache(None)
            nlp_max_iter = max(300, max_iter * 3) if weights.get("actuator_dynamics", False) else max(200, max_iter * 2)
            qp_solver = None
        else:
            nlp_max_iter = max_iter
            qp_solver = None
        
        # Model-type suffix: ensure segment 2 never loads wrong model (act vs control-rate vs basic)
        model_suffix = "_act" if use_actuator_dynamics else ("_du" if use_control_rate else "")
        code_export_dir = os.path.join(base_export, f"tvc_seg{seg_idx}{model_suffix}_N{N}")
        json_file = os.path.join(code_export_dir, f"tvc_rocket_seg{seg_idx}{model_suffix}.json")
        model_name = f"tvc_rocket_seg{seg_idx}{model_suffix}"
        model = export_tvc_ode_model(m, I, r_thrust, model_name=model_name, use_control_rate=use_control_rate,
                                    use_actuator_dynamics=use_actuator_dynamics, actuator_tau=actuator_tau)
        ocp = build_acados_ocp(model, N, Tf, x0, xg, uref, weights, bounds, dt, terminal_weights,
                               code_export_dir=code_export_dir, json_file=json_file,
                               nlp_solver_max_iter=nlp_max_iter, qp_solver=qp_solver,
                               verbose_solve=verbose_solve)
        
        # Extend x0/xg to 16-dim (with u_prev/u_actual) for control rate or actuator model
        x0_arr = np.asarray(x0, dtype=float).flatten()
        xg_arr = np.asarray(xg, dtype=float).flatten()
        use_16 = use_control_rate or use_actuator_dynamics
        # Segment 2+: x0 from prev segment should be 16-dim; if truncated, use u_actual from prev end or uref
        if use_16 and len(x0_arr) == 12:
            x0_seg = np.concatenate([x0_arr[:12], uref])
        elif use_16 and len(x0_arr) >= 16:
            x0_seg = x0_arr[:16]  # Ensure exactly 16, drop any extra
        else:
            x0_seg = x0_arr
        if use_16 and len(xg_arr) == 12:
            xg_seg = np.concatenate([xg_arr[:12], uref])
        elif use_16 and len(xg_arr) >= 16:
            xg_seg = xg_arr[:16]
        else:
            xg_seg = xg_arr
        
        try:
            try:
                solver = AcadosOcpSolver(ocp, verbose=False, check_reuse_possible=False)
            except TypeError:
                solver = AcadosOcpSolver(ocp, verbose=False)
        except OSError as e:
            if "cannot open shared object file" in str(e) or "libqpOASES" in str(e) or "libhpipm" in str(e):
                raise RuntimeError(
                    f"Acados solver failed: {e}\n\n"
                    "Fix: set ACADOS_SOURCE_DIR (so the libraries can be preloaded), e.g.:\n"
                    "  export ACADOS_SOURCE_DIR=/path/to/acados\n"
                    "Or launch via ``python run_tvc_traj_opt.py`` which preloads the libs automatically."
                ) from e
            raise
        except Exception as e:
            raise RuntimeError(f"Acados solver creation failed: {e}") from e
        
        # Set initial state (guess + enforce constraint at runtime for segment 2)
        solver.set(0, "x", x0_seg)
        if seg_idx > 0:
            # Explicitly set lbx_0=ubx_0=x0 at runtime; ensures correct initial state constraint
            # (build-time x0 in OCP may not propagate correctly when module cache is cleared)
            try:
                solver.constraints_set(0, "lbx", np.asarray(x0_seg, dtype=float))
                solver.constraints_set(0, "ubx", np.asarray(x0_seg, dtype=float))
            except Exception as e:
                if verbose_solve:
                    print(f"  [Note] constraints_set lbx/ubx: {e}")
        
        # Segment 2 + actuator: relax QP/SQP to avoid ACADOS_MINSTEP (status 4)
        if seg_idx > 0 and use_actuator_dynamics:
            try:
                solver.options_set("qp_tau_min", 1e-8)  # allow smaller barrier param
                solver.options_set("globalization", "FIXED_STEP")
                solver.options_set("globalization_fixed_step_length", 0.7)  # conservative step
            except Exception:
                pass
        
        # Set model param p: control rate uses 1/dt; actuator uses 1/tau per channel
        if use_control_rate:
            p_val = np.array([N / Tf])
            for i in range(N):
                solver.set(i, "p", p_val)
        elif use_actuator_dynamics:
            actuator_tau_arr = np.asarray(actuator_tau).flatten()
            if len(actuator_tau_arr) < 4:
                actuator_tau_arr = np.resize(actuator_tau_arr, 4)
            p_val = np.asarray(1.0 / np.maximum(actuator_tau_arr.astype(float), 1e-6), dtype=np.float64)
            for i in range(N):
                solver.set(i, "p", p_val)
        
        # schedule_ref=True: time-scheduled ref for on-time arrival; False: constant goal ref (may arrive early)
        use_schedule_ref = weights.get("schedule_ref", True)
        if use_schedule_ref:
            x0_12 = np.asarray(x0_seg).flatten()[:12]
            xg_12 = np.asarray(xg_seg).flatten()[:12]
            uref_arr_yref = np.array(uref)
            # Segment 2 + actuator: use u_actual ref that interpolates u_a0 -> uref (smoother cost landscape)
            u_actual_ref = uref_arr_yref
            if seg_idx > 0 and use_actuator_dynamics:
                u_a0_ref = np.asarray(x0_seg[12:16], dtype=float)
            for i in range(N):
                alpha = float(i) / N
                x_ref = (1 - alpha) * x0_12 + alpha * xg_12
                if use_control_rate:
                    yref = np.concatenate([x_ref, uref_arr_yref, uref_arr_yref, np.zeros(4)])
                elif use_actuator_dynamics:
                    if seg_idx > 0:
                        # u_actual ref: interpolate from u_a0 to uref over horizon (reduces initial cost spike)
                        t_frac = float(i + 1) / max(N, 1)
                        decay = 1.0 - np.exp(-4.0 * t_frac)
                        u_actual_ref = u_a0_ref + (uref_arr_yref - u_a0_ref) * decay
                    yref = np.concatenate([x_ref, u_actual_ref, uref_arr_yref])  # [x_ref_12, u_actual_ref, uref]
                else:
                    yref = np.concatenate([x_ref, uref_arr_yref])
                solver.set(i, "yref", yref)
        
        # Initial guess: linear interpolation x0->xg
        # Segment 2 + actuator: u_actual from seg1 may differ from uref; use exponential decay for u_actual
        # to give dynamics-consistent guess (avoids ACADOS_MINSTEP from bad linear interpolation)
        uref_arr = np.array(uref)
        if seg_idx > 0 and use_actuator_dynamics:
            u_a0 = np.asarray(x0_seg[12:16], dtype=float)
            for i in range(1, N + 1):
                alpha = float(i) / N
                x_phys_guess = (1 - alpha) * x0_seg[:12] + alpha * xg_seg[:12]
                # u_actual: exponential decay from u_a0 to uref (time constant ~3*tau over horizon)
                t_frac = float(i) / max(N, 1)
                decay = 1.0 - np.exp(-5.0 * t_frac)  # reach ~99% of (uref-u_a0) by 60% of horizon
                u_actual_guess = u_a0 + (uref_arr - u_a0) * decay
                x_guess = np.concatenate([x_phys_guess, u_actual_guess])
                solver.set(i, "x", x_guess)
        else:
            for i in range(1, N + 1):
                alpha = float(i) / N
                x_guess = (1 - alpha) * x0_seg + alpha * xg_seg
                solver.set(i, "x", x_guess)
        for i in range(N):
            solver.set(i, "u", uref_arr)
        
        # Acados has no per-iteration callback; emit iter=0 before solve
        if iteration_callback is not None:
            iteration_callback(0, 0.0, 0.0, seg_idx)
        
        t0 = time.perf_counter()
        status = solver.solve()
        elapsed = time.perf_counter() - t0
        total_solve_time += elapsed
        if status != 0 and seg_idx > 0:
            # ACADOS_MINSTEP(4) common in segment 2: QP step too small, but solution often still usable
            print(f"  [Note] Segment {seg_idx+1} Acados status={status}, solution may be partial (ignore if trajectory OK)")
        
        # After solve: print iteration stats and final cost
        cost_val = solver.get_cost()
        try:
            sqp_iter = solver.get_stats("sqp_iter")
        except Exception:
            sqp_iter = 1
        total_sqp_iters += int(sqp_iter)
        if verbose_solve:
            solver.print_statistics()
            print(f"  Segment {seg_idx+1} final: cost={cost_val:.6e}, SQP iter={sqp_iter}, time={elapsed:.3f}s")
            sys.stdout.flush()
        if iteration_callback is not None:
            iteration_callback(int(sqp_iter), float(cost_val), 0.0, seg_idx)

        # Extract solution before _extract_sqp_cost_history (set_iterate may clobber primals).
        seg_u_act = None
        try:
            seg_xs = [acados_state_to_method1(np.array(solver.get(i, "x"), copy=True)) for i in range(N+1)]
            seg_us = [np.array(solver.get(i, "u"), copy=True) for i in range(N)]
            # Flatten x0 for next segment; critical for actuator/control-rate (16-dim) continuity
            x0 = np.asarray(solver.get(N, "x"), dtype=float).flatten()
            if use_actuator_dynamics:
                rows = []
                for i in range(N + 1):
                    xa = np.asarray(solver.get(i, "x"), dtype=float).flatten()
                    if xa.size < 16:
                        rows = None
                        break
                    rows.append(np.array(xa[12:16], dtype=float, copy=True))
                if rows is not None:
                    seg_u_act = np.stack(rows, axis=0)
        except Exception as e:
            if status != 0:
                # On solver failure use linear interpolation guess for seg_xs
                seg_xs = []
                for i in range(N + 1):
                    alpha = float(i) / N
                    x_ac = (1 - alpha) * x0_seg + alpha * xg_seg
                    seg_xs.append(acados_state_to_method1(np.array(x_ac, copy=True)))
                seg_us = [np.array(uref_arr, copy=True) for _ in range(N)]
                x0 = np.array(xg_seg, copy=True)
                seg_u_act = None
                print(f"  [Fallback] Segment {seg_idx+1} using initial guess (solver.get error: {e})")
            else:
                raise

        costs_list = _extract_sqp_cost_history(solver, cost_val, verbose_solve)
        if verbose_solve and len(costs_list) >= 2:
            try:
                ni = solver.get_stats("nlp_iter")
            except Exception:
                ni = "?"
            print(f"  Cost curve (nlp_iter={ni}, sqp_iter={sqp_iter}): "
                  f"iter0={costs_list[0]:.4e}, iter_last={costs_list[-1]:.4e}")

        class SimpleLogger:
            def __init__(self, costs):
                self.costs = costs if isinstance(costs, (list, tuple)) else [costs] if costs is not None else [0.0]

        all_loggers.append(SimpleLogger(costs_list))
        
        if callback is not None:
            callback(None, seg_idx, seg_xs, seg_us, all_xs, all_us)
        
        all_xs.append(seg_xs)
        all_us.append(seg_us)
        all_u_actual.append(seg_u_act)
        
        # Explicitly destroy solver to avoid Acados module cache reusing wrong dim (acados#905)
        del solver
        gc.collect()
    
    if verbose_solve and total_sqp_iters > 0:
        print(f"  [Acados] Total: SQP iter={total_sqp_iters}, time={total_solve_time:.3f}s, avg={total_solve_time/total_sqp_iters*1000:.1f}ms/iter")
        sys.stdout.flush()
    
    # Combine segments
    combined_xs = []
    combined_us = []
    u_blocks = []
    has_u_actual = bool(all_u_actual) and all(a is not None for a in all_u_actual)
    for i, (seg_xs, seg_us) in enumerate(zip(all_xs, all_us)):
        if i == 0:
            combined_xs.extend(seg_xs)
            combined_us.extend(seg_us)
            if has_u_actual:
                u_blocks.append(np.asarray(all_u_actual[i], dtype=float))
        else:
            combined_xs.extend(seg_xs[1:])
            combined_us.extend(seg_us)
            if has_u_actual:
                u_blocks.append(np.asarray(all_u_actual[i][1:], dtype=float))
    u_actual_out = np.vstack(u_blocks) if has_u_actual and u_blocks else None

    meta = {
        "total_sqp_iters": int(total_sqp_iters),
        "total_solve_time": float(total_solve_time),
    }
    return combined_xs, combined_us, all_loggers, u_actual_out, meta


def solve_with_acados_waypoints_unified(dt, waypoints, m, I, r_thrust, weights, bounds, max_iter=100,
                                        use_box_solver=False, callback=None, running_flag=None,
                                        terminal_weights=None, iteration_callback=None, verbose_solve=False,
                                        interp_initial_guess=None):
    """
    Solve with Acados using a single unified problem (all segments merged).

    Cost design — each **running** stage ``i = 0 .. N-1`` is exactly one of:
    - **Waypoint stage** (initial node and each interior waypoint arrival time): ``yref`` is that
      waypoint's full reference; ``W`` uses **terminal** state weights (``W_e``: running Q×``terminal_cost_multiplier``,
      plus ``u`` / actuator / ``du`` blocks) when ``waypoint_terminal_cost`` is True,
      else the default running ``W_reg``.
    - **Intermediate stage** (all other stages): ``yref`` state matches the **next** waypoint along
      the path (end of the current segment); **only control regularization** enters the least
      squares (state block of ``W`` is zero — no spurious pull toward origin or cheap shortcut).
      The **last** waypoint is enforced only by the **terminal** cost on ``x[:12]`` (``W_e``),
      not duplicated as a separate running waypoint stage at ``i = N``.

    Initial state guess over the horizon: by default all nodes use x0 (no spatial
    interpolation). If ``interp_initial_guess`` is True, or
    ``weights['unified_interp_initial_guess']`` is True when ``interp_initial_guess``
    is None, guess states by linear interpolation from wp_i to wp_{i+1} per segment.

    Returns ``(xs, us, loggers, us_actual, meta)`` like ``solve_with_acados_waypoints``.
    If ``weights['acados_objective'] == 'min_time'`` or ``'free_tf'``, ``unified`` is ignored for
    multi-leg problems (per-segment solves); ``meta`` carries ``plot_dt`` and segment times.
    """
    if not ACADOS_AVAILABLE:
        raise ImportError("Acados not available.")

    obj = str(weights.get("acados_objective", "tracking")).lower()
    if obj == "min_time":
        return solve_with_acados_waypoints_min_time(
            dt,
            waypoints,
            m,
            I,
            r_thrust,
            weights,
            bounds,
            max_iter=max_iter,
            use_box_solver=use_box_solver,
            callback=callback,
            running_flag=running_flag,
            terminal_weights=terminal_weights,
            iteration_callback=iteration_callback,
            verbose_solve=verbose_solve,
        )
    if obj in ("free_tf", "free_tf_external"):
        return solve_with_acados_waypoints_free_tf(
            dt,
            waypoints,
            m,
            I,
            r_thrust,
            weights,
            bounds,
            max_iter=max_iter,
            use_box_solver=use_box_solver,
            callback=callback,
            running_flag=running_flag,
            terminal_weights=terminal_weights,
            iteration_callback=iteration_callback,
            verbose_solve=verbose_solve,
        )

    if interp_initial_guess is None:
        use_interp_guess = bool(weights.get("unified_interp_initial_guess", False))
    else:
        use_interp_guess = bool(interp_initial_guess)

    m = float(m)
    I = np.array(I, dtype=float).reshape(3, 3)
    r_thrust = np.array(r_thrust, dtype=float).reshape(3,)
    
    durations = []
    for i in range(len(waypoints) - 1):
        d = waypoints[i+1][4] - waypoints[i][4]
        if d <= 0:
            raise ValueError(f"Waypoint {i+1} time must be greater than waypoint {i} time")
        durations.append(d)
    
    # Nodes per segment (same as Pinocchio unified)
    N_per_seg = [max(10, int(d / dt)) for d in durations]
    N_total = sum(N_per_seg)
    Tf_total = sum(durations)
    
    x0 = waypoint_to_acados_state(waypoints[0])
    xg = waypoint_to_acados_state(waypoints[-1])
    uref = np.array([0.0, 0.0, m*9.81, 0.0])
    
    # Unified mode needs stronger waypoint tracking (higher position weight)
    use_actuator_dynamics = weights.get("actuator_dynamics", False)
    actuator_tau = weights.get("actuator_tau", [0.05, 0.05, 0.05, 0.05])
    use_control_rate = (not use_actuator_dynamics) and weights.get("du", 0.0) > 0
    model = export_tvc_ode_model(m, I, r_thrust, use_control_rate=use_control_rate,
                                use_actuator_dynamics=use_actuator_dynamics, actuator_tau=actuator_tau)
    base_export = os.path.join(os.path.dirname(os.path.abspath(__file__)), "c_generated_code")
    unified_suffix = "_act" if use_actuator_dynamics else ("_du" if use_control_rate else "")
    code_export_dir = os.path.join(base_export, f"tvc_unified{unified_suffix}_N{N_total}")
    json_file = os.path.join(code_export_dir, f"tvc_rocket_unified{unified_suffix}.json")
    ocp = build_acados_ocp(model, N_total, Tf_total, x0, xg, uref, weights, bounds, dt, terminal_weights,
                           code_export_dir=code_export_dir, json_file=json_file,
                           verbose_solve=verbose_solve)
    
    use_16 = use_control_rate or use_actuator_dynamics
    x0_seg = np.concatenate([np.asarray(x0).flatten()[:12], uref]) if use_16 else np.asarray(x0).flatten()
    
    try:
        solver = AcadosOcpSolver(ocp, verbose=False)
    except Exception as e:
        raise RuntimeError(f"Acados solver creation failed: {e}") from e
    
    solver.set(0, "x", x0_seg)
    
    # Set model param p: control rate uses 1/dt; actuator uses 1/tau per channel
    if use_control_rate:
        p_val = np.array([N_total / Tf_total])
        for i in range(N_total):
            solver.set(i, "p", p_val)
    elif use_actuator_dynamics:
        actuator_tau_arr = np.asarray(actuator_tau).flatten()
        if len(actuator_tau_arr) < 4:
            actuator_tau_arr = np.resize(actuator_tau_arr, 4)
        p_val = 1.0 / np.maximum(actuator_tau_arr.astype(float), 1e-6)
        for i in range(N_total):
            solver.set(i, "p", p_val)
    
    # Regulation stages: exactly the running least-squares W from build_acados_ocp.
    W_reg = np.array(ocp.cost.W, dtype=float, copy=True)
    W_waypoint = _unified_waypoint_running_W_from_terminal(
        ocp, weights, use_actuator_dynamics, use_control_rate, bounds=bounds)
    use_large_weights_at_waypoints = weights.get("waypoint_terminal_cost", True)

    wp_stages = {0}
    acc = 0
    for j in range(len(N_per_seg) - 1):
        acc += N_per_seg[j]
        wp_stages.add(int(acc))

    def _wp_index_for_waypoint_stage(i):
        if i == 0:
            return 0
        c_run = 0
        for j, nj in enumerate(N_per_seg):
            c_run += nj
            if i == c_run:
                return j + 1
        raise RuntimeError(f"unified: stage {i} is not a waypoint stage")

    for i in range(N_total):
        if i in wp_stages:
            wp = waypoints[_wp_index_for_waypoint_stage(i)]
            yref = _unified_yref_waypoint(wp, uref, use_actuator_dynamics, use_control_rate)
            W_stage = W_waypoint if use_large_weights_at_waypoints else W_reg
        else:
            seg = _unified_segment_idx_for_stage(i, N_per_seg)
            wp_next = waypoints[seg + 1]
            yref = _unified_yref_waypoint(
                wp_next, uref, use_actuator_dynamics, use_control_rate)
            W_stage = _unified_W_reg_no_state_tracking(
                W_reg, use_actuator_dynamics, use_control_rate)
        solver.set(i, "yref", yref)
        try:
            solver.cost_set(i, "W", W_stage)
        except Exception:
            pass

    _sync_acados_terminal_ls_from_ocp(solver, ocp)

    # Initial guess: default constant x0; optional linear interpolation per segment
    if use_interp_guess:
        x_prev = x0_seg.copy()
        for seg_idx in range(len(durations)):
            end_wp = waypoints[seg_idx + 1]
            x_end_12 = waypoint_to_acados_state(end_wp)[:12]
            x_end = np.concatenate([x_end_12, uref]) if use_16 else x_end_12
            n_seg = N_per_seg[seg_idx]
            i0 = sum(N_per_seg[:seg_idx])
            for k in range(n_seg):
                alpha = (k + 1) / n_seg
                x_guess = (1 - alpha) * x_prev + alpha * x_end
                solver.set(i0 + k + 1, "x", x_guess)
            x_prev = x_end
    else:
        for j in range(1, N_total + 1):
            solver.set(j, "x", x0_seg.copy())
    for i in range(N_total):
        solver.set(i, "u", uref)
    
    if iteration_callback is not None:
        iteration_callback(0, 0.0, 0.0, 0)
    
    t0 = time.perf_counter()
    status = solver.solve()
    elapsed = time.perf_counter() - t0
    
    cost_val = solver.get_cost()
    try:
        sqp_iter = solver.get_stats("sqp_iter")
    except Exception:
        sqp_iter = 1

    N_shoot = int(solver.N)
    # Snapshot (x,u) before _extract_sqp_cost_history: that helper uses set_iterate() over past
    # SQP iterates and can leave the solver primal on an early iterate when store_iterates=True.
    combined_xs = [
        acados_state_to_method1(np.array(solver.get(i, "x"), dtype=np.float64, copy=True))
        for i in range(N_shoot + 1)
    ]
    combined_us = [
        np.array(solver.get(i, "u"), dtype=np.float64, copy=True).ravel() for i in range(N_shoot)
    ]
    combined_u_actual = None
    if use_actuator_dynamics:
        rows = []
        for i in range(N_shoot + 1):
            xa = np.asarray(solver.get(i, "x"), dtype=np.float64).flatten()
            if xa.size < 16:
                rows = None
                break
            rows.append(np.array(xa[12:16], dtype=np.float64, copy=True))
        if rows is not None:
            combined_u_actual = np.stack(rows, axis=0)

    if verbose_solve:
        solver.print_statistics()
        print(f"  Final: cost={cost_val:.6e}, SQP iter={sqp_iter}, time={elapsed:.3f}s")
        if sqp_iter > 0:
            print(f"  [Acados] Total: SQP iter={sqp_iter}, time={elapsed:.3f}s, avg={elapsed/sqp_iter*1000:.1f}ms/iter")
        try:
            print(f"  [unified] x_N[:3] (world p, Method1): {np.asarray(combined_xs[-1][:3], dtype=float)}")
        except Exception:
            pass
        sys.stdout.flush()
    if iteration_callback is not None:
        iteration_callback(int(sqp_iter), float(cost_val), 0.0, 0)
    
    costs_list = _extract_sqp_cost_history(solver, cost_val, verbose_solve)
    if verbose_solve and len(costs_list) >= 2:
        try:
            ni = solver.get_stats("nlp_iter")
        except Exception:
            ni = "?"
        print(f"  Cost curve (unified, nlp_iter={ni}, sqp_iter={sqp_iter}): "
              f"iter0={costs_list[0]:.4e}, iter_last={costs_list[-1]:.4e}")
    
    class SimpleLogger:
        def __init__(self, costs):
            self.costs = costs if isinstance(costs, (list, tuple)) else [costs] if costs is not None else [0.0]
    
    all_loggers = [SimpleLogger(costs_list)]
    
    if callback is not None:
        callback(None, 0, combined_xs, combined_us, [], [])
    
    return combined_xs, combined_us, all_loggers, combined_u_actual, {}


def _acados_cli_demo_problem(
    use_control_rate_smooth=False,
    use_actuator_dynamics=False,
    actuator_tau=None,
    waypoint_terminal_cost=True,
):
    """
    Shared waypoint scene and weights for CLI tests (Method 4 / 5 / 7).
    Returns ``acados_objective``-free weights; each method test sets that key.
    """
    if actuator_tau is None:
        actuator_tau = [0.05, 0.05, 0.05, 0.05]
    dt = 0.05
    waypoints = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 6.0, 0.0, 5.0],
        [4.0, 0.0, 0.0, 0.0, 10.0],
    ]
    m = 0.6
    I = np.diag([0.02, 0.02, 0.01])
    r_thrust = np.array([0, 0, -0.2])
    weights = {"p": 1.0, "v": 1.0, "R": 1.0, "yaw": 1.0, "w": 1, "u": 1.0}
    if use_control_rate_smooth:
        weights["du"] = 100.0
    if use_actuator_dynamics:
        weights["actuator_dynamics"] = True
        weights["actuator_tau"] = list(np.asarray(actuator_tau).flatten()[:4])
    weights["schedule_ref"] = True
    weights["terminal_cost_multiplier"] = 200.0
    weights["terminal_constraint"] = False
    weights["waypoint_terminal_cost"] = waypoint_terminal_cost
    bounds = {
        "th_p": (-0.15, 0.15), "th_r": (-0.15, 0.15),
        "T": (0.0, 15.0), "tau_yaw": (-1.0, 1.0),
        "state_v_horizontal_max": 1.0, "state_v_vertical_max": 1.5,
        "state_roll_max": np.radians(5.0), "state_pitch_max": np.radians(5.0),
        "state_yaw_max": np.radians(30.0),
    }
    return dt, waypoints, m, I, r_thrust, weights, bounds


def _acados_cli_print_scene_preamble(title, dt, waypoints, weights, bounds, verbose_solve, extra_lines=()):
    if len(waypoints) < 2:
        raise ValueError("Need at least 2 waypoints (start and end)")
    n_seg = len(waypoints) - 1
    print("=" * 50)
    print(title)
    print("=" * 50)
    for i, wp in enumerate(waypoints):
        t = wp[4] if len(wp) >= 5 else 0.0
        print(f"WP{i}: {wp[:3]} (t={t}s)")
    if len(waypoints) >= 2:
        seg_info = [f"{waypoints[i + 1][4] - waypoints[i][4]:.1f}s" for i in range(n_seg)]
        print(f"dt={dt}s, nominal segment durations: {seg_info}")
    for line in extra_lines:
        print(line)
    print(f"Control rate penalty: {'enabled du=' + str(weights.get('du', 0)) if weights.get('du') else 'disabled'}")
    ad = weights.get("actuator_dynamics", False)
    print(f"Actuator dynamics: {'enabled tau=' + str(weights.get('actuator_tau', [])) if ad else 'disabled'}")
    print(f"Ref mode: {'on-time arrival (schedule_ref)' if weights.get('schedule_ref', True) else 'constant goal (may arrive early)'}")
    print(f"Terminal cost: multiplier vs running={weights.get('terminal_cost_multiplier', weights.get('terminal_scale', 200.0))}")
    print(f"Terminal constraint: {weights.get('terminal_constraint', False)}")
    if verbose_solve:
        print("Iteration output: enabled (use python -u to avoid output buffering)")
    print("-" * 50)


def test_acados_method4(
    show_plot=True,
    unified=False,
    use_control_rate_smooth=False,
    verbose_solve=True,
    waypoint_terminal_cost=True,
    use_actuator_dynamics=False,
    actuator_tau=None,
    interp_initial_guess=False,
):
    """
    **Method 4:** fixed physical horizon per segment, ``build_acados_ocp`` (NONLINEAR_LS).

    ``unified=True``: single OCP via ``solve_with_acados_waypoints_unified``.
    Waypoint format: ``[x, y, z, yaw_deg, time]``.
    """
    dt, waypoints, m, I, r_thrust, weights, bounds = _acados_cli_demo_problem(
        use_control_rate_smooth=use_control_rate_smooth,
        use_actuator_dynamics=use_actuator_dynamics,
        actuator_tau=actuator_tau,
        waypoint_terminal_cost=waypoint_terminal_cost,
    )
    weights = dict(weights)
    weights.setdefault("acados_objective", "tracking")

    extra = [
        "Formulation: Method 4 (fixed Tf, nonlinear LS)",
        "Mode: unified (single OCP)" if unified else "Mode: segment (per-segment OCP)",
    ]
    _acados_cli_print_scene_preamble(
        f"Acados Method 4 — {len(waypoints)} waypoints, {len(waypoints) - 1} segments",
        dt, waypoints, weights, bounds, verbose_solve, extra_lines=extra,
    )

    solver_fn = solve_with_acados_waypoints_unified if unified else solve_with_acados_waypoints
    solve_kw = dict(
        dt=dt, waypoints=waypoints, m=m, I=I, r_thrust=r_thrust,
        weights=weights, bounds=bounds, verbose_solve=verbose_solve,
    )
    if unified:
        solve_kw["interp_initial_guess"] = interp_initial_guess
    _ret = solver_fn(**solve_kw)
    xs, us, loggers, us_actual = _ret[:4]

    _acados_cli_print_solve_summary(dt, waypoints, xs, us, loggers)
    if show_plot:
        _plot_result(
            xs, us, waypoints, dt, loggers, bounds, us_actual=us_actual,
            suptitle="TVC Trajectory - Acados (Method 4)",
        )
    return xs, us, loggers, us_actual


def test_acados_method5(
    show_plot=True,
    verbose_solve=True,
    waypoint_terminal_cost=True,
    use_actuator_dynamics=False,
    actuator_tau=None,
    use_control_rate_smooth=False,
):
    """
    **Method 5:** minimum time per segment, ``build_acados_ocp_min_time`` (pseudo-time + ``T_seg`` state).
    Returns ``(xs, us, loggers, us_actual, meta)`` with ``meta['optimal_segment_times']``.
    """
    dt, waypoints, m, I, r_thrust, base_w, bounds = _acados_cli_demo_problem(
        use_control_rate_smooth=use_control_rate_smooth,
        use_actuator_dynamics=use_actuator_dynamics,
        actuator_tau=actuator_tau,
        waypoint_terminal_cost=waypoint_terminal_cost,
    )
    weights = dict(base_w)
    weights["acados_objective"] = "min_time"
    weights.setdefault("min_time_T_min", 0.15)
    weights.setdefault("min_time_T_max_scale", 1.0)
    weights.setdefault("min_time_weight", 1.0)

    extra = [
        "Formulation: Method 5 (min-time, free segment duration T_seg)",
    ]
    _acados_cli_print_scene_preamble(
        f"Acados Method 5 — {len(waypoints)} waypoints, {len(waypoints) - 1} segments",
        dt, waypoints, weights, bounds, verbose_solve, extra_lines=extra,
    )

    _ret = solve_with_acados_waypoints(
        dt=dt, waypoints=waypoints, m=m, I=I, r_thrust=r_thrust,
        weights=weights, bounds=bounds, verbose_solve=verbose_solve,
    )
    xs, us, loggers, us_actual, meta = _ret
    plot_dt = float(meta.get("plot_dt", dt))
    opt_T = meta.get("optimal_segment_times", [])
    if opt_T:
        print(f"  Optimal segment times T*: {[float(t) for t in opt_T]} s")

    _acados_cli_print_solve_summary(plot_dt, waypoints, xs, us, loggers)
    if show_plot:
        _plot_result(
            xs, us, waypoints, dt, loggers, bounds, us_actual=us_actual,
            suptitle="TVC Trajectory - Acados (Method 5 min-time)",
            plot_dt=plot_dt,
            plot_meta=meta,
        )
    return xs, us, loggers, us_actual, meta


def test_acados_method7(
    show_plot=True,
    verbose_solve=True,
    waypoint_terminal_cost=True,
    use_actuator_dynamics=False,
    actuator_tau=None,
    use_control_rate_smooth=False,
):
    """
    **Method 7:** free final time with **EXTERNAL** cost, ``build_acados_ocp_free_tf_external``.
    Returns ``(xs, us, loggers, us_actual, meta)``.
    """
    dt, waypoints, m, I, r_thrust, base_w, bounds = _acados_cli_demo_problem(
        use_control_rate_smooth=use_control_rate_smooth,
        use_actuator_dynamics=use_actuator_dynamics,
        actuator_tau=actuator_tau,
        waypoint_terminal_cost=waypoint_terminal_cost,
    )
    weights = dict(base_w)
    weights["acados_objective"] = "free_tf_external"
    weights.setdefault("min_time_T_min", 0.15)
    weights.setdefault("min_time_T_max_scale", 1.0)
    weights.setdefault("free_tf_w_T", 1.0)
    weights.setdefault("free_tf_w_tvc", 0.01)
    weights.setdefault("free_tf_w_tau_yaw", 0.01)
    weights.setdefault("free_tf_w_terminal_time", 10.0)
    weights.setdefault("free_tf_include_state_terminal", True)

    extra = [
        "Formulation: Method 7 (free tf state, EXTERNAL running/terminal cost)",
    ]
    _acados_cli_print_scene_preamble(
        f"Acados Method 7 — {len(waypoints)} waypoints, {len(waypoints) - 1} segments",
        dt, waypoints, weights, bounds, verbose_solve, extra_lines=extra,
    )

    _ret = solve_with_acados_waypoints(
        dt=dt, waypoints=waypoints, m=m, I=I, r_thrust=r_thrust,
        weights=weights, bounds=bounds, verbose_solve=verbose_solve,
    )
    xs, us, loggers, us_actual, meta = _ret
    plot_dt = float(meta.get("plot_dt", dt))
    opt_T = meta.get("optimal_segment_times", [])
    if opt_T:
        print(f"  Optimal segment tf*: {[float(t) for t in opt_T]} s")

    _acados_cli_print_solve_summary(plot_dt, waypoints, xs, us, loggers)
    if show_plot:
        _plot_result(
            xs, us, waypoints, dt, loggers, bounds, us_actual=us_actual,
            suptitle="TVC Trajectory - Acados (Method 7 free-tf EXTERNAL)",
            plot_dt=plot_dt,
            plot_meta=meta,
        )
    return xs, us, loggers, us_actual, meta


def _acados_cli_print_solve_summary(dt, waypoints, xs, us, loggers):
    print(f"Solve done: {len(xs)} state points, {len(us)} controls, {len(loggers)} segments")
    print(f"Start position: {xs[0][:3]}")
    end_wp = waypoints[-1]
    print(f"End position: {xs[-1][:3]} (target: {end_wp[:3]})")
    err_end = np.linalg.norm(np.array(xs[-1][:3]) - np.array(end_wp[:3]))
    print(f"  End error: {err_end:.4f}m")
    if len(waypoints) >= 3:
        dt_chk = dt
        t0 = waypoints[0][4] if len(waypoints[0]) >= 5 else 0.0
        for i in range(1, len(waypoints) - 1):
            idx = int((waypoints[i][4] - t0) / dt_chk)
            if idx < len(xs):
                at_wp = xs[idx][:3]
                err = np.linalg.norm(np.array(at_wp) - np.array(waypoints[i][:3]))
                print(f"  WP{i} at: {at_wp} | error: {err:.4f}m")
    for i, lg in enumerate(loggers):
        if lg and lg.costs:
            print(f"  Segment {i+1} final cost: {lg.costs[-1]:.6e}")
    print("=" * 50)


def test_three_waypoints(show_plot=True, unified=False, use_control_rate_smooth=False, verbose_solve=True,
                        waypoint_terminal_cost=True, use_actuator_dynamics=False, actuator_tau=None,
                        interp_initial_guess=False):
    """
    Backward-compatible alias: same as ``test_acados_method4`` (fixed-time Acados / Method 4).
    """
    return test_acados_method4(
        show_plot=show_plot,
        unified=unified,
        use_control_rate_smooth=use_control_rate_smooth,
        verbose_solve=verbose_solve,
        waypoint_terminal_cost=waypoint_terminal_cost,
        use_actuator_dynamics=use_actuator_dynamics,
        actuator_tau=actuator_tau,
        interp_initial_guess=interp_initial_guess,
    )


def _plot_result(xs, us, waypoints, dt, loggers, bounds=None, us_actual=None, suptitle=None, plot_dt=None,
                 plot_meta=None):
    """Plot trajectory with the same dashboard layout as tvc_traj_opt_gui.
    ``plot_dt``: optional average grid step (e.g. Method 5/7 ``meta['plot_dt']``) when not using piecewise times.
    ``plot_meta``: optional dict with ``optimal_segment_times``, ``segment_boundary_indices`` (min-time / multi-seg).
    """
    _script_dir = Path(__file__).resolve().parent
    if str(_script_dir) not in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if _script_dir not in sys.path:
            sys.path.insert(0, str(_script_dir))
    from tvc_common import segment_boundaries_from_waypoints, physical_time_grid_per_shooting_segment
    dt_grid = float(plot_dt) if plot_dt is not None else float(dt)
    sbo_meta = (plot_meta or {}).get("segment_boundary_indices")
    ots_meta = (plot_meta or {}).get("optimal_segment_times")
    time_states = None
    boundaries = None
    if (
        ots_meta
        and sbo_meta
        and len(ots_meta) == len(sbo_meta)
        and len(xs) == int(sbo_meta[-1]) + 1
    ):
        time_states = physical_time_grid_per_shooting_segment(ots_meta, sbo_meta)
        boundaries = [min(int(b), len(xs) - 1) for b in sbo_meta]
    if boundaries is None:
        boundaries = [min(b, len(xs) - 1) for b in segment_boundaries_from_waypoints(waypoints or [], dt_grid)]
    if boundaries:
        print(f"  Segment boundary indices: {boundaries} ({len(boundaries)} segments)")
    wp_list = waypoints if waypoints and all(len(wp) >= 5 for wp in waypoints) else [
        [wp[0], wp[1], wp[2], 0.0, i * dt_grid] for i, wp in enumerate(waypoints or [])
    ]

    if loggers:
        print("Cost curve (before plot):")
        for i, lg in enumerate(loggers):
            if lg and lg.costs:
                print(f"  Segment {i+1}: n={len(lg.costs)}, cost[0]={lg.costs[0]:.6e}, cost[-1]={lg.costs[-1]:.6e}")
                if len(lg.costs) <= 20:
                    print(f"    costs = {[f'{c:.4e}' for c in lg.costs]}")
                else:
                    print(f"    costs[:5] = {[f'{c:.4e}' for c in lg.costs[:5]]} ... costs[-3:] = {[f'{c:.4e}' for c in lg.costs[-3:]]}")

    from tvc_traj_gui_plots import plot_gui_style_results
    solve_meta = {"plot_dt": dt_grid, "dt_grid": dt_grid}
    plot_gui_style_results(
        xs, us, dt_grid,
        waypoints=wp_list,
        optimization_bounds=bounds,
        all_loggers=loggers,
        suptitle=suptitle or "TVC Trajectory - Acados (Method 4)",
        show=True,
        us_actual=us_actual,
        solve_meta=solve_meta,
        segment_boundaries_override=boundaries if time_states is not None else None,
        time_states=time_states,
    )


if __name__ == "__main__":
    if not ACADOS_AVAILABLE:
        print("Acados not available. Install: pip install casadi; build acados from source.")
        print("See: https://docs.acados.org/installation/")
        exit(1)

    import argparse
    parser = argparse.ArgumentParser(description="TVC Acados trajectory optimization test")
    parser.add_argument(
        "--method",
        type=int,
        choices=(4, 5, 7),
        default=4,
        help="4=fixed Tf LS (build_acados_ocp); 5=min-time T_seg (build_acados_ocp_min_time); "
        "7=free tf EXTERNAL (build_acados_ocp_free_tf_external)",
    )
    parser.add_argument("--no-plot", action="store_true", default=False, help="Do not show plot")
    parser.add_argument("--unified", action="store_true", default=False,
                        help="Method 4 only: single OCP (ignored for Method 5/7)")
    parser.add_argument("--no-waypoint-terminal", action="store_true",
                        help="Do not enforce waypoint terminal cost. With --unified keeps unified; else multi-WP uses segment mode")
    parser.add_argument("--smooth", action="store_true", help="Add control rate penalty for smoother u")
    parser.add_argument("--actuator", action="store_true", default=False,
                        help="Enable first-order actuator dynamics (tau*u_dot = u_cmd - u_actual per channel)")
    parser.add_argument("--actuator-tau", type=str, default="0.5,0.5,1.0,0.1",
                        help="Actuator time constants [pitch,roll,T,yaw] in seconds (default: 0.05,0.05,0.05,0.05)")
    parser.add_argument("--quiet", action="store_true", help="Do not print SQP iteration stats and cost")
    parser.add_argument("--interp-initial-guess", action="store_true", default=True,
                        help="Unified mode only: linearly interpolate x initial guess along each segment")
    args = parser.parse_args()

    actuator_tau_list = [float(x.strip()) for x in args.actuator_tau.split(",") if x.strip()]
    if args.actuator and len(actuator_tau_list) < 4:
        actuator_tau_list = actuator_tau_list + [0.05] * (4 - len(actuator_tau_list))

    common_kw = dict(
        show_plot=not args.no_plot,
        verbose_solve=not args.quiet,
        waypoint_terminal_cost=not args.no_waypoint_terminal,
        use_actuator_dynamics=args.actuator,
        actuator_tau=actuator_tau_list,
        use_control_rate_smooth=args.smooth,
    )
    if args.method in (5, 7) and args.unified:
        print("Note: --unified applies to Method 4 only; ignored for Method 5/7.")
    if args.method == 4:
        test_acados_method4(
            unified=args.unified,
            interp_initial_guess=args.interp_initial_guess,
            **common_kw,
        )
    elif args.method == 5:
        test_acados_method5(**common_kw)
    else:
        test_acados_method7(**common_kw)
