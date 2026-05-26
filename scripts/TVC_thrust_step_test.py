import numpy as np
import matplotlib.pyplot as plt


def first_order_response(t, u, tau):
    """
    Simulate first-order system:
        tau * y_dot + y = u
    using forward Euler integration.
    """
    y = np.zeros_like(u)
    dt = t[1] - t[0]

    for k in range(1, len(t)):
        y[k] = y[k-1] + dt * (u[k-1] - y[k-1]) / tau

    return y


def build_step_profile(total_time, dt, initial_value, events):
    """
    Build a piecewise-constant input profile.

    Parameters
    ----------
    total_time : float
        Total simulation time [s]
    dt : float
        Sampling time [s]
    initial_value : float
        Initial input value
    events : list of tuples
        Each tuple is (time, new_value), meaning from 'time' onward
        the command becomes 'new_value'
    """
    t = np.arange(0, total_time + dt, dt)
    u = np.ones_like(t) * initial_value

    for event_time, new_value in events:
        u[t >= event_time] = new_value

    return t, u


def experiment_A(step_interval=5.0, step_size_1=20.0, step_size_2=40.0):
    """
    Experiment A:
    5 s:   0 -> 200
    then every step_interval s:
    +step_size_1 x2, -step_size_1 x4, +step_size_1 x2,
    then +step_size_2 x1, -step_size_2 x2, +step_size_2 x1
    """
    hover = 200.0
    events = []

    current_time = 5.0
    current_thrust = 0.0

    # Jump to hover
    current_thrust = hover
    events.append((current_time, current_thrust))

    def add_step(delta):
        nonlocal current_time, current_thrust, events
        current_time += step_interval
        current_thrust += delta
        events.append((current_time, current_thrust))

    # +step_size_1 x2
    for _ in range(2):
        add_step(+step_size_1)

    # -step_size_1 x4
    for _ in range(4):
        add_step(-step_size_1)

    # +step_size_1 x2
    for _ in range(2):
        add_step(+step_size_1)

    # +step_size_2 x1
    add_step(+step_size_2)

    # -step_size_2 x2
    for _ in range(2):
        add_step(-step_size_2)

    # +step_size_2 x1
    add_step(+step_size_2)

    total_time = current_time + 8.0
    # step_size_1: 8 steps, step_size_2: 4 steps (first step_size_2 at 5 + 9*step_interval)
    t_s1_start = 5.0 + step_interval
    t_s1_end = 5.0 + 9 * step_interval  # start of first step_size_2 step
    regions = [(t_s1_start, t_s1_end, step_size_1), (t_s1_end, total_time, step_size_2)]
    return build_step_profile(total_time, 0.01, 0.0, events), regions


def experiment_B(step_interval=5.0, step_size_1=20.0, step_size_2=40.0):
    """
    Experiment B:
    5 s:   0 -> 200
    then first the 40N section:
        +step_size_2 x1, -step_size_2 x2, +step_size_2 x1
    then the 20N section:
        +step_size_1 x2, -step_size_1 x4, +step_size_1 x2
    """
    hover = 200.0
    events = []

    current_time = 5.0
    current_thrust = 0.0

    # Jump to hover
    current_thrust = hover
    events.append((current_time, current_thrust))

    def add_step(delta):
        nonlocal current_time, current_thrust, events
        current_time += step_interval
        current_thrust += delta
        events.append((current_time, current_thrust))

    # 40N section first
    add_step(+step_size_2)
    add_step(-step_size_2)
    add_step(-step_size_2)
    add_step(+step_size_2)

    # 20N section
    for _ in range(2):
        add_step(+step_size_1)

    for _ in range(4):
        add_step(-step_size_1)

    for _ in range(2):
        add_step(+step_size_1)

    total_time = current_time + 8.0
    # step_size_2: 4 steps, step_size_1: 8 steps (first step_size_1 at 5 + 5*step_interval)
    t_s2_start = 5.0 + step_interval
    t_s2_end = 5.0 + 5 * step_interval  # start of first step_size_1 step
    regions = [(t_s2_start, t_s2_end, step_size_2), (t_s2_end, total_time, step_size_1)]
    return build_step_profile(total_time, 0.01, 0.0, events), regions


def plot_experiment(t, u, y, title, regions=None):
    plt.figure(figsize=(10, 4.8))
    ax = plt.gca()
    if regions is not None:
        deltas = sorted(set(d for _, _, d in regions))
        colors = {deltas[0]: (0.85, 0.92, 1.0), deltas[-1]: (1.0, 0.92, 0.85)}  # small step: light blue, large step: light orange
        for t_start, t_end, delta in regions:
            ax.axvspan(t_start, t_end, facecolor=colors.get(delta, (0.95, 0.95, 0.95)), alpha=0.7)
    plt.plot(t, u, label='Commanded thrust')
    plt.plot(t, y, label='Actual thrust (1st-order response)')
    plt.xlabel('Time [s]')
    plt.ylabel('Thrust [N]')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


if __name__ == "__main__":
    tau = 0.5  # Example time constant [s]
    step_interval = 5.0  # Time between thrust steps [s]
    step_size_1 = 10.0  # Small thrust step magnitude [N]
    step_size_2 = 20.0  # Large thrust step magnitude [N]

    # Experiment A
    (tA, uA), regions_A = experiment_A(step_interval, step_size_1, step_size_2)
    yA = first_order_response(tA, uA, tau)
    plot_experiment(tA, uA, yA, f'Experiment A: {step_size_1}N steps first, then {step_size_2}N steps', regions_A)

    # Experiment B
    (tB, uB), regions_B = experiment_B(step_interval, step_size_1, step_size_2)
    yB = first_order_response(tB, uB, tau)
    plot_experiment(tB, uB, yB, f'Experiment B: {step_size_2}N steps first, then {step_size_1}N steps', regions_B)

    plt.show()