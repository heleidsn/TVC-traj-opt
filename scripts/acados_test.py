import os
import numpy as np
import casadi as ca

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_EXPORT_DIR = os.path.join(_THIS_DIR, "c_generated_code", "acados_test")

# =========================
# 1) Model
# =========================
model = AcadosModel()
model.name = "minimal_integrator"

# State and control
x = ca.SX.sym("x", 1)
u = ca.SX.sym("u", 1)

# Explicit dynamics: x_dot = u
xdot = ca.SX.sym("xdot", 1)
f_expl = u
f_impl = xdot - f_expl

model.x = x
model.u = u
model.xdot = xdot
model.f_expl_expr = f_expl
model.f_impl_expr = f_impl

# =========================
# 2) OCP
# =========================
ocp = AcadosOcp()
ocp.model = model

# Horizon
N = 20
Tf = 2.0
ocp.dims.N = N
ocp.solver_options.tf = Tf

# =========================
# 3) cost: LINEAR_LS
#    y = [x; u]
# =========================
ocp.cost.cost_type = "LINEAR_LS"
ocp.cost.cost_type_e = "LINEAR_LS"

# running cost: y = Vx*x + Vu*u = [x; u]
ocp.cost.Vx = np.array([[1.0],
                        [0.0]])
ocp.cost.Vu = np.array([[0.0],
                        [1.0]])
ocp.cost.W = np.diag([1.0, 0.1])
ocp.cost.yref = np.array([0.0, 0.0])

# terminal cost: y_e = x
ocp.cost.Vx_e = np.array([[1.0]])
ocp.cost.W_e = np.array([[10.0]])
ocp.cost.yref_e = np.array([0.0])

# =========================
# 4) Constraints
# =========================
x0 = np.array([1.0])
ocp.constraints.x0 = x0

# Optional: control bounds -1 <= u <= 1
ocp.constraints.lbu = np.array([-1.0])
ocp.constraints.ubu = np.array([ 1.0])
ocp.constraints.idxbu = np.array([0])

# =========================
# 5) Solver options
# =========================
ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
ocp.solver_options.integrator_type = "ERK"
ocp.solver_options.nlp_solver_type = "SQP"

# =========================
# 6) Build solver and solve
# =========================
os.makedirs(_CODE_EXPORT_DIR, exist_ok=True)
try:
    ocp.code_gen_opts.code_export_directory = _CODE_EXPORT_DIR
except AttributeError:
    ocp.code_export_directory = _CODE_EXPORT_DIR
solver = AcadosOcpSolver(ocp, json_file=os.path.join(_CODE_EXPORT_DIR, "acados_ocp.json"))

status = solver.solve()
print("status =", status)

# =========================
# 7) Extract solution
# =========================
x_traj = np.zeros((N+1, 1))
u_traj = np.zeros((N, 1))

for i in range(N):
    x_traj[i] = solver.get(i, "x")
    u_traj[i] = solver.get(i, "u")
x_traj[N] = solver.get(N, "x")

print("x_traj =\n", x_traj)
print("u_traj =\n", u_traj)

# plot the trajectory
import matplotlib.pyplot as plt
plt.plot(x_traj)
plt.plot(u_traj)
plt.show()