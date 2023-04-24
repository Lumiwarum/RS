import numpy as np
import sympy as sp 
from scipy.integrate import odeint
from visualize import visualize_double_pendulum
from matplotlib import pyplot as plt
# Define kinematics parameters
l1, l2 = 1, 0.8  # length of pendulums
m1, m2 = 0.5, 1
k = 200
g = 9.81

t = sp.symbols('t')
q1 = sp.Function('q_1')(t)
q2 = sp.Function('q_2')(t)
q3 = sp.Function('q_3')(t)

q = sp.Matrix([q1, q2, q3])
dq = q.diff(t)

r1 = sp.Matrix([
    l1 * sp.cos(q1),
    l1 * sp.sin(q1)
])
r2 = sp.Matrix([
    (l2+q3) * sp.cos(q2) + r1[0],
    (l2+q3) * sp.sin(q2) + r1[1]
])

J_r1 = sp.Matrix([
    [-l1 * sp.sin(q1), 0, 0],
    [l1 * sp.cos(q1), 0, 0]
])

J_r2 = sp.Matrix([
    [-l1 * sp.sin(q1), -1*(l2+q3) * sp.sin(q2), sp.cos(q2)],
    [l1 * sp.cos(q1), (l2+q3) * sp.cos(q2), sp.sin(q2)]
])

assert J_r1 == r1.jacobian(q) 
assert J_r2.simplify() == r2.jacobian(q).simplify()

r1_dot = sp.Matrix([
    -l1 * sp.sin(q1) * q1.diff(t),
    l1 * sp.cos(q1) * q1.diff(t)
])

r2_dot = sp.Matrix([
    -(l2+q3) * sp.sin(q2) * q2.diff(t) - l1 * sp.sin(q1) * q1.diff(t) + sp.cos(q2) * q3.diff(t),
    (l2+q3) * sp.cos(q2) * q2.diff(t) + l1 * sp.cos(q1) * q1.diff(t) + sp.sin(q2) * q3.diff(t)
])

assert r1_dot == J_r1 @ dq 
assert r2_dot == J_r2 @ dq

M_q = m1 * J_r1.T @ J_r1 + m2 * J_r2.T @ J_r2
M_q.simplify()
K = (1/2 * dq.T @ M_q @ dq)[0]
# K2 = (1/2 * m1 * r1_dot.T @ r1_dot + 1/2 * m2 * r2_dot.T @ r2_dot)[0]
# They are equal
# assert (K - K2).simplify() == 0

P = (m1 * g * r1[1] + m2 * g * r2[1]+ 1/2 * k * q3**2)

L = K - P


# -------------------------------------------------------------
# dLddq = L.diff(dq)
# assert (dLddq - M_q @ dq).simplify() == sp.Matrix([0, 0, 0])

# dLddqdt = dLddq.diff(t)
# assert (dLddqdt - M_q.diff(t) @ dq - M_q @ dq.diff(t)).simplify() == sp.Matrix([0, 0, 0])

C = M_q.diff(t) @ dq - K.diff(q)
C.simplify()
g_q = P.diff(q)
g_q.simplify()

b1, b2, b3 = 0.2, 0.2, 0.2
Q = sp.Matrix([
    0,
    0,
    0
])


ddq = M_q.inv() @ (-C - g_q + Q)

lambda_ddq = sp.lambdify((*q, *dq), ddq)
lambda_U = sp.lambdify(((*q, *dq)), K + P)

def f(x, t):
    # x = (q1, q2, dq1, dq2)
    q1, q2, q3, dq1, dq2, dq3 = x
    return (dq1, dq2, dq3, *lambda_ddq(*x))

# the time parameters
t_stop = 10  # how many seconds to simulate
dt = 1/200
t = np.arange(0, t_stop, dt)

# define angles
q0 = [0, 0, 0, 0, 0, 0]
q_sol = np.array(odeint(f, q0, t))

print(q_sol.shape)

# calculate cartesian points of joints via forward kinematics
points_1 = l1*np.cos(q_sol[:, 0]), l1*np.sin(q_sol[:, 0])
points_2 = (l2+q_sol[:, 2])*np.cos(q_sol[:, 1]) + points_1[0], (l2+q_sol[:, 2])*np.sin(q_sol[:, 1]) + points_1[1]

joint_points = [points_1, points_2]

# Stats to print
stats = [[r'time $t$ ', t],
         [r'norm $\|\mathbf{r}_e\|$ ', np.linalg.norm(np.array(points_2), axis=0)]]  # put energy here

plt.plot(t, q_sol)
plt.legend(['q1', 'q2', 'q3', 'dq1', 'dq2', 'dq3'])
plt.savefig('2.5.png')
plt.show()
plt.plot(t, lambda_U(q_sol[:, 0], q_sol[:, 1], q_sol[:, 2], q_sol[:, 3], q_sol[:, 4], q_sol[:, 5]))
plt.legend(['U'])
plt.savefig('2.4.png')


# animate motion of double pendulum
visualize_double_pendulum(joint_points,
                          stats=stats,
                          save=False,  # save as html animation
                          axes=False,
                          show=True,
                          trace_len=0.2,
                          dt=dt)