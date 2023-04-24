import numpy as np
import matplotlib.pyplot as plt
from visualize import visualize_rotation


def quat_product(p, q):
  pq = np.concatenate(([p[0]*q[0]-p[1:4].T @ q[1:4]], p[0]*q[1:4]+ q[0]*p[1:4]+np.cross(p[1:4],q[1:4])))
  return pq

# transfomration from quaternion to rotation matrix
def quat2rot(q):
    q0, q1, q2, q3 = q
    rotation_matrix = np.array(
        [[1 - 2 * q1 ** 2 - 2 * q2 ** 2, 2 * q0 * q1 - 2 * q2 * q3, 2 * q0 * q2 + 2 * q1 * q3],
         [2*q0*q1+2*q2*q3, 1 - 2 * q0 ** 2 - 2 *q2 ** 2, 2 * q1 * q2 - 2 * q0 * q3],
         [2 * q0 * q2 - 2 * q1 * q3, 2 * q2 * q1 + 2 * q0 * q3, 1 - 2 * q0 ** 2 - 2 * q1 ** 2]]
    )
    return rotation_matrix

m = 0.1 # [kg] - mass of the motors 
M = 0.5 # [kg] mass of the drone body
h = 0.08 # [m] high of the body
R = 0.05 # [m] radius of the body
l = 0.3 # [m] length of arm 
k_t = 0.05 # thrust coefficient 
k_m = 0.001 # torque coefficient 

B = np.array([[0, l*k_t, 0, -l*k_t],
              [-l*k_t, 0, l*k_t, 0],
              [k_m, -k_m, k_m, -k_m]])

Jx = M * (1/12) * (h**2 + 3* R **2) +2 * (l**2) *m
Jy = M * (1/12) * (h**2 + 3* R **2) +2 * (l**2) *m
Jz = M * (1/2) * (R **2) +4 * (l**2) *m

J = np.diag([Jx, Jy, Jz])

def prop_torque(pro_speed):
  return B @ pro_speed

def quad_system(t, state, pro_speed):
  # processing the state
  q1, q2, q3, q4, w1, w2, w3 =state
  Q = np.array([q1, q2, q3, q4])
  Omega = np.array([w1, w2, w3])

  torque = prop_torque(pro_speed)

  # dynamics
  dQ = quat_product((1/2) * Q, np.concatenate(([0], Omega)))
  domega = np.linalg.inv(J) @ (torque-np.cross(Omega, J @ Omega))

  return np.concatenate((dQ,  domega))

import math
# runge-kutta fourth-order numerical integration
def rk4(func, tk, _yk, _dt, pro_speed):
  
    # evaluate derivative at several stages within time interval
    f1 = func(tk, _yk, pro_speed)
    f2 = func(tk + _dt / 2, _yk + (f1 * (_dt / 2)), pro_speed)
    f3 = func(tk + _dt / 2, _yk + (f2 * (_dt / 2)), pro_speed)
    f4 = func(tk + _dt, _yk + (f3 * _dt), pro_speed)

    # return an average of the derivative over tk, tk + dt
    return _yk + (_dt / 6) * (f1 + (2 * f2) + (2 * f3) + f4)

def normalize(v, tolerance=0.00001):
    mag2 = sum(n * n for n in v)
    if abs(mag2 - 1.0) > tolerance:
        mag = math.sqrt(mag2)
        v = tuple(n / mag for n in v)
    return np.array(v)

def plot_graphs(t, sol, pro_sp):
  sol = np.array(sol)
  fig = plt.figure(figsize = (20, 10))
  axs = fig.subplots(2, 2)
  for i in range(2):
    for j in range(2):
      axs[i, j].set_title(f'q{i+2*j}')
      axs[i, j].plot(t, sol[:,i+2*j], 'b', label='actual')
      #axs[i, j].plot(error_t, error[:, i+2*j], 'r', label='error')
      #axs[i, j].plot(t, qd[:, i+2*j], 'b--', label='taget')
      axs[i, j].legend()
  fig.suptitle(f'speed = {pro_sp}')
  plt.show()

def simulate_quadrotor(propellers_speeds, t):
  Q0 = np.array([1, 0, 0, 0])
  omega0 = np.array([0, 0, 0])
  
  
  state0 = np.concatenate((Q0, omega0))
  
  
  dt = t[1]-t[0]
  sol = []
  for i, time in enumerate(t):
    state0 = rk4(quad_system, time, state0, dt, propellers_speeds)
    Q = state0[:4]
    Q = normalize(Q)
    state0[:4] = Q
    sol.append(state0)
  return sol


U1 = 3
U2 = U1 *2
propeller_speeds = np.array([U1, U2, U1, U2]) * 100
t = np.linspace(0, 5, 5000)

sol = simulate_quadrotor(propeller_speeds, t)
plot_graphs(t, sol, propeller_speeds)

N = len(t)
dt = t[1]-t[0]
# 
rm = np.zeros([N, 3, 3])
quaternion_norm = np.zeros(N)


for i, time in enumerate(t):
  quat = sol[i][:4]
  quat_norm = np.linalg.norm(quat)
  quaternion_norm[i] = quat_norm
  rm[i, :, :] = quat2rot(quat)  # np.eye(3)




# ////////////////////////////////////////////////
# animate the rotation matrix and two scalar stats

stats = [[r'time $t$ ', t],
         [r'quat norm $\|q\|$ ', quaternion_norm]]

visualize_rotation(rm,
                   stats=stats, 
                   save = True, 
                   dt = dt)