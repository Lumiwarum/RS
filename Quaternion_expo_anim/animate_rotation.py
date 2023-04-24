import numpy as np
from visualize import visualize_rotation


def quat_product(p, q):
  pq = np.concatenate(([p[0]*q[0]-p[1:4].T @ q[1:4]], p[0]*q[1:4]+ q[0]*p[1:4]+np.cross(p[1:4],q[1:4])))
  return pq

def quaternion_exp(q):
  vnorm = np.linalg.norm(q[1:4])
  q0 = np.exp(q[0]) * np.cos(vnorm)
  qv = (q0 * np.sin(vnorm) / vnorm) * q[1:4]
  return np.concatenate(([q0], qv))


def expo_sys(state, dt, omega_hat):
  new_omega = (1/2) * omega_hat * dt
  quat_omega = quaternion_exp(new_omega)
  return quat_product(state, quat_omega)


# transfomration from quaternion to rotation matrix
def quat2rot(q):
    q0, q1, q2, q3 = q
    rotation_matrix = np.array(
        [[1 - 2 * q1 ** 2 - 2 * q2 ** 2, 2 * q0 * q1 - 2 * q2 * q3, 2 * q0 * q2 + 2 * q1 * q3],
         [2*q0*q1+2*q2*q3, 1 - 2 * q0 ** 2 - 2 *q2 ** 2, 2 * q1 * q2 - 2 * q0 * q3],
         [2 * q0 * q2 - 2 * q1 * q3, 2 * q2 * q1 + 2 * q0 * q3, 1 - 2 * q0 ** 2 - 2 * q1 ** 2]]
    )
    return rotation_matrix

state0 = np.array([1, 0, 0, 5])
state0 = state0/(np.linalg.norm(state0)-1e-5)
print(np.linalg.norm(state0))

omega0 = np.array([np.pi, np.pi/2, np.pi/4])
omega_hat = np.concatenate(([0], omega0))
t = np.linspace(0, 10, 1000)

N = len(t)
# 
rm = np.zeros([N, 3, 3])
quaternion_norm = np.zeros(N)

dt = t[1]-t[0]
sol = []
axil_sol = []
for i, time in enumerate(t):
  state0 = expo_sys(state0, dt, omega_hat)
  sol.append(state0)
  quat = state0
  quat_norm = np.linalg.norm(quat)
    
    # store the quaternion norm and rotation matrices
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