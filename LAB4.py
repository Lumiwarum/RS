import numpy as np
import sympy as sp
from scipy.integrate import odeint
from visualize import visualize_double_pendulum
import matplotlib.pyplot as plt
# Define kinematics parameters
lengths = 0.8, 1.2  # length of pendulums

# Add your dynamics parameters here
masses = [1, 1]

g= 9.81

M = lambda theta1, theta2: np.array([[(masses[0]+masses[1])* (lengths[0] **2)+ masses[1]*(lengths[1]**2)+2* masses[1]*lengths[0]*lengths[1]*np.cos(theta2), masses[1]*(lengths[1]**2)+masses[1]*lengths[0]*lengths[1]*np.cos(theta2)],
                                     [ masses[1]*(lengths[1]**2)+masses[1]*lengths[0]*lengths[1]*np.cos(theta2), masses[1] * (lengths[1]**2) ]])

C = lambda theta1, theta2, dtheta1, dtheta2: np.array([[0, -masses[1]*lengths[0]*lengths[1]*np.sin(theta2)*(2*dtheta1+dtheta2)],
                                                       [(1/2)*masses[1]*lengths[0]*lengths[1]*np.sin(theta2)*(2*dtheta1+dtheta2),-(1/2)*masses[1]*lengths[0]*lengths[1]*np.sin(theta2)*(dtheta1)]])

g_func = lambda theta1, theta2: -g * np.array([[(masses[0]+masses[1])*lengths[0] * np.sin(theta1)+ masses[1]*lengths[1]*np.sin(theta1+theta2)],
                                               [ masses[1]*lengths[1]*np.sin(theta1+theta2)]])


fric_coeff = [0.9, 0.9]
friction = lambda dtheta1, dtheta2: np.array([[fric_coeff[0]* dtheta1],
                                              [fric_coeff[1]* dtheta2]])
# the time parameters
t_stop = 50  # how many seconds to simulate
dt = 0.01
t = np.arange(0, t_stop, dt)
# print(t)
# define angles
#

def system(t, state):
    q1, q2, dq1, dq2 = state
    dq = np.array([[dq1], [dq2]])
    M_cur = M(q1, q2)
    C_cur = C(q1, q2, dq1, dq2)
    g_cur = g_func(q1, q2)
    fric_cur = friction(dq1, dq2)
    ddq1, ddq2 = np.linalg.inv(M_cur) @(-C_cur@dq -g_cur)#-fric_cur)
    return dq1, dq2, ddq1, ddq2

state0 = np.array([np.pi/2, 0, 0, 0])
sol = odeint(system,state0,t, tfirst = True)
print(sol.shape)

thetas = np.pi+np.sin(2*t), np.cos(2*t)

# calculate cartesian points of joints via forward kinematics
# points_1 = lengths[0]*np.sin(thetas[0]), lengths[0]*np.cos(thetas[0])
# points_2 = lengths[1]*np.sin(thetas[0]+thetas[1]) + \
#     points_1[0], lengths[1]*np.cos(thetas[0]+thetas[1]) + points_1[1]
# joint_points = [points_1, points_2]


points_1 = lengths[0]*np.sin(sol[:,0]), lengths[0]*np.cos(sol[:,0])
points_2 = lengths[1]*np.sin(sol[:,0]+sol[:,1]) + \
    points_1[0], lengths[1]*np.cos(sol[:,0]+sol[:,1]) + points_1[1]
joint_points = [points_1, points_2]

T = (1/2) *((masses[0]+masses[1])* (lengths[0]**2) * (sol[:,2]**2)+masses[1]* (lengths[1]**2) * (sol[:,3]+sol[:,2])**2) + masses[1]*lengths[0]*lengths[1]*sol[:,2]*(sol[:,2]+sol[:,3])*np.cos(sol[:,1])
U= -(masses[0]+masses[1])*g*lengths[0] * np.cos(sol[:,0])- masses[1]*g*lengths[1] * np.cos(sol[:,0]+sol[:,1])

plt.plot(t,T)
plt.plot(t,U)
plt.show()
# Stats to print
stats = [[r'time $t$ ', t],
         [r'norm $\|\mathbf{r}_e\|$ ', np.linalg.norm(np.array(points_2), axis=0)]]  # put energy here

# animate motion of double pendulum
visualize_double_pendulum(joint_points,
                          stats=stats,
                          save=False,  # save as html animation
                          axes=False,
                          show=True,
                          trace_len=0.2,
                          dt=dt)
