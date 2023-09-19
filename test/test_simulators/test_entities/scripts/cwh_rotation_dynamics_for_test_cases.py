"""
# -------------------------------------------------------------------------------
# Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
# Reinforcement Learning Core (CoRL) Runtime Assurance Extensions
#
# This is a US Government Work not subject to copyright protection in the US.
#
# The use, dissemination or disclosure of data in this file is subject to
# limitation or restriction. See accompanying README and LICENSE for details.
# -------------------------------------------------------------------------------
"""

import numpy as np
from scipy.integrate import solve_ivp
import random

M = 12             # mass
J = 0.0573         # inertia
N = 0.001027       # mean motion
AVL = 0.03490685   # angular velocity limit

def cwh_derivative(t, state, u):
    x, y, theta, x_dot, y_dot, theta_dot = state
    ft, tz = u

    fx = ft*np.cos(theta)
    fy = ft*np.sin(theta)

    x_ddot = 2*N*y_dot + 3*(N**2)*x + fx/M
    y_ddot = -2*N*x_dot + fy/M
    theta_ddot = tz/J

    if theta_dot >= AVL:
        theta_dot = AVL
        theta_ddot = min(0,theta_ddot)
    elif theta_dot <= -AVL:
        theta_dot = -AVL
        theta_ddot = max(0,theta_ddot)

    state_dot = np.array([x_dot, y_dot, theta_dot, x_ddot, y_ddot, theta_ddot], dtype=float)
    return state_dot


def solve_cwh_traj(x0, u, t):
    state0 = np.array(x0, dtype=float)
    sol = solve_ivp(cwh_derivative, (0, t), state0, args=(u,), max_step=0.01)
    y = sol.y
    position_str = ', '.join([str(x) for x in y[0:2, -1]])
    # wrap the theta angle about 0
    wrapped_theta = wrap_angle(y[2],0)
    theta_str = ', '.join([str(wrapped_theta[len(wrapped_theta)-1])])
    velocity_str = ', '.join([str(x) for x in y[3:5, -1]])
    theta_dot_str = ', '.join([str(x) for x in y[5:, -1]])

    print(f"- position == `[{position_str}]`\n- theta == `[{theta_str}]`")
    print(f"- velocity == `[{velocity_str}]`\n- thetadot == `[{theta_dot_str}]`")

def wrap_angle(angle,center):
    """
    Wrap the angle about the requested center point
    input in radians
    """
    wrapped_angle = ((angle + np.pi) % (2 * np.pi)) - np.pi + center
    return wrapped_angle

#position = (np.random.rand(2) * 2000 - 1000).tolist()
#theta = (np.random.rand(1) * 2*2*np.pi - 2*np.pi).tolist()
#velocity = (np.random.rand(2) * 20 - 10).tolist()
#thetadot = (np.random.rand(1) * 2*0.0349 - 0.0349).tolist()
#thrust = (np.random.rand(1) * 2 - 1).tolist()
#moment = (np.random.rand(1) * 2*0.001 - 0.001).tolist()
#u = thrust + moment
#t = random.randint(10, 100)
position = np.array([0,0]).tolist()
velocity = np.array([0,0]).tolist()
theta = np.array([0]).tolist()
thetadot = np.array([0]).tolist()
u = np.array([-1,  -0.001]).tolist()
t = 10.0

print("initial conditions")
print(f"- position = `[{', '.join([str(x) for x in position])}]`")
print(f"- theta = `[{', '.join([str(x) for x in theta])}]`")
print(f"- velocity = `[{', '.join([str(x) for x in velocity])}]`")
print(f"- thetadot = `[{', '.join([str(x) for x in thetadot])}]`")
print(f"- control = `[{', '.join([str(x) for x in u])}]`")

print(f"\nSolution @ t = {t}")
solve_cwh_traj(position + theta + velocity + thetadot, u, t)
