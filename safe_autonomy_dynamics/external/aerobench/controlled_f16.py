"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Safe Autonomy Dynamics.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

Stanley Bak
Python Version of LQR controlled F-16
ODE derivative code (controlled F16)
"""

from math import cos, sin

import numpy as np
from numpy import deg2rad

from safe_autonomy_dynamics.external.aerobench.lowlevel.low_level_controller import LowLevelController
from safe_autonomy_dynamics.external.aerobench.lowlevel.subf16_model import subf16_model


def controlled_f16(x_f16, u_ref, llc, f16_model='morelli', v2_integrators=False):
    """returns the LQR-controlled F-16 state derivatives and more"""

    assert isinstance(x_f16, np.ndarray)
    assert isinstance(llc, LowLevelController)
    assert u_ref.size == 4

    assert f16_model in ['stevens', 'morelli'], f'Unknown F16_model: {f16_model}'

    x_ctrl, u_deg = llc.get_u_deg(u_ref, x_f16)

    # Note: Control vector (u) for subF16 is in units of degrees
    xd_model, nz, ny, _, _ = subf16_model(x_f16[0:13], u_deg, f16_model)

    if v2_integrators:
        # integrators from matlab v2 model
        ps = xd_model[6] * cos(xd_model[1]) + xd_model[8] * sin(xd_model[1])

        ny_r = ny + xd_model[8]
    else:
        # Nonlinear (Actual): ps = p * cos(alpha) + r * sin(alpha)
        ps = x_ctrl[4] * cos(x_ctrl[0]) + x_ctrl[5] * sin(x_ctrl[0])

        # Calculate (side force + yaw rate) term
        ny_r = ny + x_ctrl[5]

    xd = np.zeros((x_f16.shape[0], ))
    xd[:len(xd_model)] = xd_model

    # integrators from low-level controller
    start = len(xd_model)
    end = start + llc.get_num_integrators()
    int_der = llc.get_integrator_derivatives(u_ref, nz, ps, ny_r)
    xd[start:end] = int_der

    # Convert all degree values to radians for output
    u_rad = np.zeros((7, ))  # throt, ele, ail, rud, Nz_ref, ps_ref, Ny_r_ref

    u_rad[0] = u_deg[0]  # throttle

    for i in range(1, 4):
        u_rad[i] = deg2rad(u_deg[i])

    u_rad[4:7] = u_ref[0:3]  # inner-loop commands are 4-7

    return xd, u_rad, nz, ps, ny_r
