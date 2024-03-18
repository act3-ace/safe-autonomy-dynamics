"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Safe Autonomy Dynamics.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

Stanley Bak
clf16.py for F-16 model

This is the objective function for finding the trim condition of the initial states
"""

from math import asin, sin

from safe_autonomy_dynamics.external.aerobench.lowlevel.conf16 import conf16
from safe_autonomy_dynamics.external.aerobench.lowlevel.subf16_model import subf16_model
from safe_autonomy_dynamics.external.aerobench.lowlevel.tgear import tgear


def clf16(s, x, u, const, model='stevens', adjust_cy=True):
    '''
    objective function of the optimization to find the trim conditions

    x and u get modified in-place
    returns the cost
    '''

    _, singam, _, _, tr, _, _, _, thetadot, _, _, orient = const
    gamm = asin(singam)

    if len(s) == 3:
        u[0] = s[0]
        u[1] = s[1]
        x[1] = s[2]
    else:
        u[0] = s[0]
        u[1] = s[1]
        u[2] = s[2]
        u[3] = s[3]
        x[1] = s[4]
        x[3] = s[5]
        x[4] = s[6]

    #
    # Get the current power and constraints
    #
    x[12] = tgear(u[0])
    [x, u] = conf16(x, u, const)

    xd = subf16_model(x, u, model, adjust_cy)[0]  # noqa: E731

    #
    # Steady Level flight
    #
    if orient == 1:
        r = 100.0 * (xd[0]**2 + xd[1]**2 + xd[2]**2 + xd[6]**2 + xd[7]**2 + xd[8]**2)

    #
    # Steady Climb
    #
    if orient == 2:
        r = 500.0 * (xd[11] - x[0] * sin(gamm))**2 + xd[0]**2 + 100.0 * (xd[1]**2 + xd[2]**2) + \
            10.0 * (xd[6]**2 + xd[7]**2 + xd[8]**2)

    #
    # Coord Turn
    #
    if orient == 3:
        r = xd[0] * xd[0] + 100.0 * (xd[1] * xd[1] + xd[2] * xd[2] + xd[11] * xd[11]) + \
            10.0 * (xd[6] * xd[6] + xd[7] * xd[7] + xd[8] * xd[8]) + 500.0 * (xd[5] - tr)**2

    #
    # Pitch Pull Up
    #

    if orient == 4:
        r = 500.0 * (xd[4] - thetadot)**2 + xd[0]**2 + 100.0 * (xd[1]**2 + xd[2]**2) + \
            10.0 * (xd[6]**2 + xd[7]**2 + xd[8]**2)

    #
    # Scale r if it is less than 1
    #
    if r < 1.0:
        r = r**0.5

    return r
