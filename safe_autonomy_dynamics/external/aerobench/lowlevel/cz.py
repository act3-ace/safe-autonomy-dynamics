"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Safe Autonomy Dynamics.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

Stanley Bak
Python F-16 GCAS
Cz function
"""

import numpy as np

from safe_autonomy_dynamics.external.aerobench.util import fix, sign


def cz(alpha, beta, el):
    'cz function'

    a = np.array([.770, .241, -.100, -.415, -.731, -1.053, -1.355, -1.646, -1.917, -2.120, -2.248, -2.229], dtype=float).T

    s = .2 * alpha
    k = fix(s)

    if k <= -2:
        k = -1

    if k >= 9:
        k = 8

    da = s - k
    l = k + fix(1.1 * sign(da))  # noqa: E741
    l = l + 3  # noqa: E741
    k = k + 3
    s = a[k - 1] + abs(da) * (a[l - 1] - a[k - 1])

    return s * (1 - (beta / 57.3)**2) - .19 * (el / 25)
