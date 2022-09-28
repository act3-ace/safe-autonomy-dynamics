"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Safe Autonomy Dynamics.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

Stanley Bak
F16 GCAS in Python
dnda function
"""

import numpy as np

from safe_autonomy_dynamics.external.aerobench.util import fix, sign


def dnda(alpha, beta):
    'dnda function'

    a = np.array(
        [
            [.001, -.027, -.017, -.013, -.012, -.016, .001, .017, .011, .017, .008, .016], [
                .002, -.014, -.016, -.016, -.014, -.019, -.021, .002, .012, .016, .015, .011
            ], [-.006, -.008, -.006, -.006, -.005, -.008, -.005, .007, .004, .007, .006, .006], [
                -.011, -.011, -.010, -.009, -.008, -.006, .000, .004, .007, .010, .004, .010
            ], [-.015, -.015, -.014, -.012, -.011, -.008, -.002, .002, .006, .012, .011, .011], [
                -.024, -.010, -.004, -.002, -.001, .003, .014, .006, -.001, .004, .004, .006
            ], [-.022, .002, -.003, -.005, -.003, -.001, -.009, -.009, -.001, .003, -.002, .001]
        ],
        dtype=float
    ).T

    s = .2 * alpha
    k = fix(s)

    if k <= -2:
        k = -1

    if k >= 9:
        k = 8

    da = s - k
    l = k + fix(1.1 * sign(da))  # noqa: E741
    s = .1 * beta
    m = fix(s)
    if m <= -3:
        m = -2

    if m >= 3:
        m = 2

    db = s - m
    n = m + fix(1.1 * sign(db))
    l = l + 3  # noqa: E741
    k = k + 3
    m = m + 4
    n = n + 4
    t = a[k - 1, m - 1]
    u = a[k - 1, n - 1]
    v = t + abs(da) * (a[l - 1, m - 1] - t)
    w = u + abs(da) * (a[l - 1, n - 1] - u)

    return v + (w - v) * abs(db)
