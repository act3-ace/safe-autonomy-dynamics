'''
Stanley Bak
F-16 GCAS in Python
cl function
'''

import numpy as np

from safe_autonomy_dynamics.external.aerobench.util import fix, sign


def cl(alpha, beta):
    'cl function'

    a = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [-.001, -.004, -.008, -.012, -.016, -.022, -.022, -.021, -.015, -.008, -.013, -.015], \
    [-.003, -.009, -.017, -.024, -.030, -.041, -.045, -.040, -.016, -.002, -.010, -.019], \
    [-.001, -.010, -.020, -.030, -.039, -.054, -.057, -.054, -.023, -.006, -.014, -.027], \
    [.000, -.010, -.022, -.034, -.047, -.060, -.069, -.067, -.033, -.036, -.035, -.035], \
    [.007, -.010, -.023, -.034, -.049, -.063, -.081, -.079, -.060, -.058, -.062, -.059], \
    [.009, -.011, -.023, -.037, -.050, -.068, -.089, -.088, -.091, -.076, -.077, -.076]], dtype=float).T

    s = .2 * alpha
    k = fix(s)

    if k <= -2:
        k = -1

    if k >= 9:
        k = 8

    da = s - k
    l = k + fix(1.1 * sign(da))
    s = .2 * abs(beta)
    m = fix(s)
    if m == 0:
        m = 1

    if m >= 6:
        m = 5

    db = s - m
    n = m + fix(1.1 * sign(db))
    l = l + 3
    k = k + 3
    m = m + 1
    n = n + 1
    t = a[k-1, m-1]
    u = a[k-1, n-1]
    v = t + abs(da) * (a[l-1, m-1] - t)
    w = u + abs(da) * (a[l-1, n-1] - u)
    dum = v + (w - v) * abs(db)

    return dum * sign(beta)
