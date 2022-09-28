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
"""


def cy(beta, ail, rdr):
    'cy function'

    return -.02 * beta + .021 * (ail / 20) + .086 * (rdr / 30)
