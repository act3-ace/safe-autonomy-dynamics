"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core AFSIM Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module implements a 3D point mass spacecraft with Clohessy-Wilshire physics dynamics in non-inertial orbital
Hill's reference frame.
"""

from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from safe_autonomy_dynamics.base_models import BaseEntity, BaseEntityValidator, BaseLinearODESolverDynamics

M_DEFAULT = 12
N_DEFAULT = 0.001027


class CWHSpacecraftValidator(BaseEntityValidator):
    """
    Validator for CWHSpacecraft kwargs.

    Parameters
    ----------
    x: [float]
        Length 1, x position value
    y: [float]
        Length 1, y position value
    z: [float]
        Length 1, z position value
    x_dot: [float]
        Length 1, x velocity value
    y_dot: [float]
        Length 1, y velocity value
    z_dot: [float]
        Length 1, z velocity value

    Raises
    ------
    ValueError
        Improper list lengths for parameters 'x', 'y', 'z', 'x_dot', 'y_dot', 'z_dot'
    """
    x: float = 0
    y: float = 0
    z: float = 0
    x_dot: float = 0
    y_dot: float = 0
    z_dot: float = 0


class CWHSpacecraft(BaseEntity):
    """
    3D point mass spacecraft with +/- xyz thrusters and Clohessy-Wiltshire dynamics in Hill's reference frame.

    States
        x
        y
        z
        x_dot
        y_dot
        z_dot

    Controls
        thrust_x
            range = [-1, 1] Newtons
        thrust_y
            range = [-1, 1] Newtons
        thrust_z
            range = [-1, 1] Newtons

    Parameters
    ----------
    m: float
        Mass of spacecraft in kilograms, by default 12.
    n: float
        Orbital mean motion of Hill's reference frame's circular orbit in rad/s, by default 0.001027.
    integration_method: str
        Numerical integration method passed to dynamics model. See BaseODESolverDynamics.
    kwargs:
        Additional keyword arguments passed to CWHSpacecraftValidator.
    """

    def __init__(self, m=M_DEFAULT, n=N_DEFAULT, integration_method="RK45", **kwargs):
        dynamics = CWHDynamics(m=m, n=n, integration_method=integration_method)
        self._state = np.array([])

        control_map = {
            'thrust_x': 0,
            'thrust_y': 1,
            'thrust_z': 2,
        }

        super().__init__(dynamics, control_default=np.zeros((3, )), control_min=-1, control_max=1, control_map=control_map, **kwargs)

    @classmethod
    def _get_config_validator(cls):
        return CWHSpacecraftValidator

    def __eq__(self, other):
        if isinstance(other, CWHSpacecraft):
            eq = (self.velocity == other.velocity).all()
            eq = eq and (self.position == other.position).all()
            eq = eq and (self.orientation.as_euler("zyx") == other.orientation.as_euler("zyx")).all()
            return eq
        return False

    def _build_state(self):
        state = np.array(
            [self.config.x, self.config.y, self.config.z] + [self.config.x_dot, self.config.y_dot, self.config.z_dot], dtype=np.float32
        )

        return state

    @property
    def x(self):
        """get x"""
        return self._state[0]

    @property
    def y(self):
        """get y"""
        return self._state[1]

    @property
    def z(self):
        """get z"""
        return self._state[2]

    @property
    def x_dot(self):
        """get x_dot, the velocity component in the x direction"""
        return self._state[3]

    @property
    def y_dot(self):
        """get y_dot, the velocity component in the y direction"""
        return self._state[4]

    @property
    def z_dot(self):
        """get z_dot, the velocity component in the z direction"""
        return self._state[5]

    @property
    def position(self):
        """get 3d position vector"""
        return self._state[0:3].copy()

    @property
    def orientation(self):
        """
        Get orientation of CWHSpacecraft, which is always an identity rotation as a point mass model doesn't rotate.

        Returns
        -------
        scipy.spatial.transform.Rotation
            Rotation transformation of the entity's local reference frame basis vectors in the global reference frame.
            i.e. applying this rotation to [1, 0, 0] yields the entity's local x-axis in the global frame.
        """
        # always return a no rotation quaternion as points do not have an orientation
        return Rotation.from_quat([0, 0, 0, 1])

    @property
    def velocity(self):
        """Get 3d velocity vector"""
        return self._state[3:6].copy()


class CWHDynamics(BaseLinearODESolverDynamics):
    """
    State transition implementation of 3D Clohessy-Wiltshire dynamics model.

    Parameters
    ----------
    m: float
        Mass of spacecraft in kilograms, by default 12
    n: float
        Orbital mean motion of Hill's reference frame's circular orbit in rad/s, by default 0.001027
    kwargs:
        Additional keyword arguments passed to parent class BaseLinearODESolverDynamics constructor
    """

    def __init__(self, m=M_DEFAULT, n=N_DEFAULT, **kwargs):
        self.m = m  # kg
        self.n = n  # rads/s

        A, B = generate_cwh_matrices(self.m, self.n, '3d')

        super().__init__(A=A, B=B, **kwargs)


def generate_cwh_matrices(m: float, n: float, mode: str = '2d') -> Tuple[np.ndarray, np.ndarray]:
    """Generates A and B Matrices from Clohessy-Wiltshire linearized dynamics of dx/dt = Ax + Bu

    Parameters
    ----------
    m : float
        mass in kg of spacecraft
    n : float
        orbital mean motion in rad/s of current Hill's reference frame
    mode : str, optional
        dimensionality of dynamics matrices. '2d' or '3d', by default '2d'

    Returns
    -------
    np.ndarray
        A dynamics matrix
    np.ndarray
        B dynamics matrix
    """
    assert mode in ['2d', '3d'], "mode must be on of ['2d', '3d']"
    if mode == '2d':
        A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [3 * n**2, 0, 0, 2 * n],
            [0, 0, -2 * n, 0],
        ], dtype=np.float64)

        B = np.array([
            [0, 0],
            [0, 0],
            [1 / m, 0],
            [0, 1 / m],
        ], dtype=np.float64)
    else:
        A = np.array(
            [
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [3 * n**2, 0, 0, 0, 2 * n, 0],
                [0, 0, 0, -2 * n, 0, 0],
                [0, 0, -n**2, 0, 0, 0],
            ],
            dtype=np.float64
        )

        B = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1 / m, 0, 0],
            [0, 1 / m, 0],
            [0, 0, 1 / m],
        ], dtype=np.float64)

    return A, B
