"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Safe Autonomy Dynamics.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module implements 1d, 2d, and 3d point mass integrators.
"""

import abc
from typing import Tuple

import numpy as np

from safe_autonomy_dynamics.base_models import BaseEntity, BaseEntityValidator, BaseLinearODESolverDynamics

M_DEFAULT = 1
DAMPING_DEFAULT = 0


class BaseIntegratorValidator(BaseEntityValidator):
    """
    Validator for Integrator kwargs.

    Parameters
    ----------
    position : list[float]
        Length 3 list of x, y, z position values.
    velocity : list[float]
        Length 3 list of x, y, z velocity values.

    Raises
    ------
    ValueError
        Improper list lengths for parameters 'position', 'velocity'
    """
    x: float = 0
    y: float = 0
    z: float = 0
    xdot: float = 0
    ydot: float = 0
    zdot: float = 0


class BaseIntegrator(BaseEntity):
    """
    Base inerface for Integrator Entities
    """

    def __init__(self, dynamics, control_default, control_min, control_max, control_map, **kwargs):
        super().__init__(
            dynamics=dynamics,
            control_default=control_default,
            control_min=control_min,
            control_max=control_max,
            control_map=control_map,
            **kwargs
        )

    @classmethod
    def _get_config_validator(cls):
        return BaseIntegratorValidator

    def __eq__(self, other):
        if isinstance(other, BaseIntegrator):
            eq = (self.velocity == other.velocity).all()
            eq = eq and (self.position == other.position).all()
            return eq
        return False

    @property
    @abc.abstractmethod
    def position(self):
        """get position vector"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def velocity(self):
        """Get velocity vector"""
        raise NotImplementedError


class Integrator1d(BaseIntegrator):
    """
    1d integrator simulation entity

    States
        x
        x_dot

    Controls
        thrust_x
            default range = [-1, 1]

    Parameters
    ----------
    m: float
        Mass of integrator, by default 1.
    trajectory_samples : int
        number of trajectory samples the generate and store on steps
    integration_method: str
        Numerical integration method passed to dynamics model. See BaseODESolverDynamics.
    kwargs:
        Additional keyword arguments passed to BaseIntegrator.
    """

    def __init__(self, m=M_DEFAULT, damping=DAMPING_DEFAULT, trajectory_samples=0, integration_method="RK45", **kwargs):
        dynamics = IntegratorDynamics(
            m=m, damping=damping, mode='1d', trajectory_samples=trajectory_samples, integration_method=integration_method
        )
        self._state = np.array([])

        control_map = {
            'thrust_x': 0,
        }

        super().__init__(dynamics, control_default=np.zeros((1, )), control_min=-1, control_max=1, control_map=control_map, **kwargs)

    def __eq__(self, other):
        if isinstance(other, Integrator1d):
            eq = (self.velocity == other.velocity).all()
            eq = eq and (self.position == other.position).all()
            return eq
        return False

    def _build_state(self):
        state = np.array([self.config.x, self.config.xdot], dtype=np.float32)

        return state

    @property
    def x(self):
        """get x"""
        return self._state[0]

    @property
    def y(self):
        """get y"""
        return 0

    @property
    def z(self):
        """get z"""
        return 0

    @property
    def x_dot(self):
        """get x_dot, the velocity component in the x direction"""
        return self._state[1]

    @property
    def y_dot(self):
        """get y_dot, the velocity component in the y direction"""
        return 0

    @property
    def z_dot(self):
        """get z_dot, the velocity component in the z direction"""
        return 0

    @property
    def position(self):
        """get position vector"""
        return self._state[0].copy()

    @property
    def velocity(self):
        """Get velocity vector"""
        return self._state[1].copy()


class Integrator2d(BaseIntegrator):
    """
    1d integrator simulation entity

    States
        x
        y
        x_dot
        y_dot

    Controls
        thrust_x
            default range = [-1, 1]
        thrust_y
            default range = [-1, 1]

    Parameters
    ----------
    m: float
        Mass of integrator, by default 1.
    trajectory_samples : int
        number of trajectory samples the generate and store on steps
    integration_method: str
        Numerical integration method passed to dynamics model. See BaseODESolverDynamics.
    kwargs:
        Additional keyword arguments passed to BaseIntegrator.
    """

    def __init__(self, m=M_DEFAULT, damping=DAMPING_DEFAULT, trajectory_samples=0, integration_method="RK45", **kwargs):
        dynamics = IntegratorDynamics(
            m=m, damping=damping, mode='2d', trajectory_samples=trajectory_samples, integration_method=integration_method
        )
        self._state = np.array([])

        control_map = {
            'thrust_x': 0,
            'thrust_y': 0,
        }

        super().__init__(dynamics, control_default=np.zeros((2, )), control_min=-1, control_max=1, control_map=control_map, **kwargs)

    def __eq__(self, other):
        if isinstance(other, Integrator1d):
            eq = (self.velocity == other.velocity).all()
            eq = eq and (self.position == other.position).all()
            return eq
        return False

    def _build_state(self):
        state = np.array([self.config.x, self.config.y] + [self.config.xdot, self.config.ydot], dtype=np.float32)

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
        return 0

    @property
    def x_dot(self):
        """get x_dot, the velocity component in the x direction"""
        return self._state[2]

    @property
    def y_dot(self):
        """get y_dot, the velocity component in the y direction"""
        return self._state[3]

    @property
    def z_dot(self):
        """get z_dot, the velocity component in the z direction"""
        return 0

    @property
    def position(self):
        """get position vector"""
        return self._state[0:2].copy()

    @property
    def velocity(self):
        """Get velocity vector"""
        return self._state[2:4].copy()


class Integrator3d(BaseIntegrator):
    """
    1d integrator simulation entity

    States
        x
        y
        z
        x_dot
        y_dot
        z_dot

    Controls
        thrust_x
            default range = [-1, 1]
        thrust_y
            default range = [-1, 1]
        thrust_z
            default range = [-1, 1]

    Parameters
    ----------
    m: float
        Mass of integrator, by default 1.
    trajectory_samples : int
        number of trajectory samples the generate and store on steps
    integration_method: str
        Numerical integration method passed to dynamics model. See BaseODESolverDynamics.
    kwargs:
        Additional keyword arguments passed to BaseIntegrator.
    """

    def __init__(self, m=M_DEFAULT, damping=DAMPING_DEFAULT, trajectory_samples=0, integration_method="RK45", **kwargs):
        dynamics = IntegratorDynamics(
            m=m, damping=damping, mode='3d', trajectory_samples=trajectory_samples, integration_method=integration_method
        )
        self._state = np.array([])

        control_map = {
            'thrust_x': 0,
            'thrust_y': 0,
            'thrust_z': 0,
        }

        super().__init__(dynamics, control_default=np.zeros((3, )), control_min=-1, control_max=1, control_map=control_map, **kwargs)

    def __eq__(self, other):
        if isinstance(other, Integrator1d):
            eq = (self.velocity == other.velocity).all()
            eq = eq and (self.position == other.position).all()
            return eq
        return False

    def _build_state(self):
        state = np.array(
            [self.config.x, self.config.y, self.config.z] + [self.config.xdot, self.config.ydot, self.config.zdot], dtype=np.float32
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
        """get position vector"""
        return self._state[0:3].copy()

    @property
    def velocity(self):
        """Get velocity vector"""
        return self._state[3:6].copy()


class IntegratorDynamics(BaseLinearODESolverDynamics):
    """
    State transition implementation of 3D integrator dynamics model.

    Parameters
    ----------
    m: float
        Mass of object, by default 1
    damping: float
        linear velocity damper. Default is zero
    mode : str, optional
        dimensionality of dynamics matrices. '1d', '2d', or '3d', by default '1d'
    kwargs:
        Additional keyword arguments passed to parent class BaseLinearODESolverDynamics constructor
    """

    def __init__(self, m=M_DEFAULT, damping=DAMPING_DEFAULT, mode='1d', **kwargs):
        self.m = m
        self.damping = damping

        A, B = generate_dynamics_matrices(self.m, self.damping, mode)

        super().__init__(A=A, B=B, **kwargs)


def generate_dynamics_matrices(m: float, damping: float = 0, mode: str = '1d') -> Tuple[np.ndarray, np.ndarray]:
    """Generates A and B Matrices for linearized dynamics of dx/dt = Ax + Bu

    Parameters
    ----------
    m : float
        mass of object
    damping : float, optional
        linear velocity damper. Default is zero
    mode : str, optional
        dimensionality of dynamics matrices. '1d', '2d', or '3d', by default '1d'

    Returns
    -------
    np.ndarray
        A dynamics matrix
    np.ndarray
        B dynamics matrix
    """
    assert mode in ['1d', '2d', '3d'], "mode must be on of ['1d', '2d', '3d']"
    if mode == '1d':
        A = np.array([
            [0, 1],
            [0, -damping],
        ], dtype=np.float64)

        B = np.array([
            [0],
            [1 / m],
        ], dtype=np.float64)
    elif mode == '2d':
        A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, -damping, 0],
            [0, 0, 0, -damping],
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
                [0, 0, 0, -damping, 0, 0],
                [0, 0, 0, 0, -damping, 0],
                [0, 0, 0, 0, 0, -damping],
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
