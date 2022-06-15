"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core AFSIM Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module implements a spacecraft with Clohessy-Wilshire physics dynamics in non-inertial orbital
Hill's reference frame along with rotational dynamics. 2D scenario models in-plane (x-y) translational
motion and rotation about the z axis. 3D scenario is pending.
"""

import abc
import math
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from safe_autonomy_dynamics.base_models import BaseEntity, BaseEntityValidator, BaseLinearODESolverDynamics

M_DEFAULT = 12
J_DEFAULT = 0.0573
JW_DEFAULT = 4.1e-5
AAL_DEFAULT = 0.017453
AVL_DEFAULT = 0.034907
ALW_DEFAULT = 181.3
VLW_DEFAULT = 576
N_DEFAULT = 0.001027


class BaseCWHRotationSpacecraftValidator(BaseEntityValidator):
    """
    Validator for CWHRotationSpacecraft kwargs.

    Parameters
    ----------
    position : list[float]
        Length 3 list of x, y, z position values.
    velocity : list[float]
        Length 3 list of x, y, z velocity values.
    orientation: list[float]
        Length 4 list of quaternion orientation values.
    angular velocity: 3 list[float]
        Length 3 list of x, y, z angular velocity values.

    Raises
    ------
    ValueError
        Improper list lengths for parameters 'position', 'velocity', 'orientation', 'angular velocity'
    """
    x: float = 0
    y: float = 0
    z: float = 0
    xdot: float = 0
    ydot: float = 0
    zdot: float = 0
    q1: float = 0
    q2: float = 0
    q3: float = 0
    q4: float = 0
    wx: float = 0
    wy: float = 0
    wz: float = 0


class BaseCWHRotationSpacecraft(BaseEntity):
    """
    Base class for a spacecraft with translational Clohessy-Wiltshire dynamics in Hill's reference frame
    using thruster actuation, and rotational dynamics using reaction wheel actuation

    States
        x
        y
        z
        q1
        q2
        q3
        q4 (scalar)
        x_dot
        y_dot
        z_dot
        wx
        wy
        wz

    Controls
        thrust_x, thrust_y, thrust_z
            range = [-1, 1] Newtons
        moment_x, moment_y, moment_z
            range = [-0.001, 0.001] Newton-Meters

    Parameters
    ----------
    kwargs:
        Additional keyword arguments passed to parent class BaseEntity
    """

    def __init__(self, dynamics, control_default, control_min=-np.inf, control_max=np.inf, control_map=None, **kwargs):
        super().__init__(
            dynamics=dynamics,
            control_default=control_default,
            control_min=control_min,
            control_max=control_max,
            control_map=control_map,
            **kwargs
        )
        self.partner = None

    @classmethod
    def _get_config_validator(cls):
        return BaseCWHRotationSpacecraftValidator

    def __eq__(self, other):
        if isinstance(other, BaseCWHRotationSpacecraft):
            eq = (self.velocity == other.velocity).all()
            eq = eq and (self.position == other.position).all()
            eq = eq and (self.orientation.as_quat == other.orientation.as_quat).all()
            eq = eq and (self.angular_velocity == other.angular_velocity).all()
            return eq
        return False

    @abc.abstractmethod
    def _build_state(self):
        """form state vector"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def x(self):
        """get x"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def y(self):
        """get y"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def z(self):
        """get z"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def q1(self):
        """get first element of quaternion"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def q2(self):
        """get second element of quaternion"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def q3(self):
        """get third element of quaternion"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def q4(self):
        """get fourth element of quaternion"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def x_dot(self):
        """get x_dot, the velocity component in the x direction"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def y_dot(self):
        """get y_dot, the velocity component in the y direction"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def z_dot(self):
        """get z_dot, the velocity component in the z direction"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def wx(self):
        """get wx, the angular velocity component about the x axis"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def wy(self):
        """get wy, the angular velocity component about the y axis"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def wz(self):
        """get wz, the angular velocity component about the z axis"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def position(self):
        """get 3d position vector"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def orientation(self):
        """
        Get orientation of CWHRotationSpacecraft

        Returns
        -------
        scipy.spatial.transform.Rotation
            Rotation transformation of the entity's local reference frame basis vectors in the global reference frame.
            i.e. applying this rotation to [1, 0, 0] yields the entity's local x-axis in the global frame.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def velocity(self):
        """Get 3d velocity vector"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def angular_velocity(self):
        """Get 3d angular velocity vector"""
        raise NotImplementedError


class CWHRotation2dSpacecraftValidator(BaseCWHRotationSpacecraftValidator):
    """
    Validator for CWHRotation2dSpacecraft kwargs.

    Parameters
    ----------
    position : list[float]
        Length 2 list of x, y position values.
    theta: list[float]
        Length 1 list of angular position value
    velocity : list[float]
        Length 2 list of x, y velocity values.
    angular velocity: 1 list[float]
        Length 3 list of z angular velocity values.

    Raises
    ------
    ValueError
        Improper list lengths for parameters 'position', 'theta', 'velocity', 'angular velocity'
    """
    x: float = 0
    y: float = 0
    x_dot: float = 0
    y_dot: float = 0
    theta: float = 0
    wz: float = 0


class CWHRotation2dSpacecraft(BaseCWHRotationSpacecraft):
    """
    Spacecraft with 2D translational Clohessy-Wiltshire dynamics in Hill's reference frame.
    In-plane motion (x,y) using +/- xy thrusters.

    1D rotational dynamics (about z) using a +/- z reaction wheel

    States
        x
        y
        theta
        x_dot
        y_dot
        theta_dot

    Controls
        thrust_x
            range = [-1, 1] Newtons
        moment_z
            range = [-1, 1] Newton-Meters

    Parameters
    ----------
    m: float
        Mass of spacecraft in kilograms, by default 12.
    J: float
        Inertia of spacecraft in kg*m^2
    AAL: float
        Angular acceleration limit in rad/s^2
    AVL: float
        Angular velocity limit in rad/s
    JW: float
        Inertia of reaction wheel in kg*m^2
    ALW: float
         Acceleration limit of reaction wheel in rad/s^2
    VLW: float
         Velocity limit of reaction wheel in rad/s
    n: float
        Orbital mean motion of Hill's reference frame's circular orbit in rad/s, by default 0.001027.
    integration_method: str
        Numerical integration method passed to dynamics model. See BaseODESolverDynamics.
    kwargs:
        Additional keyword arguments passed to parent class BaseCWHRotatoinSpacecraft.
    """

    def __init__(
        self,
        m=M_DEFAULT,
        J=J_DEFAULT,
        aal=AAL_DEFAULT,
        avl=AVL_DEFAULT,
        Jw=JW_DEFAULT,
        alw=ALW_DEFAULT,
        vlw=VLW_DEFAULT,
        n=N_DEFAULT,
        integration_method="RK45",
        **kwargs
    ):
        self._state = np.array([])

        self.m = m  # kg
        self.J = J  # kg*m^2
        self.aal = aal  # rad/s^2
        self.avl = avl  # rad/s
        self.Jw = Jw  # kg*m^2
        self.alw = alw  # rad/s^2
        self.vlw = vlw  # rad/s
        self.n = n  # rads/s
        """ Define limits for angular acceleration, angular velocity, and control inputs """
        ang_acc_limit = min(self.aal, self.Jw * self.alw / self.J)
        ang_vel_limit = min(self.avl, self.Jw * self.vlw / self.J)

        control_default = np.zeros((2, ))
        control_min = np.array([-1, -ang_acc_limit * self.J])
        control_max = np.array([1, ang_acc_limit * self.J])
        control_map = {
            'thrust_x': 0,
            'moment_z': 1,
        }
        """ Create instance of dynamics class """
        dynamics = CWHRotation2dDynamics(m=m, J=J, avl=ang_vel_limit, aal=ang_acc_limit, n=n, integration_method=integration_method)

        super().__init__(
            dynamics, control_default=control_default, control_min=control_min, control_max=control_max, control_map=control_map, **kwargs
        )

    @classmethod
    def _get_config_validator(cls):
        return CWHRotation2dSpacecraftValidator

    def __eq__(self, other):
        if isinstance(other, CWHRotation2dSpacecraft):
            eq = (self.velocity == other.velocity).all()
            eq = eq and (self.position == other.position).all()
            eq = eq and (self.orientation.as_quat == other.orientation.as_quat).all()
            eq = eq and (self.angular_velocity == other.angular_velocity).all()
            return eq
        return False

    def _build_state(self):
        """form state vector"""
        state = np.array(
            [self.config.x, self.config.y, self.config.theta] + [self.config.xdot, self.config.ydot, self.config.wz], dtype=np.float32
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
        return 0

    @property
    def theta(self):
        """get theta"""
        return self._state[2]

    @property
    def q1(self):
        """get first element of quaternion"""
        return 0

    @property
    def q2(self):
        """get second element of quaternion"""
        return 0

    @property
    def q3(self):
        """get third element of quaternion"""
        return 0

    @property
    def q4(self):
        """get fourth element of quaternion"""
        return 0

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
        return 0

    @property
    def wx(self):
        """get wx, the angular velocity component about the x axis"""
        return 0

    @property
    def wy(self):
        """get wy_dot, the angular velocity component about the y axis"""
        return 0

    @property
    def wz(self):
        """get wz_dot, the angular velocity component about the z axis"""
        return self._state[5]

    @property
    def position(self):
        """get 3d position vector"""
        position = np.array([self.x, self.y, 0])
        return position

    @property
    def orientation(self):
        """
        Get orientation of CWHRotationSpacecraft

        Returns
        -------
        scipy.spatial.transform.Rotation
            Rotation transformation of the entity's local reference frame basis vectors in the global reference frame.
            i.e. applying this rotation to [1, 0, 0] yields the entity's local x-axis in the global frame.
        """
        return Rotation.from_quat(self._state[3:6])

    @property
    def velocity(self):
        """Get 3d velocity vector"""
        velocity = np.array([self.x_dot, self.y_dot, self.z_dot])
        return velocity

    @property
    def angular_velocity(self):
        """Get 3d angular velocity vector"""
        return np.array([0, 0, self.wz])


class CWHRotation2dDynamics(BaseLinearODESolverDynamics):
    """
    State transition implementation of 3D Clohessy-Wiltshire dynamics model.

    Parameters
    ----------
    m: float
        Mass of spacecraft in kilograms, by default 12
    J: float
        Inertia of spacecraft in kg*m^2
    AVL: float
         Angular velocity limit in rad/s
    AAL: float
         Angular acceleration limit in rad/s^2
    n: float
        Orbital mean motion of Hill's reference frame's circular orbit in rad/s, by default 0.001027
    kwargs:
        Additional keyword arguments passed to parent class BaseLinearODESolverDynamics constructor
    """

    def __init__(self, m=M_DEFAULT, J=J_DEFAULT, avl=AVL_DEFAULT, aal=AAL_DEFAULT, n=N_DEFAULT, **kwargs):
        self.m = m  # kg
        self.J = J  # kg*m^2
        self.ang_vel_limit = avl  # rad/s
        self.ang_acc_limit = aal  # rad/s^2
        self.n = n  # rads/s

        A, B = generate_cwhrotation_matrices(self.m, self.n, '2d')

        super().__init__(A=A, B=B, **kwargs)

    def _compute_state_dot(self, t: float, state: np.ndarray, control: np.ndarray) -> np.ndarray:

        x, y, theta, x_dot, y_dot, theta_dot = state
        # Form separate state vector for translational state
        pos_vel_state_vec = np.array([x, y, x_dot, y_dot], dtype=np.float64)
        # Compute the rotated thrust vector
        thrust_vector = control[0] * np.array([math.cos(theta), math.sin(theta)])
        # Compute derivatives
        pos_vel_derivative = np.matmul(self.A, pos_vel_state_vec) + np.matmul(self.B, thrust_vector)
        theta_dot_dot = control[1] / self.J
        # check angular acceleration and velocity limit
        if theta_dot_dot > self.ang_acc_limit:
            theta_dot_dot = self.ang_acc_limit
        elif theta_dot_dot < -self.ang_acc_limit:
            theta_dot_dot = -self.ang_acc_limit

        if theta_dot >= self.ang_vel_limit:
            theta_dot_dot = min(0, theta_dot_dot)
            theta_dot = self.ang_vel_limit
        elif theta_dot <= -self.ang_vel_limit:
            theta_dot_dot = max(0, theta_dot_dot)
            theta_dot = -self.ang_vel_limit
        # Form array of state derivatives
        state_derivative = np.array(
            [pos_vel_derivative[0], pos_vel_derivative[1], theta_dot, pos_vel_derivative[2], pos_vel_derivative[3], theta_dot_dot],
            dtype=np.float32
        )

        return state_derivative


def generate_cwhrotation_matrices(m: float, n: float, mode: str = '2d') -> Tuple[np.ndarray, np.ndarray]:
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
