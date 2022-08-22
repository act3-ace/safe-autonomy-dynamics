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

import math
from typing import Union

import numpy as np
from scipy.spatial.transform import Rotation

from safe_autonomy_dynamics.base_models import BaseEntityValidator, BaseODESolverDynamics, BaseRotationEntity
from safe_autonomy_dynamics.cwh import M_DEFAULT, N_DEFAULT, generate_cwh_matrices

INERTIA_DEFAULT = 0.0573
INERTIA_WHEEL_DEFAULT = 4.1e-5
ANG_ACC_LIMIT_DEFAULT = 0.017453
ANG_VEL_LIMIT_DEFAULT = 0.034907
ACC_LIMIT_WHEEL_DEFAULT = 181.3
VEL_LIMIT_WHEEL_DEFAULT = 576


class CWHRotation2dSpacecraftValidator(BaseEntityValidator):
    """
    Validator for CWHRotation2dSpacecraft kwargs.

    Parameters
    ----------
    x: [float]
       Length 1, x position value. m
    y: [float]
       Length 1, y position value. m
    theta: [float]
       Length 1, rotation angle value. rad
    x_dot: [float]
       Length 1, x velocity value. m/s
    y_dot: [float]
       Length 1, y velocity value. m/s
    theta_dot: [float]
       Length 1, rotation rate value. rad/s

    Raises
    ------
    ValueError
        Improper list lengths for parameters 'x', 'y', 'theta', 'x_dot', 'y_dot', 'theta_dot'
    """
    x: float = 0
    y: float = 0
    theta: float = 0
    x_dot: float = 0
    y_dot: float = 0
    wz: float = 0


class CWHRotation2dSpacecraft(BaseRotationEntity):
    """
    Spacecraft with 2D translational Clohessy-Wiltshire dynamics in Hill's reference frame.
    In-plane motion (x,y) using +/- x thruster rotated to desired direction

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
        thrust_y
            range = [-1, 1] Newtons
        moment_z
            range = [-0.001, 0.001] Newton-Meters

    Parameters
    ----------
    m: float
        Mass of spacecraft in kilograms, by default 12.
    inertia: float
        Inertia of spacecraft in kg*m^2
    ang_acc_limit: float
        Angular acceleration limit in rad/s^2
    ang_vel_limit: float
        Angular velocity limit in rad/s
    inertia_wheel: float
        Inertia of reaction wheel in kg*m^2
    acc_limit_wheel: float
         Acceleration limit of reaction wheel in rad/s^2
    vel_limit_wheel: float
         Velocity limit of reaction wheel in rad/s
    n: float
        Orbital mean motion of Hill's reference frame's circular orbit in rad/s, by default 0.001027.
    integration_method: str
        Numerical integration method passed to dynamics model. See BaseODESolverDynamics.
    kwargs:
        Additional keyword arguments passed to parent class BaseRotationEntity.
    """

    def __init__(
        self,
        m=M_DEFAULT,
        inertia=INERTIA_DEFAULT,
        ang_acc_limit=ANG_ACC_LIMIT_DEFAULT,
        ang_vel_limit=ANG_VEL_LIMIT_DEFAULT,
        inertia_wheel=INERTIA_WHEEL_DEFAULT,
        acc_limit_wheel=ACC_LIMIT_WHEEL_DEFAULT,
        vel_limit_wheel=VEL_LIMIT_WHEEL_DEFAULT,
        n=N_DEFAULT,
        integration_method="RK45",
        **kwargs
    ):
        self._state = np.array([])
        self.partner = {}

        self.m = m  # kg
        self.inertia = inertia  # kg*m^2
        self.ang_acc_limit = ang_acc_limit  # rad/s^2
        self.ang_vel_limit = ang_vel_limit  # rad/s
        self.inertia_wheel = inertia_wheel  # kg*m^2
        self.acc_limit_wheel = acc_limit_wheel  # rad/s^2
        self.vel_limit_wheel = vel_limit_wheel  # rad/s
        self.n = n  # rads/s
        """ Define limits for angular acceleration, angular velocity, and control inputs """
        ang_acc_limit = min(self.ang_acc_limit, self.inertia_wheel * self.acc_limit_wheel / self.inertia)
        ang_vel_limit = min(self.ang_vel_limit, self.inertia_wheel * self.vel_limit_wheel / self.inertia)

        control_default = np.zeros((3, ))
        control_min = np.array([-1, -1, -ang_acc_limit * self.inertia])
        control_max = np.array([1, 1, ang_acc_limit * self.inertia])
        control_map = {
            'thrust_x': 0,
            'thrust_y': 1,
            'moment_z': 2,
        }
        """ Create instance of dynamics class """
        dynamics = CWHRotation2dDynamics(
            m=m, inertia=inertia, ang_acc_limit=ang_acc_limit, ang_vel_limit=ang_vel_limit, n=n, integration_method=integration_method
        )

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

    def register_partner(self, partner: BaseRotationEntity):
        """
        Register another entity as this entity's partner. Defines line of communication between entities.

        Parameters
        ----------
        partner: BaseEntity
            Entity with line of communication to this entity.

        Returns
        -------
        None
        """
        self.partner[partner.name] = partner

    def _build_state(self):
        """form state vector"""
        state = np.array(
            [self.config.x, self.config.y, self.config.theta] + [self.config.x_dot, self.config.y_dot, self.config.wz], dtype=np.float32
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
    def q1(self):
        """get first element of quaternion"""
        return self.quaternion[0]

    @property
    def q2(self):
        """get second element of quaternion"""
        return self.quaternion[1]

    @property
    def q3(self):
        """get third element of quaternion"""
        return self.quaternion[2]

    @property
    def q4(self):
        """get fourth element of quaternion (scalar)"""
        return self.quaternion[3]

    @property
    def theta(self):
        """get theta"""
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
        """get z_dot, the velocity component in the z axis"""
        return 0

    @property
    def wx(self):
        """get wz_dot, the angular velocity component about the x axis"""
        return 0

    @property
    def wy(self):
        """get wz_dot, the angular velocity component about the y axis"""
        return 0

    @property
    def wz(self):
        """get wz_dot, the angular velocity component about the z axis"""
        return self._state[5]

    @property
    def position(self):
        """get 3d position vector"""
        position = np.array([self.x, self.y, self.z])
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
        return Rotation.from_euler("ZYX", [self.theta, 0, 0])

    @property
    def quaternion(self):
        """get 4d quaternion
           Quaternion order (scalar in 4th element) matches scipy convention of [x,y,z,w]
        """
        return self.orientation.as_quat()

    @property
    def velocity(self):
        """Get 3d velocity vector"""
        return np.array([self.x_dot, self.y_dot, self.z_dot])

    @property
    def angular_velocity(self):
        """Get 3d angular velocity vector"""
        return np.array([self.wx, self.wy, self.wz])


class CWHRotation2dDynamics(BaseODESolverDynamics):
    """
    State transition implementation of 3D Clohessy-Wiltshire dynamics model.

    Parameters
    ----------
    m: float
        Mass of spacecraft in kilograms, by default 12
    inertia: float
        Inertia of spacecraft in kg*m^2
    ang_acc_limit: float
         Angular acceleration limit in rad/s^2
    ang_vel_limit: float
         Angular velocity limit in rad/s
    n: float
        Orbital mean motion of Hill's reference frame's circular orbit in rad/s, by default 0.001027
    kwargs:
        Additional keyword arguments passed to parent class BaseODESolverDynamics constructor
    """

    def __init__(
        self,
        m=M_DEFAULT,
        inertia=INERTIA_DEFAULT,
        ang_acc_limit=ANG_ACC_LIMIT_DEFAULT,
        ang_vel_limit=ANG_VEL_LIMIT_DEFAULT,
        n=N_DEFAULT,
        state_max: Union[float, np.ndarray] = None,
        state_min: Union[float, np.ndarray] = None,
        angle_wrap_centers: np.ndarray = None,
        **kwargs
    ):
        self.m = m  # kg
        self.inertia = inertia  # kg*m^2
        self.ang_acc_limit = ang_acc_limit  # rad/s^2
        self.ang_vel_limit = ang_vel_limit  # rad/s
        self.n = n  # rads/s

        A, B = generate_cwh_matrices(self.m, self.n, '2d')

        assert len(A.shape) == 2, f"A must be square matrix. Instead got shape {A.shape}"
        assert len(B.shape) == 2, f"A must be square matrix. Instead got shape {B.shape}"
        assert A.shape[0] == A.shape[1], f"A must be a square matrix, not dimension {A.shape}"
        assert A.shape[1] == B.shape[0], (
            "number of columns in A must match the number of rows in B." + f" However, got shapes {A.shape} for A and {B.shape} for B"
        )

        self.A = np.copy(A)
        self.B = np.copy(B)

        if state_min is None:
            state_min = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -self.ang_vel_limit])

        if state_max is None:
            state_max = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, self.ang_vel_limit])

        if angle_wrap_centers is None:
            angle_wrap_centers = np.array([None, None, 0, None, None, None], dtype=float)

        super().__init__(state_min=state_min, state_max=state_max, angle_wrap_centers=angle_wrap_centers, **kwargs)

    def _compute_state_dot(self, t: float, state: np.ndarray, control: np.ndarray) -> np.ndarray:

        x, y, theta, x_dot, y_dot, theta_dot = state
        # Form separate state vector for translational state
        pos_vel_state_vec = np.array([x, y, x_dot, y_dot], dtype=np.float64)
        # Compute the rotated thrust vector
        thrust_vector = (
            control[0] * np.array([math.cos(theta), math.sin(theta)]) + control[1] * np.array([-math.sin(theta), math.cos(theta)])
        )
        # Compute derivatives
        pos_vel_derivative = np.matmul(self.A, pos_vel_state_vec) + np.matmul(self.B, thrust_vector)
        theta_dot_dot = control[2] / self.inertia
        # check angular velocity limit
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
