"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Safe Autonomy Dynamics.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module implements a spacecraft with 3D Clohessy-Wilshire physics dynamics in non-inertial orbital
Hill's reference frame along with 3D rotational dynamics using quaternions for attitude representation.
"""

from typing import Union

import numpy as np
import pint
from pydantic import AfterValidator
from scipy.spatial.transform import Rotation
from typing_extensions import Annotated

from safe_autonomy_dynamics.base_models import (
    BaseControlAffineODESolverDynamics,
    BaseEntityValidator,
    BaseRotationEntity,
    BaseUnits,
    build_unit_conversion_validator_fn,
)
from safe_autonomy_dynamics.cwh import generate_cwh_matrices
from safe_autonomy_dynamics.utils import number_list_to_np

M_DEFAULT = 12
INERTIA_MATRIX_DEFAULT = np.matrix([[0.0573, 0.0, 0.0], [0.0, 0.0573, 0.0], [0.0, 0.0, 0.0573]])
INERTIA_WHEEL_DEFAULT = 4.1e-5
ANG_ACC_LIMIT_DEFAULT = 0.017453
ANG_VEL_LIMIT_DEFAULT = 0.034907
ACC_LIMIT_WHEEL_DEFAULT = 181.3
VEL_LIMIT_WHEEL_DEFAULT = 576
THRUST_CONTROL_LIMIT_DEFAULT = 1.0
N_DEFAULT = 0.001027


class SixDOFSpacecraftValidator(BaseEntityValidator):
    """
    Validator for SixDOFSpacecraft kwargs.

    Parameters
    ----------
    x: float or pint.Quantity
       x position value
    y: float or pint.Quantity
       y position value
    z: float or pint.Quantity
       z position value
    x_dot: float or pint.Quantity
       x velocity value
    y_dot: float or pint.Quantity
       y velocity value
    z_dot: float or pint.Quantity
       z velocity value
    q1: float
       first element of quaternion - rotation from body to Hill frame
    q2: float
       second element of quaternion value - rotation from body to Hill frame
    q3: float
       third element of quaternion value - rotation from body to Hill frame
    q4: float
       fourth element of quaternion value (scalar) - rotation from body to Hill frame
       Placing the scalar as the 4th element matches the convention used by scipy
    wx: float or pint.Quantity
       x axis local body reference frame angular rate value
    wy: float or pint.Quantity
       y axis local body reference frame angular rate value
    wz: float or pint.Quantity
       z axis local body reference frame angular rate value

    Raises
    ------
    ValueError
        Improper list lengths for parameters 'x', 'y', 'z', 'x_dot', 'y_dot', 'z_dot', 'q1', 'q2', 'q3', 'q4', 'wx', 'wy', 'wz'
    """
    x: Annotated[Union[float, pint.Quantity], AfterValidator(build_unit_conversion_validator_fn('meters'))] = 0
    y: Annotated[Union[float, pint.Quantity], AfterValidator(build_unit_conversion_validator_fn('meters'))] = 0
    z: Annotated[Union[float, pint.Quantity], AfterValidator(build_unit_conversion_validator_fn('meters'))] = 0
    x_dot: Annotated[Union[float, pint.Quantity], AfterValidator(build_unit_conversion_validator_fn('meters/second'))] = 0
    y_dot: Annotated[Union[float, pint.Quantity], AfterValidator(build_unit_conversion_validator_fn('meters/second'))] = 0
    z_dot: Annotated[Union[float, pint.Quantity], AfterValidator(build_unit_conversion_validator_fn('meters/second'))] = 0
    q1: float = 0
    q2: float = 0
    q3: float = 0
    q4: float = 0
    wx: Annotated[Union[float, pint.Quantity], AfterValidator(build_unit_conversion_validator_fn('radians/second'))] = 0
    wy: Annotated[Union[float, pint.Quantity], AfterValidator(build_unit_conversion_validator_fn('radians/second'))] = 0
    wz: Annotated[Union[float, pint.Quantity], AfterValidator(build_unit_conversion_validator_fn('radians/second'))] = 0


class SixDOFSpacecraft(BaseRotationEntity):  # pylint: disable=too-many-public-methods
    """
    Spacecraft with 3D Clohessy-Wiltshire translational dynamics, in Hill's frame and 3D rotational dynamics

    States
        x, y, z
        q1, q2, q3, q4
        x_dot, y_dot, z_dot
        wx, wy, wz

    Controls
        thrust_x
            range = [-1, 1] Newtons
        thrust_y
            range = [-1, 1] Newtons
        thrust_z
            range = [-1, 1] Newtons
        moment_x
            range = [-0.001, 0.001] Newton-Meters
        moment_y
            range = [-0.001, 0.001] Newton-Meters
        moment_z
            range = [-0.001, 0.001] Newton-Meters

    Parameters
    ----------
    m: float
        Mass of spacecraft in kilograms, by default 12
    inertia_matrix: float
        Inertia matrix of spacecraft (3x3) in kg*m^2
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
    thrust_control_limit: float
        Thrust control limit in N
    body_frame_thrust: bool
        Flag indicating the reference frame for the control thrust vector: True- Body frame, False - Hill's frame
        by default, True
    n: float
        Orbital mean motion of Hill's reference frame's circular orbit in rad/s, by default 0.001027.
    trajectory_samples : int
        number of trajectory samples the generate and store on steps
    integration_method: str
        Numerical integration method passed to dynamics model. See BaseODESolverDynamics.
    kwargs:
        Additional keyword arguments passed to parent class BaseRotationSpacecraft.
        body_frame_thrust: bool
            Flag indicating the reference frame for the control thrust vector: True- Body frame, False - Hill's frame
    """

    base_units = BaseUnits("meters", "seconds", "radians")

    def __init__(
        self,
        sim=None,
        m=M_DEFAULT,
        inertia_matrix=INERTIA_MATRIX_DEFAULT,
        ang_acc_limit=ANG_ACC_LIMIT_DEFAULT,
        ang_vel_limit=ANG_VEL_LIMIT_DEFAULT,
        inertia_wheel=INERTIA_WHEEL_DEFAULT,
        acc_limit_wheel=ACC_LIMIT_WHEEL_DEFAULT,
        vel_limit_wheel=VEL_LIMIT_WHEEL_DEFAULT,
        thrust_control_limit=THRUST_CONTROL_LIMIT_DEFAULT,
        body_frame_thrust=True,
        n=N_DEFAULT,
        trajectory_samples=0,
        integration_method="RK45",
        **kwargs
    ):
        self._state = np.array([])

        # Define limits for angular acceleration, angular velocity, and control inputs
        ang_acc_limit = number_list_to_np(ang_acc_limit, shape=(3, ))  # rad/s^2
        ang_vel_limit = number_list_to_np(ang_vel_limit, shape=(3, ))  # rad/s

        acc_limit_combined = np.zeros((3, ))
        vel_limit_combined = np.zeros((3, ))
        control_limit = np.zeros((6, ))
        for i in range(3):
            acc_limit_combined[i] = min(ang_acc_limit[i], inertia_wheel * acc_limit_wheel / inertia_matrix[i, i])
            vel_limit_combined[i] = min(ang_vel_limit[i], inertia_wheel * vel_limit_wheel / inertia_matrix[i, i])
            control_limit[i] = thrust_control_limit
            control_limit[i + 3] = acc_limit_combined[i] * inertia_matrix[i, i]

        control_default = np.zeros((6, ))
        control_min = -1 * control_limit
        control_max = control_limit
        control_map = {
            'thrust_x': 0,
            'thrust_y': 1,
            'thrust_z': 2,
            'moment_x': 3,
            'moment_y': 4,
            'moment_z': 5,
        }
        """ Create instance of dynamics class """
        dynamics = SixDOFDynamics(
            m=m,
            inertia_matrix=inertia_matrix,
            ang_acc_limit=acc_limit_combined,
            ang_vel_limit=vel_limit_combined,
            n=n,
            body_frame_thrust=body_frame_thrust,
            trajectory_samples=trajectory_samples,
            integration_method=integration_method,
        )
        self.lead = None

        super().__init__(
            dynamics, control_default=control_default, control_min=control_min, control_max=control_max, control_map=control_map, **kwargs
        )
        self._sim = sim

    @classmethod
    def _get_config_validator(cls):
        return SixDOFSpacecraftValidator

    def __eq__(self, other):
        if isinstance(other, SixDOFSpacecraft):
            eq = (self.velocity == other.velocity).all()
            eq = eq and (self.position == other.position).all()
            eq = eq and (self.quaternion == other.quaternion).all()
            eq = eq and (self.angular_velocity == other.angular_velocity).all()
            return eq
        return False

    def register_lead(self, lead: BaseRotationEntity):
        """
        Register another entity as this entity's lead. Defines line of communication between entities.

        Parameters
        ----------
        lead: BaseRotationEntity
            Entity with line of communication to this entity.

        Returns
        -------
        None
        """
        self.lead = lead

    def _build_state(self):
        """form state vector"""
        state = np.array(
            [self.config.x, self.config.y, self.config.z] + [self.config.q1, self.config.q2, self.config.q3, self.config.q4] +
            [self.config.x_dot, self.config.y_dot, self.config.z_dot] + [self.config.wx, self.config.wy, self.config.wz],
            dtype=np.float32
        )

        return state

    @property
    def x(self):
        return self._state[0]

    @property
    def y(self):
        return self._state[1]

    @property
    def z(self):
        return self._state[2]

    @property
    def q1(self):
        """get first element of quaternion"""
        return self._state[3]

    @property
    def q2(self):
        """get second element of quaternion"""
        return self._state[4]

    @property
    def q3(self):
        """get third element of quaternion"""
        return self._state[5]

    @property
    def q4(self):
        """get fourth element of quaternion (scalar)"""
        return self._state[6]

    @property
    def x_dot(self):
        """get x_dot, the velocity component in the x direction"""
        return self._state[7]

    @property
    def x_dot_with_units(self):
        """Get x_dot as a pint.Quantity with units"""
        return self.ureg.Quantity(self.x_dot, self.base_units.velocity)

    @property
    def y_dot(self):
        """get y_dot, the velocity component in the y direction"""
        return self._state[8]

    @property
    def y_dot_with_units(self):
        """Get y_dot as a pint.Quantity with units"""
        return self.ureg.Quantity(self.y_dot, self.base_units.velocity)

    @property
    def z_dot(self):
        """get z_dot, the velocity component in the z axis"""
        return self._state[9]

    @property
    def z_dot_with_units(self):
        """Get z_dot as a pint.Quantity with units"""
        return self.ureg.Quantity(self.z_dot, self.base_units.velocity)

    @property
    def wx(self):
        return self._state[10]

    @property
    def wy(self):
        return self._state[11]

    @property
    def wz(self):
        return self._state[12]

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

            In this implementation local frame is body, global frame is Hill's frame
            Quaternion order (scalar in 4th element) matches scipy convention of [x,y,z,w]
        """
        return Rotation.from_quat([self.q1, self.q2, self.q3, self.q4])

    @property
    def quaternion(self):
        """get 4d quaternion"""
        return self.orientation.as_quat()

    @property
    def velocity(self):
        """Get 3d velocity vector"""
        return np.array([self.x_dot, self.y_dot, self.z_dot])

    @property
    def angular_velocity(self):
        """Get 3d angular velocity vector"""
        return np.array([self.wx, self.wy, self.wz])

    def set_sim(self, sim):
        """sets internal sim reference

        Parameters
        ----------
        sim
            sim to set internal sim reference to
        """
        self._sim = sim

    def entity_relative_position(self, entity_name) -> np.ndarray:
        """Returns the position of another entitiy relative to this entities position

        Parameters
        ----------
        entity_name: str
            name of entity to get relative position of

        Returns
        -------
        np.ndarray
            3d relative position of other entity
        """
        other_entity = self._sim.sim_entities[entity_name]
        return other_entity.position - self.position

    def entity_relative_velocity(self, entity_name) -> np.ndarray:
        """Returns the position of another entitiy relative to this entities position

        Parameters
        ----------
        entity_name: str
            name of entity to get relative position of

        Returns
        -------
        np.ndarray
            3d relative position of other entity
        """
        other_entity = self._sim.sim_entities[entity_name]
        return other_entity.velocity - self.velocity


class SixDOFDynamics(BaseControlAffineODESolverDynamics):
    """
    State transition implementation of 3D Clohessy-Wiltshire dynamics model and 3D rotational dynamics model.

    Parameters
    ----------
    m: float
        Mass of spacecraft in kilograms, by default 12
    inertia_matrix: float
        Inertia matrix of spacecraft (3x3) in kg*m^2
    ang_acc_limit: float, list, np.ndarray
        Angular acceleration limit in rad/s^2. If array_like, applied to x, y, z elementwise
    ang_vel_limit: float, list, np.ndarray
        Angular velocity limit in rad/s. If array_like, applied to x, y, z elementwise
    thrust_control_limit: float
        Thrust control limit in N
    n: float
        Orbital mean motion of Hill's reference frame's circular orbit in rad/s, by default 0.001027
    body_frame_thrust: bool
        Flag indicating the reference frame for the control thrust vector: True- Body frame, False - Hill's frame
        by default, True
    kwargs:
        Additional keyword arguments passed to parent class BaseLinearODESolverDynamics constructor
    """

    def __init__(
        self,
        m=M_DEFAULT,
        inertia_matrix=INERTIA_MATRIX_DEFAULT,
        ang_acc_limit=ANG_ACC_LIMIT_DEFAULT,
        ang_vel_limit=ANG_VEL_LIMIT_DEFAULT,
        n=N_DEFAULT,
        body_frame_thrust=True,
        state_max: Union[float, np.ndarray, None] = None,
        state_min: Union[float, np.ndarray, None] = None,
        angle_wrap_centers: Union[np.ndarray, None] = None,
        **kwargs
    ):
        self.m = m  # kg
        self.inertia_matrix = inertia_matrix  # kg*m^2
        self.n = n  # rads/s
        self.body_frame_thrust = body_frame_thrust
        self.control_thrust_Hill = np.zeros(3, )

        ang_acc_limit = number_list_to_np(ang_acc_limit, shape=(3, ))  # rad/s^2
        ang_vel_limit = number_list_to_np(ang_vel_limit, shape=(3, ))  # rad/s

        A, B = generate_cwh_matrices(self.m, self.n, '3d')

        assert len(A.shape) == 2, f"A must be square matrix. Instead got shape {A.shape}"
        assert len(B.shape) == 2, f"A must be square matrix. Instead got shape {B.shape}"
        assert A.shape[0] == A.shape[1], f"A must be a square matrix, not dimension {A.shape}"
        assert A.shape[1] == B.shape[0], (
            "number of columns in A must match the number of rows in B." + f" However, got shapes {A.shape} for A and {B.shape} for B"
        )

        self.A = np.copy(A)
        self.B = np.copy(B)

        if state_min is None:
            state_min = np.array(
                [
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -ang_vel_limit[0],
                    -ang_vel_limit[1],
                    -ang_vel_limit[2]
                ]
            )

        if state_max is None:
            state_max = np.array(
                [
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    ang_vel_limit[0],
                    ang_vel_limit[1],
                    ang_vel_limit[2]
                ]
            )

        super().__init__(state_min=state_min, state_max=state_max, angle_wrap_centers=angle_wrap_centers, **kwargs)

    def state_transition_system(self, state: np.ndarray) -> np.ndarray:

        x, y, z, q1, q2, q3, q4, x_dot, y_dot, z_dot, wx, wy, wz = state

        # Compute translational derivatives
        pos_vel_state_vec = np.array([x, y, z, x_dot, y_dot, z_dot], dtype=np.float64)
        pos_vel_derivative = self.A @ pos_vel_state_vec

        # Compute rotational derivatives
        q_derivative = np.zeros((4, ))
        w_derivative = np.zeros((3, ))
        q_derivative[0] = 0.5 * (q4 * wx - q3 * wy + q2 * wz)
        q_derivative[1] = 0.5 * (q3 * wx + q4 * wy - q1 * wz)
        q_derivative[2] = 0.5 * (-q2 * wx + q1 * wy + q4 * wz)
        q_derivative[3] = 0.5 * (-q1 * wx - q2 * wy - q3 * wz)
        w_derivative[0] = 1 / self.inertia_matrix[0, 0] * ((self.inertia_matrix[1, 1] - self.inertia_matrix[2, 2]) * wy * wz)
        w_derivative[1] = 1 / self.inertia_matrix[1, 1] * ((self.inertia_matrix[2, 2] - self.inertia_matrix[0, 0]) * wx * wz)
        w_derivative[2] = 1 / self.inertia_matrix[2, 2] * ((self.inertia_matrix[0, 0] - self.inertia_matrix[1, 1]) * wx * wy)

        # Form derivative array
        state_derivative = np.array(
            [
                pos_vel_derivative[0],
                pos_vel_derivative[1],
                pos_vel_derivative[2],
                q_derivative[0],
                q_derivative[1],
                q_derivative[2],
                q_derivative[3],
                pos_vel_derivative[3],
                pos_vel_derivative[4],
                pos_vel_derivative[5],
                w_derivative[0],
                w_derivative[1],
                w_derivative[2]
            ],
            dtype=np.float32
        )
        return state_derivative

    def state_transition_input(self, state: np.ndarray) -> np.ndarray:
        quat = state[3:7]

        w_derivative = np.array(
            [[1 / self.inertia_matrix[0, 0], 0, 0], [0, 1 / self.inertia_matrix[1, 1], 0], [0, 0, 1 / self.inertia_matrix[2, 2]]]
        )

        # Convert the control thrust to Hill's frame prior to application in the CWH equations
        if self.body_frame_thrust:
            r1 = 1 / self.m * self.apply_quat(np.array([1, 0, 0]), quat)
            r2 = 1 / self.m * self.apply_quat(np.array([0, 1, 0]), quat)
            r3 = 1 / self.m * self.apply_quat(np.array([0, 0, 1]), quat)
            vel_derivative = np.array([[r1[0], r2[0], r3[0]], [r1[1], r2[1], r3[1]], [r1[2], r2[2], r3[2]]])
        else:
            vel_derivative = self.B[3:6, :]

        g = np.vstack(
            (
                np.zeros((7, 6)),
                np.hstack((vel_derivative, np.zeros(vel_derivative.shape))),
                np.hstack((np.zeros(w_derivative.shape), w_derivative))
            )
        )

        return g

    def apply_quat(self, x: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """
        Apply quaternion rotation to 3d vector

        Parameters
        ----------
        x : np.ndarray
            vector of length 3
        quat : np.ndarray
            quaternion vector of form [x, y, z, w]

        Returns
        -------
        np.ndarray
            rotated vector of length 3
        """
        p = np.insert(x, 0, 0, axis=0)
        r = np.array([quat[3], quat[0], quat[1], quat[2]])
        r_p = np.array([quat[3], -quat[0], -quat[1], -quat[2]])
        rotated_x = self.hamilton_product(self.hamilton_product(r, p), r_p)[1:]
        return rotated_x

    def hamilton_product(self, r, q):
        """Hamilton product between 2 vectors"""
        return np.array(
            [
                r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3],
                r[0] * q[1] + r[1] * q[0] + r[2] * q[3] - r[3] * q[2],
                r[0] * q[2] - r[1] * q[3] + r[2] * q[0] + r[3] * q[1],
                r[0] * q[3] + r[1] * q[2] - r[2] * q[1] + r[3] * q[0]
            ]
        )
