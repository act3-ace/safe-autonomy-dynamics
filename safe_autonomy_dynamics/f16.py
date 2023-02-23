"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Safe Autonomy Dynamics.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module implements 3D 6dof F16 entities using flight dynamics from AeroBenchVVPython
(https://github.com/stanleybak/AeroBenchVVPython)
"""

import math

import numpy as np
from scipy.spatial.transform import Rotation

from safe_autonomy_dynamics.base_models import BaseEntity, BaseEntityValidator, BaseODESolverDynamics
from safe_autonomy_dynamics.external.aerobench.controlled_f16 import controlled_f16
from safe_autonomy_dynamics.external.aerobench.lowlevel.low_level_controller import LowLevelController


class AeroBenchF16StateVariables:
    """F-16 state variables"""
    V = 0
    ATTACK = 1
    SIDESLIP = 2
    ROLL = 3
    PITCH = 4
    YAW = 5
    ROLL_RATE = 6
    PITCH_RATE = 7
    YAW_RATE = 8
    X_POS = 9
    Y_POS = 10
    Z_POS = 11
    POWER = 12


class AeroBenchF16Validator(BaseEntityValidator):
    """
    Base validator for AeroBenchF16.

    Parameters
    ----------
    x : float
        Initial x position
    y : float
        Initial y position
    z : float
        Initial z position
    heading : float
        Initial angle of velocity vector relative to x-axis. Right hand rule sign convention.
    v : float
        Initial velocity magnitude, aka speed, of dubins entity.

    """
    x: float = 0
    y: float = 0
    z: float = 0
    v: float = 100
    attack_angle: float = 0
    sideslip_angle: float = 0
    roll: float = 0
    pitch: float = 0
    yaw: float = 0
    roll_rate: float = 0
    pitch_rate: float = 0
    yaw_rate: float = 0
    power: float = 0


class AeroBenchF16(BaseEntity):
    """
    Base interface for AeroBench Entities.
    """

    def __init__(self, integration_method='RK45', model_str='morelli', v2_integrators=False, **kwargs):
        self.partner = None

        # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
        state_min = np.array([-np.inf] * 13, dtype=np.float32)
        state_max = np.array([np.inf] * 13, dtype=np.float32)
        angle_wrap_centers = np.array([None, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None], dtype=np.float32)

        # control = [Nz, ps, Ny_r, throttle]
        control_min = np.array([-np.inf] * 4, dtype=np.float32)
        control_max = np.array([np.inf] * 4, dtype=np.float32)
        control_default = np.array([0] * 4, dtype=np.float32)

        dynamics = AeroBenchF16Dynamics(
            state_min=state_min,
            state_max=state_max,
            angle_wrap_centers=angle_wrap_centers,
            integration_method=integration_method,
            model_str=model_str,
            v2_integrators=v2_integrators
        )

        super().__init__(dynamics, control_default=control_default, control_min=control_min, control_max=control_max, **kwargs)

    @classmethod
    def _get_config_validator(cls):
        return AeroBenchF16Validator

    def _build_state(self) -> np.ndarray:
        return np.array(
            [
                self.config.v,
                self.config.attack_angle,
                self.config.sideslip_angle,
                self.config.roll,
                self.config.pitch,
                self.config.yaw,
                self.config.roll_rate,
                self.config.pitch_rate,
                self.config.yaw_rate,
                self.config.x,
                self.config.y,
                self.config.z,
                self.config.power
            ],
            dtype=np.float32
        )

    def __eq__(self, other):
        if isinstance(other, AeroBenchF16):
            eq = (self.velocity == other.velocity).all()
            eq = eq and (self.position == other.position).all()
            eq = eq and (self.acceleration == other.acceleration).all()
            eq = eq and (self.orientation.as_euler("ZYX") == other.orientation.as_euler("ZYX")).all()
            eq = eq and self.heading == other.heading
            eq = eq and self.roll == other.roll
            eq = eq and self.gamma == other.gamma
            eq = eq and self.v == other.v
            return eq
        return False

    def register_partner(self, partner: BaseEntity):
        """
        Register another entity as this aircraft's partner. Defines line of communication between entities.

        Parameters
        ----------
        partner: BaseEntity
            Entity with line of communication to this aircraft.

        Returns
        -------
        None
        """
        self.partner = partner

    @property
    def x(self):
        return self.state[AeroBenchF16StateVariables.X_POS]

    @property
    def y(self):
        return self.state[AeroBenchF16StateVariables.Y_POS]

    @property
    def z(self):
        return self.state[AeroBenchF16StateVariables.Z_POS]

    @property
    def position(self):
        position = np.array([self.x, self.y, self.z])
        return position

    @property
    def v(self):
        """Get v, the velocity magnitude. aka speed."""
        return self.state[AeroBenchF16StateVariables.V]

    @property
    def attack(self):
        """Get alpha, the angle of attack."""
        return self.state[AeroBenchF16StateVariables.ATTACK]

    @property
    def yaw(self):
        """Get yaw."""
        return self.state[AeroBenchF16StateVariables.YAW]

    @property
    def pitch(self):
        """Get pitch."""
        return self.state[AeroBenchF16StateVariables.PITCH]

    @property
    def roll(self):
        """Get roll."""
        return self.state[AeroBenchF16StateVariables.ROLL]

    @property
    def yaw_rate(self):
        """Get yaw_rate."""
        return self.state[AeroBenchF16StateVariables.YAW_RATE]

    @property
    def pitch_rate(self):
        """Get pitch_rate."""
        return self.state[AeroBenchF16StateVariables.PITCH_RATE]

    @property
    def roll_rate(self):
        """Get roll_rate."""
        return self.state[AeroBenchF16StateVariables.ROLL_RATE]

    @property
    def heading(self):
        """
        Get heading, the angle of velocity relative to the x-axis projected to the xy-plane.
        Right hand rule sign convention.
        """
        # TODO: check if yaw and heading are equivalent (does heading = yaw - sideslip?)
        return self.yaw

    @property
    def gamma(self):
        """
        Get gamma, aka flight path angle, the angle of the velocity vector relative to the xy-plane.
        Right hand rule sign convention.
        """
        return self.pitch - self.attack

    @property
    def acceleration(self):
        """Get 3d acceleration vector"""
        acc = self.state_dot[AeroBenchF16StateVariables.V]
        acc = acc * (self.velocity / self.v)  # acc * unit velocity
        return acc

    @property
    def velocity(self):
        """Get 3d velocity vector"""
        velocity = np.array(
            [
                self.v * math.cos(self.heading) * math.cos(self.gamma),
                self.v * math.sin(self.heading) * math.cos(self.gamma),
                -1 * self.v * math.sin(self.gamma),
            ],
            dtype=np.float32,
        )
        return velocity

    @property
    def orientation(self):
        """
        Get orientation of entity.

        Returns
        -------
        scipy.spatial.transform.Rotation
            Rotation tranformation of the entity's local reference frame basis vectors in the global reference frame.
            i.e. applying this rotation to [1, 0, 0] yields the entity's local x-axis (i.e. direction of nose) in the global frame.
            For Dubins, derived from yaw, pitch, roll attributes.
        """
        return Rotation.from_euler("ZYX", [self.yaw, self.pitch, self.roll])


class AeroBenchF16Dynamics(BaseODESolverDynamics):
    """
    State transition implementation of AeroBenchVV F16.

    Parameters
    ----------
    g : float
        gravitational acceleration constant if ft/s^2
    kwargs
        Additional keyword args passed to parent BaseODESolverDynamics constructor
    """

    def __init__(self, model_str="morelli", v2_integrators=False, **kwargs):
        self.model_str = model_str
        self.v2_integrators = v2_integrators
        self.llc = LowLevelController()
        super().__init__(**kwargs)

    def _compute_state_dot(self, t: float, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        # append integral error states to state vector
        s = np.zeros(len(state) + self.llc.get_num_integrators())
        s[:len(state)] = state

        # compute state derivative
        xd = controlled_f16(s, control, self.llc, self.model_str, self.v2_integrators)[0]

        # remove integral error states from state vector
        xd = xd[:-self.llc.get_num_integrators()]
        return xd
