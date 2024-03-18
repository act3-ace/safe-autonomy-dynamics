"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Safe Autonomy Dynamics.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module implements 2D and 3D Aircraft entities with Dubins physics dynamics models.
"""

import abc
from typing import Union

import numpy as np
import pint
from pydantic import AfterValidator
from scipy.spatial.transform import Rotation
from typing_extensions import Annotated

from safe_autonomy_dynamics.base_models import (
    BaseControlAffineODESolverDynamics,
    BaseEntity,
    BaseEntityValidator,
    BaseUnits,
    build_unit_conversion_validator_fn,
)


class BaseDubinsAircraftValidator(BaseEntityValidator):
    """
    Base validator for Dubins Aircraft implementations.

    Parameters
    ----------
    x : float or pint.Quantity
        Initial x position
    y : float or pint.Quantity
        Initial y position
    z : float or pint.Quantity
        Initial z position
    heading : float or pint.Quantity
        Initial angle of velocity vector relative to x-axis. Right hand rule sign convention.
    v : float or pint.Quantity
        Initial velocity magnitude, aka speed, of dubins entity.

    Raises
    ------
    ValueError
        Improper list length for parameter 'position'
    """

    x: Annotated[Union[float, pint.Quantity], AfterValidator(build_unit_conversion_validator_fn('feet'))] = 0
    y: Annotated[Union[float, pint.Quantity], AfterValidator(build_unit_conversion_validator_fn('feet'))] = 0
    z: Annotated[Union[float, pint.Quantity], AfterValidator(build_unit_conversion_validator_fn('feet'))] = 0
    heading: Annotated[Union[float, pint.Quantity], AfterValidator(build_unit_conversion_validator_fn('radians'))] = 0
    v: Annotated[Union[float, pint.Quantity], AfterValidator(build_unit_conversion_validator_fn('ft/s'))] = 200


class BaseDubinsAircraft(BaseEntity):
    """
    Base interface for Dubins Entities.
    """

    base_units = BaseUnits("feet", "seconds", "radians")

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
        return BaseDubinsAircraftValidator

    def __eq__(self, other):
        if isinstance(other, BaseDubinsAircraft):
            eq = (self.velocity == other.velocity).all()
            eq = eq and (self.position == other.position).all()
            eq = eq and (self.orientation.as_euler("ZYX") == other.orientation.as_euler("ZYX")).all()
            eq = eq and self.heading == other.heading
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
    @abc.abstractmethod
    def v(self):
        """Get v, the velocity magnitude. aka speed."""
        raise NotImplementedError

    @property
    def v_with_units(self):
        """Get v as a pint.Quantity with units"""
        return self.ureg.Quantity(self.v, self.base_units.velocity)

    @property
    def yaw(self):
        """
        Get yaw. Equivalent to heading for Dubins model

        Intrinsic Euler Angle Z of ZYX
        """
        return self.heading

    @property
    def yaw_with_units(self):
        """Get yaw as a pint.Quantity with units"""
        return self.ureg.Quantity(self.yaw, self.base_units.angle)

    @property
    def pitch(self):
        """
        Get pitch. Equivalent to gamma for Dubins model

        Intrinsic Euler Angle Y of ZYX
        """
        return self.gamma

    @property
    def pitch_with_units(self):
        """Get pitch as a pint.Quantity with units"""
        return self.ureg.Quantity(self.pitch, self.base_units.angle)

    @property
    @abc.abstractmethod
    def roll(self):
        """
        Get roll.

        Intrinsic Euler Angle X of ZYX
        """
        raise NotImplementedError

    @property
    def roll_with_units(self):
        """Get roll as a pint.Quantity with units"""
        return self.ureg.Quantity(self.roll, self.base_units.angle)

    @property
    @abc.abstractmethod
    def heading(self):
        """
        Get heading, the angle of velocity vector projected to the xy-plane wrt the x-axis.
        Right hand rule sign convention.
        """
        raise NotImplementedError

    @property
    def heading_with_units(self):
        """Get heading as a pint.Quantity with units"""
        return self.ureg.Quantity(self.heading, self.base_units.angle)

    @property
    @abc.abstractmethod
    def gamma(self):
        """
        Get gamma, aka flight path angle, the angle of the velocity vector relative to the xy-plane.
        Right hand rule sign convention.
        """
        raise NotImplementedError

    @property
    def gamma_with_units(self):
        """Get gamma as a pint.Quantity with units"""
        return self.ureg.Quantity(self.gamma, self.base_units.angle)

    @property
    @abc.abstractmethod
    def acceleration(self):
        """Get 3d acceleration vector"""
        raise NotImplementedError

    @property
    def acceleration_with_units(self):
        """Get acceleration as a pint.Quantity with units"""
        return self.ureg.Quantity(self.acceleration, self.base_units.acceleration)

    @property
    def velocity(self):
        """Get 3d velocity vector"""
        velocity = np.array(
            [
                self.v * np.cos(self.heading) * np.cos(self.gamma),
                self.v * np.sin(self.heading) * np.cos(self.gamma),
                -1 * self.v * np.sin(self.gamma),
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
            Rotation tranformation yielding the entity's local reference frame basis vectors in global reference frame coordinates.
            i.e. applying this rotation to [1, 0, 0] yields the entity's local x-axis (i.e. direction of nose) in the global frame.
            For Dubins, derived from yaw, pitch, roll attributes.
        """
        return Rotation.from_euler("ZYX", [self.yaw, self.pitch, self.roll])


############################
# 2d Dubins Implementation #
############################


class Dubins2dAircraft(BaseDubinsAircraft):
    """
    2D Dubins Aircraft Simulation Entity.

    States
        x
        y
        heading
            range = [-pi, pi] rad
        v
            default range = [200, 400] ft/s

    Controls
        heading_rate
            default range = [-pi/18, pi/18] rad/s (i.e. +/- 10 deg/s)
        acceleration
            default range = [-96.5, 96.5] ft/s^2

    Parameters
    ----------
    trajectory_samples : int
        number of trajectory samples the generate and store on steps
    integration_method: str
        Numerical integration method passed to dynamics model. See BaseODESolverDynamics
    v_min : float
        min velocity state value, by default 200
        ft/s
    v_max : float
        max velocity state value, by default 400
        ft/s
    heading_rate_control_min : float
        min heading rate control value. Control Values outside this bound will be clipped, by default -pi/18 (-10 deg / s)
        radians / s
    heading_rate_control_max : float
        max heading rate control value. Control Values outside this bound will be clipped, by default pi/18 (10 deg / s)
        radians / s
    acceleration_control_min : float
        min acceleration control value. Control Values outside this bound will be clipped, by default -96.5
        ft / s^2
    acceleration_control_max : float
        max acceleration control value. Control Values outside this bound will be clipped, by default 96.5
        ft / s^2
    kwargs
        Additional keyword args passed to BaseDubinsAircraftValidator
    """

    def __init__(
        self,
        trajectory_samples=0,
        integration_method="RK45",
        v_min=200,
        v_max=400,
        heading_rate_control_min=-0.174533,
        heading_rate_control_max=0.174533,
        acceleration_control_min=-96.5,
        acceleration_control_max=96.5,
        **kwargs
    ):

        state_min = np.array([-np.inf, -np.inf, -np.inf, v_min], dtype=np.float32)
        state_max = np.array([np.inf, np.inf, np.inf, v_max], dtype=np.float32)
        angle_wrap_centers = np.array([None, None, 0, None], dtype=np.float32)

        control_default = np.zeros((2, ))
        control_min = np.array([heading_rate_control_min, acceleration_control_min])
        control_max = np.array([heading_rate_control_max, acceleration_control_max])
        control_map = {
            'heading_rate': 0,
            'acceleration': 1,
        }

        dynamics = Dubins2dDynamics(
            trajectory_samples=trajectory_samples,
            state_min=state_min,
            state_max=state_max,
            angle_wrap_centers=angle_wrap_centers,
            integration_method=integration_method,
        )

        super().__init__(
            dynamics, control_default=control_default, control_min=control_min, control_max=control_max, control_map=control_map, **kwargs
        )

    def __eq__(self, other):
        if isinstance(other, Dubins2dAircraft):
            eq = (self.velocity == other.velocity).all()
            eq = eq and (self.position == other.position).all()
            eq = eq and (self.acceleration == other.acceleration).all()
            eq = eq and (self.orientation.as_euler("ZYX") == other.orientation.as_euler("ZYX")).all()
            eq = eq and self.heading == other.heading
            eq = eq and self.roll == other.roll
            eq = eq and self.gamma == other.gamma
            return eq
        return False

    def _build_state(self):
        return np.array([self.config.x, self.config.y, self.config.heading, self.config.v], dtype=np.float32)

    @property
    def x(self):
        return self._state[0]

    @x.setter
    def x(self, value):
        self._state[0] = value

    @property
    def y(self):
        return self._state[1]

    @y.setter
    def y(self, value):
        self._state[1] = value

    @property
    def z(self):
        return 0

    @property
    def heading(self):
        return self._state[2]

    @heading.setter
    def heading(self, value):
        self._state[2] = value

    @property
    def v(self):
        return self._state[3]

    @v.setter
    def v(self, value):
        self._state[3] = value

    @property
    def position(self):
        position = np.zeros((3, ))
        position[0:2] = self._state[0:2]
        return position

    @property
    def gamma(self):
        """
        Get gamma, aka flight path angle, the angle of the velocity vector relative to the xy-plane.
        Right hand rule sign convention.
        Always 0 for Dubins 2D.
        """
        return 0

    @property
    def roll(self):
        """
        Get roll. Always 0 for Dubins 2D.
        """
        return 0

    @property
    def acceleration(self):
        acc = self.state_dot[3]
        acc = acc * (self.velocity / self.v)  # acc * unit velocity
        return acc


class Dubins2dDynamics(BaseControlAffineODESolverDynamics):
    """
    State transition implementation of non-linear 2D Dubins dynamics model.
    """

    def state_transition_system(self, state: np.ndarray) -> np.ndarray:
        _, _, heading, v = state

        x_dot = v * np.cos(heading)
        y_dot = v * np.sin(heading)

        state_dot = np.array([x_dot, y_dot, 0, 0])

        return state_dot

    def state_transition_input(self, state: np.ndarray) -> np.ndarray:
        g = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
        return g


############################
# 3D Dubins Implementation #
############################


class Dubins3dAircraftValidator(BaseDubinsAircraftValidator):
    """
    Validator for Dubins3dAircraft.

    Parameters
    ----------
    gamma : float or pint.Quantity
        Initial gamma value of Dubins3dAircraft in radians
    roll : float or pint.Quantity
        Initial roll value of Dubins3dAircraft in radians
    """
    gamma: Annotated[Union[float, pint.Quantity], AfterValidator(build_unit_conversion_validator_fn('radians'))] = 0
    roll: Annotated[Union[float, pint.Quantity], AfterValidator(build_unit_conversion_validator_fn('radians'))] = 0


class Dubins3dAircraft(BaseDubinsAircraft):
    """
    3D Dubins Aircraft Simulation Entity.

    States
        x
        y
        z
        heading
            range = [-pi, pi] rad
        gamma
            default range = [-pi/9, pi/9] rad
        roll
            default range = [-pi/3, pi/3] rad
        v
            default range = [200, 400] ft/s

    Controls
        gamma_rate
            default range = [-pi/18, pi/18] rad/s (i.e. +/- 10 deg/s)
        roll_rate
            default range = [-pi/36, pi/36] rad/s (i.e. +/- 5 deg/s)
        acceleration
            default range = [-96.5, 96.5] ft/s^2

    Parameters
    ----------
    trajectory_samples : int
        number of trajectory samples the generate and store on steps
    integration_method: str
        Numerical integration method passed to dynamics model. See BaseODESolverDynamics
    v_min : float
        min velocity state value, by default 200
        ft/s
    v_max : float
        max velocity state value, by default 400
        ft/s
    gamma_min : float
        min gamma state value allowed by dynamical system equation. System state flow outside of this bound will be clipped.
        By default, -pi/9 (-20 deg)
        radians
    gamma_max : float
        max gamma state value allowed by dynamical system equation. System state flow outside of this bound will be clipped.
        By default, pi/9 (20 deg)
        radians
    roll_min : float
        min roll state value allowed by dynamical system equation. System state flow outside of this bound will be clipped.
        By default, -pi/3 (-60 deg)
        radians
    roll_max : float
        max roll state value allowed by dynamical system equation. System state flow outside of this bound will be clipped.
        By default, pi/3 (60 deg)
        radians
    gamma_rate_control_min : float
        min gamma rate control value. Control Values outside this bound will be clipped, by default -pi/18 (-10 deg / s)
        radians / s
    gamma_rate_control_max : float
        max gamma rate control value. Control Values outside this bound will be clipped, by default pi/18 (10 deg / s)
        radians / s
    roll_rate_control_min : float
        min roll rate control value. Control Values outside this bound will be clipped, by default -pi/36 (-5 deg / s)
        radians / s
    roll_rate_control_max : float
        max roll rate control value. Control Values outside this bound will be clipped, by default pi/36 (5 deg / s)
        radians / s
    acceleration_control_min : float
        min acceleration control value. Control Values outside this bound will be clipped, by default -96.5
        ft / s^2
    acceleration_control_max : float
        max acceleration control value. Control Values outside this bound will be clipped, by default 96/5
        ft / s^2
    kwargs
        Additional keyword args passed to BaseDubinsAircraftValidator
    """

    def __init__(
        self,
        trajectory_samples=0,
        integration_method='RK45',
        v_min=200,
        v_max=400,
        gamma_min=-0.349066,
        gamma_max=0.349066,
        roll_min=-1.047198,
        roll_max=1.047198,
        gamma_rate_control_min=-0.174533,
        gamma_rate_control_max=0.174533,
        roll_rate_control_min=-0.087266,
        roll_rate_control_max=0.087266,
        acceleration_control_min=-96.5,
        acceleration_control_max=96.5,
        **kwargs
    ):

        state_min = np.array([-np.inf, -np.inf, -np.inf, -np.inf, gamma_min, roll_min, v_min], dtype=np.float32)
        state_max = np.array([np.inf, np.inf, np.inf, np.inf, gamma_max, roll_max, v_max], dtype=np.float32)
        angle_wrap_centers = np.array([None, None, None, 0, 0, 0, None], dtype=np.float32)

        control_default = np.zeros((3, ))
        control_min = np.array([gamma_rate_control_min, roll_rate_control_min, acceleration_control_min])
        control_max = np.array([gamma_rate_control_max, roll_rate_control_max, acceleration_control_max])
        control_map = {
            'gamma_rate': 0,
            'roll_rate': 1,
            'acceleration': 2,
        }

        dynamics = Dubins3dDynamics(
            trajectory_samples=trajectory_samples,
            state_min=state_min,
            state_max=state_max,
            angle_wrap_centers=angle_wrap_centers,
            integration_method=integration_method,
        )

        super().__init__(
            dynamics, control_default=control_default, control_min=control_min, control_max=control_max, control_map=control_map, **kwargs
        )

    def __eq__(self, other):
        if isinstance(other, Dubins3dAircraft):
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

    @classmethod
    def _get_config_validator(cls):
        return Dubins3dAircraftValidator

    def _build_state(self):
        return np.array(
            [self.config.x, self.config.y, self.config.z, self.config.heading, self.config.gamma, self.config.roll, self.config.v],
            dtype=np.float32
        )

    @property
    def x(self):
        return self._state[0]

    @x.setter
    def x(self, value):
        self._state[0] = value

    @property
    def y(self):
        return self._state[1]

    @y.setter
    def y(self, value):
        self._state[1] = value

    @property
    def z(self):
        return self._state[2]

    @z.setter
    def z(self, value):
        self._state[2] = value

    @property
    def heading(self):
        return self._state[3]

    @heading.setter
    def heading(self, value):
        self._state[3] = value

    @property
    def gamma(self):
        return self._state[4]

    @gamma.setter
    def gamma(self, value):
        self._state[4] = value

    @property
    def roll(self):
        return self._state[5]

    @roll.setter
    def roll(self, value):
        self._state[5] = value

    @property
    def v(self):
        return self._state[6]

    @v.setter
    def v(self, value):
        self._state[6] = value

    @property
    def position(self):
        position = self._state[0:3].copy()
        return position

    @property
    def acceleration(self):
        acc = self.state_dot[6]
        acc = acc * (self.velocity / self.v)  # acc * unit velocity
        return acc


class Dubins3dDynamics(BaseControlAffineODESolverDynamics):
    """
    State transition implementation of non-linear 3D Dubins dynamics model.

    Parameters
    ----------
    g : float
        gravitational acceleration constant if ft/s^2
    kwargs
        Additional keyword args passed to parent BaseODESolverDynamics constructor
    """

    def __init__(self, g=32.17, **kwargs):
        self.g = g
        super().__init__(**kwargs)

    def state_transition_system(self, state: np.ndarray) -> np.ndarray:
        _, _, _, heading, gamma, roll, v = state

        x_dot = v * np.cos(heading) * np.cos(gamma)
        y_dot = v * np.sin(heading) * np.cos(gamma)
        z_dot = -1 * v * np.sin(gamma)
        heading_dot = (self.g / v) * np.tan(roll)  # g = 32.17 ft/s^2

        state_dot = np.array([x_dot, y_dot, z_dot, heading_dot, 0, 0, 0])

        return state_dot

    def state_transition_input(self, state: np.ndarray) -> np.ndarray:
        g = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        return g
