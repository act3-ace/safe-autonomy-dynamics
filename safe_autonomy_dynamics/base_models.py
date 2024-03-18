"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Safe Autonomy Dynamics.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module provides base implementations for entities in the saferl simulator
"""
from __future__ import annotations

import abc
import warnings
from types import ModuleType
from typing import TYPE_CHECKING, Callable, Tuple, Union

import numpy as np
import pint
import scipy.integrate
import scipy.spatial
from pint import _typing as pintt
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    import jax
    import jax.numpy as jnp
    from jax.experimental.ode import odeint
else:
    try:
        import jax
        import jax.numpy as jnp
        from jax.experimental.ode import odeint
    except ImportError:
        jax = None
        jnp = None
        odeint = None


class BaseEntityValidator(BaseModel):
    """
    Validator for BaseEntity's config member.

    Parameters
    ----------
    name : str
        Name of entity
    """
    name: str
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


def build_unit_conversion_validator_fn(unit: Union[str, pint.Unit]) -> Callable[[Union[float, pint.Quantity]], float]:
    """
    Builds a function for optionally converting an arbitrary unit



    Parameters
    ----------
    unit : Union[str, pint.Unit]
        unit to convert value to in returned Callabe

    Returns
    -------
    Callable[[Union[float, pint.Quantity]], float]
        A function that converts pint.Quantity inputs to the specified unit and returns the magnitude as a float.
        If the input is not a pint.Quantity, it is assumed to be a numeric type in the correct unit and simply cast as float
    """

    def fn(x: Union[float, pint.Quantity]) -> float:
        if isinstance(x, pint.Quantity):
            try:
                return float(x.to(unit).magnitude)
            except pint.errors.DimensionalityError as e:
                # Convert to an error type handled by Pydantic so that conversion issues show up as a Pydantic
                # validation error for easier debugging rather than an error originating here
                raise ValueError from e
        return float(x)

    return fn


class BaseUnits:
    """Provides unit system definitions for entities
    """

    def __init__(self, length: Union[str, pint.Unit], time: Union[str, pint.Unit], angle: Union[str, pint.Unit]):
        self.length: pint.Unit = pint.Unit(length)
        self.time: pint.Unit = pint.Unit(time)
        self.angle: pint.Unit = pint.Unit(angle)

        self.velocity: pint.Unit = self.length / self.time
        self.angular_velocity: pint.Unit = self.angle / self.time

        self.acceleration: pint.Unit = self.length / (self.time**2)
        self.angular_acceleration: pint.Unit = self.angle / (self.time**2)


class BaseEntity(abc.ABC):
    """
    Base implementation of a dynamics controlled entity within the saferl sim.

    Parameters
    ----------
    dynamics : BaseDynamics
        Dynamics object for computing state transitions
    control_default: np.ndarray
        Default control vector used when no action is passed to step(). Typically 0 or neutral for each actuator.
    control_min: np.ndarray
        Optional minimum allowable control vector values. Control vectors that exceed this limit are clipped.
    control_max: np.ndarray
        Optional maximum allowable control vector values. Control vectors that exceed this limit are clipped.
    control_map: dict
        Optional mapping for actuator names to their indices in the state vector.
        Allows dictionary action inputs in step().
    """

    base_units = BaseUnits('meters', 'seconds', 'radians')

    def __init__(self, dynamics, control_default, control_min=-np.inf, control_max=np.inf, control_map=None, **kwargs):
        self.config = self._get_config_validator()(**kwargs)
        self.name = self.config.name
        self.dynamics = dynamics

        self.control_default = control_default
        self.control_min = control_min
        self.control_max = control_max
        self.control_map = control_map

        self._state = self._build_state()
        self.state_dot = np.zeros_like(self._state)

        self.ureg: pint.UnitRegistry = pint.get_application_registry()

    @classmethod
    def _get_config_validator(cls):
        return BaseEntityValidator

    @abc.abstractmethod
    def _build_state(self) -> np.ndarray:
        raise NotImplementedError

    def step(self, step_size, action=None):
        """
        Executes a state transition simulation step for the entity.

        Parameters
        ----------
        step_size : float
            Duration of simulation step in seconds
        action : Union(dict, list, np.ndarray), optional
            Control action taken by entity, by default None resulting in a control of control_default
            When list or ndarray, directly used and control vector for dynamics model
            When dict, unpacked into control vector. Requires control_map to be defined.
        Raises
        ------
        KeyError
            Raised when action dict key not found in control map
        ValueError
            Raised when action is not one of the required types
        """

        if action is None:
            control = self.control_default.copy()
        else:
            if isinstance(action, dict):
                assert self.control_map is not None, "Cannot use dict-type action without a control_map " \
                                                     "(see BaseEntity __init__())"
                control = self.control_default.copy()
                for action_name, action_value in action.items():
                    if action_name not in self.control_map:
                        raise KeyError(
                            f"action '{action_name}' not found in entity's control_map, "
                            f"please use one of: {self.control_map.keys()}"
                        )

                    control[self.control_map[action_name]] = action_value
            elif isinstance(action, list):
                control = np.array(action, dtype=np.float32)
            elif isinstance(action, np.ndarray):
                control = action.copy()
            elif jnp is not None and isinstance(action, jnp.ndarray):  # pylint: disable=used-before-assignment
                control = action.copy()
            else:
                raise ValueError("action must be type dict, list, np.ndarray or jnp.ndarray")

        # enforce control bounds
        if (np.any(control < self.control_min) or np.any(control > self.control_max)):
            warnings.warn(f"Control input exceeded limits. Clipping to range ({self.control_min}, {self.control_max})")
        control = np.clip(control, self.control_min, self.control_max)

        # compute new state if dynamics were applied
        self.state, self.state_dot = self.dynamics.step(step_size, self.state, control)

    @property
    def state(self) -> np.ndarray:
        """
        Returns copy of entity's state vector.

        Returns
        -------
        np.ndarray
            copy of state vector
        """
        return self._state.copy()

    @state.setter
    def state(self, value: np.ndarray):
        self._state = value.copy()

    @property
    @abc.abstractmethod
    def x(self) -> float:
        """get x"""
        raise NotImplementedError

    @property
    def x_with_units(self) -> pint.Quantity:
        """get x as a pint.Quantity with units"""
        return self.ureg.Quantity(self.x, self.base_units.length)

    @property
    @abc.abstractmethod
    def y(self) -> float:
        """get y"""
        raise NotImplementedError

    @property
    def y_with_units(self) -> pint.Quantity:
        """get y as a pint.Quantity with units"""
        return self.ureg.Quantity(self.y, self.base_units.length)

    @property
    @abc.abstractmethod
    def z(self) -> float:
        """get z"""
        raise NotImplementedError

    @property
    def z_with_units(self) -> pint.Quantity:
        """get z as a pint.Quantity with units"""
        return self.ureg.Quantity(self.z, self.base_units.length)

    @property
    @abc.abstractmethod
    def position(self) -> np.ndarray:
        """get 3d position vector"""
        raise NotImplementedError

    @property
    def position_with_units(self) -> pintt.Quantity[np.ndarray]:
        """get position as a pint.Quantity with units"""
        return self.ureg.Quantity(self.position, self.base_units.length)

    @property
    @abc.abstractmethod
    def orientation(self) -> scipy.spatial.transform.Rotation:
        """
        Get orientation of entity.

        Returns
        -------
        scipy.spatial.transform.Rotation
            Rotation transformation of the entity's local reference frame basis vectors in the global reference frame.
            i.e. applying this rotation to [1, 0, 0] yields the entity's local x-axis in the global frame.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def velocity(self) -> np.ndarray:
        """Get 3d velocity vector"""
        raise NotImplementedError

    @property
    def velocity_with_units(self) -> pintt.Quantity[np.ndarray]:
        """get velocity as a pint.Quantity with units"""
        return self.ureg.Quantity(self.velocity, self.base_units.velocity)


class BaseRotationEntity(BaseEntity):
    """
    Base implementation of a base entity with rotational states within the saferl sim.

    Parameters
    ----------
    dynamics : BaseDynamics
        Dynamics object for computing state transitions
    control_default: np.ndarray
        Default control vector used when no action is passed to step(). Typically 0 or neutral for each actuator.
    control_min: np.ndarray
        Optional minimum allowable control vector values. Control vectors that exceed this limit are clipped.
    control_max: np.ndarray
        Optional maximum allowable control vector values. Control vectors that exceed this limit are clipped.
    control_map: dict
        Optional mapping for actuator names to their indices in the state vector.
        Allows dictionary action inputs in step().
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

    @property
    @abc.abstractmethod
    def q1(self) -> float:
        """get first element of quaternion"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def q2(self) -> float:
        """get second element of quaternion"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def q3(self) -> float:
        """get third element of quaternion"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def q4(self) -> float:
        """get fourth element of quaternion (scalar)"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def quaternion(self) -> np.ndarray:
        """get 4d quaternion"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def wx(self) -> float:
        """get wx, the angular velocity component about the local body frame x axis"""
        raise NotImplementedError

    @property
    def wx_with_unit(self) -> pint.Quantity:
        """get wx as a pint.Quantity with units"""
        return self.ureg.Quantity(self.wx, self.base_units.angular_velocity)

    @property
    @abc.abstractmethod
    def wy(self) -> float:
        """get wy, the angular velocity component about the local body frame y axis"""
        raise NotImplementedError

    @property
    def wy_with_unit(self) -> pint.Quantity:
        """get wy as a pint.Quantity with units"""
        return self.ureg.Quantity(self.wy, self.base_units.angular_velocity)

    @property
    @abc.abstractmethod
    def wz(self) -> float:
        """get wz, the angular velocity component about the local body frame z axis"""
        raise NotImplementedError

    @property
    def wz_with_unit(self) -> pint.Quantity:
        """get wz as a pint.Quantity with units"""
        return self.ureg.Quantity(self.wz, self.base_units.angular_velocity)

    @property
    @abc.abstractmethod
    def angular_velocity(self) -> np.ndarray:
        """get 3d angular velocity vector"""
        raise NotImplementedError

    @property
    def angular_velocity_with_units(self) -> pintt.Quantity[np.ndarray]:
        """get 3d angular velocity vector as pint.Quantity with units"""
        return self.ureg.Quantity(self.angular_velocity, self.base_units.angular_velocity)


class BaseDynamics(abc.ABC):
    """
    State transition implementation for a physics dynamics model. Used by entities to compute their next state when
    their step() method is called.

    Parameters
    ----------
    state_min : float or np.ndarray
        Minimum allowable value for the next state. State values that exceed this are clipped.
        When a float, represents single limit applied to entire state vector.
        When an ndarray, each element represents the limit to the corresponding state vector element.
    state_max : float or np.ndarray
        Maximum allowable value for the next state. State values that exceed this are clipped.
        When a float, represents single limit applied to entire state vector.
        When an ndarray, each element represents the limit to the corresponding state vector element.
    angle_wrap_centers: np.ndarray
        Enables circular wrapping of angles. Defines the center of circular wrap such that angles are within
        [center+pi, center-pi].
        When None, no angle wrapping applied.
        When ndarray, each element defines the angle wrap center of the corresponding state element.
        Wrapping not applied when element is NaN.
    use_jax : bool
        True if using jax version of numpy/scipy. By default, False
    """

    def __init__(
        self,
        state_min: Union[float, np.ndarray] = -np.inf,
        state_max: Union[float, np.ndarray] = np.inf,
        angle_wrap_centers: Union[np.ndarray, None] = None,
        use_jax: bool = False,
    ):
        self.state_min = state_min
        self.state_max = state_max
        self.angle_wrap_centers = angle_wrap_centers
        self.use_jax = use_jax

        self.np: ModuleType
        if use_jax:
            if jax is None:  # pylint: disable=used-before-assignment
                raise ImportError("Failed to import jax. Make sure to install jax if using the `use_jax` option")
            self.np = jnp
        else:
            self.np = np

    def step(self, step_size: float, state: np.ndarray, control: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the dynamics state transition from the current state and control input.

        Parameters
        ----------
        step_size : float
            Duration of the simulation step in seconds.
        state : np.ndarray
            Current state of the system at the beginning of the simulation step.
        control : np.ndarray
            Control vector of the dynamics model.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of the system's next state and the state's instantaneous time derivative at the end of the step
        """
        next_state, state_dot = self._step(step_size, state, control)
        next_state = self.np.clip(next_state, self.state_min, self.state_max)
        next_state = self._wrap_angles(next_state)
        return next_state, state_dot

    def _wrap_angles(self, state: Union[np.ndarray, jnp.ndarray]):
        if self.angle_wrap_centers is not None:
            needs_wrap = self.np.logical_not(self.np.isnan(self.angle_wrap_centers))

            wrapped_state = ((state + np.pi) % (2 * np.pi)) - np.pi + self.angle_wrap_centers

            output_state = self.np.where(needs_wrap, wrapped_state, state)
        else:
            output_state = state

        return output_state

    @abc.abstractmethod
    def _step(self, step_size: float, state: np.ndarray, control: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class BaseODESolverDynamics(BaseDynamics):
    """
    State transition implementation for generic Ordinary Differential Equation dynamics models.
    Computes next state through numerical integration of differential equation.

    Parameters
    ----------
    trajectory_samples : int
        number of trajectory samples the generate and store on steps
    state_dot_min : float or np.ndarray
        Minimum allowable value for the state time derivative. State derivative values that exceed this are clipped.
        When a float, represents single limit applied to entire state vector.
        When an ndarray, each element represents the limit to the corresponding state vector element.
        By default, -inf
    state_dot_max : float or np.ndarray
        Maximum allowable value for the state time derivative. State derivative values that exceed this are clipped.
        When a float, represents single limit applied to entire state vector.
        When an ndarray, each element represents the limit to the corresponding state vector element.
        By default, +inf
    integration_method : string
        Numerical integration method used by dynamics solver. One of ['RK45', 'RK45_JAX', 'Euler'].
        'RK45' is slow but very accurate.
        'RK45_JAX' is very accurate, and fast when JIT compiled but otherwise very slow. 'use_jax' must be set to True.
        'Euler' is fast but very inaccurate.
    kwargs
        Additional keyword arguments passed to parent BaseDynamics constructor.
    """

    def __init__(
        self,
        trajectory_samples: int = 0,
        state_dot_min: Union[float, np.ndarray] = -np.inf,
        state_dot_max: Union[float, np.ndarray] = np.inf,
        integration_method="RK45",
        **kwargs
    ):
        self.integration_method = integration_method
        self.state_dot_min = state_dot_min
        self.state_dot_max = state_dot_max

        assert isinstance(trajectory_samples, int), "trajectory_samples must be an integer"
        self.trajectory_samples = trajectory_samples

        self.trajectory = None
        self.trajectory_t = None

        super().__init__(**kwargs)

    def compute_state_dot(self, t: float, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """
        Computes the instantaneous time derivative of the state vector

        Parameters
        ----------
        t : float
            Time in seconds since the beginning of the simulation step.
            Note, this is NOT the total simulation time but the time within the individual step.
        state : np.ndarray
            Current state vector at time t.
        control : np.ndarray
            Control vector.

        Returns
        -------
        np.ndarray
            Instantaneous time derivative of the state vector.
        """
        state_dot = self._compute_state_dot(t, state, control)
        state_dot = self._clip_state_dot_direct(state_dot)
        state_dot = self._clip_state_dot_by_state_limits(state, state_dot)
        return state_dot

    @abc.abstractmethod
    def _compute_state_dot(self, t: float, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _clip_state_dot_direct(self, state_dot):
        return self.np.clip(state_dot, self.state_dot_min, self.state_dot_max)

    def _clip_state_dot_by_state_limits(self, state, state_dot):
        lower_bounded_states = state <= self.state_min
        upper_bounded_states = state >= self.state_max

        lower_bounded_clipped = self.np.clip(state_dot, 0, np.inf)
        upper_bounded_clipped = self.np.clip(state_dot, -np.inf, 0)

        state_dot = self.np.where(lower_bounded_states, lower_bounded_clipped, state_dot)
        state_dot = self.np.where(upper_bounded_states, upper_bounded_clipped, state_dot)

        return state_dot

    def _step(self, step_size, state, control):

        if self.integration_method == "RK45":

            t_eval = None
            if self.trajectory_samples > 0:
                t_eval = np.linspace(0, step_size, self.trajectory_samples + 1)[1:]

            sol = scipy.integrate.solve_ivp(self.compute_state_dot, (0, step_size), state, args=(control, ), t_eval=t_eval)

            self.trajectory = sol.y.T
            self.trajectory_t = sol.t

            next_state = sol.y[:, -1]  # save last timestep of integration solution
            state_dot = self.compute_state_dot(step_size, next_state, control)
        elif self.integration_method == "RK45_JAX":
            if not self.use_jax:
                raise ValueError("use_jax must be set to True if using RK45_JAX")

            assert self.trajectory_samples <= 0, "trajectory sampling not currently supported with rk45 jax integration"

            sol = odeint(  # pylint: disable=used-before-assignment
                self.compute_state_dot_jax, state, jnp.linspace(0., step_size, 11), control
            )
            next_state = sol[-1, :]  # save last timestep of integration solution
            state_dot = self.compute_state_dot(step_size, next_state, control)
        elif self.integration_method == "Euler":
            assert self.trajectory_samples <= 0, "trajectory sampling not currently supported with euler integration"
            state_dot = self.compute_state_dot(0, state, control)
            next_state = state + step_size * state_dot
        else:
            raise ValueError(f"invalid integration method '{self.integration_method}'")

        return next_state, state_dot

    def compute_state_dot_jax(self, state, t, control):
        """Compute state dot for jax odeint
        """
        return self._compute_state_dot(t, state, control)


class BaseControlAffineODESolverDynamics(BaseODESolverDynamics):
    """
    State transition implementation for control affine Ordinary Differential Equation dynamics models of the form
        dx/dt = f(x) + g(x)u.

    At Each point in the numerical integration processes, f(x) and g(x) are computed at the integration point

    Computes next state through numerical integration of differential equation.

    Parameters
    ----------
    kwargs
        Additional keyword arguments passed to parent BaseODESolverDynamics constructor.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _compute_state_dot(self, t: float, state: np.ndarray, control: np.ndarray):
        state_dot = self.state_transition_system(state) + self.state_transition_input(state) @ control
        return state_dot

    @abc.abstractmethod
    def state_transition_system(self, state: np.ndarray) -> np.ndarray:
        """Computes the system state contribution to the system state's time derivative

        i.e. implements f(x) from dx/dt = f(x) + g(x)u

        Parameters
        ----------
        state : np.ndarray
            Current state vector of the system.

        Returns
        -------
        np.ndarray
            state time derivative contribution from the current system state
        """
        raise NotImplementedError

    @abc.abstractmethod
    def state_transition_input(self, state: np.ndarray) -> np.ndarray:
        """Computes the control input matrix contribution to the system state's time derivative

        i.e. implements g(x) from dx/dt = f(x) + g(x)u

        Parameters
        ----------
        state : np.ndarray
            Current state vector of the system.

        Returns
        -------
        np.ndarray
            input matrix in state space representation time derivative
        """
        raise NotImplementedError


class BaseLinearODESolverDynamics(BaseControlAffineODESolverDynamics):
    """
    State transition implementation for generic Linear Ordinary Differential Equation dynamics models of the form
    dx/dt = Ax+Bu.
    Computes next state through numerical integration of differential equation.

    Parameters
    ----------
    A : np.ndarray
        State transition matrix. A of dx/dt = Ax + Bu. Should be dimension len(n) x len(n)
    B : npndarray
        Control input matrix. B of dx/dt = Ax + Bu. Should be dimension len(n) x len(u)
    kwargs
        Additional keyword arguments passed to parent BaseVectorizedODESolverDynamics constructor.
    """

    def __init__(self, A: np.ndarray, B: np.ndarray, **kwargs):
        super().__init__(**kwargs)

        assert len(A.shape) == 2, f"A must be square matrix. Instead got shape {A.shape}"
        assert len(B.shape) == 2, f"A must be square matrix. Instead got shape {B.shape}"
        assert A.shape[0] == A.shape[1], f"A must be a square matrix, not dimension {A.shape}"
        assert A.shape[1] == B.shape[0], (
            "number of columns in A must match the number of rows in B." + f" However, got shapes {A.shape} for A and {B.shape} for B"
        )

        self.A = self.np.copy(A)
        self.B = self.np.copy(B)

    def state_transition_system(self, state: np.ndarray) -> np.ndarray:
        return self.A @ state

    def state_transition_input(self, state: np.ndarray) -> np.ndarray:
        return self.B
