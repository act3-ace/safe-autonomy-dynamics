"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Safe Autonomy Dynamics.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module implements a sun model in non-inertial orbital Hill's reference frame.
"""

from typing import Union

import numpy as np
import pint
import scipy
from pydantic import AfterValidator
from typing_extensions import Annotated

from safe_autonomy_dynamics.base_models import (
    BaseEntity,
    BaseEntityValidator,
    BaseODESolverDynamics,
    BaseUnits,
    build_unit_conversion_validator_fn,
)
from safe_autonomy_dynamics.cwh import N_DEFAULT


class SunEntityValidator(BaseEntityValidator):
    """
    Validator for SunEntity kwargs.

    Parameters
    ----------
    theta: float or pint.Quantity
       Length 1, rotation angle value. rad

    Raises
    ------
    ValueError
        Improper list lengths for parameter 'theta'
    """
    theta: Annotated[Union[float, pint.Quantity], AfterValidator(build_unit_conversion_validator_fn('radians'))] = 0


class SunEntity(BaseEntity):
    """
    Sun in Hill's reference frame.
    Assumed to rotate in x-y plane with angular velocity "n"

    States
        theta

    Parameters
    ----------
    n: float
        Orbital mean motion of Hill's reference frame's circular orbit in rad/s, by default 0.001027.
    integration_method: str
        Numerical integration method passed to dynamics model. See BaseODESolverDynamics.
    use_jax : bool
        True if using jax version of numpy/scipy in dynamics model. By default, False
    kwargs:
        Additional keyword arguments passed to parent class BaseRotationEntity.
    """

    base_units = BaseUnits("meters", "seconds", "radians")

    def __init__(self, n=N_DEFAULT, integration_method="RK45", use_jax: bool = False, **kwargs):
        self._state = np.array([])

        self.n = n  # rads/s
        """ Create instance of dynamics class """
        dynamics = SunDynamics(n=n, integration_method=integration_method, use_jax=use_jax)

        super().__init__(dynamics, control_default=np.array([]), **kwargs)

    @classmethod
    def _get_config_validator(cls):
        return SunEntityValidator

    def __eq__(self, other):
        if isinstance(other, SunEntity):
            return True
        return False

    def _build_state(self):
        """form state vector"""
        state = np.array([self.config.theta], dtype=np.float32)

        return state

    @property
    def theta(self):
        """get theta"""
        return self._state[0]

    @property
    def x(self) -> float:
        """get x"""
        raise NotImplementedError

    @property
    def y(self) -> float:
        """get y"""
        raise NotImplementedError

    @property
    def z(self) -> float:
        """get z"""
        raise NotImplementedError

    @property
    def position(self) -> np.ndarray:
        """get 3d position vector"""
        raise NotImplementedError

    @property
    def orientation(self) -> scipy.spatial.transform.Rotation:
        """
        Get orientation of entity.
        """
        raise NotImplementedError

    @property
    def velocity(self) -> np.ndarray:
        """Get 3d velocity vector"""
        raise NotImplementedError


class SunDynamics(BaseODESolverDynamics):
    """Dynamics for the sun. Assumed to rotate in x-y plane"""

    def __init__(self, n=N_DEFAULT, **kwargs):
        self.n = n  # rads/s

        super().__init__(**kwargs)

    def _compute_state_dot(self, t: float, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        return np.array([self.n])
