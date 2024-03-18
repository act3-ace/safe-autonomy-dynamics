"""
# -------------------------------------------------------------------------------
# Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
# Reinforcement Learning Core (CoRL) Runtime Assurance Extensions
#
# This is a US Government Work not subject to copyright protection in the US.
#
# The use, dissemination or disclosure of data in this file is subject to
# limitation or restriction. See accompanying README and LICENSE for details.
# -------------------------------------------------------------------------------
"""
import pint
import pytest

from safe_autonomy_dynamics.base_models import build_unit_conversion_validator_fn

def test_unit_conversion_validator_raised_value_error():
    """Given a requirement to have a quanity with a specific unit and a quantity that
       cannot convert to the required unit, when the validator attempts to validate
       the quantity, then a ValueError is raised rather than a pint 
       DimensionalityError to align with Pydantic validation
    """
    position_unit_validator = build_unit_conversion_validator_fn("meters")
    velocity = pint.Quantity(1.0, "m/s")

    with pytest.raises(ValueError):
        position_unit_validator(velocity)