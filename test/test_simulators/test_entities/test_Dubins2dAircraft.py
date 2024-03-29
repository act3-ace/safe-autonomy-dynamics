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

This module defines tests for the Dubins2dAircraft acted_entity.

Author: John McCarroll
"""

import pytest
import os

from safe_autonomy_dynamics.dubins import Dubins2dAircraft
from test.test_simulators.test_entities.conftest import evaluate
from test.conftest import read_test_cases, delimiter


# Define test assay
test_cases_file_path = os.path.join(os.path.split(__file__)[0], "../../test_cases/Dubins2dAircraft_test_cases.yaml")
# TODO: parameterized_fixture_keywords have become common...
parameterized_fixture_keywords = ["attr_init",
                                  "init_kwargs",
                                  "control",
                                  "num_steps",
                                  "attr_targets",
                                  "error_bound",
                                  "proportional_error_bound",
                                  "angles"]
test_configs, IDs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)


# override entity fixture
@pytest.fixture
def entity(initial_entity_state, init_kwargs):
    entity = Dubins2dAircraft(name="tests", **init_kwargs)

    if initial_entity_state is not None:
        entity.state = initial_entity_state

    return entity


@pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, indirect=True, ids=IDs)
def test_Dubins2dAircraft(acted_entity, control, num_steps, attr_targets, error_bound, proportional_error_bound, angles):
    if proportional_error_bound is None:
        proportional_error_bound = 0
    evaluate(acted_entity, attr_targets, angles=angles, error_bound=error_bound, proportional_error_bound=proportional_error_bound)