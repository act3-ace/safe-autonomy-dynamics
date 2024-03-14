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

This module defines tests for relative distance and position metrics between entities.

Author: Aditesh Kumar
"""

import os
import numpy as np
import pytest

from safe_autonomy_dynamics.cwh import CWHRotation2dSpacecraft, CWHSpacecraft, SixDOFSpacecraft
from test.conftest import delimiter, read_test_cases

# Define test assay
test_cases_file_path = os.path.join(os.path.split(__file__)[0], "../../test_cases/entity_relative_metrics.yaml")
parameterized_fixture_keywords = ["this_entity_name", "other_entity_name", "expected_relative_pos", "expected_relative_vel", "error_bound"]
test_configs, IDs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)


class MockSimulator():
    """A stub simulator that can interface with CWH models."""

    def __init__(self, entities) -> None:
        self.sim_entities = {e.name: e for e in entities}
        for entity in self.sim_entities.values():
            entity.set_sim(self)


@pytest.fixture
def simulator():
    entities = [
        CWHSpacecraft(name='cwh', x=0, y=0, z=0, x_dot=0, y_dot=0, z_dot=0),
        CWHRotation2dSpacecraft(name='rot2d', x=536.324, y=-738.112, x_dot=-3.7, y_dot=-17, theta=1.2, wz=0.03),
        SixDOFSpacecraft(name='sixdof', x=-995.99, y=-536.324, z=738.112, x_dot=17, y_dot=-25.3, z_dot=3.7, wx=-0.0065, wy=0.03, wz=0.0123),
    ]
    return MockSimulator(entities)


@pytest.fixture
def this_entity_name(request):
    return request.param


@pytest.fixture
def other_entity_name(request):
    return request.param


@pytest.fixture
def expected_relative_pos(request):
    return request.param


@pytest.fixture
def expected_relative_vel(request):
    return request.param


@pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, indirect=True, ids=IDs)
def test_entity_relative_position(
    simulator, this_entity_name, other_entity_name, expected_relative_pos, expected_relative_vel, error_bound
):
    this_entity = simulator.sim_entities[this_entity_name]
    rel_pos = this_entity.entity_relative_position(other_entity_name)
    assert (np.abs(expected_relative_pos - rel_pos) < error_bound).all()


@pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, indirect=True, ids=IDs)
def test_entity_relative_velocity(
    simulator, this_entity_name, other_entity_name, expected_relative_pos, expected_relative_vel, error_bound
):
    this_entity = simulator.sim_entities[this_entity_name]
    rel_vel = this_entity.entity_relative_velocity(other_entity_name)
    assert (np.abs(expected_relative_vel - rel_vel) < error_bound).all()