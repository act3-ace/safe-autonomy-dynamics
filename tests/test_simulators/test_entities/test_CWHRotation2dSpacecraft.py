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

This module defines tests for the CWHRotationSpacecraft entity.

Author: John McCarroll, Andy Barth
"""

import pytest
import os

from safe_autonomy_dynamics.cwh import CWHRotation2dSpacecraft
from tests.test_simulators.test_entities.conftest import evaluate
from tests.conftest import read_test_cases, delimiter


# Define test assay
test_cases_file_path = os.path.join(os.path.split(__file__)[0], "../../test_cases/CWHRotation2dSpacecraft_test_cases.yaml")
parameterized_fixture_keywords = ["attr_init",
                                  "init_kwargs",
                                  "control",
                                  "num_steps",
                                  "attr_targets",
                                  "error_bound"]
test_configs, IDs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)

parameterized_fixture_keywords.append("use_jax")
test_configs =  [config.copy() + [False] for config in test_configs] + [config.copy() + [True] for config in test_configs]
IDs += [id + "_jax" for id in IDs]

# override entity fixture
@pytest.fixture
def entity(initial_entity_state, use_jax, init_kwargs):
    entity = CWHRotation2dSpacecraft(name="tests", use_jax=use_jax, **init_kwargs)
    if initial_entity_state is not None:
        entity.state = initial_entity_state

    return entity


@pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, indirect=True, ids=IDs)
def test_CWHRotation2dSpacecraft(acted_entity, control, num_steps, attr_targets, error_bound):
    evaluate(acted_entity, attr_targets, error_bound=error_bound)

