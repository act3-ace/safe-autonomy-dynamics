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

This module defines tests of equality between different entities.

Author: Aditesh Kumar
"""

import importlib
import numpy as np
import os
import pytest

from test.conftest import read_test_cases, delimiter


# Define test assay
test_cases_file_path = os.path.join(os.path.split(__file__)[0], "../../test_cases/compare_entities_test_cases.yaml")
parameterized_fixture_keywords = ["this_entity", "other_entity", "expect_equal"]
test_configs, IDs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)


def build_entity(entity_name, entity_config):
    mod_name, _, obj_name = entity_config['entity_class'].rpartition('.')
    entity_cls = getattr(importlib.import_module(mod_name), obj_name)
    entity = entity_cls(name=entity_name, **entity_config.get('init_kwargs', {}))
    if 'init_state' in entity_config and isinstance(entity_config['init_state'], dict):
        initial_state = []
        for value in entity_config['init_state'].values():
            initial_state += value if type(value) in [list, np.ndarray] else [value]
        entity.state = np.array(initial_state, dtype=entity.state.dtype)
    return entity


@pytest.fixture
def this_entity(request):
    return build_entity('this', request.param)


@pytest.fixture
def other_entity(request):
    return build_entity('other', request.param)


@pytest.fixture
def expect_equal(request):
    return request.param


@pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, indirect=True, ids=IDs)
def test_compare_entities(this_entity, other_entity, expect_equal):
    assert (this_entity == other_entity) == expect_equal
