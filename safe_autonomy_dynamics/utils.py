"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Safe Autonomy Dynamics.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

Provides general purpose util functions for the safe autonomy dynamics library
"""

from typing import Tuple, Union

import numpy as np


def number_list_to_np(input_val: Union[float, int, list, np.ndarray], shape: Tuple[int], dtype=np.float64):
    """
    Converts dynamic number, list or np.ndarray to a np.ndarray of a particular shape a dtype
    If np.ndarray is passed, performs shape and dtype checking

    Parameters
    ----------
    input : float, int, list, np.ndarray
        input to be converted to standardized np.ndarray
    shape : Tuple[int]
        shape of desired np.ndarray
    dtype : data-type, optional
        dtype of desired np.ndarray, by default np.float64

    Returns
    -------
    np.ndarray
        converted np.ndarray from input
    """
    if isinstance(input_val, np.ndarray):
        output = input_val
    elif isinstance(input_val, list):
        output = np.ndarray(input_val, dtype=dtype)
    elif isinstance(input_val, (float, int)):
        output = float(input_val) * np.ones(shape, dtype=dtype)
    else:
        raise TypeError(f"input_val is type {type(input_val)}, must be Number, list, or np.ndarray")

    assert output.shape == shape, f"input_val is of shape {output.shape} instead of {shape}"
    assert output.dtype == dtype, f"input_val dtype of type {output.dtype} instead of {dtype}"

    return output
