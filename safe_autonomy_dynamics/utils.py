import numpy as np
from numbers import Number

def number_list_to_np(input, shape, dtype=np.float64):
    if isinstance(input, np.ndarray):
        output = input
    elif isinstance(input, list):
        output = np.ndarray(input, dtype=dtype)
    elif isinstance(input, (Number)):
        output = input * np.ones(shape, dtype=dtype)
        
    assert output.shape == shape, f"input is of shape {input.shape} instead of {shape}"
    assert output.dtype == dtype, f"input of type {type(input.dtype)} instead of {dtype}"

    return output
