# User Guide

## Purpose

The safe-autonomy-dynamics package provides an API for dynamic systems supported by a library of common functions used to access and update system dynamics. These dynamics are used to build simulated environments which behave like real-world systems for the purpose of safe autonomy research and development (though their use is not limited to the safety domain). The package also includes a zoo of air and space domain dynamics modules tailored for simulating aerospace systems. The team intends to grow the zoo as new dynamic systems are studied or simulation requirements change.

## Overview

safe-autonomy-dynamics is designed around the concept of an entity.  An entity has a state and exhibits dynamics that modifies that state over time.  Entities can also have actions that they perform that will effect the state transition over time.  Entities can be used as representation of an agent in a simulation.

## Usage

The following is an example of how to utilize safe-autonomy-dynamics in your own software.

```python
from safe_autonomy_dynamics.dubins import Dubins2dAircraft

# Create entity w/ initial state
aircraft = Dubins2dAircraft(name="trainer", v=200)

# Transition the state over a time step w/ a given action
aircraft.step(200, {'acceleration': 2})

```

## Additional Resources

- [Developers Guide](developer-guide.md)
- [API](api/index.md)
