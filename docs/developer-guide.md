# Developer Guide

## Design Patterns

safe-autonomy-dynamics is designed around the concept of an entity, which has associated dynamics that it exhibits.  All entities extend from [```BaseEntity```](api/base_models.md#safe_autonomy_dynamics.base_models.BaseEntity), which provides the functionaltiy common to all entities.  Once extended from [```BaseEntity```](api/base_models.md#safe_autonomy_dynamics.base_models.BaseEntity), the distinct entity can exhibit the dynamics particular to that entity.

Each entity defines a Validator that contains the state of the entity, such as position information or movement information.  Each Validator extends from [```BaseEntityValidator```](api/base_models.md#safe_autonomy_dynamics.base_models.BaseEntityValidator), which extends a Pydantic ```Model``` and provides the ability to validate the state of the entity.  For example, the validator can ensure that state fields expected to hold position information are actually holding position information with appropriate units, rather than accidentally having velocity information.

## Testing

### Unit Tests

Prior to running the safe-autonomy-dynamics unit tests, ensure that testing dependencies are installed by running

```shell
poetry install --with test
```

This command will install the dependencies to your current virtual environment or an automatically generated virtual environment, if you are not in your base environment.  Unfortunately, pip does not allow for installation of dependency groups from a pyproject.toml.  So, Poetry must be used to establish an appropriate environment.  Once the testing dependencies have been installed, you can execute the unit tests by running

```shell
poetry run pytest
```

The use of Poetry to run pytest ensures that the correct environment is used to execute the unit tests.  In some cases, trying to run pytest directly in your virtual environment will fail due to an ImportError of a dependency, despite having the dependency already installed in the environment.  Another option for executing the unit tests is to run

```shell
python -m pytest
```

## Releasing

Releases of safe-autonomy-dynamics are handled by the CI/CD pipeline.  New releases of safe-autonomy-dynamics will be created with each update to the main branch of the repository.  The CI/CD pipeline will handle verifying the state of the code, and upon successful verification and testing, will publish a package of safe-autonomy-dynamics to an appropriate package repository, such as PyPI.
