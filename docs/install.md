## Installation

The following instructions detail how to install
the safe-autonomy-dynamics library on your local system.
It is recommended to install the python modules within
a [virtualenv](https://virtualenv.pypa.io/en/stable/#)
or [conda](https://docs.conda.io/projects/conda/en/latest/index.html) environment.
safe-autonomy-dynamics utilizes [Poetry](https://python-poetry.org/) to handle installation.
Poetry can install safe-autonomy-dynamics into an auto-generated virtualenv or within the currently active environment.

## Installing safe-autonomy-dynamics

Clone a copy of the safe-autonomy-dynamics repo onto your local
machine via SSH (recommended):

```shell
git clone git@github.com:act3-ace/safe-autonomy-dynamics.git
```

or HTTPS:

```shell
git clone https://github.com/act3-ace/safe-autonomy-dynamics.git
```

Install the safe-autonomy-dynamics module into your
environment using `poetry`:

```shell
cd safe-autonomy-dynamics
poetry install
```

Poetry will handle installing appropriate versions of the dependencies for safe-autonomy-dynamics into your environment, if they aren't already installed.  Poetry will install an editable version of safe-autonomy-dynamics to the environment.

If you'd like jax support, specify the jax extra:

```shell
poetry install -E jax
```

## Questions or Issues?
If you have any trouble installing the safe-autonomy-dynamics
package in your local environment, please feel free to
submit an [issue](https://github.com/act3-ace/safe-autonomy-dynamics/issues).

For more information on what's available in safe-autonomy-dynamics,
see our [API](api/index.md).
