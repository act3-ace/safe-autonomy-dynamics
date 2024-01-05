# safe-autonomy-dynamics

## Intro

The safe-autonomy-dynamics package provides an API for dynamic systems supported by a library of common functions used to access and update system dynamics. These dynamics are used to build simulated environments which behave like real-world systems for the purpose of safe autonomy research and development (though their use is not limited to the safety domain). The package also includes a zoo of air and space domain dynamics modules tailored for simulating aerospace systems. The team intends to grow the zoo as new dynamic systems are studied or simulation requirements change.

## Docs

Library documentation and api reference located [here](https://rta.github.com/act3-ace/safe-autonomy-stack/safe-autonomy-dynamics).

## Installation

The following instructions detail how to install
the safe-autonomy-dynamics library on your local system.
It is recommended to install the python modules within
a [virtualenv](https://virtualenv.pypa.io/en/stable/#)
or [conda](https://docs.conda.io/projects/conda/en/latest/index.html) environment.
safe-autonomy-dynamics utilizes [Poetry](https://python-poetry.org/) to handle installation.
Poetry can install safe-autonomy-dynamics into an auto-generated virtualenv or within the currently active environment.

### Installing safe-autonomy-dynamics

Clone a copy of the safe-autonomy-dynamics repo onto your local
machine via SSH (recommended):

```shell
git clone git@github.com:act3-ace/safe-autonomy-dynamics.git
```

or HTTPS:

```shell
git clone https://github.com/act3-ace/safe-autonomy-dynamics.git
```

#### Installing safe-autonomy-dynamics with pip

To install the safe-autonomy-dynamics module into your
environment using `pip`:

```shell
cd safe-autonomy-dynamics
pip install .
```

For a local development version, please install
using the `-e, --editable` option:

```shell
pip install -e .
```

If you'd like jax support, specify the jax extra:

```shell
pip install .[jax]
```

#### Installing safe-autonomy-dynamics with Poetry

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

## Build Docs Locally

First make sure the mkdocs requirements are installed

```shell
poetry install --with docs
```

Now, build the documentation and serve it locally. By default, you should be able to reach the docs on your local web browser at `127.0.0.1:8000`

```shell
rm -r site
poetry run mkdocs build
cp -r docs/. site/
poetry run mkdocs serve
```

## Public Release

Approved for public release; distribution is unlimited. Case Number: AFRL-2023-6155

A prior version of this repository was approved for public release. Case Number: AFRL-2022-3202

## Team

Umberto Ravaioli,
Kyle Dunlap,
Jamie Cunningham,
John McCarroll,
Kerianne Hobbs,
Charles Keating
