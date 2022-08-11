# safe-autonomy-dynamics

## Intro
The safe-autonomy-dynamics package provides an API for dynamic systems supported by a library of common functions used to access and update system dynamics. These dynamics are used to build simulated environments which behave like real-world systems for the purpose of safe autonomy research and development (though their use is not limited to the safety domain). The package also includes a zoo of air and space domain dynamics modules tailored for simulating aerospace systems. The team intends to grow the zoo as new dynamic systems are studied or simulation requirements change.

## Docs
Library documentation and api reference located [here](https://act3-ace.github.io/safe-autonomy-dynamics/).

## Installation
The following instructions detail how to install 
the safe-autonomy-dynamics library on your local system.
It is recommended to install the python modules within 
a [virtualenv](https://virtualenv.pypa.io/en/stable/#)
or [conda](https://docs.conda.io/projects/conda/en/latest/index.html) environment.

### Installing safe-autonomy-dynamics
Clone a copy of the safe-autonomy-dynamics source code 
onto your local machine via SSH:
```shell
git clone git@github.com:act3-ace/safe-autonomy-dynamics.git
```
or HTTPS:
```shell
git clone https://github.com/act3-ace/safe-autonomy-dynamics.git
```

Install the safe-autonomy-dynamics module into your 
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

## Build Docs Locally

First make sure the mkdocs requirements are installed 

```shell
pip install -r mkdocs-requirements.txt
```

Now, build the documentation and serve it locally. By default, you should be able to reach the docs on your local web browser at `127.0.0.1:8000`

```shell
rm -r site
mkdocs build
cp -r docs/. site/
mkdocs serve
```

## Public Release
Distribution A. Approved for public release; distribution is unlimited. Case Number: AFRL-2022-3202

## Team
Umberto Ravaioli,
Kyle Dunlap,
Jamie Cunningham,
John McCarroll,
Kerianne Hobbs
