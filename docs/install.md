# Installation
The following instructions detail how to install 
the safe-autonomy-dynamics library on your local system.
It is recommended to install python modules within 
a [virtualenv](https://virtualenv.pypa.io/en/stable/#)
or [conda](https://docs.conda.io/projects/conda/en/latest/index.html)
environment.

## Installing safe-autonomy-dynamics
Clone a copy of the safe-autonomy-dynamics repo onto your local
machine via SSH (recommended):
```shell
git clone git@git.act3-ace.com:rta/safe-autonomy-dynamics.git
```
or HTTPS:
```shell
git clone https://git.act3-ace.com/rta/safe-autonomy-dynamics.git
```

Install the safe-autonomy-sims module into your local
environment using `pip`:
```shell
pip install path/to/safe-autonomy-dynamics/
```

## Development
For a local development version, you can install the
safe-autonomy-dynamics package using `pip`'s 
`-e, --editable` option:
```shell
pip install -e path/to/safe-autonomy-dynamics/
```
This will install the package in an editable mode within
your environment, allowing any changes you make to the
source to persist.

## Questions or Issues?
If you have any trouble installing the safe-autonomy-dynamics
package in your local environment, please feel free to
submit an [issue](https://git.act3-ace.com/rta/safe-autonomy-dynamics/-/issues).

For more information on what's available in safe-autonomy-dynamics,
see our [API](api/index.md).
