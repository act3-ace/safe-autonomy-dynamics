#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import os
from pathlib import Path

from setuptools import setup, find_packages


def parse_requirements(filename: str):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


reqs = parse_requirements("requirements.txt")


version = {}
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = None
with open(os.path.join(base_dir, 'version.py')) as fp:
     exec(fp.read(), version)

if __name__ == '__main__':
    jax_reqs = parse_requirements("requirements-jax.txt")
    test_reqs = parse_requirements("requirements-compliance.txt") + jax_reqs
    docs_require = parse_requirements("requirements-mkdocs.txt")

    setup(
        name="safe-autonomy-dynamics",
        author="ACT3",
        description="ACT3 Safe Autonomy Dynamics Models",

        long_description=Path("README.md").read_text(),
        long_description_content_type="text/markdown",

        url="https://github.com/act3-ace/safe-autonomy-dynamics",

        license="",

        setup_requires=[
            'setuptools_scm',
            'pytest-runner'
        ],
        version=version["__version__"],

        # add in package_data
        include_package_data=True,
        package_data={
            'safe_autonomy_dynamics': ['*.yml', '*.yaml']
        },

        packages=find_packages(),

        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],

        install_requires=reqs,

        extras_require={
            "testing":  test_reqs,
            "docs":  docs_require,
            "jax": jax_reqs,
        },
        python_requires='>=3.8',
    )


