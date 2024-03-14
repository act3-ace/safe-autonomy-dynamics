## [1.2.1](https://github.com/act3-ace/safe-autonomy-dynamics/compare/v1.2.0...v1.2.1) (2024-03-14)


### Bug Fixes

* proper equality check of quaternions in CWHRotation2dSpacecraft ([5477bdd](https://github.com/act3-ace/safe-autonomy-dynamics/commit/5477bddc822844b7ba63211a9a34cba087b7e068))
* sixdof quat comparison + add unit tests ([e81d717](https://github.com/act3-ace/safe-autonomy-dynamics/commit/e81d717d253fa80c0397a8b8020129e39ae28ce4))

# [1.2.0](https://github.com/act3-ace/safe-autonomy-dynamics/compare/v1.1.0...v1.2.0) (2024-01-05)


### Bug Fixes

* **dependencies:** Expanded dependency version requirements to a range that includes versions of common dependencies needed by CoRL 2.9.0 ([ce46983](https://github.com/act3-ace/safe-autonomy-dynamics/commit/ce46983bfbae42b312d0143729b42fb535157815))
* **release:** Removed non-existant dependency groups from release.sh ([66f1e02](https://github.com/act3-ace/safe-autonomy-dynamics/commit/66f1e02c0230839c433206d87f08b6d1c78abf32))
* **release:** test release ([b1e48b6](https://github.com/act3-ace/safe-autonomy-dynamics/commit/b1e48b6aac33dae2793643e32d93393cbbba8b06))
* **release:** test release ([3a72e10](https://github.com/act3-ace/safe-autonomy-dynamics/commit/3a72e10afd5a5d3e0085588ff9f84f88d43054f5))
* **release:** updated permissions on release.sh to allow execution ([98c6d7b](https://github.com/act3-ace/safe-autonomy-dynamics/commit/98c6d7bf96523764dd6ca2ec0ea855b8a664e7e0))
* use pint application registry ([116611f](https://github.com/act3-ace/safe-autonomy-dynamics/commit/116611f12938d9dc05a412889769501b377d3296))


### Features

* Add sun entity ([3406bee](https://github.com/act3-ace/safe-autonomy-dynamics/commit/3406beee877fd9fe99780bec0aaafa4e7d460264))

# 1.1.0 (2024-01-05)


### Bug Fixes

* consistency on intrinsic orientation angles
* **dependencies:** moved developer dependencies out of main requirements file
* Remove partners
* removed url components of semantic release commit
* Use jax 0.4.3


### Features

* dubins custom state and control limits
* Entity validators no longer allow unexpected parameters
* initial aerobench dynamics implementation
* Jax ODE Solver
* optional jax version of dynamics with optional jax import. np is now a class member and can be set to either numy or jax.numpy
* Optional trajectory of intermediate points sampled along dynamics step
* Pint units and constructor init params and attributes with units supported
* warning on action clipping

## 0.11.3 (2023-04-05)


### Bug Fixes

* removed url components of semantic release commit

## 0.11.2 (2023-04-05)


### Bug Fixes

* Remove partners

## 0.11.1 (2023-03-28)


### Bug Fixes

* Use jax 0.4.3

# 0.11.0 (2023-03-25)


### Features

* dubins custom state and control limits

# 0.10.0 (2023-03-18)


### Features

* Entity validators no longer allow unexpected parameters

# 0.9.0 (2023-03-17)


### Features

* Optional trajectory of intermediate points sampled along dynamics step

# 0.8.0 (2023-03-16)


### Features

* Pint units and constructor init params and attributes with units supported

# 0.7.0 (2023-02-23)


### Bug Fixes

* consistency on intrinsic orientation angles


### Features

* warning on action clipping

# 0.6.0 (2023-02-14)


### Features

* Jax ODE Solver

# 0.5.0 (2023-02-09)


### Features

* optional jax version of dynamics with optional jax import. np is now a class member and can be set to either numy or jax.numpy

## 0.4.1 (2023-01-19)


### Bug Fixes

* **dependencies:** moved developer dependencies out of main requirements file, closes #23

# 0.4.0 (2022-09-22)


### Features

* initial aerobench dynamics implementation

# 0.3.0 (2022-08-13)


### Features

* **cwh:** 6dof model

# 0.2.0 (2022-08-01)


### Features

* **version:** updates for semantic release

## 0.0.1003 (2022-07-28)


### Bug Fixes

* change to 0.1.0
* change version to the realistic number

## 0.0.1002 (2022-07-18)


### Bug Fixes

* **gitlab-ci:** no allow failure mkdocs
* **gitlab-ci:** update mkdocs
* **mkdocs-requirements:** pin mkdocsstrings
* **version:** update version for initial release
