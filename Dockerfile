##########################################################################################
# Dependent tags
##########################################################################################

ARG ACT3_OCI_REGISTRY=reg.git.act3-ace.com

##########################################################################################
# Dependent images
##########################################################################################

#########################################################################################
# develop stage contains base requirements. Used as base for all other stages
#########################################################################################

ARG IMAGE_REPO_BASE
FROM ${IMAGE_REPO_BASE}docker.io/python:3.8 as develop

ARG PIP_INDEX_URL

#Sets up apt mirrors to replace the default registries
RUN echo "deb ${APT_MIRROR_URL} stable main contrib non-free" > /etc/apt/sources.list && \
echo "deb-src ${APT_MIRROR_URL} stable main contrib non-free" >> /etc/apt/sources.list

# install SA dynamics requirements
# hadolint ignore=DL3013
RUN python --version && \
    python -m pip install --no-cache-dir --upgrade pip

#########################################################################################
# build stage packages the source code
#########################################################################################

FROM develop as build
ENV SA_DYNAMICS_ROOT=/opt/libsa-dynamics

WORKDIR /opt/project
COPY . .

RUN python setup.py bdist_wheel -d ${SA_DYNAMICS_ROOT} && \
    python -m pip install --no-cache-dir .


#########################################################################################
# CI/CD stages. DO NOT make any stages after cicd
#########################################################################################

# the CI/CD pipeline uses the last stage by default so set your stage for CI/CD here with FROM your_ci_cd_stage as cicd
# this image should be able to run and test your source code
# python CI/CD jobs assume a python executable will be in the PATH to run all testing, documentation, etc.
FROM develop as cicd
