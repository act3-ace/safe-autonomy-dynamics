#########################################################################################
# develop stage contains base requirements. Used as base for all other stages
#########################################################################################

ARG IMAGE_REPO_BASE
FROM ${IMAGE_REPO_BASE}docker.io/python:3.8 as develop

ARG PIP_INDEX_URL
ARG APT_MIRROR_URL
ARG SECURITY_MIRROR_URL

#Sets up apt mirrors to replace the default registries
RUN if [ -n "$APT_MIRROR_URL" ] ; then sed -i "s|http://archive.ubuntu.com|${APT_MIRROR_URL}|g" /etc/apt/sources.list ; fi && \
if [ -n "$SECURITY_MIRROR_URL" ] ; then sed -i "s|http://security.ubuntu.com|${SECURITY_MIRROR_URL}|g" /etc/apt/sources.list ; fi

#########################################################################################
# build stage packages the source code
#########################################################################################

FROM develop as build
ENV SA_DYNAMICS_ROOT=/opt/libact3-sa-dynamics

WORKDIR /opt/project
COPY . .

RUN python setup.py bdist_wheel -d ${SA_DYNAMICS_ROOT} && \
    pip install --no-cache-dir .

#########################################################################################
# package stage 
#########################################################################################

# the package stage contains everything required to install the project from another container build
# NOTE: a kaniko issue prevents the source location from using a ENV variable. must hard code path

FROM scratch as package
COPY --from=build /opt/libact3-sa-dynamics /opt/libact3-sa-dynamics


#########################################################################################
# CI/CD stages. DO NOT make any stages after cicd
#########################################################################################

# the CI/CD pipeline uses the last stage by default so set your stage for CI/CD here with FROM your_ci_cd_stage as cicd
# this image should be able to run and test your source code
# python CI/CD jobs assume a python executable will be in the PATH to run all testing, documentation, etc.

FROM build as cicd
