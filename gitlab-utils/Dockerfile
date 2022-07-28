# The pycity_scheduling framework's Dockerfile to be used by the Gitlab-CI.
# Download, build, and install all required base packages and then setup the pycity_scheduling framework environment.


FROM ubuntu:focal AS base


ENTRYPOINT /bin/bash


ENV CMAKE_BUILD_PARALLEL_LEVEL="8"
ENV MAKEFLAGS "-j8"
ENV DEBIAN_FRONTEND noninteractive
ENV TZ Etc/UTC

RUN apt-get update &&\
    apt-get install -y wget software-properties-common curl apt-utils git mpich

# Download and install the SCIP optimization solver (SCIPOptSuite 8.0.0).
# Important note: You are allowed to retrieve a copy of SCIP for research purposes as a member of a noncommercial and academic institution only.
# To get a copy of SCIP, you need to certify that you are a member of a noncommercial, academic institution and accept the ZIB Academic License.
# Commercial use requires a commercial license.
# For further information visit: https://scipopt.org/#scipoptsuite

RUN wget --quiet https://www.scipopt.org/download/release/SCIPOptSuite-8.0.0-Linux-ubuntu.deb &&\
    apt-get install -y gcc g++ gfortran liblapack3 libtbb2 libcliquer1 libopenblas-dev libgslcblas0 patchelf &&\
    apt install -y $SCIPAMPL_BASE_DIR/SCIPOptSuite-8.0.0-Linux-ubuntu.deb &&\
    rm $SCIPAMPL_BASE_DIR/SCIPOptSuite-8.0.0-Linux-ubuntu.deb

# Download and install the Gurobi optimization solver (Gurobi 9.1.2).
# Important note: You are allowed to retrieve a copy of Gurobi for research purposes as a member of a non-commercial and academic institution only.
# To get a copy of Gurobi, you need to request for an academic license at Gurobi Optimization, LLC.
# Commercial use requires a commercial license.
# For further information visit: https://www.gurobi.com/
RUN mkdir -p /opt/gurobi
ENV GUROBI_BASE_DIR="/opt/gurobi"
ENV GUROBI_HOME="${GUROBI_BASE_DIR}/gurobi912/linux64"

RUN cd $GUROBI_BASE_DIR &&\
    wget --quiet https://packages.gurobi.com/9.1/gurobi9.1.2_linux64.tar.gz &&\
    tar xfz gurobi9.1.2_linux64.tar.gz &&\
    rm $GUROBI_BASE_DIR/gurobi9.1.2_linux64.tar.gz &&\
	chmod -R 755 $GUROBI_BASE_DIR/gurobi912 &&\
    cd /

ENV PATH="${PATH}:$GUROBI_HOME/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$GUROBI_HOME/lib"

# clone and patch pyomo
RUN git clone --branch 5.7.3 --depth 1 https://github.com/Pyomo/pyomo
ADD gitlab-utils/pyomo_scipampl.patch /pyomo/
RUN cd /pyomo && \
    git apply pyomo_scipampl.patch &&\
    cd /

FROM base as python37

# Install Python version 3.7 together with some core package:
RUN cd / &&\
    add-apt-repository -y ppa:deadsnakes/ppa &&\
    apt-get update &&\
    apt-get install -y python3.7-venv python3-pip &&\
    cd /



RUN cd $GUROBI_HOME &&\
    python3.7 setup.py install &&\
    cd /

# Add requirements to docker image
ADD requirements.txt /

RUN python3.7 -m pip install /pyomo
RUN python3.7 -m pip install --no-cache-dir -r /requirements.txt

FROM base as python38

# Install Python version 3.8 together with some core package:
RUN cd / &&\
    add-apt-repository -y ppa:deadsnakes/ppa &&\
    apt-get update &&\
    apt-get install -y python3.8-venv python3-pip &&\
    cd /



RUN cd $GUROBI_HOME &&\
    python3.8 setup.py install &&\
    cd /

# Add requirements to docker image
ADD requirements.txt /

RUN python3.8 -m pip install /pyomo
RUN python3.8 -m pip install --no-cache-dir -r /requirements.txt

FROM base as python39

# Install Python version 3.9 together with some core package:
RUN cd / &&\
    add-apt-repository -y ppa:deadsnakes &&\
    apt-get update &&\
    apt-get install -y python3.9-venv python3-pip &&\
    cd /



RUN cd $GUROBI_HOME &&\
    python3.9 setup.py install &&\
    cd /

# Add requirements to docker image
ADD requirements.txt /

RUN python3.9 -m pip install /pyomo
RUN python3.9 -m pip install --no-cache-dir -r /requirements.txt