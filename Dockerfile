FROM ubuntu:24.04

WORKDIR /workdir/dlio

# Combine apt-get update and install into a single RUN command to reduce layers
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    wget bc curl vim-tiny git iproute2 sysstat mpich libc6 libhwloc-dev zlib1g-dev cmake && \
    rm -rf /var/lib/apt/lists/*

# Install uv and set PATH
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Update PATH
ENV PATH="/root/.local/bin:${PATH}"

# Now use uv to install python and initialize our environment 
RUN uv python install 3.12.9 && \
    uv venv && \
    uv init

# Install Python dependencies
#COPY requirements.txt /workdir/dlio
# Copy everything except what our .dockerignore ignores
COPY . /workdir/dlio 
RUN rm -rf /workdir/dlio/dlio_benchmark.orig

#RUN uv pip install git+https://github.com/argonne-lcf/dlio_benchmark.git@main 
#RUN uv pip install https://github.com/russfellows/dlio_benchmark_s3.git

#ARG REPO_URL="https://github.com/russfellows/dlio_benchmark_s3.git"
#ARG BRANCH="main"
#RUN git clone --branch ${BRANCH} ${REPO_URL} .
#RUN rm -rf /workdir/dlio/.git 

# Add extra URL so we pick up latest from nvidia at pypi
RUN uv pip install --extra-index-url https://pypi.nvidia.com/ -r requirements.txt

# Removed the python between run and setup.py
#RUN uv run setup.py build && \
#    uv run setup.py install 

# New way to build 
RUN uv build && uv pip install ./dist/*.whl 

# STEP 10: Copy the Rust executable to /usr/local/bin
COPY dlio_s3_rust/release/s3Rust-cli /usr/local/bin/

# STEP 12: Install the Python wheel using uv pip install
RUN uv pip install ./dlio_s3_rust/wheels/dlio_s3_rust-*-cp312-cp312-manylinux_2_39_x86_64.whl

# Clean up unnecessary files to reduce image size
RUN apt-get clean && \
    chmod +x ./dlio_benchmark/*.py && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN rm -f environment-ppc.yaml hello.py pyproject.toml.old requirements.txt.old setup.py.old

#source .venv/bin/activate
#ENV PATH="/workdir/dlio/.venv/bin/activate:${PATH}"

ENTRYPOINT /bin/bash
