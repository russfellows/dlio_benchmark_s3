# Use a suitable Ubuntu base image
FROM ubuntu:24.04

# Set the working directory for the main project
WORKDIR /workdir/dlio

# Set a non-interactive frontend for apt commands
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# Includes tools for building, git for cloning, and libraries needed by the benchmark/dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget curl vim-tiny git iproute2 sysstat mpich libmpich-dev libc6 \
    libhwloc-dev zlib1g-dev cmake build-essential **ca-certificates** && \
    rm -rf /var/lib/apt/lists/*


# Install Rust toolchain using rustup (standard method)
# This adds cargo and rustc to the PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"


# Install uv (a fast Python package installer and runner)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Install Python 3.12 and initialize uv virtual environment
RUN uv python install 3.12.9 && \
    uv venv && \
    uv init

# Install maturin - needed to build Rust-based Python wheels
# We install it in the uv venv so it's available for the build step
RUN uv pip install maturin

# --- Add the uv venv bin directory to the PATH ---
# This is crucial to make maturin (and other venv executables) directly available
ENV PATH="/workdir/dlio/.venv/bin:${PATH}"

# --- Build and Install the main dlio_benchmark project ---

# Copy the main dlio_benchmark project source code from your build context
# Ensure your .dockerignore excludes unnecessary files
COPY . /workdir/dlio

# Install the main project's Python dependencies from requirements.txt
# This includes the NVIDIA DALI package from the extra index
RUN uv pip install --extra-index-url https://pypi.nvidia.com/ -r requirements.txt

# Build and install the main dlio_benchmark project using its pyproject.toml/setup.py
# uv build uses the build backend to create a wheel, then uv pip install installs it
RUN uv build && uv pip install ./dist/*.whl

# --- Build and Install dlio_s3_rust project ---

# Create a separate working directory for the rust project build
# This keeps its build artifacts separate from the main project
WORKDIR /tmp/dlio_s3_rust

# Clone the dlio_s3_rust repository source code
# Replace 'main' with the desired branch if needed
RUN git clone --depth 1 https://github.com/russfellows/dlio_s3_rust.git .

# Build the Rust executable and the Python wheel

# Set PYO3 compatibility flag.
ENV PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1

# Build the Rust CLI executable.
RUN uv run cargo build --release

# Copy the CLI executable to /usr/local/bin.
RUN cp target/release/s3Rust-cli /usr/local/bin/s3Rust-cli

# Build and install the Python extension into the virtual environment.
RUN maturin build --release --features extension-module
#RUN uv run maturin build --release --features extension-module
#RUN uv run maturin develop --release --features extension-module

# Install the built Python wheel using uv
# The wheel will be in target/wheels/
RUN uv pip install target/wheels/*.whl

# --- Final Cleanup ---

# Clean up temporary build directories and package lists to reduce image size
RUN rm -rf /tmp/dlio_s3_rust /var/lib/apt/lists/* /root/.cargo /root/.rustup

# Ensure the main benchmark scripts are executable (adjust path if needed)
# WORKDIR is currently /tmp/dlio_s3_rust, let's switch back to the main project dir
WORKDIR /workdir/dlio
RUN chmod +x dlio_benchmark/*.py

# Define the default command to run when the container starts
ENTRYPOINT ["/workdir/dlio/docker-entrypoint.sh"]
CMD ["/bin/bash"]
