#!/bin/bash

# Fix our PATH, we no longer have .cargo/bin in our path, so we remove it
export PATH=/workdir/dlio/.venv/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# --- Initialization Commands ---
echo "Running initialization command: uv run main.py"
uv run main.py

# Now we want to activate our virtual environment, but we can't here, need to in bash somehow?
#source .venv/bin/activate

# Add any other initialization commands here

# --- Execute the main command ---
# This will be whatever is passed to the container (e.g., /bin/bash)
echo "Initialization complete. Starting requested command: $@"
exec "$@"
