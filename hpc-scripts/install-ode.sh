#!/bin/bash
# Builds and installs ODE (Open Dynamics Engine) to ~/.local without root access.
set -euo pipefail

PREFIX="$HOME/.local"

echo "  Building ODE from source..."
rm -rf /tmp/ode-build 2>/dev/null; \
git clone https://bitbucket.org/odedevs/ode.git /tmp/ode-build && \
mkdir -p /tmp/ode-build/build && \
cd /tmp/ode-build/build && \
cmake .. -DCMAKE_INSTALL_PREFIX="$PREFIX" -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release && \
make -j4 && \
make install

echo "  ODE installed to $PREFIX"
