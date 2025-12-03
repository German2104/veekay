#!/usr/bin/env bash
set -euo pipefail

# Simple helper to build and run the Vulkan testbed
# Usage: ./run_testbed.sh [Release|Debug]

BUILD_TYPE="${1:-Release}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

echo "[run_testbed] Configuring (${BUILD_TYPE})..."
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"

echo "[run_testbed] Building testbed target..."
cmake --build "${BUILD_DIR}" --target testbed --config "${BUILD_TYPE}"

echo "[run_testbed] Running..."
exec "${BUILD_DIR}/testbed/testbed"
