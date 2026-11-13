#!/usr/bin/env bash
#
# Fetch the third-party dependencies that are not vendored in git.
#
# Only headers are needed at build time; the Vulkan loader itself comes from the
# system (Debian/Ubuntu: libvulkan-dev).
#
# Usage: scripts/fetch_deps.sh [--force]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VULKAN_HEADERS_TAG="${VULKAN_HEADERS_TAG:-v1.4.368}"
DEST="${ROOT}/third_party/vulkan-headers"

force=0
[[ "${1:-}" == "--force" ]] && force=1

if [[ -d "${DEST}/include/vulkan" && ${force} -eq 0 ]]; then
    echo "Vulkan-Headers already present at ${DEST}"
    exit 0
fi

if [[ ${force} -eq 1 ]]; then
    rm -rf "${DEST}"
fi

mkdir -p "${ROOT}/third_party"
echo "Cloning Vulkan-Headers ${VULKAN_HEADERS_TAG} (Apache-2.0)..."
git clone --depth 1 --branch "${VULKAN_HEADERS_TAG}" \
    https://github.com/KhronosGroup/Vulkan-Headers.git "${DEST}"

echo "Done: ${DEST}"
