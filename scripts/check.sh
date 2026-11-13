#!/usr/bin/env bash
#
# Single quality gate for VulTorch.
#
# Everything in this project must compile and run with ZERO warnings:
#   * C++ is built with -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor
#     -Wcast-qual and -Werror.
#   * Every compute shader is recompiled with glslangValidator and any
#     diagnostic is fatal.
#   * Python is byte-compiled and tested with warnings promoted to errors.
#
# Usage:  scripts/check.sh
# Env:    PYTHON=<interpreter>  (default: .venv/bin/python3 if present)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "${PYTHON:-}" ]]; then
    PY="$PYTHON"
elif [[ -x ".venv/bin/python3" ]]; then
    PY=".venv/bin/python3"
else
    PY="python3"
fi

echo "==> C++ backend (warnings are errors)"
cmake -S . -B build -DVULTORCH_WERROR=ON >/dev/null
cmake --build build -j"$(nproc 2>/dev/null || echo 4)"

echo "==> Codec CLI"
make -C tools clean >/dev/null
make -C tools

echo "==> Compute shaders"
shader_failed=0
for f in shaders/*.comp; do
    out="$(glslangValidator --target-env vulkan1.3 -V "$f" -o "/tmp/vultorch_$(basename "$f").spv" 2>&1)" || shader_failed=1
    if grep -qiE "warning|error" <<<"$out"; then
        echo "    $f"
        echo "$out" | sed 's/^/      /'
        shader_failed=1
    fi
done
if (( shader_failed )); then
    echo "shader validation failed" >&2
    exit 1
fi

echo "==> Python syntax (warnings are errors)"
"$PY" -W error -m compileall -q vultorch tests examples benchmarks test_lossless_roundtrip.py

echo "==> Tests (warnings are errors)"
"$PY" -W error -m pytest -q

validation_layer=""
for candidate in \
    /usr/share/vulkan/explicit_layer.d/VkLayer_khronos_validation.json \
    /usr/share/vulkan/implicit_layer.d/VkLayer_khronos_validation.json; do
    if [[ -f "$candidate" ]]; then
        validation_layer="VK_LAYER_KHRONOS_validation"
        break
    fi
done

if [[ -n "$validation_layer" ]]; then
    echo "==> Vulkan validation layer"
    set +e
    VK_INSTANCE_LAYERS="$validation_layer" \
        "$PY" -m pytest tests/test_execution_engine.py -q 2> /tmp/vultorch_validation.log
    set -e
    if grep -qE "VUID-|Validation Error" /tmp/vultorch_validation.log; then
        echo "validation layer reported errors:" >&2
        grep -E "VUID-|Validation Error" /tmp/vultorch_validation.log | sort -u | head -40 >&2
        exit 1
    fi
    echo "    clean"
else
    echo "==> Vulkan validation layer (skipped: not installed)"
fi

echo
echo "All checks passed with zero warnings."
