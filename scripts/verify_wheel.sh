#!/bin/bash
# Post-build gate: SASS arch coverage + MPI-leak check + metadata dump.
# Runs after the wheel artifact is already uploaded, so a verify failure never loses a build.
set -euo pipefail
OUT="$GITHUB_WORKSPACE/artifacts"
WHL=$(ls "$OUT"/torch-*.whl)
VFY=$(mktemp -d)
unzip -o -q "$WHL" 'torch/lib/libtorch_cuda.so' 'torch/lib/libtorch_cpu.so' -d "$VFY"

if [ -x "$OUT/cuobjdump" ]; then
  "$OUT/cuobjdump" --list-elf "$VFY/torch/lib/libtorch_cuda.so" | grep -oE 'sm_[0-9]+' | sort -uV > "$VFY/archs"
else
  strings -n 6 "$VFY/torch/lib/libtorch_cuda.so" | grep -oE 'sm_[0-9]+' | sort -uV > "$VFY/archs"
fi
echo "--- embedded SASS archs:"; cat "$VFY/archs"
for a in sm_75 sm_80 sm_86 sm_90 sm_100 sm_103 sm_120; do
  grep -qx "$a" "$VFY/archs" || { echo "MISSING $a"; exit 1; }
done
if ldd "$VFY/torch/lib/libtorch_cpu.so" 2>/dev/null | grep -qi mpi; then echo "MPI LEAK"; exit 1; fi
echo "--- wheel metadata (deps of interest):"
unzip -p "$WHL" '*.dist-info/METADATA' | grep -E '^(Name|Version):|Requires-Dist: (nvidia-cublas|nvidia-cudnn|.*triton)' || true
echo "--- verification OK: all archs present, no MPI leak"
rm -rf "$VFY"
