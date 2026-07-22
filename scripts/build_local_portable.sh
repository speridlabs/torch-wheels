#!/bin/bash
# Local fallback build: torch multiarch (adds sm_103/B300) on any x86_64 machine, no GPU needed.
# Canonical builds happen in CI (.github/workflows/build-wheels.yml) — see docs/building.md
# for why CI wheels have correct pip-package RUNPATHs and this one does not.
set -euo pipefail

TORCH_TAG=${TORCH_TAG:-v2.9.1}
WORKDIR=${WORKDIR:-$PWD/torchbuild}
ARCHS=${ARCHS:-"7.5;8.0;8.6;8.9;9.0;10.0;10.3;12.0+PTX"}
VERSION_TAG=${VERSION_TAG:-${TORCH_TAG#v}+cu130.multiarch}
JOBS=${MAX_JOBS:-$(( $(nproc) * 8 / 10 ))}
PYBIN=${PYBIN:-python3.12}

if [ ! -x /usr/local/cuda-13.0/bin/nvcc ]; then
  echo ">> CUDA 13.0 required at /usr/local/cuda-13.0 (NOT 13.2 — its CCCL breaks torch 2.9 SortStable.cu)."
  echo ">> Easiest: run inside docker image nvidia/cuda:13.0.1-devel-ubuntu24.04"
  echo ">> Or the runfile without driver:"
  echo "   wget https://developer.download.nvidia.com/compute/cuda/13.0.1/local_installers/cuda_13.0.1_580.82.07_linux.run"
  echo "   sudo sh cuda_13.0.1_*.run --toolkit --silent"
  exit 1
fi
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH

mkdir -p "$WORKDIR"; cd "$WORKDIR"
if [ ! -d pytorch ]; then
  git clone --branch "$TORCH_TAG" --depth 1 --recursive https://github.com/pytorch/pytorch
fi

command -v "$PYBIN" >/dev/null || PYBIN=python3
[ -d buildenv ] || "$PYBIN" -m venv buildenv
source buildenv/bin/activate
pip install -q --upgrade pip
cd pytorch
pip install -q -r requirements.txt cmake ninja
pip install -q nvidia-cudnn-cu13

CUDNN_PKG=$(python -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__))")
export CUDNN_ROOT=$CUDNN_PKG
export CUDNN_INCLUDE_DIR=$CUDNN_PKG/include
export CUDNN_LIBRARY=$CUDNN_PKG/lib/libcudnn.so.9

export TORCH_CUDA_ARCH_LIST="$ARCHS"
export USE_CUDA=1 USE_CUDNN=1 BUILD_TEST=0 USE_MPI=0
export MAX_JOBS=$JOBS
export PYTORCH_BUILD_VERSION=$VERSION_TAG
export PYTORCH_BUILD_NUMBER=1

# TRAP: changing ARCHS vs a previous build requires starting clean —
# the cmake cache does NOT self-invalidate and you get the OLD wheel re-tagged:
#   rm -rf build dist

echo "=== BUILD START $(date -u '+%H:%M:%S') | tag=$TORCH_TAG | archs=$ARCHS | jobs=$MAX_JOBS ==="
nvcc --version | tail -1
python setup.py bdist_wheel
echo "=== BUILD END $(date -u '+%H:%M:%S') ==="
ls -lh dist/

echo ">> MPI leak check (must be empty):"
TMP=$(mktemp -d); unzip -o -q dist/torch-*.whl 'torch/lib/libtorch_cpu.so' 'torch/lib/libtorch_cuda.so' -d "$TMP"
ldd "$TMP/torch/lib/libtorch_cpu.so" | grep -i mpi || true
echo ">> Embedded SASS archs (expect sm_75 80 86 89 90 100 103 120):"
"$CUDA_HOME/bin/cuobjdump" --list-elf "$TMP/torch/lib/libtorch_cuda.so" 2>/dev/null | grep -oE 'sm_[0-9]+' | sort -uV | uniq -c || \
  strings -n 6 "$TMP/torch/lib/libtorch_cuda.so" | grep -oE '\-arch sm_[0-9]+' | sort -uV
rm -rf "$TMP"
