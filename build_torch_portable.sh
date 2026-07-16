#!/bin/bash
# build_torch_portable.sh — compila torch 2.9.1 multiarch (incluye sm_103/B300)
# en CUALQUIER maquina x86_64 SIN GPU. Ver README_BUILD.md para requisitos y trampas.
set -euo pipefail

WORKDIR=${WORKDIR:-$PWD/torchbuild}
ARCHS=${ARCHS:-"7.5;8.0;8.6;8.9;9.0;10.0;10.3;12.0+PTX"}   # = lista de fornax cu130 + PTX
VERSION_TAG=${VERSION_TAG:-2.9.1+cu130.multiarch}
JOBS=${MAX_JOBS:-$(( $(nproc) * 8 / 10 ))}                  # ~2 GB RAM por job

# --- 0) toolkit CUDA 13.0 EXACTO (no 13.2: su CCCL rompe SortStable.cu de torch 2.9) ---
if [ ! -x /usr/local/cuda-13.0/bin/nvcc ]; then
  echo ">> Falta CUDA 13.0 en /usr/local/cuda-13.0."
  echo ">> Opcion A (recomendada): correr en imagen docker nvidia/cuda:13.0.1-devel-ubuntu24.04"
  echo ">> Opcion B: runfile sin driver:"
  echo "   wget https://developer.download.nvidia.com/compute/cuda/13.0.1/local_installers/cuda_13.0.1_580.82.07_linux.run"
  echo "   sudo sh cuda_13.0.1_*.run --toolkit --silent"
  exit 1
fi
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH

# --- 1) fuente ---
mkdir -p "$WORKDIR"; cd "$WORKDIR"
if [ ! -d pytorch ]; then
  git clone --branch v2.9.1 --depth 1 --recursive https://github.com/pytorch/pytorch
fi

# --- 2) entorno de build: python 3.12 OBLIGATORIO (la wheel debe ser cp312) ---
PY=python3.12; command -v $PY >/dev/null || PY=python3
$PY -c 'import sys; assert sys.version_info[:2]==(3,12), "hace falta python 3.12 (cp312)"'
[ -d buildenv ] || $PY -m venv buildenv
source buildenv/bin/activate
pip install -q --upgrade pip
cd pytorch
pip install -q -r requirements.txt cmake ninja
pip install -q nvidia-cudnn-cu13    # cuDNN 9.x para CUDA 13 (para linkar)

# --- 3) cuDNN del pip (no hace falta cuDNN de sistema) ---
CUDNN_PKG=$(python -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__))")
export CUDNN_ROOT=$CUDNN_PKG
export CUDNN_INCLUDE_DIR=$CUDNN_PKG/include
export CUDNN_LIBRARY=$CUDNN_PKG/lib/libcudnn.so.9

# --- 4) flags de build ---
export TORCH_CUDA_ARCH_LIST="$ARCHS"
export USE_CUDA=1 USE_CUDNN=1 BUILD_TEST=0 USE_MPI=0
export MAX_JOBS=$JOBS
export PYTORCH_BUILD_VERSION=$VERSION_TAG
export PYTORCH_BUILD_NUMBER=1

# TRAMPA: si cambias ARCHS respecto a un build previo, hay que partir de cero
# (la cache de cmake NO se invalida sola y sale la wheel vieja re-etiquetada):
#   rm -rf build dist

echo "=== BUILD START $(date -u '+%H:%M:%S') | archs=$ARCHS | jobs=$MAX_JOBS ==="
nvcc --version | tail -1
python setup.py bdist_wheel
echo "=== BUILD END $(date -u '+%H:%M:%S') ==="
ls -lh dist/

# --- 5) verificacion SIN GPU (SASS embebido) ---
python - <<'EOF'
import zipfile, re, glob
whl = glob.glob('dist/torch-*.whl')[0]
data = zipfile.ZipFile(whl).read('torch/version.py').decode()
print(data)
EOF
echo ">> Comprobacion de archs embebidas en libtorch_cuda.so:"
TMP=$(mktemp -d); unzip -o -q dist/torch-*.whl 'torch/lib/libtorch_cuda.so' -d "$TMP"
"$CUDA_HOME/bin/cuobjdump" --list-elf "$TMP/torch/lib/libtorch_cuda.so" 2>/dev/null | grep -oE 'sm_[0-9]+' | sort -uV | uniq -c || \
  strings -n 6 "$TMP/torch/lib/libtorch_cuda.so" | grep -oE '\-arch sm_[0-9]+' | sort -uV
rm -rf "$TMP"
echo ">> Deben salir: sm_75 80 86 89 90 100 103 120. La validacion JIT-cero final: solo en GPU destino."
