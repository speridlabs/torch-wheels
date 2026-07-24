#!/bin/bash
# Builds one multiarch torch wheel inside PyTorch's own manylinux builder image.
# Inputs (env): TORCH_VERSION (2.9.1|2.11.0), PY (3.10|3.11|3.12)
# Output: $GITHUB_WORKSPACE/artifacts/torch-*.whl (+ ptxas, cuobjdump extracted from the image)
set -euo pipefail

case "$TORCH_VERSION" in
  2.9.1)
    IMAGE="pytorch/manylinux2_28-builder:cuda13.0-7ab3bdbdeaf1449fc537661c324f277d86b2ff34"
    ARCH_SED='s/TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;10.0;12.0+PTX"/TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;10.0;10.3;12.0+PTX"/'
    EXTRA_REQS="nvidia-cuda-nvrtc==13.0.48; platform_system == 'Linux' | nvidia-cuda-runtime==13.0.48; platform_system == 'Linux' | nvidia-cuda-cupti==13.0.48; platform_system == 'Linux' | nvidia-cudnn-cu13>=9.24.0.43; platform_system == 'Linux' | nvidia-cublas>=13.6.0.2; platform_system == 'Linux' | nvidia-cufft==12.0.0.15; platform_system == 'Linux' | nvidia-curand==10.4.0.35; platform_system == 'Linux' | nvidia-cusolver==12.0.3.29; platform_system == 'Linux' | nvidia-cusparse==12.6.2.49; platform_system == 'Linux' | nvidia-cusparselt-cu13==0.8.0; platform_system == 'Linux' | nvidia-nccl-cu13==2.27.7; platform_system == 'Linux' | nvidia-nvshmem-cu13==3.3.24; platform_system == 'Linux' | nvidia-nvtx==13.0.39; platform_system == 'Linux' | nvidia-nvjitlink==13.0.39; platform_system == 'Linux' | nvidia-cufile==1.15.0.42; platform_system == 'Linux'"
    ;;
  2.11.0)
    IMAGE="pytorch/manylinux2_28-builder:cuda13.0-2592876440f755b3c151b1fd9d09dbe6c7cac38a"
    ARCH_SED='s/TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;10.0"/TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;10.0;10.3"/'
    EXTRA_REQS="cuda-toolkit[nvrtc,cudart,cupti,cufft,curand,cusolver,cusparse,cufile,nvjitlink,nvtx]==13.0.2; platform_system == 'Linux' | cuda-bindings>=13.0.3,<14; platform_system == 'Linux' | nvidia-cublas>=13.6.0.2; platform_system == 'Linux' | nvidia-cudnn-cu13>=9.24.0.43; platform_system == 'Linux' | nvidia-cusparselt-cu13==0.8.0; platform_system == 'Linux' | nvidia-nccl-cu13==2.28.9; platform_system == 'Linux' | nvidia-nvshmem-cu13==3.4.5; platform_system == 'Linux'"
    ;;
  *) echo "unknown TORCH_VERSION=$TORCH_VERSION"; exit 1 ;;
esac

VERSION="${TORCH_VERSION}+cu130.multiarch"
SRC="$GITHUB_WORKSPACE/pytorch"
OUT="$GITHUB_WORKSPACE/artifacts"
mkdir -p "$OUT"
df -h / | tail -1

docker pull -q "$IMAGE"

cid=$(docker run --detach --tty \
  -v "$SRC:/pytorch" -v "$OUT:/artifacts" -w / \
  -e BINARY_ENV_FILE=/tmp/env \
  -e BUILD_ENVIRONMENT=linux-binary-manywheel \
  -e DESIRED_CUDA=cu130 \
  -e "DESIRED_PYTHON=$PY" \
  -e GPU_ARCH_TYPE=cuda \
  -e GPU_ARCH_VERSION=13.0 \
  -e PACKAGE_TYPE=manywheel \
  -e PYTORCH_FINAL_PACKAGE_DIR=/artifacts \
  -e PYTORCH_ROOT=/pytorch \
  -e "PYTORCH_EXTRA_INSTALL_REQUIREMENTS=$EXTRA_REQS" \
  -e SKIP_ALL_TESTS=1 \
  -e "MAX_JOBS=${BUILD_MAX_JOBS:-8}" \
  "$IMAGE")
trap 'docker stop "$cid" >/dev/null 2>&1 || true' EXIT

docker exec -t -w /pytorch "$cid" bash -c 'bash .circleci/scripts/binary_populate_env.sh'
docker exec -t "$cid" sed -i "$ARCH_SED" /pytorch/.ci/manywheel/build_cuda.sh
echo "--- arch list after patch:"
docker exec -t "$cid" grep -n 'TORCH_CUDA_ARCH_LIST="7' /pytorch/.ci/manywheel/build_cuda.sh || true
docker exec -t "$cid" bash -c "grep -q '10.3' /pytorch/.ci/manywheel/build_cuda.sh"

docker exec -t "$cid" bash -c "printf '%s\n' 'export OVERRIDE_PACKAGE_VERSION=$VERSION' >> /tmp/env && source /tmp/env && bash /pytorch/.ci/manywheel/build.sh"

tmpc=$(docker create "$IMAGE")
docker cp "$tmpc:/usr/local/cuda-13.0/bin/ptxas" "$OUT/ptxas" || echo "warn: ptxas extraction failed (triton companion will need a fallback)"
docker cp "$tmpc:/usr/local/cuda-13.0/bin/cuobjdump" "$OUT/cuobjdump" || echo "warn: cuobjdump extraction failed (verify uses strings fallback)"
docker rm -f "$tmpc" >/dev/null

WHL=$(ls "$OUT"/torch-*.whl)
echo "--- built: $(basename "$WHL") ($(du -h "$WHL" | cut -f1))"
