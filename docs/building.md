# Building the wheels

## CI (canonical path)

[`.github/workflows/build-wheels.yml`](../.github/workflows/build-wheels.yml) builds the full matrix and uploads each wheel to the matching `v<TORCH>-cu130-multiarch` release. Single SKUs can be built by narrowing the manual-run inputs — useful for retries, since a full PyTorch build takes hours per wheel.

Build recipe in one line: **PyTorch's own manylinux builder image + PyTorch's own `.ci/manywheel` build scripts**, with exactly one override — `TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;10.0;10.3;12.0+PTX"` — plus the `+cu130.multiarch` version stamp. Everything else (feature flags, bundled-library resolution via pip `nvidia-*` packages and `$ORIGIN`-relative RUNPATHs, dependency pins, manylinux compliance, stripping) is inherited from upstream, which is what makes these wheels drop-in equivalent to official ones.

## Local / manual build (fallback)

[`scripts/build_local_portable.sh`](../scripts/build_local_portable.sh) builds on any x86_64 machine **without a GPU** (nvcc is a cross-compiler; a GPU is only needed for final validation). ~60 GB disk; wall time scales with cores (192 cores ≈ 1.5–3 h for 8 archs; 32 cores ≈ 5–6× that).

> The local script produces a working wheel but does **not** replicate official packaging: it links against the machine's CUDA toolkit and bakes `/usr/local/cuda-13.0/...` into the RUNPATH, so the wheel resolves the **system** cuBLAS/cuDNN and silently ignores the pip `nvidia-*` packages (upgrading them has no effect unless you override with `LD_LIBRARY_PATH`). The CI path does not have this defect. Use local builds for experiments, CI builds for the fleet.

## The known traps (all learned the hard way)

1. **`USE_MPI=0` always.** Machines with OpenMPI installed (AWS DLAMIs ship it in `/opt/amazon/openmpi`) get auto-detected, and `libtorch_cpu.so` ends up linked against `libmpi.so.40` — the wheel then fails to import anywhere MPI isn't installed. Official wheels don't link MPI. Post-build check: `ldd torch/lib/libtorch_cpu.so | grep -i mpi` must be empty.
2. **CUDA 13.0 exactly for torch 2.9.1 — not 13.2.** CUDA 13.2's CCCL/CUB is newer than torch 2.9.1 expects and fails compiling `SortStable.cu` (`operator+=` error in `dispatch_segmented_radix_sort`). 13.0 compiles clean.
3. **Changing `TORCH_CUDA_ARCH_LIST` requires `rm -rf build dist`.** The cmake/ninja cache does not invalidate on arch-list changes; without cleaning you get the *old* wheel re-tagged (telltale: a "build" that finishes in ~2 minutes).
4. **Python version = wheel tag.** The build venv's Python decides cp310/cp311/cp312; it must match the consumers.
5. **cuDNN for linking comes from pip** — no system cuDNN needed: `pip install nvidia-cudnn-cu13` and point `CUDNN_ROOT`/`CUDNN_INCLUDE_DIR`/`CUDNN_LIBRARY` at the package.
6. **The CUDA-13 set ships together.** In a consumer venv: this torch + `torchvision` from the cu130 index + any of your own cu130-built extension wheels. Mixing a cu128 piece fails at import (ABI mismatch / undefined symbols).
7. **Runtime companions.** CI wheels enforce the fixed NVIDIA libraries (`nvidia-cublas>=13.6`, `nvidia-cudnn-cu13>=9.24`) and the patched-ptxas triton through wheel metadata — nothing to do. Wheels from the *local* script carry no such pins: there you need `pip install -U nvidia-cublas nvidia-cudnn-cu13` and `TRITON_PTXAS_PATH=<CUDA ≥12.9>/bin/ptxas` for `torch.compile`. See [why-multiarch.md](why-multiarch.md).

## Verification

Without a GPU (build machine):

```bash
unzip -p dist/torch-*.whl torch/lib/libtorch_cuda.so > /tmp/ltc.so
cuobjdump --list-elf /tmp/ltc.so | grep -oE 'sm_[0-9]+' | sort -uV
# expect: sm_75 80 86 89 90 100 103 120
```

On each target GPU class:

```bash
python -c "import torch; print(torch.cuda.get_arch_list())"   # must list sm_103
CUDA_CACHE_PATH=$(mktemp -d) python your_workload.py           # then: du -sh that dir
# ~0 bytes for torch ops => no driver recompilation => victory
```
