# torch-wheels

Official [PyTorch](https://github.com/pytorch/pytorch) wheels rebuilt with **native B300 / Blackwell Ultra support (`sm_103`)** — the one thing upstream doesn't ship. Unmodified source, official feature set and packaging; the only delta is the arch list (`+8.9 Ada, +10.3 B300`). Every GPU the official wheel supports still works, bit-identical.

**Why:** on B300, official wheels pay a one-time driver recompilation of every kernel (~20–60 s stall + hundreds of MB of `~/.nv/ComputeCache`, re-paid per uncached pod) and `torch.compile` is broken on torch 2.9 ([pytorch#163801](https://github.com/pytorch/pytorch/issues/163801)). These wheels fix all of it at install time. Details: [docs/why-multiarch.md](docs/why-multiarch.md).

## Wheels

| release | torch | CUDA | python |
|---|---|---|---|
| `v2.9.1-cu130-multiarch` | 2.9.1 | 13.0 | cp310 · cp311 · cp312 |
| `v2.11.0-cu130-multiarch` | 2.11.0 | 13.0 | cp310 · cp311 · cp312 |

Linux x86_64, glibc ≥ 2.28, host driver ≥ 580.65. `+cu130.multiarch` satisfies any `torch>=2.9` requirement. Each release also has `triton-*+cu130ptxas` companion wheels.

## Install

```bash
pip install https://github.com/speridlabs/torch-wheels/releases/download/v2.9.1-cu130-multiarch/torch-2.9.1%2Bcu130.multiarch-cp312-cp312-manylinux_2_28_x86_64.whl
```

uv (`pyproject.toml`) — every library in your tree that needs `torch` resolves against it:

```toml
[project]
dependencies = ["torch==2.9.1"]

[tool.uv.sources]
torch  = { url = "https://github.com/speridlabs/torch-wheels/releases/download/v2.9.1-cu130-multiarch/torch-2.9.1%2Bcu130.multiarch-cp312-cp312-manylinux_2_28_x86_64.whl" }
triton = { url = "https://github.com/speridlabs/torch-wheels/releases/download/v2.9.1-cu130-multiarch/triton-3.5.1%2Bcu130ptxas-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl" }
```

Installing brings everything needed, no config:

- metadata pins `nvidia-cublas>=13.6` and `nvidia-cudnn-cu13>=9.24` (official pins older versions that also recompile on B300)
- the `triton` companion is stock triton with a CUDA-13 `ptxas`, fixing `torch.compile` on B300 (skip it on other GPUs if you want; escape hatch: `TRITON_PTXAS_PATH`)
- match the wheel's python (cp310/311/312) — all filenames on the release page

## Measured results (official vs these wheels, fresh pods, cold start / driver cache / `torch.compile`)

| GPU | torch | official | **these wheels** |
|---|---|---|---|
| B300 | 2.9.1 | 21.6 s / 300 MB / ❌ | **0.65 s / 0 MB / ✅** |
| B300 | 2.11.0 | 21.7 s / 296 MB / ✅ | **1.18 s / 0.5 MB / ✅** |
| H100 | 2.9.1 | 0.97 s / 0.4 MB / ✅ | 0.76 s / 0.3 MB / ✅ |
| H200 | 2.9.1 | 0.90 s / 0.4 MB / ✅ | 0.73 s / 0.3 MB / ✅ |
| RTX 5090 | 2.9.1 | 0.49 s / 0 MB / ✅ | 0.41 s / 0 MB / ✅ |

Torch-native ops bitwise-identical between builds on every GPU; cuBLAS/cuDNN ops within float tolerance; steady-state throughput equal. On non-B300 GPUs it's an exact drop-in.

Rented pods with drivers < 580: on datacenter GPUs install `cuda-compat-13-0` in the container; GeForce needs a 580+ host.

## Notes

- **Extensions**: pairs with anything built for (same torch minor, CUDA 13.0) — keep the whole cu130 set together (e.g. `torchvision` from the cu130 index).
- **Verify a wheel** (no GPU): `unzip -p torch-*.whl torch/lib/libtorch_cuda.so > /tmp/t.so && cuobjdump --list-elf /tmp/t.so | grep -c sm_103`
- **Building**: CI builds from unmodified upstream tags ([docs/building.md](docs/building.md) has the local path + traps).
- **License**: PyTorch BSD-3-Clause; unmodified-source builds carrying upstream license files. Build scripts MIT.
