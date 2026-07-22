# torch-wheels

Official-equivalent [PyTorch](https://github.com/pytorch/pytorch) wheels with **native NVIDIA B300 / Blackwell Ultra support (`sm_103`)** — the one thing upstream wheels don't ship.

Every wheel here is built from the **unmodified pytorch source tag** with the official feature set (CUDA 13.0, cuDNN, NCCL, distributed, flash attention). The only delta vs. the official wheel is the GPU architecture list:

```
official cu130:  7.5;8.0;8.6;9.0;10.0;12.0+PTX
this repo:       7.5;8.0;8.6;8.9;9.0;10.0;10.3;12.0+PTX
                              ^^^          ^^^^
                              Ada native   B300 native
```

Nothing is dropped — every GPU the official wheel supports is still supported, with identical numerics (verified bit-identical greedy decoding on B300, see [docs/why-multiarch.md](docs/why-multiarch.md)).

## Why

On B300 (compute capability 10.3), official PyTorch wheels work but pay a **one-time driver recompilation of every sm_100 kernel** on first use (50–67 s stalls, hundreds of MB of `~/.nv/ComputeCache`, re-paid per uncached pod). Native `sm_103` SASS eliminates it for torch's own kernels. Full investigation with measurements: [docs/why-multiarch.md](docs/why-multiarch.md).

## Wheel matrix

| release tag | torch | CUDA | python |
|---|---|---|---|
| `v2.9.1-cu130-multiarch` | 2.9.1 | 13.0 | cp310 · cp311 · cp312 |
| `v2.11.0-cu130-multiarch` | 2.11.0 | 13.0 | cp310 · cp311 · cp312 |

Filename pattern: `torch-<V>+cu130.multiarch-cp3XX-cp3XX-manylinux_2_28_x86_64.whl`. Linux x86_64 only (glibc ≥ 2.28). The `+cu130.multiarch` local version satisfies any `torch>=2.9`-style requirement. Each release also carries `triton-<X>+cu130ptxas-*.whl` companions.

## Install

Pip:

```bash
pip install https://github.com/speridlabs/torch-wheels/releases/download/v2.9.1-cu130-multiarch/torch-2.9.1%2Bcu130.multiarch-cp312-cp312-manylinux_2_28_x86_64.whl
```

uv, in a consumer project's `pyproject.toml` (this **replaces where torch comes from** — every library in your dependency tree that requires `torch` resolves against it):

```toml
[project]
dependencies = ["torch==2.9.1"]

[tool.uv.sources]
torch = [
  { url = "https://github.com/speridlabs/torch-wheels/releases/download/v2.9.1-cu130-multiarch/torch-2.9.1%2Bcu130.multiarch-cp312-cp312-manylinux_2_28_x86_64.whl", marker = "python_version == '3.12'" },
  { url = "https://github.com/speridlabs/torch-wheels/releases/download/v2.9.1-cu130-multiarch/torch-2.9.1%2Bcu130.multiarch-cp311-cp311-manylinux_2_28_x86_64.whl", marker = "python_version == '3.11'" },
]
triton = [
  { url = "https://github.com/speridlabs/torch-wheels/releases/download/v2.9.1-cu130-multiarch/triton-3.5.1%2Bcu130ptxas-cp312-cp312-manylinux_2_28_x86_64.whl", marker = "python_version == '3.12'" },
]
```

(The `triton` source is optional but recommended: it swaps in the companion wheel whose bundled `ptxas` knows `sm_103a`, making `torch.compile` work on B300 with zero configuration. Exact triton filenames per release are listed on the release page.)

## Batteries included — nothing else to configure

Native torch SASS is only half of the B300 fix; these wheels carry the other half in their metadata:

- **NVIDIA libraries are pinned forward**: the wheel requires `nvidia-cublas >= 13.6` and `nvidia-cudnn-cu13 >= 9.24` (instead of the old 13.0 / 9.13 pins in official wheels, which pay their own driver recompilation on CC 10.3). Installing the wheel installs libraries that load natively on B300 — no manual upgrades.
- **`torch.compile` works on B300 out of the box**: official triton 3.5.x bundles a `ptxas` that predates `sm_103a` ([pytorch#163801](https://github.com/pytorch/pytorch/issues/163801)). Each release includes a companion triton wheel — identical to upstream, with the bundled `ptxas` swapped for a CUDA-13 one — pulled in automatically alongside torch. (Escape hatch if you must use stock triton: `export TRITON_PTXAS_PATH=<CUDA≥12.9>/bin/ptxas`.)

Measured on an 8×B300 node (GPT-2 + ResNet-50 + Qwen2.5, fresh caches): official wheel + old libs **21–31 s** cold start → this wheel **~3 s**, bit-identical outputs, identical steady-state throughput.

## Pairing with CUDA extension packages

These wheels report torch `2.9.1` / `2.11.0` with CUDA `13.0` — any prebuilt extension wheel compiled against the same `(torch minor, CUDA major.minor)` pairing keeps working, and the whole CUDA-13 set must move together (e.g. `torchvision` from the cu130 index, plus your own cu130-built extension wheels; mixing a cu128 piece fails at import).

## Verify a wheel (no GPU needed)

```bash
unzip -p torch-*.whl torch/lib/libtorch_cuda.so > /tmp/ltc.so
cuobjdump --list-elf /tmp/ltc.so | grep -oE 'sm_[0-9]+' | sort -uV   # must include sm_103
```

## Building

Wheels are built by this repo's CI from unmodified upstream tags and uploaded to the matching release. Local / manual builds and all the known build traps: [docs/building.md](docs/building.md).

## License

PyTorch is BSD-3-Clause (© PyTorch contributors); these are unmodified-source builds and each wheel carries upstream's license files. This repo's build scripts are MIT.
