# Why these wheels exist — the B300 cold-start investigation

*Measured July 2026 on an 8× NVIDIA B300 SXM6 AC node (CC 10.3, driver 595.71), torch 2.9.1+cu130 official vs. this repo's multiarch build. Fresh `CUDA_CACHE_PATH` + fresh process per cold measurement; loaded libraries verified per-run via `/proc/self/maps`; n=3.*

## The mechanism

Official PyTorch wheels ship SASS (GPU machine code) for `…;10.0;12.0+PTX` — no `sm_103`. On a B300:

- It is **not** PTX JIT: `CUDA_DISABLE_PTX_JIT=1` changes nothing, and the wheels carry no PTX below `compute_120` (which cannot serve CC 10.3).
- It is **not** free binary compatibility either: the driver performs a **one-time, cached SASS→SASS recompilation** of each touched sm_100 kernel for CC 10.3.
- Proof by exhaustion: with `CUDA_CACHE_DISABLE=1` every process re-pays the full cost and nothing is written; with `CUDA_MODULE_LOADING=EAGER` the entire libtorch translates up-front — **144.5 s / 1.95 GB** — after which ops cost microseconds; the native-sm_103 build hits **0 cache bytes** on the same ops.

Note the 1.95 GB: one fully-translated libtorch **exceeds the driver cache's 1 GiB default cap** (`CUDA_CACHE_MAXSIZE`).

## Measured impact

Micro-benchmark (first process, fresh cache, GPU 0):

| config | torch kernels | cuBLAS fp32 | cuBLASLt bf16 | total | cache |
|---|---|---|---|---|---|
| official + old libs (cuBLAS 13.0 / cuDNN 9.13) | 2.58 s | 5.99 s | 10.15 s | **21.4 s** | 255 MB |
| official + new libs (cuBLAS 13.6 / cuDNN 9.24) | 2.59 s | 0.10 s | 0.13 s | **5.2 s** | 41 MB |
| multiarch + old libs | 0.05 s | 7.56 s | 10.66 s | **27.7 s** | 264 MB |
| **multiarch + new libs** | 0.04 s | 0.10 s | 0.13 s | **2.2 s** | 12 KB |

Real models (cold total, n=3): GPT-2 — official+new-libs 31.4 s, multiarch+new-libs **3.6 s**. ResNet-50 — 10.8 s → **2.9 s**. Real models touch hundreds of distinct kernels (GPT-2 alone wrote 423 MB of translation cache), so **both** halves of the fix are required; either alone leaves 19–31 s.

## Safety and performance equivalence

- **Bit-identical outputs** vs. the official wheel: GPT-2 fp32 greedy tokens and logits (5 decimals), GPT-2 bf16 token sequences, ResNet-50 top-5 + logits, Qwen2.5-0.5B bf16 generated text.
- **Steady-state throughput identical** (GPT-2 ~4.9 ms/iter, ResNet ~3.5 ms/iter in every config; four eager ops within ±1σ over 100 sync-timed iterations). The entire cost of the official wheel is cold-start, and the entire benefit of this one is cold-start.
- 8-GPU simultaneous cold start: ~2.2× solo time per process (CPU-side compile contention); a shared driver cache is concurrency-safe (+4%, dedups to one copy).

## The two adjacent landmines (not fixed by any torch build)

1. **NVIDIA libraries**: cuBLAS 13.0 / cuDNN 9.13 (bundled by torch 2.9.1–2.13) pay their own recompilation on CC 10.3. Fixed upstream — `pip install -U nvidia-cublas nvidia-cudnn-cu13` (13.6+ / 9.24+ load natively).
2. **`torch.compile` / Triton**: triton 3.5.1's bundled `ptxas` doesn't know `sm_103a` → `Internal Triton PTX codegen error` on B300 in *every* torch 2.9.1 build ([pytorch#163801](https://github.com/pytorch/pytorch/issues/163801)). Workaround: `TRITON_PTXAS_PATH=<any CUDA ≥12.9>/bin/ptxas`.

## Upstream status

PyTorch's official position ([RFC #172663](https://github.com/pytorch/pytorch/issues/172663)) is that CUDA-13 wheels + sm_100 compatibility *is* B300 support; no release through 2.13 (including the experimental cu132 variant) compiles `sm_103`, and none has announced plans to. When upstream ships `10.3` (or `10.0f`) in `TORCH_CUDA_ARCH_LIST` **and** bundles cuDNN ≥ 9.21, this repo retires.
