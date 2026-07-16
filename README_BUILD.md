# Kit de build: torch 2.9.1 multiarch (B300/sm_103 incluida) — SIN GPU

Compila la wheel en cualquier maquina x86_64 (RunPod H100, pod CPU, CI...).
nvcc es compilacion cruzada: NO usa GPU ni driver. La GPU solo hace falta
para la validacion final.

## Requisitos de la maquina de build
- x86_64, Ubuntu 22.04/24.04 (glibc <= la de las maquinas destino)
- Python 3.12 (la wheel debe ser cp312, igual que el venv de world_extensor)
- CUDA toolkit 13.0 EXACTO (ver trampas) — mas facil: imagen docker
  `nvidia/cuda:13.0.1-devel-ubuntu24.04`
- CPU/RAM: lo que haya (MAX_JOBS auto = 80% de cores; ~2 GB RAM/job)
- Disco: ~60 GB libres
- Tiempo de referencia: en 192 cores, single-arch ~55 min; 8 archs ~1.5-3 h.
  En una maquina de 32 cores multiplica x5-6.

## Uso
```bash
chmod +x build_torch_portable.sh
./build_torch_portable.sh
# wheel resultante: torchbuild/pytorch/dist/torch-2.9.1+cu130.multiarch-cp312-*.whl
```
Variables opcionales: `ARCHS`, `VERSION_TAG`, `MAX_JOBS`, `WORKDIR`.

## Las 6 trampas (todas verificadas a base de golpes)

0. **USE_MPI=0 SIEMPRE**: si la maquina de build tiene OpenMPI instalado (las DLAMI
   de AWS lo traen en /opt/amazon/openmpi), el build lo autodetecta y libtorch_cpu.so
   queda linkado a libmpi.so.40 => `import torch` casca en cualquier maquina sin MPI.
   Las wheels oficiales no linkan MPI; la nuestra tampoco debe. Verificacion post-build:
   `ldd torch/lib/libtorch_cpu.so | grep -i mpi` debe salir vacio.
1. **CUDA 13.0 exacto, no "13.x"**: el CCCL/cub de 13.2 es mas nuevo de lo que
   torch 2.9.1 espera y revienta compilando `SortStable.cu` (error de operator+=
   en dispatch_segmented_radix_sort). Con 13.0 compila limpio.
2. **Cambio de ARCHS => `rm -rf build dist`**: la cache de cmake/ninja NO se
   invalida al cambiar TORCH_CUDA_ARCH_LIST; sin limpiar sale una wheel VIEJA
   re-etiquetada (mismo tamano, 2 min de "build"). Sospecha si tarda poco.
3. **Python 3.12**: fornax y el venv del proyecto son cp312.
4. **cuDNN**: no hace falta de sistema; `pip install nvidia-cudnn-cu13` y
   apuntar CUDNN_ROOT/INCLUDE_DIR/LIBRARY al paquete (el script lo hace).
5. **El pack va JUNTO**: esta wheel es CUDA 13 => en el venv destino van las
   TRES piezas a la vez o nada:
   - torch (esta wheel)
   - torchvision==0.24.1 del index cu130 (la cu128 NO importa contra torch cu13)
   - fornax-2.4.0+pt29cu130 (la pt29cu128 NO casa por ABI)
   Y tras cada `uv sync` re-aplicarlas (el lock revierte a cu128) — postsync.sh.

## Por que estas archs
`ARCHS="7.5;8.0;8.6;8.9;9.0;10.0;10.3;12.0+PTX"` = exactamente el SASS que
llevan las 3 libs de fornax cu130 (verificado por ELF):
sm_75 Turing / sm_80,86 Ampere / sm_89 Ada (L4, L40S) / sm_90 Hopper (H100, H200)
/ sm_100 B200 / sm_103 B300 / sm_120 RTX PRO 6000. El +PTX es fallback de torch
para archs futuras (fornax no lo lleva: fuera de su lista NO arranca).

## Verificacion
- Sin GPU (en la maquina de build): el script ya vuelca las archs embebidas
  en libtorch_cuda.so — deben salir las 8.
- Con GPU (en cada tipo de maquina destino):
  ```bash
  python -c "import torch; print(torch.cuda.get_arch_list())"
  rm -rf ~/.nv/ComputeCache   # y correr un workload
  du -sh ~/.nv/ComputeCache   # ~0 => sin JIT, victoria
  ```
  Referencia B300: iter0 frio ~22 s (con JIT eran ~70 s); queda ~9 s/460 MB
  de fusion runtime de cuDNN/cuBLAS que es normal e inevitable.

## Contexto
- Ni torch 2.9 ni 2.13 oficiales (jul 2026) traen sm_103: en B300 JITean PTX
  (+50-67 s por proceso, cache en ~/.nv/ComputeCache). Esta wheel lo elimina.
- Emparejamiento ABI: (torch minor, CUDA major) => pt29+cu130 casan con
  cualquier torch 2.9.x/CUDA 13, oficial o casero. Si subis a torch 2.10+,
  toca recompilar fornax como pt210.
