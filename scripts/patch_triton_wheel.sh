#!/bin/bash
# Repackage the official triton wheels with a CUDA-13 ptxas so torch.compile
# works on sm_103a (pytorch#163801) without TRITON_PTXAS_PATH.
# Inputs: $1 = a built torch wheel (to read its triton pin), $2 = ptxas binary, $3 = outdir
set -euo pipefail
TORCH_WHEEL=$1
PTXAS=$2
OUTDIR=$3
mkdir -p "$OUTDIR"

PIN=$(unzip -p "$TORCH_WHEEL" '*.dist-info/METADATA' | grep -oE "^Requires-Dist: (pytorch-)?triton==[0-9.]+" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
[ -n "$PIN" ] || { echo "no triton pin found in $TORCH_WHEEL"; exit 1; }
echo "triton pin: $PIN"

pip install -q wheel
WORK=$(mktemp -d)
curl -s "https://pypi.org/pypi/triton/$PIN/json" > "$WORK/meta.json"
for CP in cp310 cp311 cp312; do
  URL=$(python3 -c "
import json
files = json.load(open('$WORK/meta.json'))['urls']
for f in files:
    if '$CP' in f['filename'] and 'x86_64' in f['filename'] and 'manylinux' in f['filename']:
        print(f['url']); break
")
  [ -n "$URL" ] || { echo "no $CP wheel for triton $PIN"; exit 1; }
  echo "--- $CP: $URL"
  curl -sL -o "$WORK/orig-$CP.whl" "$URL"
  ( cd "$WORK" && python3 -m wheel unpack "orig-$CP.whl" -d "unpack-$CP" )
  DIR=$(ls -d "$WORK/unpack-$CP"/triton-*)
  install -m 0755 "$PTXAS" "$DIR/triton/backends/nvidia/bin/ptxas"
  mv "$DIR/triton-$PIN.dist-info" "$DIR/triton-$PIN+cu130ptxas.dist-info"
  sed -i "s/^Version: $PIN$/Version: $PIN+cu130ptxas/" "$DIR/triton-$PIN+cu130ptxas.dist-info/METADATA"
  mv "$DIR" "$WORK/unpack-$CP/triton-$PIN+cu130ptxas"
  ( cd "$WORK" && python3 -m wheel pack "unpack-$CP/triton-$PIN+cu130ptxas" -d "$OUTDIR" )
done
ls -lh "$OUTDIR"
