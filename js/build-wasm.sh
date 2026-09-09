#!/usr/bin/env bash
# Build the WASM modules consumed by the npm package.
# The CPU has two variants, because -pthread requires SharedArrayBuffer and therefore
# cross-origin isolation, which some hosts (GitHub Pages) cannot serve:
#   js/wasm/cfireants-mt.js|wasm   threaded CPU, needs COOP/COEP
#   js/wasm/cfireants.js|wasm      single-threaded CPU, runs anywhere
#   js/wasm/cfireants-gpu.js|wasm  WebGPU, device supplied by JS
set -euo pipefail

cd "$(dirname "$0")/.."
OUT=js/wasm
command -v emcmake >/dev/null || { echo "emscripten not on PATH (source emsdk_env.sh)"; exit 1; }
mkdir -p "$OUT"

LD_COMMON="\
-sMODULARIZE=1 -sEXPORT_ES6=1 -sINVOKE_RUN=0 -sEXIT_RUNTIME=1 \
-sEXPORTED_RUNTIME_METHODS=['callMain','FS','ENV'] \
-sEXPORTED_FUNCTIONS=['_main','_malloc','_free'] \
-sSTACK_SIZE=4194304 -sENVIRONMENT=web,worker,node"
GROWTH="-sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB"

build() {
  local name=$1 threads=$2 gpu=${3:-OFF} dir=build-wasm-$1 cflags ldflags
  cflags="-msimd128"
  ldflags="$LD_COMMON"
  if [ "$gpu" = ON ]; then
    # emdawnwebgpu forwards to the browser's WebGPU, whose calls are all async.
    # ASYNCIFY lets the synchronous C suspend at the poll and readback points.
    # Memory growth is incompatible with the port: a growable heap is a resizable
    # ArrayBuffer, which TextDecoder rejects, and shader creation then fails
    # reading its own WGSL. Hence a fixed heap.
    cflags="$cflags --use-port=emdawnwebgpu"
    cflags="$cflags ${CFIREANTS_WGPU_EXTRA_CFLAGS:-}"
    ldflags="$ldflags --use-port=emdawnwebgpu -sASYNCIFY -sASYNCIFY_STACK_SIZE=32768 -sINITIAL_MEMORY=1073741824"
  else
    ldflags="$ldflags $GROWTH"
  fi
  if [ "$threads" = ON ]; then
    # PROXY_TO_PTHREAD moves main() off the browser main thread, keeping it free
    # to service on-demand pthread spawns. It also makes callMain return before
    # the program ends, so the JS wrapper must await Module.onExit.
    cflags="$cflags -pthread"
    ldflags="$ldflags -pthread -sPROXY_TO_PTHREAD -sPTHREAD_POOL_SIZE='navigator.hardwareConcurrency+1'"
  fi
  emcmake cmake -S . -B "$dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCFIREANTS_CUDA=OFF -DCFIREANTS_METAL=OFF -DCFIREANTS_WEBGPU="$gpu" \
    -DCFIREANTS_THREADS="$threads" -DCFIREANTS_ZLIB=OFF \
    -DCMAKE_C_FLAGS="$cflags" -DCMAKE_EXE_LINKER_FLAGS="$ldflags" >/dev/null
  cmake --build "$dir" --target cfireants_reg -j8 >/dev/null
  cp "$dir/cfireants_reg.js"   "$OUT/$name.js"
  cp "$dir/cfireants_reg.wasm" "$OUT/$name.wasm"
  # The glue hardcodes its own basenames for sidecar assets; rewrite to the new name.
  perl -pi -e "s/cfireants_reg\.(wasm|js)/$name.\$1/g" "$OUT/$name.js"
}

build cfireants-mt  ON  OFF
build cfireants     OFF OFF
build cfireants-gpu OFF ON
ls -lh "$OUT"
