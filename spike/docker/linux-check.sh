#!/usr/bin/env bash
# usage: linux-check.sh <arm64|amd64>
# ubuntu:24.04 でビルドし、ONNX Runtime を入れていない素のイメージ上で実行して静的リンクを確かめる。
# amd64 は qemu 上で rustc が SIGSEGV するため、native arm64 コンテナからクロスコンパイルする。
set -euo pipefail

arch="$1"
spike="$(cd "$(dirname "$0")/.." && pwd)"
target_dir="/work/target-linux-${arch}"

if [ "${arch}" = amd64 ]; then
  bin="${target_dir}/x86_64-unknown-linux-gnu/release/notrieve-spike"
  echo "== cross-build on ubuntu:24.04 (native arm64 -> x86_64-unknown-linux-gnu)"
  docker run --rm --platform linux/arm64 -v "${spike}:/work" -w /work ubuntu:24.04 bash -euo pipefail -c "
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl ca-certificates build-essential pkg-config gcc-x86-64-linux-gnu g++-x86-64-linux-gnu binutils-x86-64-linux-gnu >/dev/null
    curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.98.1 >/dev/null
    . \"\$HOME/.cargo/env\"
    rustup target add x86_64-unknown-linux-gnu
    CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=x86_64-linux-gnu-gcc CARGO_TARGET_DIR=${target_dir} cargo build --release --locked --target x86_64-unknown-linux-gnu
    echo '== NEEDED'
    x86_64-linux-gnu-readelf -d ${bin} | grep NEEDED
    echo '== max required symbol versions'
    x86_64-linux-gnu-objdump -T ${bin} | grep -oE '(GLIBC|GLIBCXX|CXXABI)_[0-9.]+' | sort -uV | tail -n 20
  "
else
  bin="${target_dir}/release/notrieve-spike"
  echo "== build on ubuntu:24.04 (${arch})"
  docker run --rm --platform "linux/${arch}" -v "${spike}:/work" -w /work ubuntu:24.04 bash -euo pipefail -c "
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl ca-certificates build-essential pkg-config >/dev/null
    curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.98.1 >/dev/null
    . \"\$HOME/.cargo/env\"
    CARGO_TARGET_DIR=${target_dir} cargo build --release --locked
  "
fi

for image in ubuntu:24.04 ubuntu:22.04; do
  echo "== run on ${image} (${arch})"
  docker run --rm --platform "linux/${arch}" -v "${spike}:/work:ro" -w /work "${image}" bash -c "
    ldd ${bin} | grep -i onnxruntime || echo 'no libonnxruntime dependency'
    ${bin} embed-check models/ruri-v3-130m onnx/model_int8.onnx out/reference-model_int8.json
  " && echo "== ${image} (${arch}): exit=0" || echo "== ${image} (${arch}): exit=$?"
done
