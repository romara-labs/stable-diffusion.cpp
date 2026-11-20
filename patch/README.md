# Patch Series: Romara Labs fork on upstream 532d2c1

Base (mainstream, oracle): `532d2c15c430a785a712b2d94281f04e733008a4` (leejet/stable-diffusion.cpp, 2026-11-15, `refactor(api): adopt opaque options, tensors and results (#2365)`).

Resulting tree after applying everything: `0015a710ddabd6ac64318ac548e585cda5ce2664`.

## Prepare and Compile

The main checkout contains the official base and the series in `patch/`, without the fork changes applied. Execute from the repository root to compile the fork into a separate copy:

```sh
(
set -eu
repo=$(git rev-parse --show-toplevel)
work=$(mktemp -d)
src="$work/src"

(cd "$repo/patch" && sha256sum -c SHA256SUMS)
git clone --no-checkout "$repo" "$src"
git -C "$src" switch --detach 532d2c15c430a785a712b2d94281f04e733008a4
for f in "$repo"/patch/0*.patch; do
git -C "$src" apply --index "$f"
done
test "$(git -C "$src" write-tree)" = 0015a710ddabd6ac64318ac548e585cda5ce2664
git -C "$src" submodule update --init -- ggml thirdparty/libwebp thirdparty/libwebm

cmake -S "$src" -B "$work/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$work/build" --config Release --target sd-cli sd-server --parallel 8
printf 'Build in %s\n' "$work/build"

```

For Vulkan, append `-DSD_VULKAN=ON` to the configuration command. Compiling directly from the main checkout, including via the official workflow, uses only the upstream base.

## Contents

| Patch | Files | Source in fork (base c6dfa9d) |

|---|---|---|
| 0001-backend-only-server | 16 | removal of webui, Docker, CI, README |
| 0002-server-sse-streaming | 6 | image and video streaming via SSE on the server |
| 0003-server-png-response-overhead | 4 | less overhead on PNG response, `--png-compression-level` |
| 0004-krea2-vulkan-param-transform-and-loader | 7 | Krea2 Vulkan, parameter transformation on load, loader threads |
| 0005-audio-vae-compute-context-release | 2 | release of the compute context of audio VAEs |
| 0006-comfyui-gguf-layout-and-int-metadata | 2 | GGUF in ComfyUI layout (flattened tensors) and full GGUF header reading |

| 0007-clipproj-minimax-h3-video-and-image | 15 | ClipProj (safetensors and GGUF) for MiniMax-H3 video, MiniMax-H3 image, and P-H3 image |

The division is by file groups. A file belongs to only one patch, so a patch can include a snippet of another feature (for example, the MiniMax-H3 empty prompt fix is ​​in `conditioner.hpp`, inside 0006).

## Verified
- `git apply --check --cached` and application of the 7 patches in order on `532d2c1`: passed; the tree matches the resulting tree.

- Build CPU and Vulkan (lib, sd-cli, sd-server) of the resulting tree: passed.

- Vulkan server without a template: catalog of 112 recipes, HTTP 503 on the generation and read routes of PNG compression levels 0 and 9 (10 rejected).

- ClipProj: the 8 arrays of `cliproj-h3/` (4 safetensors, 4 GGUF; 4B and 8B; ridge and `-mlp`) load and project on the CPU and Vulkan (RTX 3050) with a result equal to the numpy reference (maximum relative error 6e-9 in ridge and 5e-4 in `-mlp`); safetensors and GGUF give the same result. Test in `_kiyotaka_patchwork/c6dfa9def12f-b323dc689572/cliproj-test/`.

## Not verified
No inference was performed (no templates). CUDA, HIP, Metal and SYCL were not compiled. There is no H3 model or Qwen3-VL encoder in the machine, so ClipProj was only tested in isolation (matrix + projection), not within a video or image generation process.
