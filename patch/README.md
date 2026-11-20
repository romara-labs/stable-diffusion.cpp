# Patch Series: Romara Labs Fork on Upstream 532d2c1

## Base and Result

**Upstream base (mainstream, oracle):**

`532d2c15c430a785a712b2d94281f04e733008a4`

Repository: [leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)

Date: `2026-11-15`

Commit: `refactor(api): adopt opaque options, tensors and results (#2365)`

**Resulting tree after applying all patches:**

`0015a710ddabd6ac64318ac548e585cda5ce2664`

---

## Prepare and Compile

The main checkout contains the official upstream base and the patch series in `patch/`, without the fork changes applied.

Run the following commands from the repository root to compile the fork in a separate working copy:

```sh
(
    set -eu

    # Resolve the root directory of the main repository.
    repo=$(git rev-parse --show-toplevel)

    # Create a temporary working directory.
    work=$(mktemp -d)
    src="$work/src"

    # Verify the SHA-256 checksums of all patch files.
    (
        cd "$repo/patch"
        sha256sum -c SHA256SUMS
    )

    # Clone the repository without checking out a branch.
    git clone --no-checkout "$repo" "$src"

    # Check out the exact upstream base commit.
    git -C "$src" switch --detach \
        532d2c15c430a785a712b2d94281f04e733008a4

    # Apply all patches in numerical order to the index.
    for f in "$repo"/patch/0*.patch; do
        git -C "$src" apply --index "$f"
    done

    # Verify that the resulting Git tree matches the expected hash.
    test "$(git -C "$src" write-tree)" = \
        0015a710ddabd6ac64318ac548e585cda5ce2664

    # Initialize the required submodules.
    git -C "$src" submodule update --init -- \
        ggml \
        thirdparty/libwebp \
        thirdparty/libwebm

    # Configure the Release build.
    cmake -S "$src" -B "$work/build" \
        -DCMAKE_BUILD_TYPE=Release

    # Compile the CLI and server targets using 8 parallel jobs.
    cmake --build "$work/build" \
        --config Release \
        --target sd-cli sd-server \
        --parallel 8

    # Display the location of the generated build.
    printf 'Build in %s\n' "$work/build"
)
```

### Vulkan Build

To enable Vulkan support, append the following option to the CMake configuration command:

```sh
-DSD_VULKAN=ON
```

For example:

```sh
cmake -S "$src" -B "$work/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSD_VULKAN=ON
```

**Important:** Compiling directly from the main checkout, including through the official workflow, uses only the upstream base. The fork changes are applied exclusively in the separate temporary checkout created by the script above.

---

## Contents

The patch series contains seven patches, organized by file groups.

| Patch | Files | Description |
|---|---:|---|
| `0001-backend-only-server` | 16 | Removal of WebUI, Docker, CI and README components |
| `0002-server-sse-streaming` | 6 | Image and video streaming through SSE on the server |
| `0003-server-png-response-overhead` | 4 | Reduced PNG response overhead and `--png-compression-level` support |
| `0004-krea2-vulkan-param-transform-and-loader` | 7 | Krea2 Vulkan support, parameter transformation during loading and loader threads |
| `0005-audio-vae-compute-context-release` | 2 | Release of the audio VAE compute context |
| `0006-comfyui-gguf-layout-and-int-metadata` | 2 | GGUF support in ComfyUI layout (flattened tensors) and complete GGUF header reading |
| `0007-clipproj-minimax-h3-video-and-image` | 15 | ClipProj (Safetensors and GGUF) for MiniMax-H3 video, MiniMax-H3 image and P-H3 image |

**Patch organization**

The division is based on file groups. Each file belongs to exactly one patch.

Consequently, a patch may also contain a code snippet associated with another feature. For example, the MiniMax-H3 empty prompt fix is located in `conditioner.hpp`, which belongs to patch `0006`.

---

## Verified

The following checks and tests were completed successfully.

### 1. Patch Application and Git Tree

- `git apply --check --cached`: passed.
- Application of all seven patches in numerical order on upstream commit `532d2c1`: passed.
- Resulting Git tree matches the expected hash: `0015a710ddabd6ac64318ac548e585cda5ce2664`.

### 2. Compilation

The resulting tree was successfully compiled for both CPU and Vulkan configurations.

Verified targets:

- Library (`lib`)
- `sd-cli`
- `sd-server`

**Result:** All builds passed.

### 3. Vulkan Server

Tests were performed using the Vulkan server without a template.

Results:

- Recipe catalog: 112 recipes.
- Generation and PNG read routes: HTTP 503.
- PNG compression levels tested: `0` and `9`.
- Rejected requests: 10.

### 4. ClipProj

All eight arrays in `cliproj-h3/` were successfully loaded and projected.

Test coverage:

- 4 Safetensors arrays.
- 4 GGUF arrays.
- 4B and 8B configurations.
- Ridge and `-mlp` variants.
- CPU execution.
- Vulkan execution using an NVIDIA RTX 3050.

The results matched the NumPy reference implementation.

| Configuration | Maximum Relative Error |
|---|---:|
| Ridge | `6e-9` |
| `-mlp` | `5e-4` |

Additional verification:

- Safetensors and GGUF produced identical results.
- CPU and Vulkan projection tests completed successfully.

**Test location:**

`_kiyotaka_patchwork/c6dfa9def12f-b323dc689572/cliproj-test/`

---

## Not Verified

The following areas were not covered by the verification process:

### 1. Model Inference

No inference was performed because no templates were available.

### 2. Other Compute Backends

The following backends were not compiled:

- CUDA
- HIP
- Metal
- SYCL

### 3. Full ClipProj Integration

No H3 model or Qwen3-VL encoder was available on the test machine.

Therefore, ClipProj was tested only in isolation, covering matrix loading and projection.

It was **not tested within an actual video or image generation pipeline**.

---

## Summary

The seven-patch series applies cleanly to the specified upstream commit and produces the expected Git tree.

CPU and Vulkan builds passed, including the library, CLI and server targets. ClipProj also passed isolated CPU and Vulkan tests against the NumPy reference implementation.

However, full model inference, integration with H3/Qwen3-VL generation pipelines and compilation of CUDA, HIP, Metal and SYCL backends remain unverified.
