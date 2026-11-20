# Série de patches: fork Romara Labs sobre upstream 532d2c1

Base (mainstream, oracle): `532d2c15c430a785a712b2d94281f04e733008a4` (leejet/stable-diffusion.cpp, 2026-11-15, `refactor(api): adopt opaque options, tensors, and results (#2365)`).
Árvore resultante após aplicar tudo: `0015a710ddabd6ac64318ac548e585cda5ce2664`.

## Preparar e compilar

O checkout principal contém a base oficial e a série em `patch/`, sem as alterações do fork aplicadas. Execute da raiz do repositório para compilar o fork em uma cópia separada:

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
    printf 'Build em %s\n' "$work/build"
)
```

Para Vulkan, acrescente `-DSD_VULKAN=ON` ao comando de configuração. Compilar diretamente do checkout principal, inclusive pelo workflow oficial, usa apenas a base upstream.

## Conteúdo

| Patch | Arquivos | Origem no fork (base c6dfa9d) |
|---|---|---|
| 0001-backend-only-server | 16 | remoção do webui, Docker, CI, README |
| 0002-server-sse-streaming | 6 | stream de imagem e vídeo por SSE no servidor |
| 0003-server-png-response-overhead | 4 | menos overhead na resposta PNG, `--png-compression-level` |
| 0004-krea2-vulkan-param-transform-and-loader | 7 | Krea2 Vulkan, transformação de parâmetros no load, threads do loader |
| 0005-audio-vae-compute-context-release | 2 | liberação do contexto de compute dos VAEs de áudio |
| 0006-comfyui-gguf-layout-and-int-metadata | 2 | GGUF no layout ComfyUI (tensores achatados) e leitura de inteiros do cabeçalho GGUF |
| 0007-cliproj-minimax-h3-video-and-image | 15 | ClipProj (safetensors e GGUF) para MiniMax-H3 video, MiniMax-H3-Image e P-H3-Image |

A divisão é por grupos de arquivos. Um arquivo pertence a um só patch, então um patch pode incluir um trecho de outra feature (por exemplo, a correção de prompt vazio do MiniMax-H3 está em `conditioner.hpp`, dentro do 0006).

## Verificado
- `git apply --check --cached` e aplicação dos 7 patches em ordem sobre `532d2c1`: passou; a árvore bate com a árvore resultante.
- Build CPU e Vulkan (lib, sd-cli, sd-server) da árvore resultante: passou.
- Servidor Vulkan sem modelo: catálogo de 112 receitas, HTTP 503 nas rotas de geração e leitura dos níveis de compressão PNG 0 e 9 (10 rejeitado).
- ClipProj: as 8 matrizes de `cliproj-h3/` (4 safetensors, 4 GGUF; 4B e 8B; ridge e `-mlp`) carregam e projetam na CPU e no Vulkan (RTX 3050) com resultado igual à referência numpy (erro relativo máximo 6e-9 no ridge e 5e-4 no `-mlp`); safetensors e GGUF dão o mesmo resultado. Teste em `_kiyotaka_patchwork/c6dfa9def12f-b323dc689572/cliproj-test/`.

## Não verificado
Nenhuma inferência foi executada (sem modelos). CUDA, HIP, Metal e SYCL não foram compilados. Nenhum modelo H3 nem encoder Qwen3-VL existe na máquina, então o ClipProj só foi testado isolado (matriz + projeção), não dentro de uma geração de vídeo ou imagem.
