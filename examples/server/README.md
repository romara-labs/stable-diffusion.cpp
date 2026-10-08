# Example

The following example starts `sd-server` with a standalone diffusion model, VAE, and LLM text encoder:

```
.\bin\Release\sd-server.exe --diffusion-model  ..\models\diffusion_models\z_image_turbo_bf16.safetensors --vae ..\models\vae\ae.sft  --llm ..\models\text_encoders\qwen_3_4b.safetensors --diffusion-fa --offload-to-cpu -v --cfg-scale 1.0
```

What this example does:

* `--diffusion-model` selects the standalone diffusion model
* `--vae` selects the VAE decoder
* `--llm` selects the text encoder / language model used by this pipeline
* `--diffusion-fa` enables flash attention in the diffusion model
* `--offload-to-cpu` reduces VRAM pressure by keeping weights in RAM when possible
* `-v` enables verbose logging (equivalent to `--log-level verbose`)
* `--cfg-scale 1.0` sets the default CFG scale for generation

Logging defaults to `info`. Use `--log-level <level>` to select `debug`, `verbose`,
`info`, `warn`, or `error` (from most to least detailed). Each level includes
messages at that level and all less detailed levels. `-v` and `--verbose` are
equivalent to `--log-level verbose`. If repeated, the last logging option wins.

After the server starts successfully:

* the web UI is available at `http://127.0.0.1:1234/`
* the native async API is available under `/sdcpp/v1/...`
* the compatibility APIs are available under `/v1/...` and `/sdapi/v1/...`

If you want to use a different host or port, pass:

```bash
--listen-ip <ip> --listen-port <port>
```

# Web UI

This build ships as a **backend-only server**: no web frontend is embedded, and
building it does not require Node.js or pnpm.

The root endpoint (`/`) responds with a plain-text status message. If you want
to serve a custom UI (for example a local copy of
[sdcpp-webui](https://github.com/leejet/sdcpp-webui)), point the server at an
HTML file:

```bash
sd-server --serve-html-path ./index.html
```

The server will load and serve the specified `index.html` file at `/`. This is
useful when:

* developing or testing frontend changes
* using a custom UI
* avoiding rebuilding the binary after frontend modifications

# Usage

For detailed command-line arguments, run:

```bash
./bin/sd-server -h
```

For completely black or white images or videos, NaNs, and the `--linear-scale` /
`--attn-scale` startup options, see [Troubleshooting](../../docs/troubleshooting.md).
