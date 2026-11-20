# stable-diffusion.cpp Server

`sd-server` provides HTTP APIs for image and video generation and model
management. This fork ships a **backend-only server**: no web frontend is
embedded, and building it does not require Node.js or pnpm. Download or build
the server before starting; see the [build guide](../../docs/build.md). To use
a web page with it, serve an external one with `--serve-html-path` (see
[Using an external frontend](#using-an-external-frontend)).

- [Start the server](#start-the-server): launchers and direct execution.
- [Load a model and generate](#load-a-model-and-generate): the API-driven workflow.
- [Model management](#model-management): model directories, recipes, and startup models.
- [Server configuration](#server-configuration): launcher settings, network, logging, and help.
- [API](#api): native and compatibility endpoints.
- [Using an external frontend](#using-an-external-frontend): serve a custom Web UI.

Inference settings follow this priority: request fields, explicit server startup
options, model recipe presets, then [library defaults](../../docs/guides/c_api.md#keys).
Omitted request fields retain the configured override. In native JSON requests,
`null` clears an optional inference field so the library selects its value;
`0`, `false`, empty strings and empty numeric arrays are explicit values.
Capability `defaults` and `defaults_by_mode` contain configured overrides, with
`null` for optional settings left to the library. Effective settings are returned
in generation metadata. Output formats, queue limits and listen addresses keep
server-local defaults.

<a id="web-ui"></a>

## Start the server

From the repository root, start the launcher. It finds the server executable
and uses the repository's `models/` directory:

**Windows (PowerShell)**

```powershell
.\start-server.bat
```

You can also double-click `start-server.bat` in File Explorer to start the
server. Keep the console window open while the server is running.

**Linux/macOS**

```sh
sh ./start-server.sh
```

To start the executable directly instead, run from the build directory:

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-server.exe --model-dir ..\models
```

**Linux/macOS**

```sh
./bin/sd-server --model-dir ../models
```

The Windows path above is for a Visual Studio Release build. Adjust the path
for another build configuration or an extracted release archive. After the
server starts, <http://127.0.0.1:1234/> returns a plain-text status message;
there is no built-in Web UI.

## Load a model and generate

This backend-only server has no model-selection or generation page. Use the
[API](#api) instead:

1. Download the weights and required components listed in the matching
   [model guide](../../docs/models/README.md). Put them in the category folders
   under `models/`, such as `checkpoints/`, `diffusion_models/`, `text_encoders/`,
   `tokenizers/`, and `vae/`. See [Model directories](#model-directories) below.
2. Load models at startup ([Models loaded at startup](#models-loaded-at-startup))
   or through the native API under `/sdcpp/v1/...`, choosing a recipe and the
   component files.
3. Submit generation requests to the native async API, or to the OpenAI-compatible
   (`/v1/...`) or Stable Diffusion WebUI-compatible (`/sdapi/v1/...`) endpoints.
   See the [API reference](api.md) for request fields and examples.

LoRAs are selected through the request fields documented in the API reference.
Place LoRA files under `models/loras/` or `models/Lora/`. Prompt tags such as
`<lora:name:1>` are CLI syntax and are not parsed by the native server API.

Later sections that mention the "UI" or "browser" describe an external frontend
served with `--serve-html-path`; they do not apply to the built-in status page.

## Model management

### Model directories

Repeat `--model-dir` to share multiple model directories, including an existing
SD WebUI model directory. Run these examples from the build directory:

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-server.exe --model-dir ..\models --model-dir D:\stable-diffusion-webui\models
```

**Linux/macOS**

```sh
./bin/sd-server --model-dir ../models --model-dir /data/stable-diffusion-webui/models
```

With no `--model-dir`, the server uses `./models`. Explicit roots replace that
default and are scanned in argument order. Duplicate canonical roots and file
paths are scanned once. Every root can contain a `recipes/` directory.
Directory names are matched case-insensitively on all platforms:

| Category | Directory names |
| --- | --- |
| Checkpoints | `checkpoints`, `Stable-diffusion` |
| VAE | `vae`, `VAE` |
| LoRA | `loras`, `Lora` |
| ControlNet | `controlnet`, `ControlNet` |

Other categories keep their existing directory names. Classification uses only
the containing directory: `*.vae.safetensors`, `*.vae.ckpt`, and similar names
inside `checkpoints/` or `Stable-diffusion/` remain checkpoints.

Place model files in category directories, for example:

```text
models/
  checkpoints/
  diffusion_models/
  text_encoders/
  tokenizers/
  vae/
  clip_vision/
  loras/
  recipes/
    my-flux.json
```

Subdirectories within a model category are supported. Use **Refresh files &
recipes** after adding files or changing recipes. Scanning lists file names
without loading model weights. Category membership does not guarantee
compatibility; the model loader performs the final compatibility check.
Symlinks resolving outside the model directory are rejected. Dropdown labels
omit the category prefix while retaining subdirectory names. Identical display
paths are displayed as full file paths, with source directories before the
filename instead of a hash suffix. The same applies to duplicate LoRA paths. File IDs include a
root qualifier for additional roots; clients should use the returned IDs.
Empty optional categories do not display missing-file hints.

Place Tiny AutoEncoder (TAE) files in `vae/` and select them in the **TAE**
slot. The legacy `vae_approx/` directory is also scanned for TAE selections.

LoRAs are discovered automatically in both `loras/` and `Lora/` under every
model root. Use `--lora-model-dir` to replace automatic discovery with one
explicit LoRA directory. Duplicate names remain separate selectable entries. The LoRA selectors support
searching by name or path with the same keyboard controls as model selectors.

### Recipes

A recipe describes a model's components, supported capabilities, and generation
defaults. See the [model guides](../../docs/models/README.md) for compatible
recipes and weights. Text encoder labels in the selectors identify the expected
encoder family and size.

#### Built-in and user recipes

Built-in recipes are embedded in the binary from [recipes/](recipes/), with one JSON file per recipe. Adding or
editing a built-in requires rebuilding the server. User recipes are read from
`<model-dir>/recipes/*.json`. A valid user recipe with the same ID replaces
the entire built-in recipe. Across roots, the first valid user recipe wins;
overrides are logged. Duplicate user IDs within one root are rejected; invalid
recipes are reported without removing a valid recipe from another root or the
built-in version. Slots need only `id` and
`required`; the server supplies labels and search directories. Use `label`
only when a recipe needs a more specific display name.

Built-in additions and changes must follow the [recipe policy](recipes/README.md).
Split recipes by task weights or added capabilities, not by style or acceleration
LoRAs. Quantizations, file formats and compatible layer-count variants share a
recipe. Native encoders, projectors and connectors stay in the model's recipe.
Community editing recipes require selecting the matching LoRA separately; a
recipe does not enable it automatically.

#### Generation defaults and controls

See [generation defaults and sources](recipes/defaults.md) for the selected
publisher recommendations and reference-code defaults, with step defaults capped
at 20. Recipes link to the
original publisher's Hugging Face repository where one is available. DiT
recipes default to Euler, except for models requiring a specialized sampler.

Recipes use `features` to enable specialized controls and `disabled_features`
to turn off shared defaults. Disabled features take precedence. The UI hides
unsupported fields, empty sections and irrelevant sampler/cache options, and
omits hidden values from generation requests. See the
[feature declarations](api.md#recipe-feature-declarations) for the defaults
and complete list of supported names.

#### VAE and TAE

Compatible recipes offer a TAE slot alongside VAE. When the checkpoint does not
bundle a VAE, select at least one of VAE or TAE. The Models panel shows a reminder
below the model selectors until this requirement is met. A selected TAE is used for
encoding and decoding, including when a full VAE is also selected; use complete,
model-compatible TAE weights rather than decoder-only preview weights. Qwen
Image 2.1 uses the same recipe for its full VAE and TAEQI2.1 alternatives.

### Models loaded at startup

You can pass the same model-loading arguments used by `sd-cli` to `sd-server`.
For example, run this Z-Image Turbo configuration from the build directory:

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-server.exe `
  --model-dir ..\models `
  --diffusion-model ..\models\diffusion_models\z_image_turbo_bf16.safetensors `
  --vae ..\models\vae\ae.sft `
  --llm ..\models\text_encoders\qwen_3_4b.safetensors `
  --offload-to-cpu `
  -v `
  --cfg-scale 1.0
```

**Linux/macOS**

```sh
./bin/sd-server \
  --model-dir ../models \
  --diffusion-model ../models/diffusion_models/z_image_turbo_bf16.safetensors \
  --vae ../models/vae/ae.sft \
  --llm ../models/text_encoders/qwen_3_4b.safetensors \
  --offload-to-cpu \
  -v \
  --cfg-scale 1.0
```

`--diffusion-model`, `--vae`, and `--llm` select the model components.
`--offload-to-cpu` keeps weights in RAM when possible to reduce VRAM pressure,
`-v` enables verbose logging, and `--cfg-scale 1.0` sets the default CFG scale.

API clients can use the model loaded at startup without selecting a recipe.
The UI automatically selects **Command-line model (all features)** and uses the
startup generation defaults. This runtime-only recipe exposes all controls for the supported
image/video mode. Selecting another recipe previews a model switch; selecting
the command-line entry again restores its generation defaults. The entry is
removed when the command-line context is unloaded or replaced.

#### Identity inputs

The browser currently has no PhotoMaker identity-image or PuLID identity-embedding
upload control. Prepare the inputs using the
[PhotoMaker](../../docs/models/photo_maker.md) or [PuLID](../../docs/models/pulid.md)
guide, then supply them when starting the server. From the build directory, for example:

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-server.exe --model-dir ..\models --pm-id-images-dir ..\inputs\identity
```

**Linux/macOS**

```sh
./bin/sd-server --model-dir ../models --pm-id-images-dir ../inputs/identity
```

Select the PhotoMaker recipe and weights in the browser. For PhotoMaker v2, also
pass `--pm-id-embed-path` with the prepared embedding file. For PuLID, use
`--pulid-id-embedding` instead of the PhotoMaker input argument, then select the
PuLID recipe and weights. These startup inputs are inherited by generation requests.

#### Sharded checkpoints

For a checkpoint loaded from a complete repository directory, pass its model
loading arguments when starting `sd-server`, as in the model's CLI example.
Keep the index, tokenizer, and all shards together. The browser then uses
**Command-line model (all features)** for the model loaded at startup; selecting
an individual shard in a recipe dropdown does not load the whole repository.

### Switching models

The server keeps one model context loaded. Loading or unloading is rejected
while generation requests or queued jobs are pending. A switch validates the
selection before unloading the old context, then loads the replacement.
If loading fails, the service stays online with no active model and reports
the error. Backend and memory options supplied at startup apply to subsequent
loads. Recipe defaults apply to new loads; explicitly supplied generation
request parameters take precedence.

See [the API reference](api.md#model-management) for recipe fields, supported
slots, management endpoints, and compatibility behavior.

### Limitations

Tiny SD/Tiny SD2 and Diffusers LCM directories must first be
converted into single-file checkpoints for the file catalog. LCM checkpoints
and LoRAs use the matching SD family recipe; select LCM sampling and adjust
steps/CFG for the chosen weights.
Metadata inspection, conversion, standalone upscaling and ADetailer are CLI
workflows rather than model-loading recipes. ESRGAN and LTX latent upscalers
are selected through generation upscaling settings with `--hires-upscalers-dir`;
textual inversion uses the startup `--embd-dir` option.

SVD is not available through the current server pipeline.

## Server configuration

Reference images, masks, audio, first/last video frames, and generation options
such as upscaling are supplied with generation requests. Hardware and memory
options are startup arguments and apply to subsequent model loads.

### Launcher settings

Both source checkouts and binary release archives include `start-server.bat`
and `start-server.sh`. The launchers search for the server beside the script,
then in `bin/Release/`, `bin/`, `build/bin/Release/`, and `build/bin/`, in that order.
They print the resolved command and forward the server's exit code.

#### Custom executable

For a different build directory, set `SD_SERVER_PATH` to the executable path.
Run from the repository root:

**Windows (PowerShell)**

```powershell
$env:SD_SERVER_PATH = ".\build_vulkan\bin\Release\sd-server.exe"
.\start-server.bat --listen-port 8080
```

**Linux/macOS**

```sh
SD_SERVER_PATH="./build_vulkan/bin/sd-server" sh ./start-server.sh --listen-port 8080
```

`SD_SERVER_PATH` inherits the environment value by default. Leave it empty for
automatic discovery. An explicit invalid path does not fall back to another build.

#### Local configuration

Keep personal settings in a file beside the launchers: `start-server.local.bat`
on Windows or `start-server.local.sh` on Linux/macOS. These files are ignored by
Git and excluded from release archives. They are loaded automatically and can
set `SD_SERVER_PATH`, `MODEL_DIR`, and `COMMANDLINE_ARGS`.

**Windows (`start-server.local.bat`)**

```bat
@echo off
set "MODEL_DIR=D:\stable-diffusion-webui\models"
set "COMMANDLINE_ARGS=--listen-port 8080 --offload-to-cpu"
```

**Linux/macOS (`start-server.local.sh`)**

```sh
MODEL_DIR="/data/stable-diffusion-webui/models"
COMMANDLINE_ARGS="--listen-port 8080 --offload-to-cpu"
```

Settings are applied in this order: launcher defaults, local configuration,
then invocation arguments. Local files are shell scripts and use the
corresponding shell's syntax.

The launchers preserve the current working directory, so the default
`MODEL_DIR="./models"` is relative to where you start the launcher. To change
the first model root, set `MODEL_DIR` in the local configuration. Repeated
`--model-dir` arguments add search roots; other single-value options can override
the defaults.

In the shell script, `COMMANDLINE_ARGS` is a whitespace-separated list. Pass
values containing spaces as quoted invocation arguments instead. `MODEL_DIR`
itself supports spaces in both scripts. A leading `--version` bypasses the
configured defaults.

### Network and job queue

| Option | Default | Purpose |
| --- | --- | --- |
| `--listen-ip <ip>` | `127.0.0.1` | Server listen address |
| `--listen-port <port>` | `1234` | Server listen port |
| `--max-pending-jobs <count>` | `64` | Maximum async jobs queued or generating |
| `--completed-ttl-seconds <seconds>` | `600` | Retention time for completed async jobs |

### Logging

Logging defaults to `info`. Use `--log-level <level>` to select `debug`, `verbose`,
`info`, `warn`, or `error` (from most to least detailed). Each level includes
messages at that level and all less detailed levels. `-v` and `--verbose` are
equivalent to `--log-level verbose`. If repeated, the last logging option wins.

### Command-line help

Run from the build directory for the complete list of server and generation options:

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-server.exe --help
```

**Linux/macOS**

```sh
./bin/sd-server --help
```

## API

The server exposes these API families at the same host and port as the optional external Web UI:

| API | Path prefix |
| --- | --- |
| Native async API | `/sdcpp/v1/...` |
| OpenAI-compatible API | `/v1/...` |
| Stable Diffusion WebUI-compatible API | `/sdapi/v1/...` |

See the [API reference](api.md) for endpoints, request fields, and examples.

## Using an external frontend

The root endpoint (`/`) responds with a plain-text status message. Use
`--serve-html-path` to serve a custom `index.html` instead, for example a local
copy of [sdcpp-webui](https://github.com/leejet/sdcpp-webui). Place it at the
repository root and run from the build directory:

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-server.exe --model-dir ..\models --serve-html-path ..\index.html
```

**Linux/macOS**

```sh
./bin/sd-server --model-dir ../models --serve-html-path ../index.html
```

This lets you develop or test a frontend without rebuilding the server binary.
