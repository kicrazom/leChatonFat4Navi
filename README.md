# leChatonFat4Navi

**A local coding agent on AMD GPUs.** llama.cpp serves a GGUF model, SearXNG gives it web search,
MCP servers give it tools, and Opencode drives it, all on your own machine with no external API.

Fork of [grafiszti/leChatonFat](https://github.com/grafiszti/leChatonFat) (MIT, © Mateusz Choiński).
The original targets NVIDIA; this fork adds **ROCm and Vulkan** backends and leaves the NVIDIA path unchanged.
Built and measured on the [NaviMed-UMB](https://kicrazom.github.io/navimed-umb/lechat.html) workstation (see its Harness page),
2x Radeon AI PRO R9700.

| | |
|---|---|
| Backends | llama.cpp: NVIDIA (CUDA), AMD ROCm, AMD Vulkan, CPU; SGLang: AMD BF16; vLLM: host venv, BF16/AWQ |
| Multi-GPU | layer split across discrete cards; iGPU kept out |
| Default model | Qwen3.6-35B-A3B-UD-Q4_K_XL (22.4 GB GGUF) |
| Measured | 2x R9700, generation: Qwen3.6-35B-A3B 76.9 / 88.4 tok/s, gpt-oss-20b 124.8 / 130.6 tok/s (ROCm / Vulkan) |
| Agent | Opencode + MCP: fetch, SearXNG search, browser; tool calls tested with Bielik-11B-v3.1 (SGLang, vLLM) |

### Quick start (AMD)

```
make copy_env        # edit .env: GPU_BACKEND, AMD_GPU0/1, TENSOR_SPLIT, MODELS_DIR
make check_gpu       # which renderD is which card
make up
make smoke_test
make generate_opencode_config
```

## AMD GPUs (ROCm / Vulkan)

Set in `.env`:

```
GPU_BACKEND=rocm            # or vulkan; auto = NVIDIA if present, else ROCm if /dev/kfd exists
AMD_GPU0=/dev/dri/renderD128
AMD_GPU1=/dev/dri/renderD129
TENSOR_SPLIT=50,50          # two equal cards
MODELS_DIR=/path/to/gguf    # host dir mounted at /models
```

List **only discrete GPUs** in `AMD_GPU0/1`. Exposing all of `/dev/dri` lets an iGPU join the split.
Find the render nodes with `make check_gpu` (prints the PCI ID of each `renderD*`).

Your user must be in the `video` and `render` groups; `make up` passes their GIDs to the container.

### Measured on 2x Radeon AI PRO R9700 (gfx1201, 32 GB each)

Qwen3.6-35B-A3B-UD-Q4_K_XL, default `.env` (ctx 98304, KV cache q4_0, flash attention),
600-token prompt, 256 generated tokens, `cache_prompt=false`, 3 runs, 2026-11-13.

| Backend | Image | Prompt processing | Generation |
|---|---|---|---|
| ROCm 7.2.1 | `llama.cpp:server-rocm` (30.8 GB) | ~2,650 tok/s | 76.9 tok/s |
| Vulkan (RADV) | `llama.cpp:server-vulkan` (1.2 GB) | ~1,720 tok/s | 88.4 tok/s |

gpt-oss-20b (`ggml-org/gpt-oss-20b-GGUF`, native MXFP4, 12.1 GB), same settings,
`CHAT_TEMPLATE=` empty (the GGUF's own harmony template), `ignore_eos` to force 256 tokens:

| Backend | Prompt processing | Generation |
|---|---|---|
| ROCm 7.2.1 | ~4,300 tok/s | 124.8 tok/s |
| Vulkan (RADV) | ~2,580 tok/s | 130.6 tok/s |

As of 2026-04-26 this model did not start on R9700 with vLLM 0.19 + ROCm 7.2 (`triton_kernels`
lacked `routing_from_bitmatrix`; not re-tested since). llama.cpp does not use that path.

ROCm reads long prompts faster; Vulkan generates faster and its image is 25x smaller.
For an agent that re-reads a large context each turn, prompt processing usually dominates.

### SGLang backend (`GPU_BACKEND=sglang`)

Serves BF16 safetensors with SGLang instead of a GGUF with llama.cpp. Opencode is unchanged
(same port, `MODEL_NAME` becomes the served model id). Set in `.env`:

```
GPU_BACKEND=sglang
SGLANG_IMAGE=sglang-rdna4:v0.5.18     # an SGLang build for your GPU
SGLANG_MODELS_DIR=/path/to/hf-models
SGLANG_MODEL=bielik-11b-v31-instruct   # folder under SGLANG_MODELS_DIR
MODEL_NAME=bielik-11b-v3.1
SGLANG_TP=2
SGLANG_TOOL_PARSER=hermes              # needed for agent tool calls
```

gfx1201 needs `--attention-backend triton --disable-cuda-graph --disable-custom-all-reduce`
(set in `docker-compose.sglang.yml`). There is no public SGLang image for RDNA4; the one used
here was built locally from SGLang v0.5.18 with RDNA4 patches. Its `~/.cache/sglang` is
root-owned while the server runs as `sglang`, so JIT kernels fail with `EACCES`; the compose
file points `HOME` at `/tmp`. AWQ does not load on gfx1201 in this build (`gptq_marlin_repack`
is missing), so SGLang here means BF16.

Measured (Bielik-11B-v3.0-Instruct BF16, 600-token prompt, 256 tokens, radix cache flushed
before each run, 3 runs): TP=1 186 ms TTFT, 20.1 tok/s; TP=2 150 ms TTFT, 33.4 tok/s.

### vLLM backend (`GPU_BACKEND=vllm`)

Runs `vllm serve` from a host virtualenv as a `systemd --user` unit (`scripts/vllm-host.sh`),
because on gfx1201 the validated stack is a host install (vLLM 0.19.0+rocm721) and the public
vLLM ROCm images lag it. Serves BF16 or AWQ safetensors. Keys: `VLLM_BIN`, `VLLM_MODELS_DIR`,
`VLLM_MODEL`, `VLLM_TP`, `VLLM_CTX`, `VLLM_GPU_UTIL`, `VLLM_TOOL_PARSER`, `VLLM_VISIBLE_DEVICES`.
The script applies the gfx1201 environment floor (`VLLM_ROCM_USE_AITER=0`,
`AMD_SERIALIZE_KERNEL=1`, `HIP_LAUNCH_BLOCKING=1`, iGPU hidden) and `--enforce-eager`.

Measured (Bielik-11B-v3.0-Instruct BF16, TP=2, same prompt as above): 21.5 tok/s generation.
TTFT 205 ms on the first run, 80 ms after; the prefix cache was not reset between runs
(`/reset_prefix_cache` returned an error), so only the first TTFT is cold. The serialising
debug variables of the env floor slow decoding; SGLang above runs without them, so the two
numbers are not an engine-to-engine comparison.

### Tool calls with Bielik-11B-v3.1-Instruct

Bielik v3.1 writes tool calls as `<tool_call>{"name": ..., "arguments": ...}</tool_call>` with
extra whitespace after the opening tag. SpeakLeash's own parser is a vLLM plugin that needs
vLLM >= 0.24. On the backends here the built-in `hermes` parser works instead (tested
2026-11-13 with one call and with two parallel calls): `SGLANG_TOOL_PARSER=hermes` on SGLang
0.5.18, `VLLM_TOOL_PARSER=hermes` on vLLM 0.19. The `qwen25` parser does not recognise the
calls. Speed on 2x R9700, BF16, TP=2: 33.5 tok/s (SGLang), 22.4 tok/s (vLLM with the gfx1201
env floor); count tokens from `usage`, because after EOS v3.1 emits tokens with no text.

### Model API is local-only by default

`llama-server` runs without an API key. Upstream published it on `0.0.0.0:8001`, so anyone on the
LAN (or VPN) could use the model. This fork binds it to `127.0.0.1` (`BIND_ADDRESS` in `.env`).
For remote access set `BIND_ADDRESS` and `LLAMA_API_KEY` together, and give the client the same key
(in `opencode.json`: `"apiKey": "{env:LLAMA_API_KEY}"` under the provider `options`).

### Known issue fixed here: `TEMP` crashes ROCm

With the sampling temperature in the environment as `TEMP=0.6`, `llama-server` on ROCm segfaults in
`libamdhip64` right after "loading model". Bisecting the container environment showed `TEMP` alone
reproduces it; the HIP runtime likely reads `TEMP` as a temp-dir path. `run.sh` now runs
`unset TEMP` after the value is copied into `--temp`. Upstream runs with this variable on NVIDIA;
the NVIDIA path was not tested here.

### Other changes vs upstream

- `HF_REPO` default fixed to `unsloth/Qwen3.6-35B-A3B-GGUF` (the `-UD-GGUF` repo does not exist).
- `MODELS_DIR` makes the model directory configurable (default `./models`).

---


![img](img.png)

A Docker-based setup for running llama.cpp server with GPU acceleration, SearXNG web search, and MCP tool servers — optimized for use as a local LLM provider for the Opencode coding assistant.

## Features
- **GPU Acceleration**: Multi-GPU layer splitting with configurable tensor split
- **Docker Compose**: Easy setup and management with auto-detection (GPU/CPU)
- **Memory-fit Offload**: `--fit` auto-adjusts GPU offload and context to available VRAM
- **Configurable Inference**: Sampling parameters, penalties, reasoning mode, and more via `.env`
- **Jinja Chat Template**: Bundled chat template for improved model formatting
- **Auto-generated Config**: Generate `opencode.json` from `.env` via script
- **SearXNG Integration**: Self-hosted privacy-focused web search service
- **MCP Servers**: Browser automation and SearXNG MCP tools for Opencode
- **Simple Management**: Makefile commands for common tasks

## Prerequisites
- Docker and Docker Compose installed
- NVIDIA GPU with drivers installed, or AMD GPU with ROCm/amdgpu (optional, falls back to CPU)
- NVIDIA Container Toolkit (for GPU support)

## Quick Start

```bash
# Copy environment template
make copy_env

# Build and start the container (auto-detects GPU/CPU)
make up

# Validate the server is running
make validate
```

## Configuration

### Model
Use `make copy_env` to create `.env` file with all settings to configure:

| Variable           | Default                           | Description                                     |
|--------------------|-----------------------------------|-------------------------------------------------|
| `MODEL_NAME`       | `Qwen3.6-35B-A3B-UD-Q4_K_XL`      | GGUF model filename (without `.gguf` extension) |
| `MODEL_NAME_ALIAS` | `qwen3.6-35b`                     | Alias for the model (used in opencode.json)     |
| `HF_REPO`          | `unsloth/Qwen3.6-35B-A3B-UD-GGUF` | Hugging Face repository for model download      |
| `CTX_SIZE`         | `98304`                           | Context window size in tokens                   |
| `FLASH_ATTN`       | `on`                              | Enable flash attention                          |
| `THREADS`          | `5`                               | CPU threads for inference                       |
| `BATCH_SIZE`       | `256`                             | Prompt processing batch size                    |
| `CACHE_TYPE_K`     | `q4_0`                            | KV cache type for K tensor                      |
| `CACHE_TYPE_V`     | `q4_0`                            | KV cache type for V tensor                      |
| `NO_MMAP`          | `false`                           | Disable memory-mapped model loading             |
| `PARALLEL`         | `1`                               | Context parallelism                             |
| `CACHE_RAM`        | `4096`                            | KV cache memory budget in MiB                   |

### GPU Offload
| Variable       | Default | Description                                                        |
|----------------|---------|--------------------------------------------------------------------|
| `GPU_LAYERS`   | `all`   | Layers to offload: `all`, `auto`, or a number                      |
| `SPLIT_MODE`   | `layer` | How to split across GPUs: `layer`, `row`, `tensor`, `none`        |
| `TENSOR_SPLIT` | `40,60` | Fraction per GPU (comma-separated, e.g. `40,60`)                   |
| `MAIN_GPU`     | *(empty)* | Primary GPU index for intermediate results/KV                    |

### Memory Fit
| Variable      | Default | Description                                                    |
|---------------|---------|----------------------------------------------------------------|
| `FIT`         | `on`    | Auto-adjust unset offload/context args to fit device memory    |
| `FIT_TARGET`  | `256`   | Target free-memory margin per device in MiB                    |
| `FIT_CTX`     | `98304` | Minimum context size the fit pass may choose                   |
| `UBATCH_SIZE` | `256`   | Physical batch size for prompt processing                      |
| `POLL_BATCH`  | `0`     | Poll batch size                                                |

### Sampling & Penalties
| Variable           | Default | Description                   |
|--------------------|---------|-------------------------------|
| `TEMP`             | `0.6`   | Sampling temperature          |
| `TOP_P`            | `0.95`  | Top-p nucleus sampling        |
| `TOP_K`            | `20`    | Top-k sampling                |
| `MIN_P`            | `0.0`   | Minimum probability threshold |
| `PRESENCE_PENALTY` | `0.0`   | Presence penalty              |
| `REPEAT_PENALTY`   | `1.0`   | Repetition penalty            |

### Features
| Variable        | Default                         | Description                                                          |
|-----------------|---------------------------------|----------------------------------------------------------------------|
| `REASONING`     | `on`                            | Enable reasoning mode (`--reasoning on`)                             |
| `CHAT_TEMPLATE` | `/app/qwen_chat_template.jinja` | Path to Jinja chat template file (empty = model's built-in template) |

### Server
| Variable           | Default   | Description                    |
|--------------------|-----------|--------------------------------|
| `HOST`             | `0.0.0.0` | Bind address                   |
| `PORT_INTERNAL`    | `8080`    | Port inside Docker container   |
| `PORT_EXTERNAL`    | `8001`    | Port exposed to host           |
| `FORCE_CPU`        | `off`     | Force CPU runtime (on/off)     |
| `SEARXNG_SECRET`   | *(empty)* | Secret key for SearXNG         |

Adjust these in `.env` after copying from the template.

### Auto-generated opencode.json

Run `make generate_opencode_config` to generate `opencode.json` from `templates/opencode.json.template` using values from `.env`. This updates the provider URL, model name, model alias, and MCP configurations automatically.

The generated config includes MCP tool integration:
- **fetch**: Web fetching via the `mcp/fetch` Docker image
- **searxng-search**: Self-hosted web search via SearXNG
- **browser**: Browser automation via a custom MCP server

## Makefile Commands

| Command                         | Description                                    |
|---------------------------------|------------------------------------------------|
| `make up`                       | Build and start (auto-detects GPU/CPU)         |
| `make down`                     | Stop and remove the container                  |
| `make validate`                 | Check if the server is healthy                 |
| `make install_opencode`         | Install Opencode CLI                           |
| `make check_gpu`                | Verify GPU and NVIDIA runtime setup            |
| `make check_docker`             | Verify Docker and Docker Compose               |
| `make smoke_test`               | Run a sample chat completion request           |
| `make generate_opencode_config` | Generate `opencode.json` from `.env`           |
| `make copy_env`                 | Copy `templates/.env.template` to `.env`       |
| `make download_qwen`            | Download the default Qwen model to `./models/` |
| `make install_huggingface_cli`  | Install Hugging Face CLI                       |
| `make build-mcp-browser`        | Build the MCP browser server image             |
| `make build-searxng-mcp`        | Build the MCP SearXNG server image             |
| `make build-mcp`                | Build all MCP server images                    |

## Docker Compose Services

The project runs two services via Docker Compose:

- **llama**: The llama.cpp LLM server (main inference)
- **searxng**: Self-hosted SearXNG web search engine (`http://localhost:8081`)

Both services include healthchecks and persistent data volumes.

## Troubleshooting

### GPU Not Detected
If GPU acceleration isn't working:

1. Verify NVIDIA drivers are installed:
   ```bash
   nvidia-smi
   ```

2. Check NVIDIA Container Toolkit installation:
   ```bash
   make check_gpu
   ```

3. Restart Docker after installing the toolkit:
   ```bash
   sudo systemctl restart docker
   ```

4. The `make up` command auto-detects GPU availability; if neither `nvidia-smi` nor the NVIDIA Docker runtime is found, it falls back to CPU mode.

### Port Already in Use
If the external port is already in use, modify `PORT_EXTERNAL` in `.env`:

```bash
PORT_EXTERNAL=8002
```

### Out of Memory
If you encounter OOM errors:
- Use a smaller model
- Reduce `THREADS` or `CTX_SIZE`
- Set `FORCE_CPU=on` to disable GPU offloading
- Reduce `CACHE_RAM` or use a higher cache type (`q4_0` → `f16`)

## Data Persistence
Model data is stored in `./models/` (bind-mounted), KV cache is stored in the `llama_cache` Docker volume, and SearXNG data in the `searxng_data` volume — all persist across container restarts.

## License

This project is licensed under the [MIT License](LICENSE).

Copyright (c) 2026 Retro Crest Mateusz Choiński

Note: Any tools, models, or third-party software used in this project are subject to their own respective licenses.
