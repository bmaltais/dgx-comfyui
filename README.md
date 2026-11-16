# spark-comfyui-uv

ComfyUI on the NVIDIA DGX Spark (GB10, sm_121), installed natively with [uv](https://docs.astral.sh/uv/)
into **one folder**. One script, safe to re-run.

It is a native port of [bjarkebolding/spark-comfyui](https://github.com/bjarkebolding/spark-comfyui)
(MIT), which does the same inside a container. Same stack: cu130 PyTorch, SageAttention compiled for
sm_121, GPU onnxruntime, the unified-memory patch.

## Install on a DGX Spark

On the Spark (SSH in, or open a terminal on it):

```bash
curl -fsSL https://raw.githubusercontent.com/bmaltais/spark-comfyui-uv/main/install.sh | bash
```

It creates a `ComfyUI` folder in the directory you run it from (so `cd` to wherever you want it first),
clones this repo into it and runs `./spark-comfyui.sh install`. First run takes about 15 minutes
(PyTorch download, then a ~5 minute SageAttention compile) and ends with a one-line verdict. To name
the folder yourself: `... | bash -s -- /path/to/folder`.

Then start it:

```bash
cd ComfyUI
./spark-comfyui.sh run          # UI at http://<spark-ip>:8188
```

Stop ComfyUI with Ctrl+C. To reach the UI from your laptop, open `http://<spark-ip>:8188`. If you set
`BIND_ADDR = 127.0.0.1`, tunnel instead: `ssh -N -L 8188:127.0.0.1:8188 you@spark`, then open
`http://localhost:8188`.

### Where your models and files go

Everything of yours is in the `data/` folder inside the install folder, so it survives `update`:

| Put this | Here (inside the install folder) |
|---|---|
| Full checkpoints (`.safetensors`, SD/SDXL/Flux single-file) | `data/models/checkpoints/` |
| Diffusion models / UNets (Flux, Qwen-Image, Wan, LTX) | `data/models/diffusion_models/` |
| Text encoders (T5, Qwen, CLIP-L) | `data/models/text_encoders/` |
| VAEs | `data/models/vae/` |
| LoRAs | `data/models/loras/` |
| ControlNets, upscalers, CLIP vision, embeddings | `data/models/controlnet/`, `upscale_models/`, `clip_vision/`, `embeddings/` |
| Custom nodes | install them from the **Manager** button in the UI; they land in `data/custom_nodes/` |
| Saved workflows | `data/user/default/workflows/` |
| Images you generate | `data/output/` (inputs you upload: `data/input/`) |

Copy models in with `scp` or `rsync` from your laptop, or download them straight on the Spark with
`curl -L -o <file> <url>` or `hf download`. Models are picked up without a restart: press `r` in the
UI to refresh the lists. If your models already live elsewhere (another disk, a NAS), make any of these
folders a symlink to it; `install` leaves a symlinked folder alone.

**Update later:** run the same curl line again (it refreshes this tool and fixes whatever differs), or
`./spark-comfyui.sh update` to also move ComfyUI to the newest upstream.

**Check on it:** `./spark-comfyui.sh doctor` reports every part and changes nothing.

**Running it for days:** start it inside `tmux` or `screen` so it survives you closing the SSH session.
(There is no systemd service yet.)

**Prefer to read before running?** The installer is [`install.sh`](install.sh), about 30 lines. Or clone
and run it yourself:

```bash
git clone https://github.com/bmaltais/spark-comfyui-uv.git ComfyUI && cd ComfyUI
./spark-comfyui.sh install
```

### Requirements

A DGX Spark on DGX OS (r580 driver) with the CUDA 13.0 toolkit at `/usr/local/cuda`, plus `git`,
`curl`, `g++`, `libgl1` and `libglib2.0-0t64`, which DGX OS ships. `install` checks all of it first
and prints the exact `apt` line for anything missing. It never runs `sudo`.

### If something fails

| Message | Fix |
|---|---|
| `driver ... is older than r580` | Update DGX OS, reboot. |
| `no CUDA toolkit (nvcc)` | `sudo apt install cuda-toolkit-13-0`, or set `CUDA_HOME` in `spark-comfyui.conf`. |
| `missing libGL...` | The `sudo apt install ...` line it printed. |
| `another install, update or run holds the lock` | ComfyUI is running; stop it first. |
| Anything else | Re-run `install` (it resumes), then `doctor`, and open an issue with the output. |

## Commands

| Command | What it does |
|---|---|
| `install` | Converges to the wanted state. Each part (uv, Python, torch, ComfyUI, requirements, SageAttention, onnxruntime, patch, Manager config, data dirs) is checked against what is really installed, and only a part that differs is fixed. A second run says "Nothing to do". |
| `update` | Moves ComfyUI to the newest upstream (`COMFY_REF`), re-applies the patch, then runs `install`. A failed move rolls back to the previous commit; it is kept in `.state/comfy.previous`. |
| `run [args]` | Verifies torch, onnxruntime and a live SageAttention kernel on the GPU, then launches ComfyUI. Extra args go to `main.py`. Refuses to launch with a broken attention kernel rather than quietly running slow. |
| `doctor` | Read-only health report. Changes nothing. |

An `install`, `update` and `run` hold a lock, so you can't change the install under a running ComfyUI.

## What lives where

Everything is under the install folder (`ComfyUI/` if you used the curl line; below it is called
`<install>/`). Nothing is written to `~/.cache`, `~/.local` or the system Python.

| Path | Contents |
|---|---|
| `<install>/spark-comfyui.sh`, `install.sh`, `lib/` | this tool (a git checkout of this repo) |
| `<install>/data/` | **your content**, listed above. Passed to ComfyUI with `--base-directory`, so updates never touch it |
| `<install>/ComfyUI/` | ComfyUI's own code (a separate git checkout, managed for you; don't put files here) |
| `<install>/bin/` | uv, only if the system has none |
| `<install>/.python/`, `.venv/` | uv-managed Python 3.12 and the venv |
| `<install>/.cache/` | uv, pip, Hugging Face, torch, Triton and CUDA kernel caches |
| `<install>/build/` | SageAttention source |
| `<install>/.state/` | lock, torch constraints, build stamps |

Hugging Face downloads made by custom nodes go to `.cache/huggingface`. To move or remove the install,
move or delete the folder. (`git` credentials and shell config in `$HOME` are not touched either way.)

## Configuration

Put `KEY = VALUE` lines in `spark-comfyui.conf` next to the script, or set them in the environment
(the environment wins). Unknown keys are warned about.

| Key | Default | Meaning |
|---|---|---|
| `COMFY_REF` | `master` | Branch, tag or 40-char sha. A branch moves on `update`; a sha or tag is enforced by `install`. |
| `BIND_ADDR` | `0.0.0.0` | Interface to listen on. ComfyUI has **no login**: use `127.0.0.1` plus an SSH tunnel unless the LAN is trusted. |
| `PORT` | `8188` | |
| `SPARK_ATTENTION` | `sage` | `sage`, `sdpa`, or `ck`. Anything but `sage` skips the SageAttention build. |
| `SPARK_BF16`, `SPARK_BF16_VAE` | `1` | bf16 fast path. Set `SPARK_BF16_VAE = 0` for LTX-2.3 audio workflows. |
| `SPARK_STATIC_VRAM` | `0` | `1` disables DynamicVRAM. |
| `SPARK_RESERVE_VRAM` | unset | GB of unified memory to keep free. |
| `TORCH_VERSION`, `TORCHVISION_VERSION`, `TORCHAUDIO_VERSION` | `2.14.0`, `0.29.0`, `2.11.0` | Pins. Edit them and run `install`; the venv follows. |
| `SAGE_REF`, `SAGE_ARCH`, `ORT_WHEEL_URL`, `CUDA_HOME`, `TORCH_INDEX`, `COMFY_REPO`, `PYTHON_VERSION` | see script | Build pins and paths. |

## How a re-run stays safe

- Each part has a real check (an import, an installed version, a git sha, a live GPU kernel run), not
  a "did this step run before" flag, so a half-finished or broken install is repaired by running
  `install` again.
- Torch is held at its pin through `UV_CONSTRAINT` / `PIP_CONSTRAINT`, so a custom node's
  requirements can't replace it. `run` re-verifies torch and onnxruntime on every start and repairs
  them if a node got through anyway.
- Clones and the Manager config are written to a temporary name and renamed into place.
- `update` removes the memory patch from `model_management.py` before moving the checkout and
  re-applies it after, so git never sees a dirty tree. If the patch no longer fits the new ComfyUI,
  it rolls back and says so.

## The trade-off against the container

Upstream's container runs custom nodes as non-root with all capabilities dropped and no view of your
home. **A native install gives that up.** Every node you install through ComfyUI-Manager runs as your
user with access to your home directory. Install nodes you'd be willing to run as a script.

## Not ported

Left out on purpose; the upstream script has them if you want them: `status --watch`, `tune`
(swap off, persistence mode, clock cap), `backup` / `restore`, `prune`, `recipe`, the patch list
and the node list. Rollback is the git checkout (`.state/comfy.previous`), not an image tag.

## Status

Written against upstream at `33e521b`. Tested on a real DGX Spark (GB10, driver 580.178, CUDA 13.0, 2026-11-16): fresh install including the sm_121 SageAttention build and the aarch64 onnxruntime wheel, a no-change re-run, `doctor`, and `run` serving the UI with SageAttention active. Not yet exercised there: `update` against a newer upstream, and generating an image.

MIT. Derived from spark-comfyui, copyright (c) 2026 Bjarke Bolding; see `LICENSE`.
