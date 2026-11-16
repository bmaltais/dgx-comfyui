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

It creates a `ComfyUI` folder in the directory you run it from, so `cd` to wherever you want it first.
**That choice also picks the disk**: the models live inside it (`ComfyUI/data/models`), so run it from a
big disk (for example `cd /mnt/bigdisk`) if you plan to pull tens of GB of models. It
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

**Which ComfyUI you get:** the newest tagged **release** (e.g. `v0.45.4`), not `master`, which runs ahead
of releases. Pin something else with `COMFY_REF` (see Configuration).

**Update later:** `./spark-comfyui.sh update` moves ComfyUI to the newest release, upgrades its Python
dependencies to what that release allows (frontend package, Manager and so on; torch stays at its pin),
re-applies the patch and re-checks everything. It prints the old and new version. Running the curl line
again only refreshes this tool and repairs the install; it never moves ComfyUI to a newer release.
`doctor` tells you when a newer release exists.

**Check on it:** `./spark-comfyui.sh doctor` reports every part and changes nothing.

**Running it for days:** `./spark-comfyui.sh service` runs ComfyUI in the background as a user systemd
service: it restarts after a crash and starts at boot. `stop` stops it, `logs -f` follows it, and
`service --disable` removes it. `install` and `update` stop it while they work and start it again after.

**Prefer to read before running?** The installer is [`install.sh`](install.sh), about 30 lines. Or clone
and run it yourself:

```bash
git clone https://github.com/bmaltais/spark-comfyui-uv.git ComfyUI && cd ComfyUI
./spark-comfyui.sh install
```

### Requirements

A DGX Spark on DGX OS (r580 driver) with the CUDA 13.0 toolkit at `/usr/local/cuda`, plus `git`,
`curl`, `g++`, `libgl1` and `libglib2.0-0t64`, which DGX OS ships. `install` checks all of it first
and prints the exact `apt` line for anything missing. It never runs `sudo` itself (only `tune`, which you run on purpose, does).

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
| `update` | Moves ComfyUI to the newest release (`COMFY_REF`), upgrades its dependencies, re-applies the patch, then runs `install`. A failed move rolls back to the previous commit, which is kept in `.state/comfy.previous`. Also notes when the upstream `spark-comfyui` project has GPU-stack changes to port (`scripts/check-upstream.sh`). |
| `run [args]` | Verifies torch, onnxruntime and a live SageAttention kernel on the GPU, then launches ComfyUI. Extra args go to `main.py`. Refuses to launch with a broken attention kernel rather than quietly running slow. |
| `doctor` | Read-only health report. Changes nothing. |
| `status [--watch [SEC]]` | One-page glance: running or not, URL, versions, newer release available, GPU, unified memory, disk, queue. `--watch` is a live dashboard with sparklines (temp, power, clock, load, memory, queue). Every sample goes to `.state/status.log`, the trail to read after a silent hard reboot. |
| `service [--disable]` | Background service (user systemd unit written to `~/.config/systemd/user/`, the one thing outside the folder). Also enables lingering so it survives logout and starts at boot; if that needs `sudo` it tells you the command. |
| `stop` | Stops the service, or a foreground ComfyUI. |
| `logs [-f]` | The service's log (journald). |
| `backup [--with-output] [FILE]` | A small `.tgz` in `backups/`: workflows and settings (`data/user`), `data/input`, your `spark-comfyui.conf`, the custom-node set (git URL and commit) and a list of your models. Models are listed, never archived. Safe while running. |
| `restore FILE [--force]` | Installs if needed, puts the content back (existing files kept unless `--force`), re-clones custom nodes at their commits and lists models that are missing. |
| `tune [--clock-cap MHZ] [--persist]` <br> `tune --revert` | Host stability for the Spark, the one command that uses `sudo` and changes the machine outside this folder: swap off (on unified memory a workload near the limit thrashes swap and freezes the box instead of getting a clean out-of-memory kill), `vm.swappiness=10`, driver persistence mode, and an optional clock cap (`--clock-cap 2100` stops the silent hard reboots some units get when the GPU jumps from idle to ~85 W). `--persist` adds a boot unit (`comfyui-tune.service`) so it survives reboots. `--revert` puts everything back: swap on, clocks unlocked, persistence off, the unit and the sysctl line removed. Each step is skipped if already in place, so a re-run asks for no password it doesn't need. `doctor` warns while swap is on. |
| `recipe list\|show\|check\|install\|capture` | Everything a workflow needs, as data: its custom nodes and every model with destination, size and sha256. `install` downloads what's missing (resumable, hash-verified), installs the nodes through Manager's CLI and drops the workflow into `data/user/default/workflows`. `capture NAME` writes a recipe from a workflow that already runs here. Gated downloads read `HF_TOKEN` / `CIVITAI_TOKEN`. See `recipes/README.md`. |

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
| `COMFY_REF` | `latest` | `latest` = newest `vX.Y.Z` release tag, resolved at install and on `update`. Or a branch (`master`: moves on `update`), a tag or a 40-char sha (pinned: `install` enforces it). |
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

## Troubleshooting

Start with `./spark-comfyui.sh doctor`; every failure names its fix.

- **A custom node will not load:** read the ComfyUI log (`logs`, or the terminal running `run`). Manager installs a node's requirements with uv; torch is held at its pin, so a node demanding a different torch fails to install rather than breaking your GPU stack.
- **CPU onnxruntime sneaked in** (a node's requirements pulled it): `run` and `install` detect it and put the GPU wheel back.
- **SageAttention check fails at `run`:** it refuses to start rather than run slow. Run `./spark-comfyui.sh install` (rebuilds it, about 5 minutes), or set `SPARK_ATTENTION = sdpa`.
- **LTX-2.3 audio workflows fail with `Input type (float) and bias type (c10::BFloat16)`:** set `SPARK_BF16_VAE = 0` in `spark-comfyui.conf`. The audio VAE can't run under `--bf16-vae`; the unet and text-encoder speedups are kept.
- **Silent hard reboot during video generation:** leave `status --watch` running, reproduce, then read the last lines of `.state/status.log`. A power spike right before it dies points at overcurrent; the fix is `./spark-comfyui.sh tune --clock-cap 2100 --persist`.
- **The machine freezes near the memory limit:** swap thrash on unified memory. Run `./spark-comfyui.sh tune` (swap off) and consider `SPARK_RESERVE_VRAM`; `doctor` warns while swap is on.
- **The disk fills up:** models are the big part. `du -sh data/models/*`; the build leftovers in `build/` and `.cache/` can be deleted and are rebuilt on demand.
- **The UI is reachable by anyone on the LAN:** that is the default. Set `BIND_ADDR = 127.0.0.1` and use an SSH tunnel.

## Security notes

- **No container confinement.** Custom nodes run as you (see the next section). Everything else is unprivileged except `tune`, which you run on purpose and which uses `sudo` to change swap, GPU settings, `/etc/sysctl.conf` and, with `--persist`, `/etc/systemd/system/comfyui-tune.service`; `tune --revert` undoes all of it. The optional background service is a file in `~/.config/systemd/user/`.
- **ComfyUI has no authentication.** Manager runs in `personal_cloud` mode, fine on a trusted LAN. Do not expose the port to the internet.
- **What gets downloaded and run at install time:** uv (from astral.sh, only if absent), Python 3.12 (from uv), PyTorch (download.pytorch.org), the SageAttention source pinned to one commit, and the onnxruntime wheel pinned by sha256. `recipe install` verifies every model's sha256 before it moves into place.
- **Update the tool carefully:** `install.sh` is short and readable before you pipe it to `bash`; the repo is MIT.

## The trade-off against the container

Upstream's container runs custom nodes as non-root with all capabilities dropped and no view of your
home. **A native install gives that up.** Every node you install through ComfyUI-Manager runs as your
user with access to your home directory. Install nodes you'd be willing to run as a script.

## Not ported

Left out on purpose; the upstream script has them if you want them: `prune` and image tags (there is no image), and the patch list
and node list (a recipe installs nodes now, and Manager does it from the UI). Rollback is the git
checkout (`.state/comfy.previous`).

## Status

Written against upstream at `33e521b`. Tested on a real DGX Spark (GB10, driver 580.178, CUDA 13.0, 2026-11-16): fresh install including the sm_121 SageAttention build and the aarch64 onnxruntime wheel, a no-change re-run, `doctor`, and `run` serving the UI with SageAttention active. `update` to the newest release was run there too. Not yet exercised on the Spark: `service`, `backup`/`restore`, `recipe install`, and generating an image.

MIT. Derived from spark-comfyui, copyright (c) 2026 Bjarke Bolding; see `LICENSE`.
