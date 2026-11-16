---
name: sync-upstream
description: Check bjarkebolding/spark-comfyui for changes since the last sync and port what applies.
disable-model-invocation: true
---

# sync-upstream

This repo is a native port of upstream's container build. Upstream's `container/Dockerfile`, `gb10.sh`,
`entrypoint.sh`, `manager-config.py` and `mods/unified-memory-free/transform.py` are the sources of truth;
`dgx-comfyui.sh` and `lib/mem_unified_transform.py` are their ports. `.upstream` records the upstream
commit this repo was last brought level with (`SYNCED`).

## 1. Check

Run `scripts/check-upstream.sh` (read-only). Exit 0 means nothing to port: say so, and stop.
Exit 10 means drift. Its report lists upstream commits since `SYNCED`, pins that differ from our defaults,
and ported files that changed.

## 2. Port, one drifted item at a time

Read the upstream diff for each flagged file (`git diff <SYNCED>..<new> -- <file>` in a clone of
`UPSTREAM_REPO`) and the commit messages, then decide per item:

| Upstream change | Port as |
|---|---|
| Dockerfile `ARG` pin (torch trio, `SAGE_REF`, `ORT_WHEEL_URL`, `TORCH_CUDA_ARCH_LIST`) | the matching default near the top of `dgx-comfyui.sh` |
| `CUDA_IMAGE` moves off 13.0.x | the `nvcc` check in `chk_host`; ask first, a CUDA bump needs a field test |
| `transform.py` | copy to `lib/mem_unified_transform.py` |
| `gb10.sh` gate or guard | `gate_sage`, `gate_onnx`, `chk_*`, `cmd_run` |
| `entrypoint.sh` launch flags or steps | `cmd_run` flags and gates |
| `manager-config.py` | `fix_manager` |
| container-only (docker, health check, `tune`, prune, service, mounts, build cache) | skip, and name it as skipped |

A change that alters what a user must do or sees (a new flag, a new step, a changed default) also goes
into `README.md`. A change that moves a build input (torch, Sage, ORT) is a rebuild: run it on the Spark.

## 3. Prove it

- `shellcheck -x dgx-comfyui.sh install.sh scripts/check-upstream.sh` is clean.
- The patch applies to a fresh ComfyUI checkout twice (second run: `present`) and still parses.
- On the Spark (`gx10-f5f4` is a herdr machine: `herdr-app do space.new --machine gx10-f5f4 ...`, then
  `hc read`/`hc wait` to follow it): `./dgx-comfyui.sh install` converges, a second `install` prints
  "Nothing to do", `doctor` is healthy, and `run` answers on `:8188` with SageAttention active.
  Skipping the Spark run is allowed only when the port was text-only (README, comments); say so in the PR.

## 4. Land it

Set `SYNCED` in `.upstream` to the upstream commit reviewed, on a branch `sync-<short-sha>`, together
with the port. Open a PR whose body lists each upstream commit as ported or skipped (with the reason) and
what ran on the Spark. Leave merging to the user.
