#!/usr/bin/env bash
# dgx-comfyui.sh: ComfyUI on a DGX Spark (GB10, sm_121) in one folder, with uv.
#
#   ./dgx-comfyui.sh install            converge to the wanted state; safe to re-run, changes only what differs
#   ./dgx-comfyui.sh update             newest ComfyUI release + its dependencies, then converge
#   ./dgx-comfyui.sh run [..]           launch ComfyUI in the foreground (extra args go to main.py)
#   ./dgx-comfyui.sh doctor [--json]    read-only health report, changes nothing
#   ./dgx-comfyui.sh status [--watch [SEC] | --json]   one-page glance, a live dashboard, or one JSON snapshot
#   ./dgx-comfyui.sh service [--disable]      run ComfyUI in the background, restarted on crash and at boot
#   ./dgx-comfyui.sh stop | logs [-f]         stop it; read its service log
#   ./dgx-comfyui.sh backup [--with-output] [FILE] | restore FILE [--force]
#   ./dgx-comfyui.sh recipe list|show|check|install|capture   a workflow's nodes and models, as data
#   ./dgx-comfyui.sh tune [--clock-cap MHZ] [--persist] | tune --revert   host stability (uses sudo)
#
# Derived from https://github.com/bjarkebolding/spark-comfyui (MIT), which does the same
# in a container. Everything here lives under this script's folder: uv, Python, venv,
# ComfyUI, caches and your models. Delete the folder and it is gone.
set -euo pipefail

SELF="$(readlink -f "${BASH_SOURCE[0]}")"
ROOT="${SELF%/*}"
STATE="$ROOT/.state"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
COMFY="$ROOT/ComfyUI"
DATA="$ROOT/data"
MEM_MARKER="# spark-comfyui:mem_unified"
MEM_FILE="comfy/model_management.py"

# ---- config: environment > dgx-comfyui.conf > defaults -------------------------
CONF_KEYS=(PYTHON_VERSION TORCH_VERSION TORCHVISION_VERSION TORCHAUDIO_VERSION TORCH_INDEX
  COMFY_REPO COMFY_REF SAGE_REF SAGE_ARCH ORT_WHEEL_URL CUDA_HOME PORT BIND_ADDR
  SPARK_ATTENTION SPARK_BF16 SPARK_BF16_VAE SPARK_STATIC_VRAM SPARK_RESERVE_VRAM)
CONF_FILE="$ROOT/dgx-comfyui.conf"
[[ -f "$CONF_FILE" ]] || CONF_FILE="$ROOT/spark-comfyui.conf"   # its name before the rename
load_conf() {
  local f="$CONF_FILE" line k v
  [[ -f "$f" ]] || return 0
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"
    [[ "$line" =~ ^[[:space:]]*([A-Z_]+)[[:space:]]*=[[:space:]]*(.*[^[:space:]])[[:space:]]*$ ]] || continue
    k="${BASH_REMATCH[1]}"; v="${BASH_REMATCH[2]}"
    [[ " ${CONF_KEYS[*]} " == *" $k "* ]] || { printf '[warn] %s: unknown key %s ignored\n' "${f##*/}" "$k" >&2; continue; }
    [[ -n "${!k+x}" ]] || printf -v "$k" '%s' "$v"
  done < "$f"
}
load_conf
: "${PYTHON_VERSION:=3.12}"            # the GPU onnxruntime wheel is cp312
: "${TORCH_VERSION:=2.14.0}" "${TORCHVISION_VERSION:=0.29.0}" "${TORCHAUDIO_VERSION:=2.11.0}"
: "${TORCH_INDEX:=https://download.pytorch.org/whl/cu130}"
: "${COMFY_REPO:=https://github.com/Comfy-Org/ComfyUI.git}"
: "${COMFY_REF:=latest}"                # latest (newest release tag), or a branch, tag or 40-char sha
: "${SAGE_REF:=d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5}"
: "${SAGE_ARCH:=12.1+PTX}"
if [[ "$(uname -m)" == aarch64 ]]; then
  : "${ORT_WHEEL_URL:=https://files.pythonhosted.org/packages/fa/96/1be0b9711a614861fbf5c0c85c4da3aa2321f9da0b4734a717eb606f70ab/onnxruntime_gpu-1.29.0-cp312-cp312-manylinux_2_34_aarch64.whl#sha256=545d2966dd11208bbc98e67be4d92e54ecd793acc45c0532cc7bfd36f50692fa}"
fi
: "${ORT_WHEEL_URL:=}" "${CUDA_HOME:=}"
: "${PORT:=8188}" "${BIND_ADDR:=0.0.0.0}"
: "${SPARK_ATTENTION:=sage}" "${SPARK_BF16:=1}" "${SPARK_BF16_VAE:=1}" "${SPARK_STATIC_VRAM:=0}" "${SPARK_RESERVE_VRAM:=}"
: "${SPARK_ALLOW_NON_SPARK:=0}"         # testing only: skip the aarch64 / driver / nvcc checks

# ---- output ------------------------------------------------------------------------
CHANGES=0; PROBLEMS=0
log()  { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
did()  { CHANGES=$((CHANGES+1)); printf '  \033[33m↻\033[0m %s\n' "$*"; }
info() { printf '  \033[36m·\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m[warn]\033[0m %s\n' "$*" >&2; }
bad()  { PROBLEMS=$((PROBLEMS+1)); printf '  \033[31m✗\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

# ---- environment: every cache and tool dir inside ROOT ----------------------------------
setup_env() {
  mkdir -p "$STATE" "$ROOT/.cache" "$ROOT/bin"
  export UV_PYTHON_INSTALL_DIR="$ROOT/.python" UV_CACHE_DIR="$ROOT/.cache/uv"
  export UV_PYTHON_PREFERENCE=only-managed UV_NO_CONFIG=1 UV_HTTP_TIMEOUT=120
  export PIP_CACHE_DIR="$ROOT/.cache/pip" XDG_CACHE_HOME="$ROOT/.cache"
  export HF_HOME="$ROOT/.cache/huggingface" TORCH_HOME="$ROOT/.cache/torch"
  export TRITON_CACHE_DIR="$ROOT/.cache/triton" CUDA_CACHE_PATH="$ROOT/.cache/nv-compute"
  export CUDA_CACHE_MAXSIZE=4294967296 NCCL_P2P_DISABLE=1
  export VIRTUAL_ENV="$VENV"
  PATH="$VENV/bin:$ROOT/bin:$PATH"; export PATH
  # Keep torch where it is, whatever a custom node's requirements ask for.
  export UV_CONSTRAINT="$STATE/constraints.txt" PIP_CONSTRAINT="$STATE/constraints.txt"
  local sp="$VENV/lib/python${PYTHON_VERSION}/site-packages"
  export LD_LIBRARY_PATH="$sp/nvidia/cu13/lib:$sp/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  find_cuda || true
  if [[ -n "$CUDA_HOME" ]]; then
    export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH"
    [[ -x "$CUDA_HOME/bin/ptxas" ]] && export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas"
  fi
}
find_cuda() {
  local c
  for c in "$CUDA_HOME" /usr/local/cuda /usr/local/cuda-13.0; do
    [[ -n "$c" && -x "$c/bin/nvcc" ]] && { CUDA_HOME="$c"; return 0; }
  done
  c="$(command -v nvcc 2>/dev/null || true)"
  [[ -n "$c" ]] && { CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$c")")")"; return 0; }
  CUDA_HOME=""; return 1
}
have_gpu() { nvidia-smi -L >/dev/null 2>&1; }
dist_version() {   # installed version of a distribution in the venv, or empty
  [[ -x "$PY" ]] || return 0
  "$PY" -I -c 'import importlib.metadata as m, sys
try: print(m.version(sys.argv[1]))
except m.PackageNotFoundError: pass' "$1" 2>/dev/null || true
}
uvpip() { "$UV" pip "$@" --python "$PY"; }

# ---- components: chk_X (silent test, sets DETAIL) and fix_X (repair) ----------------------
# Each converges independently, so a re-run touches only what differs.

chk_host() {
  local m drv ver
  m="$(uname -m)"; DETAIL="$m"
  if [[ "$SPARK_ALLOW_NON_SPARK" == 1 ]]; then DETAIL="$m (checks skipped: SPARK_ALLOW_NON_SPARK=1)"; return 0; fi
  [[ "$m" == aarch64 ]] || { DETAIL="$m is not aarch64; this is for the DGX Spark"; return 1; }
  drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
  [[ -n "$drv" ]] || { DETAIL="nvidia-smi sees no GPU"; return 1; }
  (( ${drv%%.*} >= 580 )) || { DETAIL="driver $drv is older than r580 (torch cu130 needs it)"; return 1; }
  DETAIL="aarch64, driver $drv"
  if [[ "$SPARK_ATTENTION" == sage ]]; then
    find_cuda || { DETAIL="no CUDA toolkit (nvcc) found: SageAttention needs one. Set CUDA_HOME or: sudo apt install cuda-toolkit-13-0"; return 1; }
    ver="$("$CUDA_HOME/bin/nvcc" --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')"
    [[ "${ver%%.*}" -ge 13 ]] || { DETAIL="nvcc $ver at $CUDA_HOME is older than CUDA 13"; return 1; }
    DETAIL+=", nvcc $ver"
    [[ "$ver" == 13.0 ]] || warn "nvcc is $ver; upstream pins 13.0 (a newer ptxas can emit PTX the r580 driver rejects)"
    command -v g++ >/dev/null || { DETAIL+=" but g++ is missing: sudo apt install build-essential"; return 1; }
  fi
  return 0
}
chk_syslibs() {
  local miss=() l libs
  libs="$(ldconfig -p 2>/dev/null || true)"   # captured first: grep -q on a pipe trips pipefail via SIGPIPE
  for l in libGL.so.1 libglib-2.0.so.0; do grep -qF "$l" <<<"$libs" || miss+=("$l"); done
  DETAIL="libGL, libglib present"
  (( ${#miss[@]} == 0 )) || { DETAIL="missing ${miss[*]}: sudo apt install libgl1 libglib2.0-0t64"; return 1; }
}

UV="$(command -v uv 2>/dev/null || true)"
[[ -x "$ROOT/bin/uv" ]] && UV="$ROOT/bin/uv"
chk_uv() { [[ -n "$UV" && -x "$UV" ]] && DETAIL="$("$UV" --version)"; }
fix_uv() {
  command -v curl >/dev/null || die "curl is needed to fetch uv"
  curl -LsSf https://astral.sh/uv/install.sh | UV_UNMANAGED_INSTALL="$ROOT/bin" UV_NO_MODIFY_PATH=1 sh >/dev/null || return 1
  UV="$ROOT/bin/uv"
}

chk_python() {
  [[ -x "$PY" ]] || { DETAIL="no venv"; return 1; }
  local v; v="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
  DETAIL="Python $v in .venv"
  [[ "$v" == "$PYTHON_VERSION" ]] && [[ "$("$PY" -c 'import sys; print(sys.base_prefix)')" == "$ROOT/.python/"* ]]
}
fix_python() { "$UV" venv --clear --python "$PYTHON_VERSION" "$VENV" >/dev/null || return 1; }

write_constraints() {
  local want tmp="$STATE/constraints.txt.tmp"
  want="torch==$TORCH_VERSION
torchvision==$TORCHVISION_VERSION
torchaudio==$TORCHAUDIO_VERSION"
  [[ -f "$STATE/constraints.txt" && "$(cat "$STATE/constraints.txt")" == "$want" ]] && return 0
  printf '%s\n' "$want" > "$tmp" && mv -f "$tmp" "$STATE/constraints.txt"
}
chk_torch() {
  local t v a
  t="$(dist_version torch)"; v="$(dist_version torchvision)"; a="$(dist_version torchaudio)"
  DETAIL="torch ${t:-none}"
  [[ "${t%%+*}" == "$TORCH_VERSION" && "${v%%+*}" == "$TORCHVISION_VERSION" && "${a%%+*}" == "$TORCHAUDIO_VERSION" ]] || return 1
  "$PY" -c 'import torch; assert (torch.version.cuda or "").startswith("13")' 2>/dev/null
}
fix_torch() {
  uvpip install -q --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio \
    "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" "torchaudio==$TORCHAUDIO_VERSION" \
    --index-url "$TORCH_INDEX" || return 1
}

# setuptools inside torch's own declared bound (SageAttention builds against it), build tools,
# and uv inside the venv so ComfyUI-Manager finds it.
chk_tools() {
  local spec
  spec="$("$PY" -I - <<'PY'
import importlib.metadata as md
for r in (md.requires("torch") or []):
    r = r.split(";")[0].strip()
    if r.startswith("setuptools"):
        print(r); break
else:
    print("setuptools")
PY
)"
  TOOLS_SPEC="$spec"
  DETAIL="$spec, ninja, packaging, wheel, uv"
  "$PY" -I - "$spec" <<'PY' 2>/dev/null
import importlib.metadata as md, sys
from packaging.requirements import Requirement
r = Requirement(sys.argv[1])
assert md.version("setuptools") in r.specifier or not r.specifier
for p in ("ninja", "packaging", "wheel", "uv"):
    md.version(p)
PY
}
fix_tools() { uvpip install -q "$TOOLS_SPEC" ninja packaging wheel uv || return 1; }

# COMFY_REF=latest means the newest vX.Y.Z release tag upstream (not master, which runs ahead of releases).
resolve_ref() {
  local tag
  if [[ "$COMFY_REF" != latest ]]; then printf '%s' "$COMFY_REF"; return 0; fi
  tag="$(timeout 20 git ls-remote --tags --sort=-v:refname "$COMFY_REPO" 'v*' 2>/dev/null \
    | sed 's#^.*refs/tags/##' | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sed -n 1p || true)"
  [[ -n "$tag" ]] || return 1
  printf '%s' "$tag"
}
comfy_apply_ref() {   # move the checkout to the wanted ref; the caller handles the patched file
  local ref; ref="$(resolve_ref)" || { warn "cannot look up the latest ComfyUI release (offline?)"; return 1; }
  git -C "$COMFY" fetch -q --tags origin || return 1
  if git -C "$COMFY" rev-parse --verify -q "origin/$ref" >/dev/null; then
    git -C "$COMFY" checkout -q -B "$ref" "origin/$ref"
  else
    git -C "$COMFY" checkout -q --detach "$ref"
  fi
}
comfy_label() {   # "v0.45.4" for a release, else "master @ e9afbcc" ; plus the commit date
  local t; t="$(git -C "$COMFY" describe --tags --exact-match 2>/dev/null || true)"
  printf '%s (%s)' "${t:-$(git -C "$COMFY" rev-parse --abbrev-ref HEAD) @ $(git -C "$COMFY" rev-parse --short HEAD)}" \
    "$(git -C "$COMFY" log -1 --format=%cs)"
}
chk_comfy() {
  [[ -d "$COMFY/.git" ]] || { DETAIL="not cloned"; return 1; }
  DETAIL="ComfyUI $(comfy_label)"
  local want
  if [[ "$COMFY_REF" == latest ]]; then
    # A fresh install lands on the newest release; moving an existing one is `update`'s job.
    want="$(resolve_ref || true)"
    if [[ -n "$want" && "$(git -C "$COMFY" rev-parse HEAD)" != "$(git -C "$COMFY" rev-parse -q --verify "$want^{commit}" 2>/dev/null || true)" ]]; then
      DETAIL+="; newer release $want exists: run ./dgx-comfyui.sh update"
    fi
  elif [[ "$COMFY_REF" =~ ^[0-9a-f]{40}$ ]]; then
    [[ "$(git -C "$COMFY" rev-parse HEAD)" == "$COMFY_REF" ]]   # an explicit sha is a pin: install enforces it
  elif git -C "$COMFY" rev-parse -q --verify "refs/tags/$COMFY_REF" >/dev/null; then
    [[ "$(git -C "$COMFY" rev-parse HEAD)" == "$(git -C "$COMFY" rev-parse "$COMFY_REF^{commit}")" ]]   # so is a tag
  fi
}
fix_comfy() {
  if [[ ! -d "$COMFY/.git" ]]; then
    local tmp="$ROOT/ComfyUI.partial"
    rm -rf "$tmp"
    git clone -q "$COMFY_REPO" "$tmp" || return 1
    mv "$tmp" "$COMFY" || return 1
    comfy_apply_ref || return 1
  else
    move_comfy
  fi
}
# Fetch and move the checkout, undoing the patch first and redoing it after. Failure goes back to the
# previous commit instead of leaving the install on a half-moved tree.
move_comfy() {
  local prev; prev="$(git -C "$COMFY" rev-parse HEAD)"
  mem_revert
  if ! comfy_apply_ref; then
    git -C "$COMFY" checkout -q --detach "$prev" || true
    mem_apply >/dev/null || true
    die "could not move ComfyUI to $COMFY_REF; stayed on ${prev:0:9}"
  fi
  printf '%s\n' "$prev" > "$STATE/comfy.previous"
  if ! mem_apply >/dev/null; then
    git -C "$COMFY" checkout -q --detach "$prev"; mem_apply >/dev/null || true
    die "the memory patch no longer fits ComfyUI $COMFY_REF; rolled back to ${prev:0:9}. Update lib/mem_unified_transform.py."
  fi
}

# ComfyUI's own requirements. uv no-ops when satisfied, so this runs every time and reports a change
# only when the installed set really differs afterwards.
step_requirements() {   # optional arg: --upgrade, to move dependencies to the newest the release allows
  local before after
  before="$(uvpip freeze | sha256sum)"
  uvpip install -q "$@" -r "$COMFY/requirements.txt" -r "$COMFY/manager_requirements.txt" -c "$STATE/constraints.txt"
  after="$(uvpip freeze | sha256sum)"
  if [[ "$before" == "$after" ]]; then ok "ComfyUI requirements already satisfied"; else did "ComfyUI requirements installed"; fi
}

# A new ComfyUI release can move shared dependencies, so each custom node's requirements are installed
# again against them (torch stays pinned by the constraints file). One node failing is a warning, not a stop.
step_node_requirements() {
  local f n failed=0 seen=0
  for f in "$DATA"/custom_nodes/*/requirements.txt; do
    [[ -f "$f" ]] || continue
    seen=1; n="$(basename "$(dirname "$f")")"
    uvpip install -q -r "$f" -c "$STATE/constraints.txt" || { warn "requirements of custom node $n did not install"; failed=1; }
  done
  if (( ! seen )); then info "no custom node has requirements"
  elif (( failed )); then warn "some custom node requirements failed: those nodes may not load"
  else ok "custom node requirements installed"; fi
}

sage_stamp() { printf '%s|torch %s|%s' "$SAGE_REF" "$(dist_version torch)" "$SAGE_ARCH"; }
gate_sage() {
  "$PY" - <<'PY' >/dev/null 2>&1
import torch
from sageattention import sageattn
q = torch.randn(1, 8, 1024, 128, dtype=torch.float16, device="cuda")
o = sageattn(q, q, q, tensor_layout="HND")
torch.cuda.synchronize()
assert o.shape == q.shape and torch.isfinite(o).all()
PY
}
chk_sage() {
  DETAIL="SageAttention ${SAGE_REF:0:9}"
  [[ -n "$(dist_version sageattention)" ]] || { DETAIL="SageAttention not built"; return 1; }
  [[ -f "$STATE/sage.stamp" && "$(cat "$STATE/sage.stamp")" == "$(sage_stamp)" ]] || { DETAIL="SageAttention built for another ref or torch"; return 1; }
  if have_gpu; then gate_sage || { DETAIL="SageAttention kernel fails on this GPU (shadowed or miscompiled)"; return 1; }; DETAIL+=", kernel verified live"; fi
}
fix_sage() {
  local src="$ROOT/build/SageAttention"
  mkdir -p "$ROOT/build"
  if [[ ! -d "$src/.git" ]]; then
    rm -rf "$src.partial"
    git clone -q https://github.com/thu-ml/SageAttention.git "$src.partial" || return 1
    mv "$src.partial" "$src" || return 1
  fi
  if [[ "$(git -C "$src" rev-parse HEAD)" != "$SAGE_REF" ]]; then
    git -C "$src" fetch -q origin || return 1
    git -C "$src" checkout -q -f "$SAGE_REF" || return 1
  fi
  git -C "$src" clean -xfdq || return 1
  info "compiling SageAttention for sm_121 (about 5 minutes)"
  ( cd "$src"
    CXX_APPEND_FLAGS="-std=c++20" NVCC_APPEND_FLAGS="-std=c++20" TORCH_CUDA_ARCH_LIST="$SAGE_ARCH" \
    MAX_JOBS="$(nproc)" "$UV" pip install --python "$PY" -q --no-build-isolation --no-deps --reinstall-package sageattention . ) || return 1
  sage_stamp > "$STATE/sage.stamp"
}

ort_version() { local f="${ORT_WHEEL_URL%%#*}"; f="${f##*/}"; f="${f#onnxruntime_gpu-}"; printf '%s' "${f%%-*}"; }
gate_onnx() {
  "$PY" - <<'PY' >/dev/null 2>&1
import base64, numpy as np
import onnxruntime as ort
MODEL = base64.b64decode(
    "CAo6tgEKOQoBWAoBVxIBWSIEQ29udioVCgxrZXJuZWxfc2hhcGVAA0ADoAEHKhEKBHBhZH"
    "NAAUABQAFAAaABBxIKc3BhcmtfZ2F0ZSozCAEIAQgDCAMQAUIBV0okAAAAPwAAAD8AAAA/"
    "AAAAPwAAAD8AAAA/AAAAPwAAAD8AAAA/WhsKAVgSFgoUCAESEAoCCAEKAggBCgIICAoCCA"
    "hiGwoBWRIWChQIARIQCgIIAQoCCAEKAggICgIICEIECgAQEg=="
)
s = ort.InferenceSession(MODEL, providers=["CUDAExecutionProvider"])
assert s.get_providers()[0] == "CUDAExecutionProvider", s.get_providers()
y = s.run(None, {"X": np.ones((1, 1, 8, 8), dtype=np.float32)})[0]
assert np.isfinite(y).all() and abs(float(y[0, 0, 4, 4]) - 4.5) < 1e-4
PY
}
chk_ort() {
  [[ -n "$ORT_WHEEL_URL" ]] || { DETAIL="no GPU onnxruntime wheel for $(uname -m): skipped"; return 0; }
  local g c; g="$(dist_version onnxruntime-gpu)"; c="$(dist_version onnxruntime)"
  DETAIL="onnxruntime-gpu ${g:-none}"
  [[ "$g" == "$(ort_version)" && -z "$c" ]] || { [[ -n "$c" ]] && DETAIL="CPU onnxruntime $c shadows the GPU wheel"; return 1; }
  if have_gpu; then gate_onnx || { DETAIL="onnxruntime CUDA provider not live"; return 1; }; DETAIL+=", CUDA provider live"; fi
}
fix_ort() {
  uvpip uninstall -q onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true
  uvpip install -q "$ORT_WHEEL_URL" || return 1
}

chk_mem() {
  DETAIL="unified-memory get_free_memory patch"
  grep -qF "$MEM_MARKER" "$COMFY/$MEM_FILE" 2>/dev/null
}
mem_apply() {   # prints applied|present, returns non-zero if the patch cannot be made
  local f="$COMFY/$MEM_FILE" out
  grep -qF "$MEM_MARKER" "$f" && { echo present; return 0; }
  out="$(MARKER="$MEM_MARKER" "$PY" "$ROOT/lib/mem_unified_transform.py" < "$f")" || return 1
  [[ -n "$out" && "$out" != "$(cat "$f")" ]] || return 1
  printf '%s\n' "$out" > "$f.new"
  "$PY" -c 'import ast,sys; ast.parse(open(sys.argv[1], encoding="utf-8").read())' "$f.new" || { rm -f "$f.new"; return 1; }
  mv -f "$f.new" "$f"
  echo applied
}
mem_revert() { grep -qF "$MEM_MARKER" "$COMFY/$MEM_FILE" 2>/dev/null && git -C "$COMFY" checkout -q -- "$MEM_FILE" || true; }
fix_mem() { mem_apply >/dev/null; }

chk_manager() {
  DETAIL="ComfyUI-Manager config"
  [[ -f "$DATA/user/__manager/config.ini" ]] || return 1
  "$PY" -I - "$DATA/user/__manager/config.ini" <<'PY' 2>/dev/null
import configparser, sys
c = configparser.ConfigParser(); c.read(sys.argv[1]); d = c["default"]
assert d.get("network_mode") == "personal_cloud" and "downgrade_blacklist" in d and "use_uv" in d
PY
}
fix_manager() {
  mkdir -p "$DATA/user/__manager" || return 1
  "$PY" -I - "$DATA/user/__manager/config.ini" <<'PY'
import configparser, os, sys
path = sys.argv[1]
cfg = configparser.ConfigParser()
try:
    cfg.read(path)
except (configparser.Error, UnicodeDecodeError):
    os.replace(path, path + ".bak"); cfg = configparser.ConfigParser()
if not cfg.has_section("default"): cfg.add_section("default")
cfg.set("default", "network_mode", "personal_cloud")   # always: the launcher depends on it
for k, v in {"security_level": "normal", "use_uv": "True", "file_logging": "True",
             "downgrade_blacklist": "torch,torchvision,torchaudio"}.items():
    if not cfg.has_option("default", k): cfg.set("default", k, v)
tmp = path + ".tmp"
with open(tmp, "w") as f: cfg.write(f)
os.replace(tmp, path)
PY
}

DATA_DIRS=(models/checkpoints models/diffusion_models models/loras models/vae models/text_encoders
  models/clip_vision models/controlnet models/upscale_models models/embeddings custom_nodes input output user)
chk_data() {
  local d; DETAIL="data/ (models, custom_nodes, input, output, user)"
  for d in "${DATA_DIRS[@]}"; do [[ -d "$DATA/$d" ]] || return 1; done
}
fix_data() { local d; for d in "${DATA_DIRS[@]}"; do mkdir -p "$DATA/$d" || return 1; done; }

# Converge one component. In doctor mode a failing check is reported and nothing is changed.
converge() {
  local n="$1" label="$2"
  if "chk_$n"; then ok "$label: $DETAIL"; return 0; fi
  if [[ "$MODE" == doctor ]]; then bad "$label: $DETAIL"; return 1; fi
  info "$label: $DETAIL; fixing"
  if "fix_$n" && "chk_$n"; then did "$label: $DETAIL"; else bad "$label: still not right after the fix${DETAIL:+ ($DETAIL)}"; return 1; fi
}
# Environment problems only the user can fix: report, never repair.
require() {
  local n="$1" label="$2"
  if "chk_$n"; then ok "$label: $DETAIL"; else bad "$label: $DETAIL"; return 1; fi
}

# ---- commands --------------------------------------------------------------------------
acquire_lock() {
  command -v flock >/dev/null || die "flock is missing (util-linux)"
  mkdir -p "$STATE"; exec 9>"$STATE/lock"
  flock -n "$@" 9 || die "another install, update or run holds the lock (stop ComfyUI first to change it)"
}

UNIT="dgx-comfyui.service"
UNIT_FILE="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/$UNIT"
svc_active() { command -v systemctl >/dev/null && systemctl --user is-active --quiet "$UNIT" 2>/dev/null; }
SERVICE_PAUSED=0
resume_service() { if (( SERVICE_PAUSED )); then systemctl --user start "$UNIT" && printf '  \033[36m·\033[0m service restarted\n'; fi; }
pause_service() {   # the running service holds the lock; stop it while we change things, restart it after
  if svc_active; then
    SERVICE_PAUSED=1; trap resume_service EXIT
    info "stopping the $UNIT service while this runs (it restarts afterwards)"
    systemctl --user stop "$UNIT"
  fi
}

cmd_install() {
  MODE=install
  pause_service
  acquire_lock -x
  log "Host"
  require host "host" || die "fix the host problem above and re-run"
  require syslibs "system libraries" || true
  setup_env
  log "Toolchain"
  converge uv "uv" || die "cannot continue without uv"
  setup_env     # picks up a freshly installed uv
  converge python "python" || die "cannot create the venv"
  write_constraints
  log "PyTorch"
  converge torch "torch cu130" || die "cannot continue without torch"
  converge tools "build tools" || true
  log "ComfyUI"
  converge comfy "checkout" || die "cannot continue without ComfyUI"
  step_requirements
  converge mem "patch" || warn "the memory patch could not be applied (upstream moved the code?); ComfyUI still works"
  converge manager "manager" || true
  converge data "data" || true
  log "GPU stack"
  if [[ "$SPARK_ATTENTION" == sage ]]; then converge sage "sage" || true; else info "attention backend $SPARK_ATTENTION: SageAttention build skipped"; fi
  converge ort "onnx" || true
  summary install
}

cmd_update() {
  pause_service
  acquire_lock -x
  [[ -d "$COMFY/.git" ]] || die "not installed yet: run ./dgx-comfyui.sh install"
  setup_env
  log "Updating ComfyUI ($COMFY_REF)"
  local before_sha before_label frontend_before templates_before
  before_sha="$(git -C "$COMFY" rev-parse HEAD)"; before_label="$(comfy_label)"
  frontend_before="$(dist_version comfyui-frontend-package)"; templates_before="$(dist_version comfyui-workflow-templates)"
  move_comfy
  if [[ "$before_sha" == "$(git -C "$COMFY" rev-parse HEAD)" ]]; then ok "already at $before_label"
  else did "ComfyUI $before_label -> $(comfy_label); previous commit kept in .state/comfy.previous"; fi
  log "Dependencies"
  step_requirements --upgrade   # torch stays at its pin through the constraints file
  [[ "$frontend_before" == "$(dist_version comfyui-frontend-package)" ]] || info "frontend $frontend_before -> $(dist_version comfyui-frontend-package)"
  [[ "$templates_before" == "$(dist_version comfyui-workflow-templates)" ]] || info "workflow templates $templates_before -> $(dist_version comfyui-workflow-templates)"
  step_node_requirements
  # Release the lock hold before converging: install reacquires it.
  exec 9>&-
  cmd_install
  local rc=0
  [[ -x "$ROOT/scripts/check-upstream.sh" ]] && { "$ROOT/scripts/check-upstream.sh" >/dev/null 2>&1 || rc=$?; }
  if (( rc == 10 )); then
    info "the upstream spark-comfyui project has changes for the GPU stack (torch, SageAttention pins): see scripts/check-upstream.sh"
  fi
}

# The doctor's report as data: the same checks, each message collected instead of printed.
# The verdict is the `healthy` field; the exit status is 0 so a caller always gets the report.
doctor_json() {
  local items; items="$(mktemp)"
  collect() { printf '%s\t%s\n' "$1" "$2" >> "$items"; }
  log()  { collect log "$*"; }
  ok()   { collect ok "$*"; }
  did()  { CHANGES=$((CHANGES+1)); collect did "$*"; }
  info() { collect info "$*"; }
  warn() { collect warn "$*"; }
  bad()  { PROBLEMS=$((PROBLEMS+1)); collect bad "$*"; }
  summary() { :; }
  doctor_body
  python3 -I "$LIB_DIR/items_json.py" "$items" "$PROBLEMS" "$CHANGES"
  rm -f "$items"
}

cmd_doctor() {
  if [[ "${1:-}" == --json ]]; then doctor_json; return 0; fi
  doctor_body
}
doctor_body() {
  MODE=doctor; setup_env
  log "Host"
  require host "host" || true
  require syslibs "system libraries" || true
  log "Stack"
  for c in "uv:uv" "python:python" "torch:torch cu130" "tools:build tools" "comfy:ComfyUI" "mem:patch" "manager:manager" "data:data"; do converge "${c%%:*}" "${c#*:}" || true; done
  [[ "$SPARK_ATTENTION" == sage ]] && { converge sage "sage" || true; }
  converge ort "onnx" || true
  if [[ -f "$STATE/comfy.previous" ]]; then info "previous ComfyUI commit: $(cat "$STATE/comfy.previous")"; fi
  if [[ -n "$(swapon --noheadings 2>/dev/null)" ]]; then warn "swap is enabled: heavy workloads can freeze a Spark (./dgx-comfyui.sh tune)"; else ok "swap disabled on the host"; fi
  if grep -qx 'After=multi-user.target' /etc/systemd/system/comfyui-tune.service 2>/dev/null; then
    warn "comfyui-tune.service waits for the end of boot and may never run: rewrite it with ./dgx-comfyui.sh tune --persist"
  fi
  if have_gpu; then info "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"; else warn "no GPU visible: kernel gates skipped"; fi
  summary doctor
}

cmd_run() {
  acquire_lock -s
  [[ -x "$PY" && -d "$COMFY/.git" ]] || die "not installed yet: run ./dgx-comfyui.sh install"
  MODE=install; setup_env
  # Same guards the container entrypoint runs: shadowing happens after install, when a node's requirements land.
  log "GPU gates"
  if ! chk_torch; then
    warn "torch lost CUDA 13 or moved; reinstalling the pinned wheels"; fix_torch
    chk_torch || die "torch CUDA 13 check failed after repair. Host problem? Check nvidia-smi."
  fi
  have_gpu || die "nvidia-smi sees no GPU"
  "$PY" -c 'import torch; print("  torch", torch.__version__, "| CUDA", torch.version.cuda, "|", torch.cuda.get_device_name(0))' \
    || die "torch cannot initialise CUDA. Check nvidia-smi and reboot after a driver update."
  if [[ -n "$ORT_WHEEL_URL" ]] && ! chk_ort; then warn "$DETAIL; repairing"; fix_ort; chk_ort || warn "$DETAIL (preprocessors will use CPU)"; fi
  local attn=()
  case "$SPARK_ATTENTION" in
    sage) gate_sage || die "SageAttention kernel failed on this GPU; refusing to launch degraded. Run ./dgx-comfyui.sh install to rebuild, or set SPARK_ATTENTION=sdpa."
          attn=(--use-sage-attention); ok "SageAttention kernel verified live" ;;
    ck)   attn=(--use-ck-attention); info "Comfy Kitchen INT8 attention" ;;
    sdpa) info "attention: PyTorch SDPA" ;;
    *)    die "SPARK_ATTENTION must be sage, sdpa or ck (got '$SPARK_ATTENTION')" ;;
  esac
  local flags=()
  if [[ "$SPARK_BF16" == 1 ]]; then
    flags+=(--bf16-unet --bf16-text-enc); [[ "$SPARK_BF16_VAE" == 1 ]] && flags+=(--bf16-vae)
  fi
  [[ "$SPARK_STATIC_VRAM" == 1 ]] && flags+=(--disable-dynamic-vram)
  if [[ "$SPARK_RESERVE_VRAM" =~ ^[0-9]+([.][0-9]+)?$ ]]; then flags+=(--reserve-vram "$SPARK_RESERVE_VRAM"); fi
  [[ "$BIND_ADDR" == 0.0.0.0 ]] && warn "listening on every interface and ComfyUI has no login; set BIND_ADDR = 127.0.0.1 in dgx-comfyui.conf for this box only"
  log "Launching ComfyUI on $BIND_ADDR:$PORT"
  cd "$COMFY"
  echo $$ > "$STATE/comfyui.pid"   # the shell becomes python below, so this stays the server's pid
  exec python main.py --listen "$BIND_ADDR" --port "$PORT" --base-directory "$DATA" \
    --enable-manager --preview-method auto --disable-pinned-memory \
    "${attn[@]}" "${flags[@]}" "$@"
}


# ---- tune: host stability (ported from upstream, field-validated on GB10) ------------
# The only part of this tool that uses sudo and touches the host outside this folder. Opt-in.
# The inverse of cmd_tune, step for step, ordered by risk rather than by
# mirroring: the persisted unit goes FIRST so that a failure anywhere below
# leaves a box that at least stops re-applying the tuning on next boot, and
# swap goes LAST because re-enabling it is the one step that puts the freeze
# risk back, so its warning should be the last thing on screen. Same
# skip-if-already-in-the-target-state rule as tune, so a second --revert asks
# for no password it does not need.
_tune_revert() {
  log "Reverting DGX Spark system tuning"

  # 1) The persisted unit, if --persist ever wrote one.
  local unit=/etc/systemd/system/comfyui-tune.service
  if [[ -f "$unit" ]]; then
    sudo systemctl disable --now comfyui-tune.service >/dev/null 2>&1 || true
    sudo rm -f "$unit"
    sudo systemctl daemon-reload
    echo "  comfyui-tune.service removed (tuning no longer re-applies at boot)"
  else
    echo "  no persisted tuning unit"
  fi

  # 2) GPU clocks. NOTHING reports an -lgc lock range: the clocks_event_reasons
  #    fields cover the applications-clocks lever and the power cap, not this
  #    one, and GB10 reports N/A for most of the clock block anyway. So this
  #    resets unconditionally instead of pretending to detect. -rgc on a GPU
  #    that was never locked is a no-op, which is what makes that honest.
  if sudo nvidia-smi -rgc >/dev/null 2>&1; then
    echo "  GPU clocks unlocked (reset to the driver default)"
  else
    warn "could not reset GPU clocks — 'sudo nvidia-smi -rgc' failed"
  fi

  # 3) Persistence mode. This one IS queryable, so it skips cleanly.
  if [[ "$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader 2>/dev/null)" == "Enabled" ]]; then
    sudo nvidia-smi -pm 0 >/dev/null && echo "  persistence mode: off"
  else
    echo "  persistence mode: already off"
  fi

  # 4) Swappiness. tune appends one fixed line, so deleting that exact line is
  #    a complete undo of the file. The live value it displaced was never
  #    recorded anywhere, so this restores the kernel default rather than
  #    inventing a number the box may never have had.
  if grep -q '^vm\.swappiness=10$' /etc/sysctl.conf 2>/dev/null; then
    sudo sed -i '/^vm\.swappiness=10$/d' /etc/sysctl.conf
    echo "  vm.swappiness=10 removed from /etc/sysctl.conf"
    # The live value is reset only because OUR line was there. Without that
    # proof a swappiness of 10 is somebody else's setting — a drop-in under
    # /etc/sysctl.d, a distro default — and overwriting it would be this
    # command reaching past what it put there.
    if [[ "$(sysctl -n vm.swappiness 2>/dev/null)" == "10" ]]; then
      sudo sysctl -w vm.swappiness=60 >/dev/null
      echo "  vm.swappiness=60 (the kernel default)"
    fi
  else
    echo "  no vm.swappiness line of ours in /etc/sysctl.conf"
  fi

  # 5) Swap. 'swapoff -a' never touched /etc/fstab, so 'swapon -a' restores
  #    exactly what the box shipped with. A box with no swap entry has nothing
  #    to restore, and that is a fact to report, not a failure.
  local swap_back=0
  if [[ -n "$(swapon --noheadings 2>/dev/null)" ]]; then
    echo "  swap already enabled"; swap_back=1
  else
    sudo swapon -a 2>/dev/null || true
    if [[ -n "$(swapon --noheadings 2>/dev/null)" ]]; then
      echo "  swap re-enabled from /etc/fstab"; swap_back=1
    else
      echo "  no swap entry in /etc/fstab — nothing to re-enable"
    fi
  fi
  # Only warn when swap is actually back on: on a box that has none, saying
  # so would be a false alarm about a state that does not exist.
  (( swap_back )) && warn "swap is on again. On unified memory a workload that
approaches the limit thrashes instead of getting a clean OOM kill, and the box
can freeze with no logs — re-run '$SELF tune' before the next heavy generation."
  return 0
}

cmd_tune() {
  local clock_cap="" persist=0 revert=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --clock-cap) clock_cap="${2:?--clock-cap needs a MHz value, e.g. 2100}"; shift 2 ;;
      --persist)   persist=1; shift ;;
      --revert)    revert=1; shift ;;
      *) die "Unknown tune option: $1 (use --clock-cap MHZ, --persist or --revert)" ;;
    esac
  done
  if (( revert )); then
    [[ -z "$clock_cap" && "$persist" -eq 0 ]] \
      || die "--revert undoes the tuning, so it cannot be combined with
--clock-cap or --persist — run it on its own"
    _tune_revert
    return 0
  fi
  # Validate up front: a bad value would otherwise die mid-run on nvidia-smi's
  # error, skipping the remaining tune steps. 300 is the -lgc floor used below.
  if [[ -n "$clock_cap" ]]; then
    if ! [[ "$clock_cap" =~ ^[0-9]+$ ]] || (( clock_cap < 300 || clock_cap > 4000 )); then
      die "--clock-cap needs a MHz value between 300 and 4000, e.g. 2100 (got: $clock_cap)"
    fi
  fi

  log "Applying DGX Spark system tuning"

  # 1) Swap off. On unified memory, approaching the limit doesn't OOM-kill
  #    cleanly — it thrashes swap, saturates the bus and silently freezes
  #    the whole box. No swap = clean OOM kill instead. Each sudo step below
  #    is skipped if the system's already in the target state, so re-running
  #    tune (e.g. after a reboot without --persist) doesn't ask for a
  #    password it doesn't need.
  if [[ -n "$(swapon --noheadings 2>/dev/null)" ]]; then
    sudo swapoff -a
    echo "  swap disabled"
  else
    echo "  swap already disabled"
  fi
  if [[ "$(sysctl -n vm.swappiness 2>/dev/null)" != "10" ]]; then
    sudo sysctl -w vm.swappiness=10 >/dev/null
  fi
  if grep -q '^vm.swappiness=10' /etc/sysctl.conf 2>/dev/null; then
    echo "  vm.swappiness=10 (already set in /etc/sysctl.conf)"
  else
    echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf >/dev/null
    echo "  vm.swappiness=10 (written to /etc/sysctl.conf — survives reboot on its own)"
  fi

  # 2) Driver persistence mode: avoids re-init latency between runs.
  if [[ "$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader 2>/dev/null)" == "Enabled" ]]; then
    echo "  persistence mode: already on"
  else
    sudo nvidia-smi -pm 1 >/dev/null && echo "  persistence mode: on"
  fi

  # 3) Optional clock cap. GB10 idles ~14W and spikes to ~85W instantly when
  #    denoising starts; on some units that trips overcurrent protection ->
  #    instant hard reboot with NO logs. Capping to ~2100 MHz holds ~50W and
  #    eliminated the crashes in community testing (power limit via
  #    'nvidia-smi -pl' is N/A on GB10, so clocks are the only lever).
  if [[ -n "$clock_cap" ]]; then
    sudo nvidia-smi -lgc 300,"$clock_cap" \
      && echo "  GPU clocks locked to 300-${clock_cap} MHz"
  else
    echo "  (no clock cap requested — if you get silent hard reboots under"
    echo "   heavy video generation, re-run: $SELF tune --clock-cap 2100)"
  fi

  # 4) Optionally persist across reboots (swapoff and -lgc do not survive).
  if [[ "$persist" -eq 1 ]]; then
    local unit=/etc/systemd/system/comfyui-tune.service lgc_line=""
    if [[ -n "$clock_cap" ]]; then
      lgc_line="ExecStart=/usr/bin/nvidia-smi -lgc 300,$clock_cap"
    elif [[ -f "$unit" ]]; then
      # Rewriting the unit without --clock-cap must not silently drop a
      # previously persisted cap: losing it re-exposes the overcurrent
      # hard-reboot the cap exists to prevent. Carry the old line over.
      lgc_line="$(grep -m1 '^ExecStart=.*nvidia-smi -lgc' "$unit" 2>/dev/null || true)"
      [[ -n "$lgc_line" ]] \
        && echo "  keeping previously persisted clock cap (${lgc_line##*-lgc })"
    fi
    # Ordered on what the steps need, never After=multi-user.target. That
    # target is only reached once the whole boot finishes, and on a headless
    # DGX OS box plymouth-quit-wait can hold it for hours or for good: on
    # sparky the unit ran 6h after one boot and never during the next
    # (2026-10-25), with swap on the whole time. swap.target means swapoff has
    # the fstab swap to turn off; nvidia-persistenced starts with the driver.
    # WantedBy below only says "at boot", it does not order anything.
    sudo tee "$unit" >/dev/null <<UNIT
[Unit]
Description=DGX Spark tuning for ComfyUI (swap off, persistence mode, clocks)
After=swap.target nvidia-persistenced.service

[Service]
Type=oneshot
ExecStart=/usr/sbin/swapoff -a
ExecStart=/usr/bin/nvidia-smi -pm 1
$lgc_line

[Install]
WantedBy=multi-user.target
UNIT
    sudo systemctl daemon-reload
    sudo systemctl enable comfyui-tune.service >/dev/null
    echo "  tuning persisted across reboots (comfyui-tune.service)"
  else
    warn "the swap-off and clock-cap settings do NOT survive a reboot on their
own (vm.swappiness above already does) — add --persist to make all of it permanent"
  fi
}

# ---- recipes (lib/recipes.sh, ported from upstream) ---------------------------------------
# shellcheck disable=SC2034  # RECIPES_DIR is read by lib/recipes.sh
LIB_DIR="$ROOT/lib"; RECIPES_DIR="$ROOT/recipes"
hdr() { log "$@"; }
resolve_mounts() { :; }
_mount_path() { case "$1" in models) printf '%s' "$DATA/models" ;; user) printf '%s' "$DATA/user" ;; *) return 1 ;; esac; }
cmd_recipe() {
  [[ -x "$PY" && -d "$COMFY/.git" ]] || die "not installed yet: ./dgx-comfyui.sh install"
  command -v wget >/dev/null || die "wget is missing: sudo apt install wget"
  setup_env
  # shellcheck source=lib/recipes.sh
  # shellcheck source=lib/recipes.sh
  source "$LIB_DIR/recipes.sh"
  _recipe_cmd "$@"
}

# ---- status, watch ---------------------------------------------------------------------
server_pid() {
  local pid; pid="$(cat "$STATE/comfyui.pid" 2>/dev/null || true)"
  [[ -n "$pid" && -r "/proc/$pid/cmdline" ]] && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q 'main.py' && printf '%s' "$pid"
}
server_state() {   # prints: stopped | starting | up
  local pid; pid="$(server_pid || true)"
  [[ -n "$pid" ]] || { echo stopped; return; }
  if curl -fsS -m 3 -o /dev/null "http://127.0.0.1:$PORT/queue" 2>/dev/null; then echo up; else echo starting; fi
}
queue_counts() {   # "running pending", 0 0 when the server is down
  curl -fsS -m 3 "http://127.0.0.1:$PORT/queue" 2>/dev/null | "$PY" -I -c 'import json,sys
d=json.load(sys.stdin); print(len(d.get("queue_running",[])), len(d.get("queue_pending",[])))' 2>/dev/null || echo "0 0"
}
num() { [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]] && printf '%s' "$1" || printf 0; }
gpu_sample() {   # "temp power clock util" (0 where the driver reports N/A)
  local o t p c u
  o="$(nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || true)"
  IFS=',' read -r t p c u <<<"$o"
  printf '%s %s %s %s' "$(num "${t// /}")" "$(num "${p// /}")" "$(num "${c// /}")" "$(num "${u// /}")"
}
mem_sample() { free -m | awk '/^Mem:/{print $3, $2}'; }   # used total, MB
spark() {   # sparkline of the numbers given as arguments
  local -a v=("$@"); local min max x i out="" bars=(▁ ▂ ▃ ▄ ▅ ▆ ▇ █)
  (( ${#v[@]} )) || return 0
  min="${v[0]}"; max="${v[0]}"
  for x in "${v[@]}"; do (( ${x%.*} < ${min%.*} )) && min="$x"; (( ${x%.*} > ${max%.*} )) && max="$x"; done
  for x in "${v[@]}"; do
    if (( ${max%.*} == ${min%.*} )); then i=0; else i=$(( (${x%.*} - ${min%.*}) * 7 / (${max%.*} - ${min%.*}) )); fi
    out+="${bars[$i]}"
  done
  printf '%s' "$out"
}
# One read-only snapshot as JSON, for a program that manages this install (schema 1). It writes nothing
# in the install folder. Every field is present; a value that does not apply is null or "".
status_json() {
  local st pid="" rss="" svc_inst=0 svc_act=0 svc_en="" installed=0 tag="" commit="" dirty=0 branch="" unpushed="" newer=""
  local ver="" cver="" comfy_commit="" disk="" models="" want
  st="$(server_state)"; pid="$(server_pid || true)"
  [[ -n "$pid" ]] && rss="$(awk '/^VmRSS:/{printf "%d", $2/1024}' "/proc/$pid/status" 2>/dev/null || true)"
  if command -v systemctl >/dev/null && [[ -f "$UNIT_FILE" ]]; then
    svc_inst=1
    systemctl --user is-active --quiet "$UNIT" 2>/dev/null && svc_act=1
    svc_en="$(systemctl --user is-enabled "$UNIT" 2>/dev/null || true)"
  fi
  if [[ -d "$ROOT/.git" ]]; then
    tag="$(git -C "$ROOT" describe --tags --exact-match 2>/dev/null || true)"
    commit="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || true)"
    branch="$(git -C "$ROOT" symbolic-ref --short -q HEAD 2>/dev/null || true)"
    [[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=no 2>/dev/null)" ]] && dirty=1
    unpushed="$(git -C "$ROOT" rev-list --count '@{u}..HEAD' 2>/dev/null || true)"
  fi
  if [[ -d "$COMFY/.git" ]]; then
    installed=1
    cver="$(comfy_label)"; comfy_commit="$(git -C "$COMFY" rev-parse --short HEAD 2>/dev/null || true)"
    want="$(resolve_ref 2>/dev/null || true)"
    if [[ "$COMFY_REF" == latest && -n "$want" && "$(git -C "$COMFY" describe --tags --exact-match 2>/dev/null || true)" != "$want" ]]; then newer="$want"; fi
  fi
  disk="$(df -B1 --output=avail "$ROOT" 2>/dev/null | tail -1 | tr -d ' ' || true)"
  models="$(du -sb "$DATA/models" 2>/dev/null | cut -f1 || true)"
  local queue_r="" queue_p=""
  if [[ "$st" == up ]]; then read -r queue_r queue_p <<<"$(queue_counts)"; fi
  python3 -I "$LIB_DIR/jsonout.py" schema:i=1 root:s="$ROOT" installed:b="$installed" \
    tool.tag:s="$tag" tool.commit:s="$commit" tool.branch:s="$branch" tool.dirty:b="$dirty" tool.unpushed:n="$unpushed" \
    comfy.version:s="$cver" comfy.commit:s="$comfy_commit" comfy.newer_release:s="$newer" \
    torch:s="$(dist_version torch)" sageattention:s="$(dist_version sageattention)" \
    state:s="$st" pid:n="$pid" rss_mib:n="$rss" port:i="$PORT" bind:s="$BIND_ADDR" \
    queue.running:n="$queue_r" queue.pending:n="$queue_p" \
    service.installed:b="$svc_inst" service.active:b="$svc_act" service.enabled:s="$svc_en" \
    disk_free_bytes:n="$disk" models_bytes:n="$models"
}
cmd_status() {
  if [[ "${1:-}" == --json ]]; then status_json; return 0; fi
  setup_env
  local watch=0 sec=3
  if [[ "${1:-}" == --watch ]]; then watch=1; [[ "${2:-}" =~ ^[0-9]+$ ]] && sec="$2"; fi
  if (( watch )); then status_watch "$sec"; return; fi
  local st ip
  st="$(server_state)"; ip="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
  log "ComfyUI"
  case "$st" in
    up) ok "running (pid $(server_pid)): http://${ip:-localhost}:$PORT  (binds $BIND_ADDR)" ;;
    starting) info "starting: process is up, server not answering yet" ;;
    *) info "not running: ./dgx-comfyui.sh run, or ./dgx-comfyui.sh service" ;;
  esac
  if command -v systemctl >/dev/null && [[ -f "$UNIT_FILE" ]]; then
    info "service: $(systemctl --user is-active "$UNIT" 2>/dev/null || true), $(systemctl --user is-enabled "$UNIT" 2>/dev/null || true)"
  else info "service: not installed (./dgx-comfyui.sh service)"; fi
  [[ -d "$COMFY/.git" ]] || { warn "not installed yet: ./dgx-comfyui.sh install"; return 0; }
  info "version: $(comfy_label); torch $(dist_version torch); sageattention $(dist_version sageattention || true)"
  local want; want="$(resolve_ref 2>/dev/null || true)"
  if [[ "$COMFY_REF" == latest && -n "$want" && "$(git -C "$COMFY" describe --tags --exact-match 2>/dev/null || true)" != "$want" ]]; then
    info "newer release available: $want  (./dgx-comfyui.sh update)"
  fi
  log "Machine"
  read -r t p c u <<<"$(gpu_sample)"
  info "gpu: ${t}°C, ${p}W, ${c}MHz, ${u}% busy"
  read -r used total <<<"$(mem_sample)"
  info "unified memory: $((used/1024))G used of $((total/1024))G"
  info "disk: $(df -h --output=avail,pcent "$ROOT" | tail -1 | awk '{print $1" free ("$2" used)"}')   models: $(du -sh "$DATA/models" 2>/dev/null | cut -f1)"
  if read -r r q <<<"$(queue_counts)"; [[ "$st" == up ]]; then info "queue: $r running, $q pending"; fi
  [[ -f "$CONF_FILE" ]] && info "config overrides: $(grep -vE '^\s*(#|$)' "$CONF_FILE" | tr '\n' ';')"
  return 0
}
status_watch() {
  local sec="$1" max=40 t p c u used total r q n=0 st
  local -a ht=() hp=() hc_=() hu=() hm=() hg=()
  command -v tput >/dev/null && tput civis 2>/dev/null || true
  trap 'tput cnorm 2>/dev/null || true; printf "\n"; exit 0' INT TERM
  trap 'tput cnorm 2>/dev/null || true' EXIT
  : >> "$STATE/status.log"
  while true; do
    read -r t p c u <<<"$(gpu_sample)"; read -r used total <<<"$(mem_sample)"
    st="$(server_state)"; r=0; q=0; [[ "$st" == up ]] && read -r r q <<<"$(queue_counts)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$t" "$p" "$c" "$u" "$used" "$r" "$q" >> "$STATE/status.log"
    ht+=("$t"); hp+=("${p%.*}"); hc_+=("$c"); hu+=("$u"); hm+=("$((used/1024))"); hg+=("$r")
    n=${#ht[@]}; if (( n > max )); then ht=("${ht[@]:1}"); hp=("${hp[@]:1}"); hc_=("${hc_[@]:1}"); hu=("${hu[@]:1}"); hm=("${hm[@]:1}"); hg=("${hg[@]:1}"); fi
    printf '\033[H\033[2J'
    printf '\033[1m dgx-comfyui  %s  %s · every %ss · Ctrl+C to leave\033[0m\n\n' "$st" "$(comfy_label)" "$sec"
    printf '  \033[36m── GPU ──────────────────────────────────────────────\033[0m\n'
    printf '  temp   %5s°C  %s\n' "$t" "$(spark "${ht[@]}")"
    printf '  power  %5sW   %s\n' "$p" "$(spark "${hp[@]}")"
    printf '  sm clk %5sMHz %s\n' "$c" "$(spark "${hc_[@]}")"
    printf '  busy   %5s%%   %s\n' "$u" "$(spark "${hu[@]}")"
    printf '  \033[36m── SYSTEM ───────────────────────────────────────────\033[0m\n'
    printf '  unified %4sG of %sG  %s\n' "$((used/1024))" "$((total/1024))" "$(spark "${hm[@]}")"
    printf '  \033[36m── QUEUE ────────────────────────────────────────────\033[0m\n'
    printf '  running %s, pending %s  %s\n' "$r" "$q" "$(spark "${hg[@]}")"
    printf '\n  samples are kept in .state/status.log (the trail to read after a silent reboot)\n'
    sleep "$sec"
  done
}

# ---- background service ----------------------------------------------------------------
cmd_service() {
  command -v systemctl >/dev/null || die "systemd is not available here"
  if [[ "${1:-}" == --disable ]]; then
    systemctl --user disable --now "$UNIT" 2>/dev/null || true
    rm -f "$UNIT_FILE"; systemctl --user daemon-reload
    ok "service removed; ComfyUI no longer starts at boot"; return 0
  fi
  [[ -x "$PY" && -d "$COMFY/.git" ]] || die "not installed yet: ./dgx-comfyui.sh install"
  local want
  want="[Unit]
Description=ComfyUI (dgx-comfyui, $ROOT)
After=network-online.target

[Service]
Type=simple
WorkingDirectory=$ROOT
ExecStart=$SELF run $*
Restart=on-failure
RestartSec=10
TimeoutStopSec=30

[Install]
WantedBy=default.target"
  if [[ -f "$UNIT_FILE" ]] && ! grep -qF "WorkingDirectory=$ROOT" "$UNIT_FILE"; then
    die "$UNIT already serves another install ($(sed -n 's/^WorkingDirectory=//p' "$UNIT_FILE")); run its './dgx-comfyui.sh service --disable' first"
  fi
  mkdir -p "${UNIT_FILE%/*}"
  if [[ -f "$UNIT_FILE" && "$(cat "$UNIT_FILE")" == "$want" ]]; then ok "unit already in place"
  else printf '%s\n' "$want" > "$UNIT_FILE.tmp" && mv -f "$UNIT_FILE.tmp" "$UNIT_FILE"; systemctl --user daemon-reload; did "unit written to $UNIT_FILE"; fi
  # Let it run without a login session, and start at boot (needs permission to enable lingering).
  if [[ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" != yes ]]; then
    if loginctl enable-linger "$USER" 2>/dev/null; then did "lingering enabled: starts at boot, survives logout"
    else warn "could not enable lingering; run: sudo loginctl enable-linger $USER (otherwise it stops when you log out)"; fi
  fi
  systemctl --user enable --now "$UNIT" 2>&1 | sed 's/^/  /' || true
  info "follow it: ./dgx-comfyui.sh logs -f    stop: ./dgx-comfyui.sh stop    status: ./dgx-comfyui.sh status"
}
cmd_stop() {
  if svc_active; then systemctl --user stop "$UNIT"; ok "service stopped (it starts again at boot; remove it with: service --disable)"; return 0; fi
  local pid i; pid="$(server_pid || true)"
  [[ -n "$pid" ]] || { info "not running"; return 0; }
  kill -TERM "$pid"
  for i in $(seq 1 30); do [[ -d "/proc/$pid" ]] || { ok "stopped"; return 0; }; sleep 1; done
  die "pid $pid ignored SIGTERM for 30 s; stop it by hand"
}
cmd_logs() {
  command -v journalctl >/dev/null && [[ -f "$UNIT_FILE" ]] || die "no service installed; the foreground 'run' prints to its own terminal. Set one up with: ./dgx-comfyui.sh service"
  exec journalctl --user -u "$UNIT" --no-pager "${@:--n 100}"
}

# ---- backup, restore -------------------------------------------------------------------
cmd_backup() {
  local with_out=0 file="" a T parts=() rc=0
  for a in "$@"; do case "$a" in --with-output) with_out=1 ;; -*) die "unknown option $a" ;; *) file="$a" ;; esac; done
  [[ -d "$DATA" ]] || die "nothing to back up yet"
  mkdir -p "$ROOT/backups"
  file="${file:-$ROOT/backups/comfyui-backup-$(date +%Y%m%d-%H%M%S).tgz}"
  T="$(mktemp -d)"; trap 'rm -rf "$T"' RETURN
  mkdir -p "$T/manifest"
  local d url sha
  for d in "$DATA"/custom_nodes/*/; do
    [[ -d "$d" ]] || continue
    if [[ -d "$d/.git" ]]; then
      url="$(git -C "$d" remote get-url origin 2>/dev/null || true)"; sha="$(git -C "$d" rev-parse HEAD 2>/dev/null || true)"
      printf '%s\t%s\t%s\n' "$(basename "$d")" "$url" "$sha" >> "$T/manifest/custom_nodes.tsv"
    else
      warn "custom node $(basename "$d") is not a git checkout; it is not in the backup"
    fi
  done
  : >> "$T/manifest/custom_nodes.tsv"
  find -L "$DATA/models" -type f -printf '%P\t%s\n' 2>/dev/null | sort > "$T/manifest/models.tsv" || true
  for d in data/user data/input; do [[ -d "$ROOT/$d" ]] && parts+=("$d"); done
  (( with_out )) && [[ -d "$DATA/output" ]] && parts+=(data/output)
  [[ -f "$CONF_FILE" ]] && parts+=("${CONF_FILE##*/}")
  tar -czf "$file.partial" -C "$ROOT" "${parts[@]}" -C "$T" manifest || rc=$?
  (( rc <= 1 )) || { rm -f "$file.partial"; die "tar failed ($rc)"; }   # 1 = a file changed while reading: fine on a live system
  mv -f "$file.partial" "$file"
  ok "backup: $file ($(du -h "$file" | cut -f1)); models are listed, not archived"
}
cmd_restore() {
  local file="" force=0 a
  for a in "$@"; do case "$a" in --force) force=1 ;; -*) die "unknown option $a" ;; *) file="$a" ;; esac; done
  [[ -f "$file" ]] || die "usage: ./dgx-comfyui.sh restore FILE [--force]"
  tar -tzf "$file" manifest/custom_nodes.tsv >/dev/null 2>&1 || die "$file is not a dgx-comfyui backup"
  if [[ ! -x "$PY" || ! -d "$COMFY/.git" ]]; then log "Installing first"; "$SELF" install || die "install failed"; fi
  setup_env; acquire_lock -x
  local T keep=(--skip-old-files); (( force )) && keep=(--overwrite)
  T="$(mktemp -d)"; trap 'rm -rf "$T"' RETURN
  log "Restoring content"
  tar -xzf "$file" "${keep[@]}" -C "$ROOT" --exclude='manifest' --exclude='manifest/*'
  tar -xzf "$file" -C "$T" manifest
  ok "workflows, settings, inputs and config restored$( (( force )) || echo ' (existing files kept; --force overwrites)')"
  log "Custom nodes"
  local name url sha
  while IFS=$'\t' read -r name url sha; do
    [[ -n "$name" ]] || continue
    if [[ -d "$DATA/custom_nodes/$name" ]]; then ok "$name already present"; continue; fi
    [[ -n "$url" ]] || { warn "$name has no remote; reinstall it from Manager"; continue; }
    if git clone -q "$url" "$DATA/custom_nodes/$name" && git -C "$DATA/custom_nodes/$name" checkout -q "$sha"; then
      [[ -f "$DATA/custom_nodes/$name/requirements.txt" ]] && uvpip install -q -r "$DATA/custom_nodes/$name/requirements.txt" -c "$STATE/constraints.txt" || true
      did "$name cloned at ${sha:0:9}"
    else bad "$name could not be cloned from $url"; fi
  done < "$T/manifest/custom_nodes.tsv"
  log "Models"
  local rel size missing=0 total=0
  while IFS=$'\t' read -r rel size; do
    [[ -n "$rel" ]] || continue
    if [[ ! -e "$DATA/models/$rel" ]]; then missing=$((missing+1)); total=$((total+size)); printf '  missing  models/%s  (%s)\n' "$rel" "$(numfmt --to=iec "$size")"; fi
  done < "$T/manifest/models.tsv"
  if (( missing )); then info "$missing model file(s), $(numfmt --to=iec "$total"), were not in the backup: copy or download them"; else ok "every model in the backup is present"; fi
  (( PROBLEMS == 0 )) || exit 1
}

summary() {
  echo
  if (( PROBLEMS > 0 )); then printf '\033[1;31m%d problem(s) above.\033[0m\n' "$PROBLEMS" >&2
  elif [[ "$1" == doctor ]]; then printf '\033[1;32mHealthy.\033[0m\n'
  elif (( CHANGES == 0 )); then printf '\033[1;32mNothing to do: already up to date.\033[0m\n'
  else printf '\033[1;32mDone: %d change(s). Start it with ./dgx-comfyui.sh run\033[0m\n' "$CHANGES"; fi
  (( PROBLEMS == 0 )) || exit 1
}

usage() { sed -n '2,14p' "$SELF" | sed 's/^# \{0,1\}//'; }
case "${1:-}" in
  install)  shift; cmd_install "$@" ;;
  update)   shift; cmd_update "$@" ;;
  run)      shift; cmd_run "$@" ;;
  doctor)   shift; cmd_doctor "$@" ;;
  status)   shift; cmd_status "$@" ;;
  service)  shift; cmd_service "$@" ;;
  stop)     shift; cmd_stop "$@" ;;
  logs)     shift; cmd_logs "$@" ;;
  backup)   shift; setup_env; cmd_backup "$@" ;;
  recipe)   shift; cmd_recipe "$@" ;;
  tune)     shift; cmd_tune "$@" ;;
  restore)  shift; cmd_restore "$@" ;;
  ""|-h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
