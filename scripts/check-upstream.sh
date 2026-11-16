#!/usr/bin/env bash
# Read-only: report how bjarkebolding/spark-comfyui has moved since the commit in .upstream, and which
# of our pins and ported files differ from it. Exit 0 = nothing to port, 10 = drift to review.
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/.upstream"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
git clone -q "$UPSTREAM_REPO" "$T/up"
UP="$T/up"; HEAD_SHA="$(git -C "$UP" rev-parse HEAD)"
git -C "$UP" cat-file -e "$SYNCED^{commit}" 2>/dev/null || { echo "SYNCED in .upstream is not a commit of $UPSTREAM_REPO" >&2; exit 2; }
drift=0

echo "synced: ${SYNCED:0:9}   upstream now: ${HEAD_SHA:0:9}"
if [[ "$SYNCED" == "$HEAD_SHA" ]]; then echo "Upstream has not moved."; else
  echo; echo "== upstream commits since sync"; git -C "$UP" log --oneline "$SYNCED..HEAD"
fi

echo; echo "== pins: upstream Dockerfile vs our defaults (spark-comfyui.sh)"
pin() {  # upstream ARG name, our key
  local up ours
  up="$(sed -n "s/^ARG $1=//p" "$UP/container/Dockerfile" | head -1)"
  ours="$(grep -oE "\\\$\\{$2:=[^}]*\\}" "$ROOT/spark-comfyui.sh" | head -1 | sed -E "s/^.*:=//; s/\\}\$//")"
  if [[ "$up" == "$ours" ]]; then printf '  same     %-22s %s\n' "$2" "${ours:0:70}"
  else drift=1; printf '  DIFFERS  %-22s ours=%s\n           %-22s upstream=%s\n' "$2" "${ours:0:90}" "" "${up:0:90}"; fi
}
pin TORCH_VERSION TORCH_VERSION; pin TORCHVISION_VERSION TORCHVISION_VERSION; pin TORCHAUDIO_VERSION TORCHAUDIO_VERSION
pin TORCH_INDEX TORCH_INDEX; pin SAGE_REF SAGE_REF; pin TORCH_CUDA_ARCH_LIST SAGE_ARCH; pin ORT_WHEEL_URL ORT_WHEEL_URL
echo "  CUDA base image upstream: $(sed -n 's/^ARG CUDA_IMAGE=//p' "$UP/container/Dockerfile" | head -1 | cut -c1-48)  (host nvcc must match its 13.0.x line)"

echo; echo "== ported files: changed upstream since sync?"
for f in mods/unified-memory-free/transform.py container/gb10.sh container/entrypoint.sh \
         container/manager-config.py container/Dockerfile; do
  if git -C "$UP" diff --quiet "$SYNCED" HEAD -- "$f" 2>/dev/null; then printf '  unchanged  %s\n' "$f"
  else drift=1; printf '  CHANGED    %s   (git diff %s..%s -- %s)\n' "$f" "${SYNCED:0:9}" "${HEAD_SHA:0:9}" "$f"; fi
done
echo; echo "== our copy of the memory patch vs upstream's current one"
if diff -q "$UP/mods/unified-memory-free/transform.py" "$ROOT/lib/mem_unified_transform.py" >/dev/null; then echo "  identical"
else drift=1; echo "  DIFFERS"; fi

if (( drift )); then printf '\nDrift found. Detail: git clone %s and diff %s..%s\n' "$UPSTREAM_REPO" "${SYNCED:0:9}" "${HEAD_SHA:0:9}"; exit 10; fi
echo; echo "Nothing to port."
