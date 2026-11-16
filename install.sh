#!/usr/bin/env bash
# One-line installer for spark-comfyui-uv (ComfyUI on a DGX Spark, in one folder).
#
#   curl -fsSL https://raw.githubusercontent.com/bmaltais/spark-comfyui-uv/main/install.sh | bash
#
# Creates ./ComfyUI in the folder you run it from and installs there (or updates it, if it is
# already there). Pick another folder with:  ... | bash -s -- /path/to/folder
# Safe to run again: it updates this tool, then re-checks everything and fixes only what differs.
# It never uses sudo. Read it first if you like; it only clones the repo and runs ./spark-comfyui.sh install.
set -euo pipefail

REPO="${SPARK_COMFYUI_REPO:-https://github.com/bmaltais/spark-comfyui-uv.git}"
DIR="${1:-${SPARK_COMFYUI_DIR:-}}"
if [[ -z "$DIR" ]]; then
  if [[ -f "$PWD/spark-comfyui.sh" && -d "$PWD/.git" ]]; then
    DIR="$PWD"            # already inside a checkout: update it in place
  else
    DIR="$PWD/ComfyUI"
  fi
fi

die() { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }
say() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }

command -v git  >/dev/null || die "git is missing: sudo apt install git"
command -v curl >/dev/null || die "curl is missing: sudo apt install curl"

if [[ -d "$DIR/.git" ]]; then
  say "Updating the installer in $DIR"
  if [[ -n "$(git -C "$DIR" status --porcelain --untracked-files=no)" ]]; then
    printf '\033[1;33m[warn]\033[0m local edits in %s: leaving the installer files as they are\n' "$DIR" >&2
  else
    git -C "$DIR" pull --ff-only -q || die "could not fast-forward $DIR; resolve it with git, then re-run"
  fi
elif [[ -e "$DIR" && -n "$(ls -A "$DIR" 2>/dev/null)" ]]; then
  die "$DIR exists and is not an empty folder or a spark-comfyui-uv checkout; pass another folder"
else
  say "Cloning into $DIR"
  git clone -q "$REPO" "$DIR"
fi

cd "$DIR"
# Stdin may be the script itself when piped; the installer must not read from it.
exec ./spark-comfyui.sh install </dev/null
