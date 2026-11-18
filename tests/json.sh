#!/usr/bin/env bash
# Tests for the --json output (status, doctor, recipe list|show|check) and for the text preflight that
# shares its probe. Runs on any machine: it builds a throwaway install folder and stubs wget, so it
# needs no GPU, no network and no real install.   Usage: tests/json.sh
set -uo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
FAILS=0; N=0

pass() { N=$((N+1)); printf 'ok - %s\n' "$1"; }
fail() { N=$((N+1)); FAILS=$((FAILS+1)); printf 'not ok - %s\n   %s\n' "$1" "$2"; }
# check NAME PY-EXPRESSION JSON: the expression sees the parsed JSON as `d`
check() {
  local name="$1" expr="$2" json="$3" out
  if out="$(python3 -I -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if ('"$expr"') else 1)' <<<"$json" 2>&1)"; then pass "$name"
  else fail "$name" "$expr  on  ${json:0:300} $out"; fi
}

# A fake install: this tool copied in, a venv whose python is the system one, a ComfyUI checkout marker.
mk_install() {
  local d="$1"
  mkdir -p "$d/.venv/bin" "$d/ComfyUI/.git" "$d/stub" "$d/recipes"
  cp "$SRC/dgx-comfyui.sh" "$d/"; cp -r "$SRC/lib" "$d/"
  ln -s "$(command -v python3)" "$d/.venv/bin/python"
  # wget stub: prints the HTTP answer chosen by the URL (…/ok, …/gated, …/gone, …/short), nothing else.
  cat > "$d/stub/wget" <<'STUB'
#!/bin/bash
url="${@: -1}"
case "$url" in
  */ok)    printf '  HTTP/1.1 200 OK\n  Content-Length: 10\n' >&2 ;;
  */short) printf '  HTTP/1.1 200 OK\n  Content-Length: 99\n' >&2 ;;
  */gated) printf '  HTTP/1.1 401 Unauthorized\n' >&2; exit 8 ;;
  */gone)  printf '  HTTP/1.1 404 Not Found\n' >&2; exit 8 ;;
  *)       exit 4 ;;
esac
STUB
  chmod +x "$d/stub/wget"
}
# A recipe with models named have / ok / gated / gone / short, each 10 bytes.
mk_recipe() {
  python3 -I - "$1" <<'PY'
import json, sys
sha = "0" * 64
def m(f, u): return {"file": f, "dest": "loras", "url": u, "size": 10, "sha256": sha}
r = {"schema": 1, "name": "Test recipe", "description": "d", "nodes": ["some-node"], "run_flags": ["--x"],
     "models": [m("have.bin", "https://huggingface.co/a/ok"), m("ok.bin", "https://example.test/ok"),
                m("gated.bin", "https://huggingface.co/a/gated"), m("gone.bin", "https://example.test/gone"),
                m("short.bin", "https://example.test/short")]}
json.dump(r, open(sys.argv[1], "w"))
PY
}

# ---- status --json, not installed --------------------------------------------------------
U="$T/bare"; mkdir -p "$U"; cp "$SRC/dgx-comfyui.sh" "$U/"; cp -r "$SRC/lib" "$U/"
out="$("$U/dgx-comfyui.sh" status --json)"
check "status --json before install: schema, not installed, stopped" \
  'd["schema"]==1 and d["installed"] is False and d["state"]=="stopped" and d["pid"] is None and d["root"]=="'"$U"'"' "$out"
if [[ ! -e "$U/.state" && ! -e "$U/.cache" ]]; then pass "status --json writes nothing in the folder"
else fail "status --json writes nothing in the folder" "$(ls -A "$U")"; fi

# ---- status --json, installed, with a tool checkout --------------------------------------
I="$T/inst"; mk_install "$I"
git -C "$I" init -q -b main && git -C "$I" -c user.name=t -c user.email=t@t add dgx-comfyui.sh lib \
  && git -C "$I" -c user.name=t -c user.email=t@t commit -q -m init && git -C "$I" tag v9.9.9
out="$("$I/dgx-comfyui.sh" status --json)"
check "status --json: installed, clean tool checkout on its tag" \
  'd["installed"] and d["tool"]["tag"]=="v9.9.9" and d["tool"]["dirty"] is False and d["tool"]["branch"]=="main"' "$out"
echo "# edit" >> "$I/dgx-comfyui.sh"
out="$("$I/dgx-comfyui.sh" status --json)"
check "status --json: a local edit shows as dirty" 'd["tool"]["dirty"] is True' "$out"
git -C "$I" checkout -q -- dgx-comfyui.sh

# a conf value must come out as written, quotes and all
printf 'BIND_ADDR = 127.0.0.1\nPORT = 9001\n' > "$I/dgx-comfyui.conf"
out="$("$I/dgx-comfyui.sh" status --json)"
check "status --json reads dgx-comfyui.conf" 'd["bind"]=="127.0.0.1" and d["port"]==9001' "$out"

# ---- doctor --json -----------------------------------------------------------------------
out="$(SPARK_ALLOW_NON_SPARK=1 "$I/dgx-comfyui.sh" doctor --json 2>/dev/null)"; rc=$?
check "doctor --json: a report with levels, and a verdict" \
  'd["schema"]==1 and isinstance(d["items"],list) and len(d["items"])>3 and all(i["level"] in ("log","ok","did","info","warn","bad") for i in d["items"]) and d["healthy"]==(d["problems"]==0)' "$out"
if [[ $rc -eq 0 ]]; then pass "doctor --json exits 0 whatever the verdict"; else fail "doctor --json exits 0 whatever the verdict" "rc=$rc"; fi

# ---- recipes -----------------------------------------------------------------------------
mk_recipe "$I/recipes/test.json"; echo '{ not json' > "$I/recipes/broken.json"
run() { PATH="$I/stub:$PATH" "$I/dgx-comfyui.sh" "$@"; }
out="$(run recipe list --json)"
check "recipe list --json: a valid and a broken recipe" \
  '[ (r["name"],r["valid"]) for r in d["recipes"] ]==[("broken",False),("test",True)] and d["recipes"][1]["models"]==5 and d["recipes"][0]["error"]' "$out"
out="$(run recipe show test --json)"
check "recipe show --json: models, nodes, flags and the total" \
  'd["title"]=="Test recipe" and len(d["models"])==5 and d["nodes"]==[{"id":"some-node"}] and d["flags"]==["--x"] and d["total_bytes"]==50' "$out"

mkdir -p "$I/data/models/loras"; head -c 10 /dev/zero > "$I/data/models/loras/have.bin"; head -c 3 /dev/zero > "$I/data/models/loras/short.bin"
out="$(run recipe check test --json)"
check "recipe check --json: have, missing and a wrong size on disk" \
  'dict((m["file"],m["state"]) for m in d["models"])=={"have.bin":"have","ok.bin":"missing","gated.bin":"missing","gone.bin":"missing","short.bin":"size"} and d["have"]==1 and d["want"]==5' "$out"
check "recipe check --json: each source's verdict" \
  'dict((m["file"],m.get("source",{}).get("problem")) for m in d["models"])=={"have.bin":None,"ok.bin":"","gated.bin":"gated","gone.bin":"gone","short.bin":None}' "$out"
check "recipe check --json: the gated one names its token; downloads and readiness add up" \
  '[m["source"]["detail"] for m in d["models"] if m["file"]=="gated.bin"][0][0]=="HF_TOKEN" and d["download_bytes"]==30 and d["ready"] is False and d["nodes"][0]["installed"] is False' "$out"
out="$(run recipe check test --json --verify)"
check "recipe check --json --verify: a right-sized file with the wrong hash is flagged" \
  '[m["state"] for m in d["models"] if m["file"]=="have.bin"]==["hash"]' "$out"

if run recipe install test --json >/dev/null 2>&1; then fail "--json is refused on install" "it ran"; else pass "--json is refused on install"; fi

# ---- the text check still says what it said ----------------------------------------------
txt="$(run recipe check test 2>&1)"
for want in '[GATED]' '[GONE]' '[have]' '[want]' '[SIZE]'; do
  if grep -qF "$want" <<<"$txt"; then pass "recipe check text still shows $want"; else fail "recipe check text still shows $want" "$txt"; fi
done

printf '\n%d tests, %d failed\n' "$N" "$FAILS"
exit $(( FAILS > 0 ))
