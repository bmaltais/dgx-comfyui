"""Recipe reports as JSON (schema 1), built from the tagged rows lib/recipe_read.py emits.

    recipe_json.py list RECIPES_DIR             every recipe, a broken one marked invalid
    recipe_json.py show                         rows on stdin -> the recipe as data
    recipe_json.py state MODELS_DIR VERIFY      rows on stdin -> STATE rows (have, missing, size, hash)
    recipe_json.py check                        rows + STATE + PROBE + MISSINGNODE rows on stdin -> the check report

Rows are tab separated. recipe_read.py's META/NODE/FLAG/MODEL rows are the base. `state` adds
    STATE  file  dest  state  actual_size  url  size  sha256
the shell adds, for each file it still has to fetch,
    PROBE  file  kind  a  b            (what lib/recipes.sh `_recipe_probe` said)
and, for each node that is not installed,
    MISSINGNODE  id
"""
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def rows(stream):
    for line in stream:
        line = line.rstrip("\n")
        if line:
            yield line.split("\t")


def read_recipe(parsed):
    """The base object from META/NODE/FLAG/MODEL rows."""
    r = {"schema": 1, "name": "", "title": "", "description": "", "notes": "", "run_note": "",
         "workflow": "", "nodes": [], "flags": [], "models": [], "total_bytes": 0}
    for row in parsed:
        tag = row[0]
        if tag == "META" and len(row) >= 3:
            key, value = row[1], row[2]
            if key == "name":
                r["title"] = value
            elif key in r:
                r[key] = value
        elif tag == "NODE":
            r["nodes"].append({"id": row[1]})
        elif tag == "FLAG":
            r["flags"].append(row[1])
        elif tag == "MODEL":
            r["models"].append({"file": row[1], "dest": row[2], "url": row[3], "size": int(row[4]), "sha256": row[5]})
            r["total_bytes"] += int(row[4])
    return r


def cmd_list(directory):
    out = {"schema": 1, "recipes": []}
    try:
        names = sorted(n for n in os.listdir(directory) if n.endswith(".json"))
    except OSError:
        names = []
    for n in names:
        path = os.path.join(directory, n)
        proc = subprocess.run([sys.executable, "-I", os.path.join(HERE, "recipe_read.py"), path],
                              capture_output=True, text=True)
        entry = {"name": n[:-5], "valid": proc.returncode == 0, "title": "", "description": "",
                 "models": 0, "total_bytes": 0, "nodes": 0}
        if proc.returncode == 0:
            r = read_recipe(row.split("\t") for row in proc.stdout.splitlines() if row)
            entry.update(title=r["title"], description=r["description"], models=len(r["models"]),
                         total_bytes=r["total_bytes"], nodes=len(r["nodes"]))
        else:
            entry["error"] = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "invalid"
        out["recipes"].append(entry)
    return out


def cmd_state(models_dir, verify):
    for row in rows(sys.stdin):
        if row[0] != "MODEL":
            continue
        file, dest, url, size, sha = row[1:6]
        target = os.path.join(models_dir, dest, file)
        if not os.path.isfile(target):
            state, actual = "missing", 0
        else:
            actual = os.path.getsize(target)
            if actual != int(size):
                state = "size"
            elif verify == "1" and sha256_of(target) != sha:
                state = "hash"
            else:
                state = "have"
        print("\t".join(["STATE", file, dest, state, str(actual), url, size, sha]))


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cmd_check(everything):
    base = [r for r in everything if r[0] in ("META", "NODE", "FLAG", "MODEL")]
    r = read_recipe(base)
    state = {(x[1], x[2]): x for x in everything if x[0] == "STATE"}
    probe = {x[1]: x[2:] for x in everything if x[0] == "PROBE"}
    missing_nodes = {x[1] for x in everything if x[0] == "MISSINGNODE"}
    have = 0
    download = 0
    for m in r["models"]:
        s = state.get((m["file"], m["dest"]))
        m["state"] = s[3] if s else "missing"
        m["actual_size"] = int(s[4]) if s else 0
        if m["state"] == "have":
            have += 1
        if m["state"] == "missing":
            download += m["size"]
            p = probe.get(m["file"])
            if p:
                kind = p[0]
                m["source"] = {"ok": kind == "ok", "problem": "" if kind == "ok" else kind,
                               "detail": [v for v in p[1:] if v]}
    for n in r["nodes"]:
        n["installed"] = n["id"] not in missing_nodes
    r["have"] = have
    r["want"] = len(r["models"])
    r["download_bytes"] = download
    r["ready"] = all(m.get("source", {"ok": True})["ok"] for m in r["models"]) and \
        all(m["state"] in ("have", "missing") for m in r["models"])
    return r


def emit(obj):
    json.dump(obj, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")


def main(argv):
    mode = argv[0] if argv else ""
    if mode == "list":
        emit(cmd_list(argv[1]))
    elif mode == "show":
        emit(read_recipe(list(rows(sys.stdin))))
    elif mode == "state":
        cmd_state(argv[1], argv[2])
    elif mode == "check":
        emit(cmd_check(list(rows(sys.stdin))))
    else:
        raise SystemExit("recipe_json: unknown mode %r" % mode)


main(sys.argv[1:])
