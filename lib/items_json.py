"""Turn the doctor's collected report lines into JSON.

    items_json.py ITEMS_FILE PROBLEMS CHANGES

ITEMS_FILE has one `level<TAB>text` line per message (log, ok, did, info, warn, bad). The
verdict is in the JSON (`healthy`), not in the exit status, so a caller always gets the report.
"""
import json
import sys

items = []
with open(sys.argv[1], encoding="utf-8") as fh:
    for line in fh:
        level, _, text = line.rstrip("\n").partition("\t")
        items.append({"level": level, "text": text})
problems = int(sys.argv[2])
json.dump({"schema": 1, "healthy": problems == 0, "problems": problems,
           "changes": int(sys.argv[3]), "items": items}, sys.stdout, ensure_ascii=False)
sys.stdout.write("\n")
