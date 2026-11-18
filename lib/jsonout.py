"""Print one JSON object from KEY:TYPE=VALUE arguments, so the shell never builds JSON by hand.

    jsonout.py schema:i=1 installed:b=1 comfy.version:s=v0.45.4 pid:n=

TYPE is s (string), i (integer), b (1 or true is true, anything else false) or n (an
integer, or null when the value is empty). A dotted key nests: `a.b:s=x` is {"a": {"b": "x"}}.
Values come from argv only, so a path or a name with quotes or newlines cannot break the output.
"""
import json
import sys


def convert(kind, value):
    if kind == "s":
        return value
    if kind == "i":
        return int(value)
    if kind == "b":
        return value in ("1", "true", "yes")
    if kind == "n":
        return int(value) if value != "" else None
    raise SystemExit("jsonout: unknown type %r" % kind)


def main(argv):
    out = {}
    for arg in argv:
        head, sep, value = arg.partition("=")
        key, _, kind = head.rpartition(":")
        if not sep or not key:
            raise SystemExit("jsonout: bad argument %r (want KEY:TYPE=VALUE)" % arg)
        node = out
        *path, leaf = key.split(".")
        for part in path:
            node = node.setdefault(part, {})
        node[leaf] = convert(kind, value)
    json.dump(out, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")


main(sys.argv[1:])
