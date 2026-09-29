#!/usr/bin/env python3
"""Check one offline result: check_offline_result.py <results_dir> <bag> <container_exit_code>.

Passes (exit 0) only if the container exited with 0, <bag>.jsonl and <bag>_summary.json exist,
the summary says the bag was read completely and the JSONL has one valid line per processed frame.
"""
import json
import os
import sys


def check(res_dir, bag, rc):
    jsonl = os.path.join(res_dir, bag + ".jsonl")
    summ = os.path.join(res_dir, bag + "_summary.json")
    if rc != 0:
        return f"container exit code {rc}"
    if not os.path.isfile(jsonl) or not os.path.isfile(summ):
        return "result files missing"
    with open(summ, encoding="utf-8") as fh:
        s = json.load(fh)
    if not s.get("complete"):
        return (f"incomplete: read {s.get('messages_read')} of {s.get('messages_in_metadata')} messages, "
                f"processed {s.get('frames_processed')} of {s.get('frames_expected')} frames")
    with open(jsonl, encoding="utf-8") as fh:
        n = sum(1 for line in fh if line.strip() and json.loads(line))
    if n != s["frames_processed"]:
        return f"JSONL has {n} records, summary says {s['frames_processed']}"
    return None


if __name__ == "__main__":
    res_dir, bag, rc = sys.argv[1], sys.argv[2], int(sys.argv[3])
    try:
        err = check(res_dir, bag, rc)
    except Exception as exc:  # unreadable / truncated files
        err = f"cannot read results: {exc}"
    print(f"{bag}: {'OK' if err is None else 'FAILED - ' + err}", file=sys.stderr if err else sys.stdout)
    sys.exit(0 if err is None else 1)
