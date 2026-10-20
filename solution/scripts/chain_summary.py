#!/usr/bin/env python3
"""Summary of one eval_chain.sh run, optionally diffed frame by frame against a reference run.

    chain_summary.py /home/gag/i.mos/work/eval/<tag> [--ref /home/gag/i.mos/work/eval/v1-gpu] [--json out.json]

Prints Markdown: real bags (detected frames, first detection), Falcon phases, the organisers'
fake-object bag (wall / plate windows), new_data episodes, the organisers' synthetic bags,
synthetic scoreboards (via scoreboard.py) and processing time.  Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys

# Falcon (falcon_test_second_dark): the person walks 150 -> 100 m, stands at 100 m, walks 100 -> 4 m.  He is in
# view from frame 2723 at 154.5 m (D2 near-axis groups in two body layers, h 1.0 / 1.5 m, on the back-extrapolated
# walk within 0.1 m; nothing near the axis beyond 120 m in 2550-2722), so 2723-2729 is the far approach and anything
# detected before 2723 is a false positive.  The walk row keeps 2730-3180 (comparable with the v1 numbers).
FALCON = [("before 2723 (FP)", 0, 2722), ("approach 154 m", 2723, 2729), ("walk 150->100 m", 2730, 3180), ("stand 100 m", 3181, 3690),
          ("walk 100->4 m", 3691, 4400), ("after 4400", 4401, 10 ** 9)]
# cloud_with_fake_obj: objects the organisers inserted (work/specs/11_fake_style.md §4); every one intrudes into
# our gauge.  Frames 804-1509 carry no inserted object: any detection there is a false positive.
FAKE = [("wall 0-231 (98.7 m->0)", 0, 231), ("0.3 m cubes 236-532", 236, 532), ("side boxes 435-633", 435, 633),
        ("overhead boxes 534-682", 534, 682), ("plate 682-744 (<=107 m)", 682, 744), ("hanging pole 710-803", 710, 803),
        ("CLEAN 804-1509 (FP)", 804, 1509)]
SYNTH_SETS = ["val_hesai_insert", "val_hesai_insert_tunnel", "val_hesai_param", "val_hesai_curve",
              "val_falcon_param", "val_dust", "val_hesai_far_dark"]
SYN_ROOT = "/home/gag/i.mos/synthetic"


def load(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def detected(rows):
    return [r for r in rows if r.get("status") == "detected"]


def episodes(frames, gap=3):
    eps = []
    for f in sorted(frames):
        if eps and f - eps[-1][1] <= gap:
            eps[-1][1] = f
        else:
            eps.append([f, f])
    return eps


def ms_stats(rows):
    ms = sorted(r["processing_ms"] for r in rows if r.get("processing_ms") is not None)
    if not ms:
        return "-"
    return "%.0f / %.0f" % (ms[len(ms) // 2], ms[min(len(ms) - 1, int(0.95 * len(ms)))])


def dist(r):
    d = r.get("distance_m")
    return "%.1f m" % d if d is not None else "?"


def diff(rows, ref_rows):
    a = {r["frame_index"]: r.get("status") for r in rows}
    b = {r["frame_index"]: r.get("status") for r in ref_rows}
    gained = sorted(f for f in a if a[f] == "detected" and b.get(f) != "detected")
    lost = sorted(f for f in b if b[f] == "detected" and a.get(f) != "detected")
    return gained, lost


def d3_off_note(rows, min_tail=50):
    """'D3 OFF from frame X' when the run had D3 timings and then none for the last min_tail+ frames
    (pipeline d3_max_ms: three slow D3 calls disable D3 for the rest of the run - GPU contention on a shared
    machine, not a property of the code; such a run is not comparable with a D3-on reference)."""
    last_on = None
    for r in rows:
        if "d3" in ((r.get("detector") or {}).get("timings_ms") or {}):
            last_on = r["frame_index"]
    if last_on is None or not rows or rows[-1]["frame_index"] - last_on < min_tail:
        return ""
    return " **D3 OFF from frame %d**" % (last_on + 1)


def fmt_frames(fs, n=12):
    eps = episodes(fs, gap=1)
    s = ", ".join("%d" % a if a == b else "%d-%d" % (a, b) for a, b in eps[:n])
    return s + (" ..." if len(eps) > n else "")


# ---------------------------------------------------------------------------------------------------------
# Corridor stability (WP-C): how much the verified corridor ("track checked to X m") and its centre-line
# move from one frame to the next.  A jump = consecutive frames (no reset between them) whose verified_to_m
# differs by more than STAB_JUMP_M.  Centre-line jitter = |l_c(d)_k - l_c(d)_k-1| at d = 50 / 100 m, counted
# only where both frames verify the corridor to at least d (that is what RViz draws).
STAB_JUMP_M = 20.0


def _pct(vals, q):
    if not vals:
        return None
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(q * len(vals)))]


def corridor_stability(rows):
    pairs = jumps = 0
    dv, ver = [], []
    jit = {50: [], 100: []}
    prev = None
    for r in rows:
        det = r.get("detector") or {}
        v = det.get("verified_to_m")
        cur = det.get("curve") or {}
        lc = None
        if cur.get("ok") and cur.get("l0") is not None:
            lc = (cur["l0"], cur["a"], cur["b"])
        if v is not None:
            ver.append(v)
        if (prev is not None and v is not None and prev[1] is not None and not r.get("reset")
                and r.get("frame_index") == prev[0] + 1):
            pairs += 1
            dv.append(abs(v - prev[1]))
            if abs(v - prev[1]) > STAB_JUMP_M:
                jumps += 1
            if lc is not None and prev[2] is not None:
                for d in jit:
                    if v >= d and prev[1] >= d:
                        f = lambda c: c[0] + c[1] * d + c[2] * d * d
                        jit[d].append(abs(f(lc) - f(prev[2])))
        prev = (r.get("frame_index"), v, lc)
    if not ver:
        return None
    return {"frames": len(ver), "pairs": pairs, "jumps": jumps, "jump_frac": jumps / pairs if pairs else 0.0,
            "dv_p95": _pct(dv, 0.95), "v_p5": _pct(ver, 0.05), "v_p50": _pct(ver, 0.5),
            "ge100": sum(1 for x in ver if x >= 100.0) / len(ver),
            "lc_jit50_p95": _pct(jit[50], 0.95), "lc_jit100_p95": _pct(jit[100], 0.95)}


def _fmt_stab(s):
    if s is None:
        return "| - | - | - | - | - | - |"
    f = lambda x, n=1: "-" if x is None else ("%." + str(n) + "f") % x
    return "| %d | %d (%.1f %%) | %s | %s / %s | %.0f %% | %s / %s |" % (
        s["frames"], s["jumps"], 100.0 * s["jump_frac"], f(s["dv_p95"]), f(s["v_p5"]), f(s["v_p50"]),
        100.0 * s["ge100"], f(s["lc_jit50_p95"], 2), f(s["lc_jit100_p95"], 2))


def stability_section(run, ref, p, js):
    """Per recording (synthetic sets: all scenes of the set together; Falcon also per phase)."""
    groups = []    # (label, [relative jsonl paths without extension], frame range or None)
    for g in ("real", "extra", "newdata", "orgsynth"):
        d = os.path.join(run, g)
        if os.path.isdir(d):
            groups += [("%s/%s" % (g, f[:-6]), ["%s/%s" % (g, f[:-6])], None) for f in sorted(os.listdir(d)) if f.endswith(".jsonl")]
    if os.path.exists(os.path.join(run, "falcon/falcon_test_second_dark.jsonl")):
        groups.append(("falcon (all)", ["falcon/falcon_test_second_dark"], None))
        groups += [("falcon " + name, ["falcon/falcon_test_second_dark"], (lo, hi)) for name, lo, hi in FALCON[:4]]
    syn = os.path.join(run, "synth")
    if os.path.isdir(syn):
        for s in sorted(os.listdir(syn)):
            fs = sorted(f[:-6] for f in os.listdir(os.path.join(syn, s)) if f.endswith(".jsonl"))
            if fs:
                groups.append(("synth/" + s, ["synth/%s/%s" % (s, f) for f in fs], None))
    if not groups:
        return
    p("## Corridor stability (verified_to_m and centre-line, frame to frame)\n")
    p("jump = consecutive frames whose verified range differs by > %.0f m; l_c jitter = p95 of the centre-line's lateral "
      "move at 50 / 100 m between frames (only where both frames verify that far)\n" % STAB_JUMP_M)
    hdr = "| recording | frames | jumps > %.0f m | p95 dv, m | verified p5 / p50, m | >= 100 m | l_c jitter 50 / 100 m |" % STAB_JUMP_M
    p(hdr + (" ref jumps | ref >= 100 m |" if ref else ""))
    p("|---|---:|---:|---:|---|---:|---|" + ("---:|---:|" if ref else ""))
    out = {}

    def stat_of(base, rels, rng):
        agg = None
        for rel in rels:
            rows = load(os.path.join(base, rel + ".jsonl"))
            if rng:
                rows = [r for r in rows if rng[0] <= r.get("frame_index", -1) <= rng[1]]
            s = corridor_stability(rows)
            if s is None:
                continue
            if agg is None:
                agg = dict(s, _ver=[], _n=0)
            else:   # aggregate counts; percentiles of the set are taken as the max over scenes (conservative)
                for k in ("frames", "pairs", "jumps"):
                    agg[k] += s[k]
                for k in ("dv_p95", "lc_jit50_p95", "lc_jit100_p95"):
                    if s[k] is not None:
                        agg[k] = s[k] if agg[k] is None else max(agg[k], s[k])
                agg["v_p5"] = min(agg["v_p5"], s["v_p5"])
                agg["v_p50"] = None
                agg["ge100"] = (agg["ge100"] * (agg["frames"] - s["frames"]) + s["ge100"] * s["frames"]) / agg["frames"]
            agg["jump_frac"] = agg["jumps"] / agg["pairs"] if agg["pairs"] else 0.0
        if agg is not None:
            agg.pop("_ver", None)
            agg.pop("_n", None)
        return agg

    for label, rels, rng in groups:
        s = stat_of(run, rels, rng)
        line = "| %s " % label + _fmt_stab(s)
        if ref:
            r = stat_of(ref, rels, rng)
            line += (" %d (%.1f %%) | %.0f %% |" % (r["jumps"], 100.0 * r["jump_frac"], 100.0 * r["ge100"])) if r else " - | - |"
            out[label] = {"run": s, "ref": r}
        else:
            out[label] = {"run": s}
        p(line)
    p("")
    js["stability"] = out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ref")
    ap.add_argument("--json")
    a = ap.parse_args()
    ref = a.ref if a.ref and os.path.isdir(a.ref) and os.path.abspath(a.ref) != os.path.abspath(a.run) else None
    out, js = [], {}
    p = out.append
    p("# Chain summary: %s%s\n" % (os.path.basename(a.run.rstrip("/")),
                                   " (diff vs %s)" % os.path.basename(ref.rstrip("/")) if ref else ""))

    def section_diff(rel, rows):
        if not ref:
            return ""
        g, l = diff(rows, load(os.path.join(ref, rel + ".jsonl")))
        if not g and not l:
            return "same as ref"
        return "+%d [%s] / -%d [%s]" % (len(g), fmt_frames(g), len(l), fmt_frames(l))

    real_dir = os.path.join(a.run, "real")
    if os.path.isdir(real_dir):
        p("## Real bags\n\n| bag | frames | detected | first | ms med/p95 | vs ref |\n|---|---:|---:|---|---|---|")
        for f in sorted(os.listdir(real_dir)):
            if not f.endswith(".jsonl"):
                continue
            rel = "real/" + f[:-6]
            rows = load(os.path.join(a.run, rel + ".jsonl")); d = detected(rows)
            first = "%d (%s)" % (d[0]["frame_index"], dist(d[0])) if d else "-"
            p("| %s | %d | %d | %s | %s | %s%s |" % (f[:-6], len(rows), len(d), first, ms_stats(rows), section_diff(rel, rows), d3_off_note(rows)))
            js[rel] = {"frames": len(rows), "detected": len(d), "first": d[0]["frame_index"] if d else None}
        p("")

    rel = "falcon/falcon_test_second_dark"
    rows = load(os.path.join(a.run, rel + ".jsonl"))
    if rows:
        d = detected(rows); fr = [r["frame_index"] for r in d]
        first = next((r for r in d if r["frame_index"] >= 2723), None)
        p("## Falcon (%d frames, %d detected, ms %s)\n" % (len(rows), len(d), ms_stats(rows)))
        p("first detection from 2723: %s\n" % ("frame %d at %s" % (first["frame_index"], dist(first)) if first else "-"))
        p("| phase | frames | detected |\n|---|---:|---:|")
        ph = {}
        for name, lo, hi in FALCON:
            n = len([f for f in fr if lo <= f <= hi]); tot = len([r for r in rows if lo <= r["frame_index"] <= hi])
            p("| %s | %d | %d |" % (name, tot, n)); ph[name] = n
        p("\nvs ref: %s%s\n" % (section_diff(rel, rows), d3_off_note(rows)))
        js[rel] = {"detected": len(d), "first": first["frame_index"] if first else None,
                   "first_m": first.get("distance_m") if first else None, "phases": ph}

    rel = "extra/cloud_with_fake_obj"
    rows = load(os.path.join(a.run, rel + ".jsonl"))
    if rows:
        d = detected(rows); fr = [r["frame_index"] for r in d]
        p("## cloud_with_fake_obj (%d frames, %d detected, ms %s)\n" % (len(rows), len(d), ms_stats(rows)))
        p("| window | detected | first |\n|---|---:|---|")
        inside = set()
        for name, lo, hi in FAKE:
            w = [r for r in d if lo <= r["frame_index"] <= hi]; inside |= {r["frame_index"] for r in w}
            p("| %s | %d | %s |" % (name, len(w), "%d (%s)" % (w[0]["frame_index"], dist(w[0])) if w else "-"))
        outside = [f for f in fr if f not in inside]
        p("| outside windows | %d | %s |" % (len(outside), fmt_frames(outside)))
        p("\nvs ref: %s%s\n" % (section_diff(rel, rows), d3_off_note(rows)))
        js[rel] = {"detected": len(d), "outside": len(outside)}

    rel = "newdata/new_data"
    rows = load(os.path.join(a.run, rel + ".jsonl"))
    if rows:
        d = detected(rows)
        eps = episodes([r["frame_index"] for r in d])
        p("## new_data (%d frames, %d detected in %d episodes, ms %s)\n" % (len(rows), len(d), len(eps), ms_stats(rows)))
        byf = {r["frame_index"]: r for r in d}
        for lo, hi in eps:
            p("- frames %d-%d at %s" % (lo, hi, dist(byf[lo])))
        p("\nvs ref: %s%s\n" % (section_diff(rel, rows), d3_off_note(rows)))
        js[rel] = {"frames": len(rows), "detected": len(d), "episodes": eps}

    org = os.path.join(a.run, "orgsynth")
    if os.path.isdir(org):
        p("## Organisers' synthetic bags (no labels: episodes)\n")
        for f in sorted(os.listdir(org)):
            if not f.endswith(".jsonl"):
                continue
            rel = "orgsynth/" + f[:-6]; rows = load(os.path.join(a.run, rel + ".jsonl")); d = detected(rows)
            byf = {r["frame_index"]: r for r in d}
            eps = ["%d-%d (%s->%s)" % (lo, hi, dist(byf[lo]), dist(byf[hi])) for lo, hi in episodes(list(byf))]
            p("- %s (%d frames): %s; vs ref: %s%s" % (f[:-6], len(rows), ", ".join(eps) or "none", section_diff(rel, rows), d3_off_note(rows)))
            js[rel] = {"detected": len(d), "episodes": eps}
        p("")

    stability_section(a.run, ref, p, js)

    syn = os.path.join(a.run, "synth")
    if os.path.isdir(syn):
        p("## Synthetic sets (scoreboard.py)\n")
        here = os.path.dirname(os.path.abspath(__file__))
        for s in sorted(os.listdir(syn)):
            res = os.path.join(syn, s)
            jpath = os.path.join(res, "scoreboard.json")
            r = subprocess.run([sys.executable, os.path.join(here, "scoreboard.py"), "--set", os.path.join(SYN_ROOT, s),
                                "--results", res, "--out", os.path.join(res, "scoreboard.md"), "--json", jpath],
                               capture_output=True, text=True)
            if r.returncode != 0 or not os.path.exists(jpath):
                p("- %s: scoreboard failed: %s" % (s, (r.stderr or r.stdout)[-300:]))
                continue
            sb = json.load(open(jpath, encoding="utf-8"))
            js["synth/" + s] = sb
            p("### %s\n\n%s\n" % (s, open(os.path.join(res, "scoreboard.md"), encoding="utf-8").read().strip()))

    text = "\n".join(out)
    print(text)
    if a.json:
        json.dump(js, open(a.json, "w", encoding="utf-8"), indent=1)
    else:
        json.dump(js, open(os.path.join(a.run, "summary.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
