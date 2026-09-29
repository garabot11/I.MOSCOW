#!/usr/bin/env python3
"""Quick summary of a results directory: status counts, detected segments, timing."""
import json, os, sys
d = sys.argv[1]
for f in sorted(os.listdir(d)):
    if not f.endswith('.jsonl'): continue
    rows = [json.loads(l) for l in open(os.path.join(d, f))]
    if not rows: continue
    st = {}
    for r in rows: st[r['status']] = st.get(r['status'], 0) + 1
    ms = sorted(r['processing_ms'] for r in rows)
    print(f"== {f[:-6]}: frames {len(rows)} {st} ms median {ms[len(ms)//2]:.0f} p95 {ms[int(len(ms)*0.95)]:.0f} max {ms[-1]:.0f}")
    segs, cur = [], None
    for r in rows:
        if r['status'] == 'detected':
            if cur is None: cur = [r['frame_index'], r['frame_index'], [r['distance_m']], [r['relative_time_s']]]
            else: cur[1] = r['frame_index']; cur[2].append(r['distance_m']); cur[3].append(r['relative_time_s'])
        else:
            if cur: segs.append(cur); cur = None
    if cur: segs.append(cur)
    for s in segs: print(f"   detected frames {s[0]:4d}-{s[1]:4d} (t={s[3][0]:.1f}-{s[3][-1]:.1f}s) dist {min(s[2]):.0f}..{max(s[2]):.0f} m")
