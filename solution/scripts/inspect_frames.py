#!/usr/bin/env python3
"""Print candidates / rejections for given frames of a results JSONL: inspect_frames.py file.jsonl f1 f2 ..."""
import json, sys
path = sys.argv[1]; want = set(int(x) for x in sys.argv[2:])
for line in open(path):
    r = json.loads(line)
    if want and r['frame_index'] not in want: continue
    det = r['detector']
    cur = det.get('curve', {}); rail = det.get('rail', {})
    print(f"#{r['frame_index']} t={r['relative_time_s']}s {r['status']} dist={r['distance_m']} box={det['n_box']} l0={cur.get('l0')} a={cur.get('a')} b={cur.get('b')} n={cur.get('n_used')} maxd={cur.get('max_d_supported')} rail z0={rail.get('z0')} g={rail.get('grade')} c={rail.get('curv')} rms_bins={rail.get('n_bins')}")
    for c in det['candidates']:
        print(f"   CAND d={c['forward_m']:.1f} dl={c['lateral_offset_from_track']:+.2f} h={c['height_above_rail']:.2f} n={c['n_points']} ext={c['extent_dlh']} conf={c['confidence']} tid={c['track_id']} confirmed={c['confirmed']}")
    for c in det['rejected']:
        if 'too few' in c['reason']: continue
        print(f"   rej  d={c['forward_m']:.1f} dl={c['lateral_offset_from_track']:+.2f} h={c['height_above_rail']:.2f} n={c['n_points']} ext={c['extent_dlh']} :: {c['reason']}")
