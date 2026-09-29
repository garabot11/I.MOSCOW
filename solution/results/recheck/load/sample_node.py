"""Sample CPU% and RSS of the detector node every 0.5 s; the last line has its total CPU time."""
import json, sys, time
import psutil

out = open(sys.argv[1], "w", buffering=1)
node = None
while True:
    if node is None:
        for p in psutil.process_iter(["pid", "cmdline"]):
            if "lib/metro_obstacle_detector/detector_node" in " ".join(p.info["cmdline"] or []):
                node = p
                node.cpu_percent(None)
    time.sleep(0.5)
    if node is not None:
        try:
            ct = node.cpu_times()
            out.write(json.dumps({"t": round(time.monotonic(), 2), "cpu_percent": node.cpu_percent(None),
                                  "cpu_s": round(ct.user + ct.system, 3), "rss_mb": round(node.memory_info().rss / 2**20, 1),
                                  "affinity": node.cpu_affinity()}) + "\n")
        except psutil.NoSuchProcess:
            break
