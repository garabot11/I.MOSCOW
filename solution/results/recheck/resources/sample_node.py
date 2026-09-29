"""Sample CPU% (of one core) and RSS of the detector node and the bag player every 0.5 s until stopped."""
import json, sys, time
import psutil

out = open(sys.argv[1], "w", buffering=1)
procs = {}
while True:
    for p in psutil.process_iter(["pid", "cmdline"]):
        cmd = " ".join(p.info["cmdline"] or [])
        name = "node" if "lib/metro_obstacle_detector/detector_node" in cmd else "player" if ("bag play" in cmd and "ros2" in cmd) else None
        if name and p.pid not in procs:
            procs[p.pid] = (name, p)
            p.cpu_percent(None)
    time.sleep(0.5)
    for pid, (name, p) in list(procs.items()):
        try:
            out.write(json.dumps({"t": round(time.monotonic(), 2), "proc": name, "cpu_percent": p.cpu_percent(None),
                                  "rss_mb": round(p.memory_info().rss / 2**20, 1), "threads": p.num_threads()}) + "\n")
        except psutil.NoSuchProcess:
            procs.pop(pid)
