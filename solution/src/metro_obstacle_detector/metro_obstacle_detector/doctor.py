"""Self-check of the runtime: GPU, graphics, bags, DDS, CPU, RAM, disk.  Never raises, exit code 0.

    python3 -m metro_obstacle_detector.doctor [--bag-root /data ...] [--json out.json] [--full] [--quick]
    python3 -m metro_obstacle_detector.doctor --list-bags [--bag-root /data]       # table of the bags (metro bags)
    python3 -m metro_obstacle_detector.doctor --resolve-bag NAME [--bag-root /data] # JSON of one bag (metro play)

Every item is OK / WARN / FAIL with a one-line fix.  The report is written to
``/results/logs/doctor.json`` (and ``--json``); the device state (for the banner)
to ``/results/logs/device.json``; in an emergency CPU mode ``/results/ВНИМАНИЕ_GPU.txt``.

Bags: every directory under the bag root(s) (depth <= 4) that holds ``metadata.yaml``
or ``*.db3`` / ``*.mcap`` files, so a root with several dataset folders
(``/data/for_hackathon/*``, ``/data/falcon/*``) works.  Each bag is inspected in a
child process (a corrupt file can neither hang nor crash the check), reading the
metadata, the first cloud and the real message count: for SQLite3 a ``COUNT(*)``
on the read-only database (``--quick``, default), with ``--full`` every message
header through ``bag_tools.read_headers`` (one message in memory at a time; a
20 GB bag takes ~10-60 s, cached in ``/results/logs/bag_index.json`` by file
size and mtime).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional

OK, WARN, FAIL = "OK", "WARN", "FAIL"
POINTCLOUD2 = "sensor_msgs/msg/PointCloud2"
RESULTS = os.environ.get("METRO_RESULTS_DIR", "/results")
LOGS = os.path.join(RESULTS, "logs")
DEFAULT_ROOTS = [r for r in os.environ.get("METRO_DATA", "/data").split(":") if r]
BAG_FILES = ("*.db3", "*.mcap")
CACHE_VERSION = 3  # bump when inspect_bag changes what it reports


def _item(_check: str, _status: str, _msg: str, _fix: str = "", **data) -> dict:
    return {"check": _check, "status": _status, "message": _msg, "fix": _fix, "data": data}


def _run(cmd: List[str], timeout: float = 10.0, env: Optional[dict] = None) -> tuple:
    """(returncode, stdout+stderr) -- never raises; 127 = not found, 124 = timeout."""
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
                           env=env, text=True, errors="replace")
        return p.returncode, p.stdout
    except FileNotFoundError:
        return 127, "%s: not found" % cmd[0]
    except subprocess.TimeoutExpired:
        return 124, "%s: timed out after %.0f s" % (cmd[0], timeout)
    except Exception as exc:
        return 1, str(exc)


def _short(exc: BaseException) -> str:
    return (type(exc).__name__ + ": " + " ".join(str(exc).split()))[:240]


# ---------------------------------------------------------------------------- GPU
def check_gpu(prefer: str = "auto") -> List[dict]:
    try:
        from .device import DeviceManager
        quiet = type("Q", (), {"info": lambda s, m: None, "warning": lambda s, m: None})()
        dm = DeviceManager(prefer=prefer, results_dir=RESULTS if os.path.isdir(RESULTS) else None, logger=quiet)
        st = dm.status_dict()
    except Exception as exc:
        return [_item("gpu", FAIL, "device check failed: %s" % _short(exc), "rebuild the image")]
    _write_json(os.path.join(LOGS, "device.json"), st)
    arch = "arch list %s, device capability %s" % (st.get("arch_list") or "n/a (no CUDA driver visible)",
                                                    st.get("capability") or "-")
    if st["mode"] == "gpu":
        msg = "%s (sm_%s via %s), torch %s, CUDA %s, driver %s, %s/%s MB free, torch threads %s; %s" % (
            st["device_name"], (st["capability"] or "?").replace(".", ""), st["arch_note"], st["torch_version"],
            st["cuda_version"], st["driver_version"], st["mem_free_mb"], st["mem_total_mb"], st["torch_threads"], arch)
        if st["capability"] and st["arch_note"].startswith("sm_") and "binary compatible" in st["arch_note"]:
            msg += " (no native cubins for this GPU: the wheel runs the binary-compatible ones, normal for PyTorch)"
        return [_item("gpu", OK, msg, "", **st)]
    first_cmd = next((ln for ln in st["fix_hint"].splitlines() if ln.strip() and not ln.startswith("#")), "")
    level = WARN if st["mode"] == "cpu_forced" else FAIL
    msg = "%s: %s (torch %s; %s)" % (st["mode"], st["reason"], st["torch_version"], arch)
    return [_item("gpu", level, msg, first_cmd + ("  (full instructions in the banner / ВНИМАНИЕ_GPU.txt)"
                                                  if level == FAIL else ""), **st)]


# ---------------------------------------------------------------------------- graphics
def _renderer(out: str) -> str:
    m = re.search(r"OpenGL renderer string:\s*(.+)", out)
    return m.group(1).strip() if m else ""


def _nvidia_glx_present() -> bool:
    pats = ["/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so*", "/usr/lib64/libGLX_nvidia.so*", "/usr/lib/libGLX_nvidia.so*"]
    return any(glob.glob(p) for p in pats)


def check_display() -> List[dict]:
    disp = os.environ.get("DISPLAY", "")
    fix_disp = "on the host: start with ./run.sh in a desktop session (DISPLAY=:0, X cookie passed automatically)"
    if not disp:
        return [_item("display", WARN, "no DISPLAY: RViz is not started, results are shown in the terminal (metro status)",
                      fix_disp, gl_mode="none")]
    m = re.match(r"^[^:]*:(\d+)", disp)
    sock = "/tmp/.X11-unix/X%s" % m.group(1) if m else ""
    if m and not disp.split(":")[0] and not os.path.exists(sock):
        return [_item("display", WARN, "DISPLAY=%s but %s is not mounted: RViz off" % (disp, sock),
                      "docker run -v /tmp/.X11-unix:/tmp/.X11-unix:ro ... (./run.sh does it)", gl_mode="none")]
    rc, out = _run(["xdpyinfo"], timeout=8)
    if rc == 127:
        rc, out = _run(["glxinfo", "-B"], timeout=8)
    if rc != 0:
        why = out.strip().splitlines()[-1] if out.strip() else "rc %d" % rc
        return [_item("display", WARN, "cannot connect to the X server %s (%s): RViz off" % (disp, why[:120]),
                      "on the host: ./run.sh stop && ./run.sh (private Xauthority), or: xhost +si:localuser:$(id -un)",
                      gl_mode="none")]
    items = [_item("display", OK, "X server %s reachable" % disp, "")]
    tried = []
    base = dict(os.environ)
    base.pop("LIBGL_ALWAYS_SOFTWARE", None)
    candidates = [("nvidia", base)]
    if _nvidia_glx_present():
        prime = dict(base, __NV_PRIME_RENDER_OFFLOAD="1", __GLX_VENDOR_LIBRARY_NAME="nvidia")
        candidates.append(("nvidia_prime", prime))
    candidates.append(("software", dict(base, LIBGL_ALWAYS_SOFTWARE="1")))
    gl_mode, renderer = "none", ""
    for mode, env in candidates:
        rc, out = _run(["glxinfo", "-B"], timeout=12, env=env)
        rend = _renderer(out) if rc == 0 else ""
        tried.append({"mode": mode, "rc": rc, "renderer": rend})
        if not rend:
            continue
        if mode.startswith("nvidia") and "nvidia" not in rend.lower():
            continue
        gl_mode, renderer = mode, rend
        break
    if gl_mode.startswith("nvidia"):
        items.append(_item("graphics", OK, "RViz renders on the GPU: %s%s" % (
            renderer, " (PRIME offload)" if gl_mode == "nvidia_prime" else ""), "", gl_mode=gl_mode, tried=tried))
    elif gl_mode == "software":
        items.append(_item("graphics", WARN, "software OpenGL (%s): RViz works but loads the CPU" % renderer,
                           "start with GPU access: ./run.sh (adds --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all when the toolkit works)",
                           gl_mode=gl_mode, tried=tried))
    else:
        items.append(_item("graphics", WARN, "no working OpenGL (glxinfo failed): RViz off, status in the terminal",
                           "check the X server on the host: glxinfo -B", gl_mode="none", tried=tried))
    return items


# ---------------------------------------------------------------------------- bags
def _bag_signature(path: str) -> list:
    sig = []
    for pat in BAG_FILES + ("metadata.yaml",):
        for f in sorted(glob.glob(os.path.join(path, pat))):
            try:
                s = os.stat(f)
                sig.append([os.path.basename(f), s.st_size, int(s.st_mtime)])
            except OSError:
                pass
    return sig


def find_bags(roots: List[str], max_depth: int = 5) -> List[str]:
    """Bag directories under ``roots`` (a root may itself be a bag)."""
    out: List[str] = []
    for root in roots:
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            continue
        base_depth = root.rstrip("/").count("/")
        for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
            depth = dirpath.rstrip("/").count("/") - base_depth
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            if depth >= max_depth:
                dirnames[:] = []
            if "metadata.yaml" in filenames or any(f.endswith((".db3", ".mcap")) for f in filenames):
                out.append(dirpath)
                dirnames[:] = []  # a bag does not contain bags
    return sorted(set(out))


def _type_available(ty: str) -> bool:
    try:
        from rosidl_runtime_py.utilities import get_message
        get_message(ty)
        return True
    except Exception:
        return False


def inspect_bag(path: str, full: bool = False) -> dict:
    """Runs in a child process.  Metadata, storage plugin, first cloud, real message count."""
    info: Dict = {"path": path, "name": _bag_name(path), "problems": [], "notes": []}
    files = [f for pat in BAG_FILES for f in glob.glob(os.path.join(path, pat))]
    info["size_bytes"] = sum(os.path.getsize(f) for f in files if os.path.exists(f))
    meta_path = os.path.join(path, "metadata.yaml")
    meta = None
    if not os.path.exists(meta_path):
        info["problems"].append("no metadata.yaml (ros2 bag play needs it); fix: copy the bag and run "
                                "`ros2 bag reindex <copy> -s %s`" % ("mcap" if any(f.endswith(".mcap") for f in files) else "sqlite3"))
    else:
        try:
            import yaml
            with open(meta_path, encoding="utf-8") as fh:
                meta = yaml.safe_load(fh)["rosbag2_bagfile_information"]
        except Exception as exc:
            info["problems"].append("metadata.yaml unreadable: %s" % _short(exc))
    if not files:
        info["problems"].append("no .db3 / .mcap file in the directory")
    if meta is None:
        return info
    storage = meta.get("storage_identifier") or "sqlite3"
    info["storage"] = storage
    info["metadata_version"] = meta.get("version")
    info["duration_s"] = round(int((meta.get("duration") or {}).get("nanoseconds", 0)) / 1e9, 1)
    topics = {}
    for t in meta.get("topics_with_message_count") or []:
        tm = t.get("topic_metadata") or {}
        topics[tm.get("name")] = {"type": tm.get("type"), "metadata_count": int(t.get("message_count") or 0)}
    info["topics"] = topics
    clouds = [n for n, t in topics.items() if t["type"] == POINTCLOUD2 and t["metadata_count"] > 0] or \
             [n for n, t in topics.items() if t["type"] == POINTCLOUD2]
    # several clouds: the usual names first, then by message count, then by name
    pref = {"/lidar_points": 0, "/iv_points": 1, "/sensing/lidar/hesai128/pointcloud": 2}
    clouds.sort(key=lambda n: (pref.get(n, 9), -topics[n]["metadata_count"], n))
    info["cloud_topics"] = clouds
    if not clouds:
        info["problems"].append("no sensor_msgs/PointCloud2 topic (topics: %s)" % sorted(topics))
        return info
    missing_types = sorted({t["type"] for t in topics.values() if t["type"] and not _type_available(t["type"])})
    if missing_types:
        info["notes"].append("message types unknown to ROS 2 Humble: %s (metro play plays only the cloud topic)" % missing_types)
    plugin = {"sqlite3": "rosbag2_storage_default_plugins", "mcap": "rosbag2_storage_mcap"}.get(storage)
    try:
        from ament_index_python.packages import get_package_prefix
        if plugin:
            get_package_prefix(plugin)
        else:
            info["problems"].append("storage plugin '%s' is not available (supported: sqlite3, mcap)" % storage)
            return info
    except Exception:
        info["problems"].append("storage plugin %s not installed in the image" % plugin)
        return info
    from . import bag_tools
    topic = clouds[0]
    info["cloud_topic"] = topic
    if len(clouds) > 1:
        info["notes"].append("%d PointCloud2 topics %s: metro play uses %s (choose with --topic)" % (len(clouds), clouds, topic))
    try:
        first = bag_tools.read_headers(path, topic, limit=1)
    except Exception as exc:
        info["problems"].append("cannot read the bag: %s" % _short(exc))
        return info
    if not first:
        info["problems"].append("no message on %s" % topic)
        return info
    info["frame_id"] = first[0]["frame_id"]
    try:
        info["first_cloud"] = _first_cloud(path, topic)
    except Exception as exc:
        info["notes"].append("first cloud not decoded: %s" % _short(exc))
    t0 = time.monotonic()
    try:
        if full:
            hdr = bag_tools.read_headers(path, topic)
            info["real_count"] = len(hdr)
            info["count_method"] = "bag_tools.read_headers (every message)"
            stamps = [h["header_time_ns"] for h in hdr]
            back = sum(1 for a, b in zip(stamps, stamps[1:]) if b < a)
            if back:
                info["notes"].append("%d header stamps go backwards" % back)
        elif storage == "sqlite3":
            info["real_count"] = _sqlite_count(path, topic)
            info["count_method"] = "sqlite COUNT(*)"
        else:  # MCAP quick check: the per-topic count of the metadata (--full reads every header)
            info["real_count"] = topics[topic]["metadata_count"]
            info["count_method"] = "metadata.yaml (quick; metro doctor --full counts every message)"
    except Exception as exc:
        info["problems"].append("bag damaged while counting messages: %s" % _short(exc))
    info["count_s"] = round(time.monotonic() - t0, 1)
    mc = topics[topic]["metadata_count"]
    if "real_count" in info and info["real_count"] != mc:
        info["notes"].append("metadata says %d messages on %s, the bag holds %d (the real ones are played)" % (
            mc, topic, info["real_count"]))
    return info


def _sqlite_count(path: str, topic: str) -> int:
    import sqlite3
    total = 0
    for f in sorted(glob.glob(os.path.join(path, "*.db3"))):
        con = sqlite3.connect("file:%s?mode=ro&immutable=1" % f, uri=True)
        try:
            row = con.execute("SELECT COUNT(*) FROM messages m JOIN topics t ON m.topic_id = t.id WHERE t.name = ?",
                              (topic,)).fetchone()
            total += int(row[0])
        finally:
            con.close()
    return total


def _first_cloud(path: str, topic: str) -> dict:
    from . import bag_tools
    import rosbag2_py
    from .cloud_io import parse_pointcloud2_cdr
    reader, _ = bag_tools._open(path)
    try:
        reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    except Exception:
        pass
    while reader.has_next():
        name, raw, _t = reader.read_next()
        if name != topic:
            continue
        msg = parse_pointcloud2_cdr(bytes(raw))
        return {"points": int(msg.width) * int(msg.height), "width": int(msg.width), "height": int(msg.height),
                "point_step": int(msg.point_step), "fields": [f.name for f in msg.fields], "frame_id": msg.header.frame_id}
    return {}


def _inspect_in_child(path: str, full: bool, timeout: float) -> dict:
    rc, out = _run([sys.executable, "-m", "metro_obstacle_detector.doctor", "--inspect-bag", path] + (["--full"] if full else []),
                   timeout=timeout)
    for line in reversed(out.strip().splitlines()):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except Exception:
                break
    why = {124: "timed out after %.0f s" % timeout, -11: "the bag reader crashed (segmentation fault)"}.get(rc, "rc %d" % rc)
    return {"path": path, "name": os.path.basename(os.path.normpath(path)), "notes": [],
            "problems": ["cannot inspect the bag (%s): %s" % (why, out.strip().splitlines()[-1][:160] if out.strip() else "")]}


QUICK_TIMEOUT_S = 60.0    # per bag: metadata + first cloud + SQLite COUNT(*) take ~1 s even for 20 GB
FULL_TIMEOUT_S = 900.0    # per bag with --full (every message header is read)


def bag_index(roots: List[str], full: bool = False, timeout: Optional[float] = None, use_cache: bool = True) -> List[dict]:
    timeout = timeout or (FULL_TIMEOUT_S if full else QUICK_TIMEOUT_S)
    cache_path = os.path.join(LOGS, "bag_index.json")
    cache = _read_json(cache_path) or {} if use_cache else {}
    out = []
    for path in find_bags(roots):
        sig = _bag_signature(path)
        c = cache.get(path)
        if c and c.get("signature") == sig and c.get("v") == CACHE_VERSION and (not full or c.get("info", {}).get("count_method", "").startswith("bag_tools")):
            info = c["info"]
        else:
            info = _inspect_in_child(path, full, timeout)
            cache[path] = {"signature": sig, "v": CACHE_VERSION, "info": info}
        info["rel"] = _rel(path, roots)
        out.append(info)
    if use_cache:
        _write_json(cache_path, cache)
    return out


def _strip_bag(rel: str) -> str:
    """Synthetic scenes keep their bag in ``<set>/<scene>/bag``: the scene path is the bag's name."""
    if rel.endswith("/bag"):
        return rel[: -len("/bag")]
    return rel


def _bag_name(path: str) -> str:
    p = os.path.normpath(path)
    return os.path.basename(os.path.dirname(p)) if os.path.basename(p) == "bag" else os.path.basename(p)


def _rel(path: str, roots: List[str]) -> str:
    for r in roots:
        r = os.path.abspath(r)
        if path == r:
            return _bag_name(r)
        if path.startswith(r.rstrip("/") + "/"):
            return _strip_bag(os.path.relpath(path, r))
    return path


def check_bags(roots: List[str], full: bool = False) -> List[dict]:
    existing = [r for r in roots if os.path.isdir(r)]
    if not existing:
        return [_item("bags", WARN, "bag root %s not found (no recordings mounted)" % roots,
                      "./run.sh --data /path/to/dataset (mounted read-only at /data)")]
    bags = bag_index(existing, full=full)
    if not bags:
        return [_item("bags", WARN, "no rosbag2 directory under %s" % existing,
                      "./run.sh --data <dir containing bag directories>")]
    items = []
    for b in bags:
        if b.get("problems"):
            items.append(_item("bag " + b["rel"], FAIL, "; ".join(b["problems"]),
                               "" if "fix:" in "; ".join(b["problems"]) else
                               "re-copy this bag from the source (the other bags are not affected)", **b))
            continue
        fc = b.get("first_cloud") or {}
        msg = "%s, %s, %.0f s, %s (%s pts, step %s, frame %s), real %s msgs" % (
            b.get("storage"), _gb(b.get("size_bytes")), b.get("duration_s") or 0, b.get("cloud_topic"),
            fc.get("points", "?"), fc.get("point_step", "?"), b.get("frame_id"), b.get("real_count", "?"))
        if b.get("notes"):
            msg += "; " + "; ".join(b["notes"])
        items.append(_item("bag " + b["rel"], OK, msg, "", **b))
    return items


def _gb(n) -> str:
    return "%.1f GB" % (n / 1e9) if n else "?"


# ---------------------------------------------------------------------------- DDS / system
def check_dds() -> List[dict]:
    rmw = os.environ.get("RMW_IMPLEMENTATION", "") or "rmw_fastrtps_cpp (default)"
    dom = os.environ.get("ROS_DOMAIN_ID", "") or "0 (default)"
    loc = os.environ.get("ROS_LOCALHOST_ONLY", "")
    prof = os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE", "")
    items = []
    msg = "RMW %s, ROS_DOMAIN_ID %s, ROS_LOCALHOST_ONLY %s, Fast DDS profile %s" % (rmw, dom, loc or "unset", prof or "none (shared memory + UDP)")
    if prof and not os.path.exists(prof):
        items.append(_item("dds", FAIL, msg + ": profile file missing", "unset FASTRTPS_DEFAULT_PROFILES_FILE or fix the path"))
    elif loc == "1":
        items.append(_item("dds", WARN, msg + ": players outside this container (host, other containers) are not seen",
                           "start without ROS_LOCALHOST_ONLY (./run.sh does)", rmw=rmw, domain=dom))
    else:
        items.append(_item("dds", OK, msg + ("" if prof else "; a player OUTSIDE the container needs ./run.sh --udp "
                                             "(FASTRTPS_DEFAULT_PROFILES_FILE=/workspace/config/fastdds_udp_only.xml)"), "",
                           rmw=rmw, domain=dom, localhost_only=loc, profile=prof))
    try:
        st = os.statvfs("/dev/shm")
        shm_mb = st.f_frsize * st.f_blocks / 2**20
        if shm_mb < 512:
            items.append(_item("shm", WARN, "/dev/shm %.0f MB: large clouds through shared memory may fail" % shm_mb,
                               "docker run --shm-size=1g ... (./run.sh does)"))
        else:
            items.append(_item("shm", OK, "/dev/shm %.0f MB" % shm_mb, ""))
    except Exception as exc:
        items.append(_item("shm", WARN, "cannot read /dev/shm: %s" % _short(exc), "docker run --shm-size=1g"))
    return items


def check_cpu() -> List[dict]:
    from .device import available_cpus
    n = available_cpus()
    model = ""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            m = re.search(r"model name\s*:\s*(.+)", fh.read())
            model = m.group(1).strip() if m else ""
    except Exception:
        pass
    quota = ""
    try:
        with open("/sys/fs/cgroup/cpu.max", encoding="utf-8") as fh:
            q, p = fh.read().split()
            if q != "max":
                quota = ", cgroup quota %.1f CPUs" % (int(q) / int(p))
    except Exception:
        pass
    msg = "%d CPUs available to this process (host %s)%s; %s" % (n, os.cpu_count(), quota, model)
    if n < 4:
        return [_item("cpu", WARN, msg + ": fewer than 4 CPUs, 10 Hz not guaranteed", "give the container >= 8 CPUs", cpus=n)]
    return [_item("cpu", OK, msg, "", cpus=n, host_cpus=os.cpu_count(), model=model)]


def check_ram() -> List[dict]:
    try:
        mi = {}
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for ln in fh:
                k, v = ln.split(":", 1)
                mi[k] = int(v.split()[0]) * 1024
        total, avail = mi.get("MemTotal", 0), mi.get("MemAvailable", 0)
        limit = None
        try:
            with open("/sys/fs/cgroup/memory.max", encoding="utf-8") as fh:
                v = fh.read().strip()
                limit = None if v == "max" else int(v)
        except Exception:
            pass
        eff = min(avail, limit) if limit else avail
        msg = "RAM %.1f GB total, %.1f GB available%s" % (total / 2**30, avail / 2**30,
                                                         ", container limit %.1f GB" % (limit / 2**30) if limit else "")
        if eff < 3 * 2**30:
            return [_item("ram", WARN, msg + ": less than 3 GB free", "close other programs / raise docker --memory")]
        return [_item("ram", OK, msg, "", total=total, available=avail, limit=limit)]
    except Exception as exc:
        return [_item("ram", WARN, "cannot read memory info: %s" % _short(exc), "")]


def check_disk() -> List[dict]:
    items = []
    if not os.path.isdir(RESULTS):
        return [_item("disk", WARN, "%s does not exist: logs are not kept" % RESULTS, "./run.sh --results <dir>")]
    try:
        free = shutil.disk_usage(RESULTS).free
        writable = os.access(RESULTS, os.W_OK)
        msg = "%s: %.1f GB free, %s" % (RESULTS, free / 1e9, "writable" if writable else "NOT writable")
        if not writable:
            items.append(_item("disk", FAIL, msg, "on the host: chmod a+rwx <results dir> (or ./run.sh --results <own dir>)"))
        elif free < 2e9:
            items.append(_item("disk", WARN, msg + ": less than 2 GB", "free disk space on the host"))
        else:
            items.append(_item("disk", OK, msg, "", free=free))
    except Exception as exc:
        items.append(_item("disk", WARN, "cannot check %s: %s" % (RESULTS, _short(exc)), ""))
    return items


def check_software() -> List[dict]:
    parts, status, fix = [], OK, ""
    try:
        import numpy
        parts.append("numpy %s" % numpy.__version__)
    except Exception as exc:
        status, fix = FAIL, "rebuild the image"
        parts.append("numpy missing (%s)" % _short(exc))
    parts.append("ROS %s" % (os.environ.get("ROS_DISTRO") or "not sourced"))
    if not os.environ.get("ROS_DISTRO"):
        status, fix = WARN, "source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash (metro does it)"
    models = sorted(glob.glob("/workspace/models/*.pt") + glob.glob("/workspace/install/share/metro_obstacle_detector/models/*.pt"))
    parts.append("models: %s" % (", ".join(os.path.basename(m) for m in models) if models else "none in the image"))
    return [_item("software", status, ", ".join(parts), fix, models=models)]


# ---------------------------------------------------------------------------- report
def _read_json(path: str):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _write_json(path: str, data) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = "%s.%d.tmp" % (path, os.getpid())  # unique: the background bag index and metro play may write at once
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=1, default=str)
        os.replace(tmp, path)
        return True
    except Exception:
        return False


def run_all(roots: List[str], full: bool = False, prefer: str = "auto", skip_bags: bool = False) -> dict:
    items: List[dict] = []
    checks = [("gpu", lambda: check_gpu(prefer)), ("display", check_display), ("dds", check_dds), ("cpu", check_cpu),
              ("ram", check_ram), ("disk", check_disk), ("software", check_software)]
    if not skip_bags:
        checks.append(("bags", lambda: check_bags(roots, full)))
    for name, fn in checks:
        try:
            items += fn()
        except Exception as exc:  # a broken check must not hide the others
            items.append(_item(name, WARN, "check failed: %s" % _short(exc), ""))
    gl = next((i["data"].get("gl_mode") for i in items if i["check"] in ("graphics", "display") and "gl_mode" in i["data"]), "none")
    gpu = next((i for i in items if i["check"] == "gpu"), {})
    return {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {s: sum(1 for i in items if i["status"] == s) for s in (OK, WARN, FAIL)},
        "device_mode": gpu.get("data", {}).get("mode"),
        "gl_mode": gl,
        "bag_roots": roots,
        "items": items,
    }


def print_report(rep: dict, color: bool) -> None:
    col = {OK: "\033[32m", WARN: "\033[33m", FAIL: "\033[1;31m"} if color else {}
    rst = "\033[0m" if color else ""
    print("metro doctor %s   OK %d  WARN %d  FAIL %d   (GPU mode: %s, RViz GL: %s)" % (
        rep["time"], rep["summary"][OK], rep["summary"][WARN], rep["summary"][FAIL], rep["device_mode"], rep["gl_mode"]))
    for i in rep["items"]:
        print("  [%s%-4s%s] %-52s %s" % (col.get(i["status"], ""), i["status"], rst, i["check"][:52], i["message"]))
        if i["fix"] and i["status"] != OK and "fix:" not in i["message"]:
            print("  %s %-52s fix: %s" % (" " * 6, "", i["fix"]))
    sys.stdout.flush()


def print_bags(bags: List[dict]) -> None:
    if not bags:
        print("no bags found")
        return
    print("%-64s %-7s %8s %7s %-34s %8s %s" % ("bag (metro play <bag>)", "storage", "size", "dur s", "cloud topic / frame", "msgs", "state"))
    for b in bags:
        fc = b.get("first_cloud") or {}
        state = "BROKEN: " + "; ".join(b["problems"]) if b.get("problems") else ("; ".join(b.get("notes", [])) or "ok")
        topic = ("%s / %s" % (b.get("cloud_topic") or "-", b.get("frame_id") or "-"))
        print("%-64s %-7s %8s %7s %-34s %8s %s" % (b.get("rel", b.get("name"))[:64], b.get("storage", "-"),
                                                  _gb(b.get("size_bytes")), b.get("duration_s", "-"), topic[:34],
                                                  b.get("real_count", "?"), state[:200]))
        if fc:
            print("%-64s   %s points/cloud, point_step %s, fields %s" % ("", fc.get("points"), fc.get("point_step"), ",".join(fc.get("fields", []))))
    sys.stdout.flush()


def resolve_bag(name: str, roots: List[str]) -> dict:
    """A bag by path, by path relative to a root, by a path suffix (``<set>/<scene>``, ``for_hackathon/x``)
    or by directory name (unique).  A synthetic scene directory stands for its ``bag/`` subdirectory."""

    def as_bag(p: str) -> str:
        sub = os.path.join(p, "bag")
        return sub if not find_bags_here(p) and os.path.isdir(sub) else p

    def find_bags_here(p: str) -> bool:
        return os.path.exists(os.path.join(p, "metadata.yaml")) or \
            any(glob.glob(os.path.join(p, pat)) for pat in BAG_FILES)

    cands = []
    if os.path.isdir(name):
        cands = [as_bag(os.path.abspath(name))]
    else:
        for r in roots:
            p = os.path.join(r, name)
            if os.path.isdir(p):
                cands = [as_bag(os.path.abspath(p))]
                break
        if not cands:
            found = find_bags([r for r in roots if os.path.isdir(r)])
            key = name.strip("/")
            cands = [p for p in found if p.endswith("/" + key) or p.endswith("/" + key + "/bag")] or \
                    [p for p in found if key in _bag_name(p)]
    if not cands:
        return {"ok": False, "error": "bag '%s' not found under %s (metro bags lists them)" % (name, roots)}
    if len(cands) > 1:
        return {"ok": False, "error": "'%s' matches several bags: %s" % (name, [_rel(c, roots) for c in cands])}
    path = cands[0]
    if path not in find_bags([path]):
        return {"ok": False, "error": "%s is not a rosbag2 directory (no metadata.yaml / .db3 / .mcap)" % path}
    info = bag_index([path], full=False, use_cache=True)[0]
    info["rel"] = _rel(path, roots)
    info["ok"] = not info.get("problems")
    if not info["ok"]:
        info["error"] = "bag %s cannot be played: %s" % (info.get("rel"), "; ".join(info["problems"]))
    info["path"] = path
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="metro self-check (never raises, exit code 0)")
    ap.add_argument("--bag-root", action="append", default=None, help="bag root directory (repeatable; default /data)")
    ap.add_argument("--json", default="", help="also write the report here")
    ap.add_argument("--full", action="store_true", help="count every message of every bag (slow, cached)")
    ap.add_argument("--no-bags", action="store_true")
    ap.add_argument("--prefer", default="auto", choices=["auto", "gpu", "cpu"])
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="only write the JSON")
    ap.add_argument("--list-bags", action="store_true")
    ap.add_argument("--resolve-bag", default="")
    ap.add_argument("--summary-from", default="", help="print the report of an earlier run (JSON)")
    ap.add_argument("--inspect-bag", default="", help=argparse.SUPPRESS)
    try:
        args = ap.parse_args(argv)
    except SystemExit:
        return 0
    roots = args.bag_root or DEFAULT_ROOTS
    try:
        if args.inspect_bag:
            try:
                info = inspect_bag(args.inspect_bag, args.full)
            except Exception as exc:
                info = {"path": args.inspect_bag, "name": os.path.basename(args.inspect_bag), "notes": [],
                        "problems": ["cannot inspect: %s" % _short(exc)]}
            print(json.dumps(info, default=str))
            return 0
        if args.summary_from:
            rep = _read_json(args.summary_from)
            if rep:
                print_report(rep, color=not args.no_color and sys.stdout.isatty())
            return 0
        if args.list_bags:
            print_bags(bag_index([r for r in roots if os.path.isdir(r)], full=args.full) if any(os.path.isdir(r) for r in roots) else [])
            return 0
        if args.resolve_bag:
            print(json.dumps(resolve_bag(args.resolve_bag, roots), default=str))
            return 0
        rep = run_all(roots, full=args.full, prefer=args.prefer, skip_bags=args.no_bags)
        _write_json(os.path.join(LOGS, "doctor.json"), rep)
        if args.json:
            _write_json(args.json, rep)
        if not args.quiet:
            print_report(rep, color=not args.no_color and sys.stdout.isatty())
    except KeyboardInterrupt:
        print("doctor interrupted")
    except Exception as exc:
        print("doctor: unexpected problem (%s); continuing" % _short(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
