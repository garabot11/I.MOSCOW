"""GPU / emergency-CPU device manager (interface C).

    dm = DeviceManager(prefer="auto", results_dir="/results", logger=node.get_logger())
    x = x.to(dm.device)                      # torch.device("cuda:0") or ("cpu"); None without torch
    try: ... GPU work ...
    except Exception as exc:                 # CUDA error at run time
        dm.report_failure(exc)               # hot switch to cpu_emergency (logged once), rerun on dm.device
    text = dm.maybe_repeat_banner(time.monotonic())   # banner at the first call and every 300 s in emergency mode
    if text: print(text)

Modes: ``gpu``; ``cpu_emergency`` (GPU wanted but unusable: banner RU+EN, file
``<results_dir>/ВНИМАНИЕ_GPU.txt``, banner repeated every 300 s); ``cpu_forced``
(``prefer="cpu"`` or ``METRO_DEVICE=cpu``: banner once, no file, no repeats).

torch is imported lazily (only in ``probe``); the module imports without torch.
Nothing here raises: every failure becomes a mode + human-readable reason + the
exact fix.  The host launcher (``run.sh``) knows things that cannot be seen from
inside a container (toolkit installed or not); it passes them as
``METRO_GPU_REASON_CODE`` / ``METRO_GPU_REASON_DETAIL`` / ``METRO_HOST_DRIVER``.

    python3 -m metro_obstacle_detector.device [--prefer auto|gpu|cpu] [--json] [--write-state PATH] [--banner]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import threading
import time
import warnings
from typing import Optional, Tuple

from . import banner as _banner

REPEAT_S = 300.0
WARN_FILE = "ВНИМАНИЕ_GPU.txt"
MIN_DRIVER = 570          # CUDA 12.8 wheels need driver >= 570.26 (driver 580 = CUDA 13.0 is fine)
MIN_FREE_MB = 512         # below this the GPU path would run out of memory
MAX_DETAIL = 240          # characters of a technical detail appended to a reason (banner / doctor / file)

FIX_TOOLKIT = (
    "# install the NVIDIA Container Toolkit (Ubuntu 22.04):\n"
    "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg\n"
    "curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | "
    "sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | "
    "sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list\n"
    "sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit\n"
    "sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker\n"
    "# then restart the solution:\n"
    "./run.sh stop && ./run.sh")
FIX_GPUS_FLAG = (
    "# start the container with GPU access (run.sh adds it automatically when the toolkit works):\n"
    "./run.sh stop && ./run.sh\n"
    "# or by hand: docker run --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all ... metro-lidar:<tag>\n"
    "# if docker says 'could not select device driver \"\" with capabilities: [[gpu]]', install the toolkit:\n"
    + FIX_TOOLKIT)
FIX_DRIVER = (
    "# install an NVIDIA driver >= %d (580 recommended) on the host, reboot, check nvidia-smi:\n"
    "sudo apt-get update && sudo apt-get install -y nvidia-driver-580 && sudo reboot\n"
    "nvidia-smi\n"
    "./run.sh stop && ./run.sh" % MIN_DRIVER)
FIX_RESTART = (
    "# check the GPU on the host and restart the container:\n"
    "nvidia-smi\n"
    "sudo dmesg | grep -i -E 'NVRM|Xid' | tail\n"
    "./run.sh stop && ./run.sh")
FIX_IMAGE = (
    "# rebuild the image (PyTorch is installed while building):\n"
    "docker build -t metro-lidar:gpu-dev solution && ./run.sh stop && ./run.sh")
FIX_FORCED = "# start without --cpu:\n./run.sh stop && ./run.sh"
FIX_MEMORY = (
    "# free GPU memory (close other GPU programs), then restart:\n"
    "nvidia-smi\n"
    "./run.sh stop && ./run.sh")

# code -> (Russian reason, English reason, fix)
REASONS = {
    "cpu_forced": ("Режим CPU выбран вручную (--cpu / METRO_DEVICE=cpu).",
                   "CPU mode was requested (--cpu / METRO_DEVICE=cpu).", FIX_FORCED),
    "no_toolkit": ("На хосте не установлен NVIDIA Container Toolkit: Docker не может дать контейнеру доступ к GPU.",
                   "The NVIDIA Container Toolkit is not installed on the host: Docker cannot give the container access to the GPU.",
                   FIX_TOOLKIT),
    "no_gpus_flag": ("Контейнер запущен без доступа к GPU (нет --gpus all или нет NVIDIA Container Toolkit; в контейнере нет /dev/nvidia*).",
                     "The container was started without GPU access (no --gpus all, or no NVIDIA Container Toolkit; no /dev/nvidia* inside).",
                     FIX_GPUS_FLAG),
    "no_host_gpu": ("На хосте не работает драйвер NVIDIA (nvidia-smi не выполняется) или нет GPU NVIDIA.",
                    "The NVIDIA driver does not work on the host (nvidia-smi fails) or there is no NVIDIA GPU.", FIX_DRIVER),
    "old_driver": ("Слишком старый драйвер NVIDIA: нужен >= %d (CUDA 12.8)." % MIN_DRIVER,
                   "The NVIDIA driver is too old: >= %d is required (CUDA 12.8)." % MIN_DRIVER, FIX_DRIVER),
    "torch_missing": ("PyTorch не установлен или не загружается в образе.",
                      "PyTorch is not installed or cannot be imported in the image.", FIX_IMAGE),
    "cuda_init_failed": ("CUDA не инициализируется в контейнере.", "CUDA cannot be initialised in the container.", FIX_RESTART),
    "arch_unsupported": ("Архитектура GPU не поддерживается этой сборкой PyTorch.",
                         "The GPU architecture is not supported by this PyTorch build.", FIX_DRIVER),
    "kernel_failed": ("Пробный расчёт на GPU завершился ошибкой.", "The GPU test computation failed.", FIX_RESTART),
    "low_memory": ("Недостаточно свободной памяти GPU.", "Not enough free GPU memory.", FIX_MEMORY),
    "runtime_failure": ("Сбой GPU во время работы — переключено на CPU.",
                        "GPU failure at run time - switched to the CPU.", FIX_RESTART),
}


def available_cpus() -> int:
    """CPUs this process may run on (honours --cpuset-cpus / taskset; os.cpu_count() does not)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        return max(1, os.cpu_count() or 1)


def default_torch_threads(mode: str, n_cpus: Optional[int] = None) -> int:
    """Intra-op threads for torch: METRO_TORCH_THREADS, else 2 on the GPU, else all CPUs but 2 (<= 8)."""
    env = os.environ.get("METRO_TORCH_THREADS", "").strip()
    if env.isdigit() and int(env) > 0:
        return int(env)
    n = available_cpus() if n_cpus is None else n_cpus
    if mode == "gpu":
        return max(1, min(2, n))
    return max(1, min(8, n - 2))


def host_driver_version() -> Optional[str]:
    """NVIDIA kernel driver version visible in this container/host, e.g. '580.178.04'."""
    try:
        with open("/proc/driver/nvidia/version", encoding="utf-8") as fh:
            m = re.search(r"Kernel Module\s+(?:for\s+\S+\s+)?([0-9]+\.[0-9]+(?:\.[0-9]+)?)", fh.read())
            if m:
                return m.group(1)
    except Exception:
        pass
    v = os.environ.get("METRO_HOST_DRIVER", "").strip()
    return v or None


def _driver_major(v: Optional[str]) -> Optional[int]:
    try:
        return int(str(v).split(".")[0])
    except Exception:
        return None


def arch_supported(capability: Tuple[int, int], arch_list) -> Tuple[bool, str]:
    """Can a GPU of ``capability`` run kernels of a build with ``arch_list`` (torch.cuda.get_arch_list())?

    Exact cubin, a cubin of the same major and lower minor (binary compatible, e.g. sm_86 on an
    sm_89 GPU) or PTX of a lower-or-equal architecture (JIT).  Returns (ok, how).
    """
    maj, mnr = int(capability[0]), int(capability[1])
    sm, ptx = [], []
    for a in arch_list or []:
        m = re.match(r"(sm|compute)_(\d+)(\d)[a-z]?$", str(a))
        if m:
            (sm if m.group(1) == "sm" else ptx).append((int(m.group(2)), int(m.group(3))))
    if (maj, mnr) in sm:
        return True, "native sm_%d%d kernels" % (maj, mnr)
    same = sorted(c for c in sm if c[0] == maj and c[1] <= mnr)
    if same:
        return True, "sm_%d%d kernels (binary compatible with sm_%d%d)" % (same[-1] + (maj, mnr))
    lower = sorted(c for c in ptx if c <= (maj, mnr))
    if lower:
        return True, "PTX compute_%d%d (JIT-compiled)" % lower[-1]
    return False, "no kernels for sm_%d%d in %s" % (maj, mnr, list(arch_list or []))


def _short(exc: BaseException, limit: int = 240) -> str:
    text = " ".join(str(exc).split())
    return (type(exc).__name__ + (": " + text if text else ""))[:limit]


class _PrintLogger:
    def info(self, msg):
        print(msg, file=sys.stderr, flush=True)

    def warning(self, msg):
        print(msg, file=sys.stderr, flush=True)


class DeviceManager:
    """See the module docstring.  Probes at construction; never raises (KeyboardInterrupt excepted)."""

    def __init__(self, prefer: str = "auto", team_contact: str = "Telegram: IRS-MIREA",
                 results_dir: Optional[str] = None, logger=None):
        env_pref = os.environ.get("METRO_DEVICE", "").strip().lower()
        prefer = (prefer or "auto").strip().lower()
        if prefer == "auto" and env_pref in ("gpu", "cpu", "auto"):
            prefer = env_pref
        self.prefer = prefer if prefer in ("auto", "gpu", "cpu") else "auto"
        self.team_contact = team_contact or _banner.DEFAULT_CONTACT
        self.results_dir = results_dir
        self.logger = logger or _PrintLogger()
        self._lock = threading.Lock()
        self.device = None
        self.mode = "cpu_emergency"
        self.code = ""
        self.reason = ""
        self.reason_ru = ""
        self.fix_hint = ""
        self.device_name = ""
        self.torch_version = None
        self.cuda_version = None
        self.capability = None
        self.arch_list = []
        self.arch_note = ""
        self.driver_version = None
        self.mem_free_mb = None
        self.mem_total_mb = None
        self.torch_threads = None
        self.probe_ms = None
        self.gpu_failures = 0
        self.last_failure = ""
        self._last_banner_s: Optional[float] = None
        self._torch = None
        self.probe()

    # ------------------------------------------------------------------ helpers
    def _log(self, level: str, msg: str) -> None:
        # one call site per severity: rclpy loggers key the severity by call site and refuse a change
        try:
            if level == "error":
                (getattr(self.logger, "error", None) or self.logger.info)(msg)
            elif level == "warning":
                (getattr(self.logger, "warning", None) or getattr(self.logger, "warn", None) or self.logger.info)(msg)
            else:
                self.logger.info(msg)
        except Exception:
            pass

    def _set_cpu(self, mode: str, code: str, detail: str = "") -> None:
        ru, en, fix = REASONS.get(code, REASONS["cuda_init_failed"])
        detail = " ".join(str(detail).split())[:MAX_DETAIL]
        self.mode, self.code = mode, code
        self.reason = en + ((" " + detail) if detail else "")
        self.reason_ru = ru + ((" " + detail) if detail else "")
        self.fix_hint = fix
        t = self._torch
        self.device = t.device("cpu") if t is not None else None

    def _import_torch(self):
        import importlib
        return importlib.import_module("torch")

    def _cpu_reason_without_cuda(self, cuda_warning: str) -> Tuple[str, str]:
        """(code, detail) when torch sees no usable CUDA device."""
        code = os.environ.get("METRO_GPU_REASON_CODE", "").strip()
        detail = " ".join(os.environ.get("METRO_GPU_REASON_DETAIL", "").split())[:MAX_DETAIL]
        if code in REASONS and code not in ("cpu_forced",):
            return code, detail
        dev_nodes = glob.glob("/dev/nvidia[0-9]*")
        drv = self.driver_version
        major = _driver_major(drv)
        if major is not None and major < MIN_DRIVER:
            return "old_driver", "Installed: %s." % drv
        if not dev_nodes:
            return "no_gpus_flag", ""
        low = cuda_warning.lower()
        if "too old" in low or "insufficient" in low:
            return "old_driver", ("Installed: %s. " % drv if drv else "") + cuda_warning[:200]
        return "cuda_init_failed", (cuda_warning[:240] if cuda_warning else "torch.cuda.is_available() is False.")

    # ------------------------------------------------------------------ interface C
    def probe(self) -> None:
        """import torch, CUDA available, test kernel (matmul + conv2d), free memory, arch supported.

        Never raises, except KeyboardInterrupt (Ctrl-C must still stop the program)."""
        t0 = time.perf_counter()
        try:
            self._probe()
        except BaseException as exc:  # never escape (KeyboardInterrupt included: a probe must not kill the node)
            if isinstance(exc, KeyboardInterrupt):
                raise
            self._set_cpu("cpu_emergency", "cuda_init_failed", _short(exc))
        self.probe_ms = round((time.perf_counter() - t0) * 1e3, 1)
        self._apply_threads()
        if self.mode == "gpu":
            self._log("info", "device: GPU %s (sm_%s, %s), torch %s, CUDA %s, %s MB free" % (
                self.device_name, (self.capability or "?").replace(".", ""), self.arch_note, self.torch_version,
                self.cuda_version, self.mem_free_mb))
            self._remove_warn_file()
        else:
            self._log("warning", "device: %s - %s" % (self.mode, self.reason))
            if self.mode == "cpu_emergency":
                self._write_warn_file()
            else:  # cpu_forced: a warning file of an earlier emergency run would be stale
                self._remove_warn_file()

    def _probe(self) -> None:
        self.driver_version = host_driver_version()
        try:
            torch = self._import_torch()
        except Exception as exc:
            self._torch = None
            self._set_cpu("cpu_forced" if self.prefer == "cpu" else "cpu_emergency",
                          "cpu_forced" if self.prefer == "cpu" else "torch_missing", _short(exc))
            return
        self._torch = torch
        self.torch_version = str(getattr(torch, "__version__", "?"))
        self.cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
        try:
            self.arch_list = list(torch.cuda.get_arch_list())
        except Exception:
            self.arch_list = []
        if self.prefer == "cpu":
            self._set_cpu("cpu_forced", "cpu_forced")
            return
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                ok = bool(torch.cuda.is_available())
            except Exception as exc:
                ok = False
                caught.append(warnings.WarningMessage(_short(exc), RuntimeWarning, "", 0))
        cuda_warning = " ".join(" ".join(str(w.message).split()) for w in caught)
        if not ok:
            code, detail = self._cpu_reason_without_cuda(cuda_warning)
            self._set_cpu("cpu_emergency", code, detail)
            return
        try:
            self.device_name = str(torch.cuda.get_device_name(0))
            cap = tuple(int(v) for v in torch.cuda.get_device_capability(0))
            self.capability = "%d.%d" % cap
        except Exception as exc:
            self._set_cpu("cpu_emergency", "cuda_init_failed", _short(exc))
            return
        ok, how = arch_supported(cap, self.arch_list) if self.arch_list else (True, "arch list unknown")
        self.arch_note = how
        if not ok:
            self._set_cpu("cpu_emergency", "arch_unsupported", "%s (sm_%d%d): %s." % (self.device_name, cap[0], cap[1], how))
            return
        dev = torch.device("cuda:0")
        try:
            self._kernel_check(torch, dev)
        except Exception as exc:
            self._set_cpu("cpu_emergency", "kernel_failed", "%s: %s" % (self.device_name, _short(exc)))
            return
        try:
            free, total = torch.cuda.mem_get_info(0)
            self.mem_free_mb, self.mem_total_mb = int(free // 2**20), int(total // 2**20)
        except Exception:
            self.mem_free_mb = self.mem_total_mb = None
        if self.mem_free_mb is not None and self.mem_free_mb < MIN_FREE_MB:
            self._set_cpu("cpu_emergency", "low_memory", "%d MB free of %s MB (need >= %d MB)." % (
                self.mem_free_mb, self.mem_total_mb, MIN_FREE_MB))
            return
        self.mode, self.code, self.reason, self.reason_ru, self.fix_hint = "gpu", "", "", "", ""
        self.device = dev

    @staticmethod
    def _kernel_check(torch, dev) -> None:
        """matmul + conv2d on the GPU, synchronised, both compared with the CPU result; raises on failure."""
        a = torch.randn(256, 256, device=dev)
        b = a @ a.t()
        x = torch.randn(1, 8, 64, 256, device=dev)
        w = torch.randn(16, 8, 3, 3, device=dev)
        y = torch.nn.functional.conv2d(x, w, padding=1)
        torch.cuda.synchronize()
        for name, got, ref in (("matmul", b.cpu(), a.cpu() @ a.cpu().t()),
                               ("conv2d", y.cpu(), torch.nn.functional.conv2d(x.cpu(), w.cpu(), padding=1))):
            if not bool(torch.isfinite(got).all()):
                raise RuntimeError("GPU %s result is not finite" % name)
            # loose tolerance: TF32 / different accumulation order on the GPU
            rel = float((got - ref).abs().max()) / (float(ref.abs().max()) + 1e-6)
            if not rel < 1e-2:
                raise RuntimeError("GPU %s result differs from the CPU result (relative error %.3g)" % (name, rel))

    def _apply_threads(self) -> None:
        t = self._torch
        if t is None:
            return
        try:
            n = default_torch_threads(self.mode)
            t.set_num_threads(n)
            self.torch_threads = int(t.get_num_threads())
        except Exception:
            pass

    def report_failure(self, exc: BaseException) -> None:
        """A GPU operation failed at run time: switch to cpu_emergency (logged once, banner due now)."""
        with self._lock:
            self.gpu_failures += 1
            self.last_failure = _short(exc)
            if self.mode != "gpu":
                return
            self._set_cpu("cpu_emergency", "runtime_failure", self.last_failure)
            self._apply_threads()
            self._last_banner_s = None
        self._log("warning", "device: GPU failure at run time (%s) - hot switch to the CPU (cpu_emergency)" % self.last_failure)
        self._write_warn_file()

    def banner_text(self, lang: str = "both") -> str:
        """RU+EN emergency banner incl. reason, fix, contact ('' in GPU mode)."""
        if self.mode == "gpu":
            return ""
        return _banner.render_from_state(self.status_dict(), lang=lang) or ""

    def maybe_repeat_banner(self, now_s: float) -> Optional[str]:
        """Banner at the first call and every REPEAT_S seconds in emergency mode (cpu_forced: first call only)."""
        with self._lock:
            if self.mode == "gpu":
                return None
            if self._last_banner_s is None:
                self._last_banner_s = float(now_s)
                return self.banner_text()
            if self.mode == "cpu_emergency" and float(now_s) - self._last_banner_s >= REPEAT_S:
                self._last_banner_s = float(now_s)
                return self.banner_text()
            return None

    def status_dict(self) -> dict:
        return {
            "mode": self.mode,
            "code": self.code,
            "reason": self.reason,
            "reason_ru": self.reason_ru,
            "fix_hint": self.fix_hint,
            "team_contact": self.team_contact,
            "prefer": self.prefer,
            "device": str(self.device) if self.device is not None else None,
            "device_name": self.device_name,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "capability": self.capability,
            "arch_list": list(self.arch_list),
            "arch_note": self.arch_note,
            "driver_version": self.driver_version,
            "mem_free_mb": self.mem_free_mb,
            "mem_total_mb": self.mem_total_mb,
            "torch_threads": self.torch_threads,
            "cpus_available": available_cpus(),
            "probe_ms": self.probe_ms,
            "gpu_failures": self.gpu_failures,
            "last_failure": self.last_failure,
        }

    # ------------------------------------------------------------------ warning file
    def _warn_path(self) -> Optional[str]:
        return os.path.join(self.results_dir, WARN_FILE) if self.results_dir else None

    def _write_warn_file(self) -> None:
        path = self._warn_path()
        if not path:
            return
        try:
            os.makedirs(self.results_dir, exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(time.strftime("%Y-%m-%d %H:%M:%S %Z") + "\n\n")
                fh.write(self.banner_text() + "\n\n")
                fh.write(json.dumps(self.status_dict(), ensure_ascii=False, indent=1) + "\n")
            os.replace(tmp, path)
        except Exception as exc:
            self._log("warning", "device: cannot write %s: %s" % (path, _short(exc)))

    def _remove_warn_file(self) -> None:
        path = self._warn_path()
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="probe the GPU / CPU runtime and print the device state")
    ap.add_argument("--prefer", default="auto", choices=["auto", "gpu", "cpu"])
    ap.add_argument("--results-dir", default=None, help="write ВНИМАНИЕ_GPU.txt here in emergency mode")
    ap.add_argument("--json", action="store_true", help="print status_dict() as JSON")
    ap.add_argument("--write-state", default="", help="also write status_dict() to this JSON file")
    ap.add_argument("--banner", action="store_true", help="print the banner in CPU modes")
    ap.add_argument("--color", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="no log line on stderr")
    args = ap.parse_args(argv)
    try:
        logger = type("Q", (), {"info": lambda s, m: None, "warning": lambda s, m: None})() if args.quiet else None
        dm = DeviceManager(prefer=args.prefer, results_dir=args.results_dir, logger=logger)
        st = dm.status_dict()
        if args.write_state:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(args.write_state)), exist_ok=True)
                with open(args.write_state, "w", encoding="utf-8") as fh:
                    json.dump(st, fh, ensure_ascii=False, indent=1)
            except Exception as exc:
                print("cannot write %s: %s" % (args.write_state, _short(exc)), file=sys.stderr)
        if args.json:
            print(json.dumps(st, ensure_ascii=False, indent=1))
        elif not args.banner:
            print("%s: %s" % (dm.mode, dm.device_name or dm.reason))
        if args.banner and dm.mode != "gpu":
            print(_banner.render_from_state(st, color=args.color), flush=True)
    except Exception as exc:  # pragma: no cover - never a traceback
        print("device check failed: %s" % _short(exc), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
