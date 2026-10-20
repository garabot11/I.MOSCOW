"""DeviceManager (interface C) with a fake torch: no torch, CUDA unavailable, kernel failure,
unsupported arch, low memory, run-time failure -> hot switch, banner repeat, warning file."""
import json
import os
import sys
import types

import pytest

from metro_obstacle_detector import banner, device
from metro_obstacle_detector.device import DeviceManager, arch_supported


class _Dev:
    def __init__(self, kind):
        self.type = kind.split(":")[0]
        self.kind = kind

    def __str__(self):
        return self.kind

    def __eq__(self, other):
        return str(self) == str(other)


def fake_torch(available=True, capability=(8, 9), arch=("sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"),
               free_mb=12000, warn=None):
    t = types.ModuleType("torch")
    t.__version__ = "2.11.0+cu128"
    t.version = types.SimpleNamespace(cuda="12.8")
    t.device = _Dev
    t._threads = [1]
    t.set_num_threads = lambda n: t._threads.__setitem__(0, n)
    t.get_num_threads = lambda: t._threads[0]

    def is_available():
        if warn:
            import warnings
            warnings.warn(warn, UserWarning)
        return available

    t.cuda = types.SimpleNamespace(
        is_available=is_available,
        get_arch_list=lambda: list(arch),
        get_device_name=lambda i=0: "NVIDIA GeForce RTX 4070 Ti SUPER",
        get_device_capability=lambda i=0: capability,
        mem_get_info=lambda i=0: (free_mb * 2**20, 16376 * 2**20),
        synchronize=lambda: None,
    )
    return t


@pytest.fixture
def env(monkeypatch, tmp_path):
    for k in ("METRO_DEVICE", "METRO_GPU_REASON_CODE", "METRO_GPU_REASON_DETAIL", "METRO_HOST_DRIVER", "METRO_TORCH_THREADS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(device, "host_driver_version", lambda: None)
    monkeypatch.setattr(device.glob, "glob", lambda pat: [])
    monkeypatch.setattr(DeviceManager, "_kernel_check", staticmethod(lambda torch, dev: None))
    return tmp_path


class Quiet:
    def __init__(self):
        self.lines = []

    def info(self, m):
        self.lines.append(("info", m))

    def warning(self, m):
        self.lines.append(("warning", m))


def use(monkeypatch, torch_mod):
    if torch_mod is None:
        def boom(self):
            raise ImportError("No module named 'torch'")
        monkeypatch.setattr(DeviceManager, "_import_torch", boom)
    else:
        monkeypatch.setattr(DeviceManager, "_import_torch", lambda self: torch_mod)


def test_import_without_torch_and_no_torch_mode(monkeypatch, env):
    use(monkeypatch, None)
    dm = DeviceManager(results_dir=str(env), logger=Quiet())
    assert dm.device is None and dm.mode == "cpu_emergency" and dm.code == "torch_missing"
    assert "docker build" in dm.fix_hint
    assert (env / device.WARN_FILE).exists()


def test_gpu_mode(monkeypatch, env):
    use(monkeypatch, fake_torch())
    (env / device.WARN_FILE).write_text("stale")
    dm = DeviceManager(results_dir=str(env), logger=Quiet())
    assert dm.mode == "gpu" and str(dm.device) == "cuda:0" and dm.reason == "" and dm.fix_hint == ""
    st = dm.status_dict()
    assert st["capability"] == "8.9" and st["torch_version"] == "2.11.0+cu128" and st["cuda_version"] == "12.8"
    assert "sm_86" in st["arch_note"] and "binary compatible" in st["arch_note"]
    assert st["torch_threads"] <= 2
    assert not (env / device.WARN_FILE).exists()  # stale warning of an earlier run removed
    assert dm.banner_text() == "" and dm.maybe_repeat_banner(0.0) is None


def test_cuda_unavailable_no_device_nodes(monkeypatch, env):
    use(monkeypatch, fake_torch(available=False))
    dm = DeviceManager(results_dir=str(env), logger=Quiet())
    assert dm.mode == "cpu_emergency" and dm.code == "no_gpus_flag" and str(dm.device) == "cpu"
    assert "--gpus all" in dm.reason and "--gpus all" in dm.fix_hint and "nvidia-container-toolkit" in dm.fix_hint
    text = (env / device.WARN_FILE).read_text(encoding="utf-8")
    assert "Свяжитесь с командой IRS-MIREA в Telegram: IRS-MIREA" in text
    assert "Contact team IRS-MIREA on Telegram: IRS-MIREA" in text
    assert '"mode": "cpu_emergency"' in text


def test_reason_from_host_launcher(monkeypatch, env):
    monkeypatch.setenv("METRO_GPU_REASON_CODE", "no_toolkit")
    monkeypatch.setenv("METRO_GPU_REASON_DETAIL", "docker info: runtimes runc")
    use(monkeypatch, fake_torch(available=False))
    dm = DeviceManager(logger=Quiet())
    assert dm.code == "no_toolkit" and "Container Toolkit" in dm.reason and "runtimes runc" in dm.reason
    assert "sudo nvidia-ctk runtime configure --runtime=docker" in dm.fix_hint


def test_long_detail_is_capped(monkeypatch, env):
    monkeypatch.setenv("METRO_GPU_REASON_CODE", "no_toolkit")
    monkeypatch.setenv("METRO_GPU_REASON_DETAIL", "docker runtimes: " + "§" * 5500)
    use(monkeypatch, fake_torch(available=False))
    dm = DeviceManager(logger=Quiet())
    assert len(dm.reason) < len(device.REASONS["no_toolkit"][1]) + device.MAX_DETAIL + 2
    assert dm.banner_text().count("§") <= 2 * device.MAX_DETAIL  # RU and EN part


def test_old_driver(monkeypatch, env):
    monkeypatch.setattr(device, "host_driver_version", lambda: "535.183.01")
    monkeypatch.setattr(device.glob, "glob", lambda pat: ["/dev/nvidia0"])
    use(monkeypatch, fake_torch(available=False, warn="The NVIDIA driver on your system is too old"))
    dm = DeviceManager(logger=Quiet())
    assert dm.code == "old_driver" and "535.183.01" in dm.reason and "nvidia-driver-580" in dm.fix_hint


def test_cuda_init_failure_keeps_torch_warning(monkeypatch, env):
    monkeypatch.setattr(device.glob, "glob", lambda pat: ["/dev/nvidia0"])
    use(monkeypatch, fake_torch(available=False, warn="CUDA initialization: CUDA unknown error"))
    dm = DeviceManager(logger=Quiet())
    assert dm.code == "cuda_init_failed" and "CUDA unknown error" in dm.reason


def test_kernel_failure(monkeypatch, env):
    use(monkeypatch, fake_torch())

    def bad(torch, dev):
        raise RuntimeError("CUDA error: no kernel image is available for execution on the device")
    monkeypatch.setattr(DeviceManager, "_kernel_check", staticmethod(bad))
    dm = DeviceManager(logger=Quiet())
    assert dm.mode == "cpu_emergency" and dm.code == "kernel_failed" and "no kernel image" in dm.reason
    assert str(dm.device) == "cpu"


def test_unsupported_arch_and_low_memory(monkeypatch, env):
    use(monkeypatch, fake_torch(capability=(6, 1)))
    dm = DeviceManager(logger=Quiet())
    assert dm.code == "arch_unsupported" and "sm_61" in dm.reason
    use(monkeypatch, fake_torch(free_mb=100))
    dm = DeviceManager(logger=Quiet())
    assert dm.code == "low_memory" and "100 MB" in dm.reason


def test_forced_cpu(monkeypatch, env):
    use(monkeypatch, fake_torch())
    monkeypatch.setenv("METRO_DEVICE", "cpu")
    (env / device.WARN_FILE).write_text("stale")
    dm = DeviceManager(results_dir=str(env), logger=Quiet())
    assert dm.mode == "cpu_forced" and str(dm.device) == "cpu"
    assert not (env / device.WARN_FILE).exists()
    assert dm.maybe_repeat_banner(0.0) and dm.maybe_repeat_banner(1000.0) is None  # once, no repeats
    assert "--cpu" in dm.banner_text()


def test_runtime_failure_hot_switch_logs_once(monkeypatch, env):
    use(monkeypatch, fake_torch())
    log = Quiet()
    dm = DeviceManager(results_dir=str(env), logger=log)
    assert dm.mode == "gpu"
    dm.report_failure(RuntimeError("CUDA error: an illegal memory access was encountered"))
    dm.report_failure(RuntimeError("CUDA error: again"))
    assert dm.mode == "cpu_emergency" and dm.code == "runtime_failure" and str(dm.device) == "cpu"
    assert dm.gpu_failures == 2 and "illegal memory access" in dm.reason
    assert sum(1 for lvl, m in log.lines if "hot switch" in m) == 1
    assert (env / device.WARN_FILE).exists()
    assert dm.status_dict()["torch_threads"] >= 1


def test_banner_repeat_every_300_s(monkeypatch, env):
    use(monkeypatch, fake_torch(available=False))
    dm = DeviceManager(logger=Quiet())
    assert dm.maybe_repeat_banner(10.0)
    assert dm.maybe_repeat_banner(200.0) is None
    assert dm.maybe_repeat_banner(309.9) is None
    assert dm.maybe_repeat_banner(310.0)
    assert dm.maybe_repeat_banner(400.0) is None


def test_banner_content_both_languages(monkeypatch, env):
    use(monkeypatch, fake_torch(available=False))
    dm = DeviceManager(logger=Quiet())
    text = dm.banner_text()
    for needle in ("GPU НЕДОСТУПЕН", "аварийном режиме на CPU", "Д3 в каждом кадре", "каждый третий кадр", "Причина:",
                   "Как включить GPU", "Свяжитесь с командой IRS-MIREA в Telegram: IRS-MIREA",
                   "GPU UNAVAILABLE", "emergency CPU mode", "D3 runs on every frame", "every third frame", "frame rate",
                   "Reason:", "How to enable the GPU", "Contact team IRS-MIREA on Telegram: IRS-MIREA", "./run.sh"):
        flat = " ".join(ln.lstrip("# ").strip() for ln in text.splitlines())  # banner lines are wrapped
        assert needle.lower() in flat.lower(), needle
    rows = text.splitlines()
    assert all(r.startswith("#") for r in rows) and rows[0] == "#" * banner.WIDTH
    assert "sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker" in text  # never wrapped
    assert "\033" not in text
    ru = dm.banner_text("ru")
    assert "Telegram: IRS-MIREA" in ru and "GPU UNAVAILABLE" not in ru


def test_banner_cli_from_state(monkeypatch, env, capsys):
    use(monkeypatch, fake_torch(available=False))
    dm = DeviceManager(logger=Quiet())
    p = env / "device.json"
    p.write_text(json.dumps(dm.status_dict()), encoding="utf-8")
    assert banner.main(["--state", str(p)]) == 0
    assert "IRS-MIREA" in capsys.readouterr().out
    assert banner.main(["--state", str(env / "missing.json")]) == 0


def test_arch_supported():
    arch = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]
    assert arch_supported((12, 0), arch) == (True, "native sm_120 kernels")
    ok, how = arch_supported((8, 9), arch)
    assert ok and "sm_86" in how
    assert arch_supported((6, 1), arch)[0] is False
    assert arch_supported((8, 9), ["sm_80", "compute_89"])[0]


def test_threads_follow_affinity(monkeypatch):
    monkeypatch.delenv("METRO_TORCH_THREADS", raising=False)
    assert device.default_torch_threads("cpu_emergency", 8) == 6
    assert device.default_torch_threads("cpu_emergency", 2) == 1
    assert device.default_torch_threads("gpu", 8) == 2
    monkeypatch.setenv("METRO_TORCH_THREADS", "3")
    assert device.default_torch_threads("cpu_emergency", 8) == 3


def test_real_torch_if_present(tmp_path, monkeypatch):
    """In the image: the real torch import path; without a GPU it must be an emergency CPU mode, never an exception."""
    pytest.importorskip("torch")
    monkeypatch.delenv("METRO_DEVICE", raising=False)
    dm = DeviceManager(results_dir=str(tmp_path), logger=Quiet())
    assert dm.mode in ("gpu", "cpu_emergency")
    assert dm.device is not None and dm.torch_version
    if dm.mode == "cpu_emergency":
        assert dm.reason and dm.fix_hint and (tmp_path / device.WARN_FILE).exists()
