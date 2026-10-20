"""GPU gate for the neural detector Д3: can this machine train and run the U-Net on its GPU?

    python solution/train/gpu_gate.py [--json out.json]

Checks (each printed as OK/FAIL with the exact error):
  1. torch / CUDA build versions, device name, compute capability, arch list of the wheel
     (the judges' RTX 4070 Ti SUPER needs sm_89, our RTX 5060 Laptop sm_120);
  2. a conv net forward + backward + optimizer step in fp32 on the GPU;
  3. the same under torch.autocast(float16) with a GradScaler (AMP training);
  4. fp16 inference (model.half()) compared with fp32;
  5. the same network on the CPU (always; this is the emergency path).
Exit code 0 when the GPU checks pass, 1 otherwise (CPU-only machines exit 1 with the reason).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback


def _net(torch):
    nn = torch.nn
    return nn.Sequential(
        nn.Conv2d(21, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(64, 3, 1),
    )


def _train_steps(torch, device, amp: bool, steps: int = 5):
    torch.manual_seed(0)
    net = _net(torch).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    x = torch.randn(4, 21, 64, 256, device=device)
    y = torch.randint(0, 3, (4, 64, 256), device=device)
    losses = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = net(x)
            loss = torch.nn.functional.cross_entropy(out.float(), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(float(loss.detach()))
    if device.type == "cuda":
        torch.cuda.synchronize()
    grad_ok = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in net.parameters())
    return net, losses, grad_ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", help="write the result dict here")
    args = ap.parse_args(argv)
    res = {"checks": {}}

    def check(name, fn):
        t0 = time.perf_counter()
        try:
            info = fn()
            res["checks"][name] = {"ok": True, "info": info, "s": round(time.perf_counter() - t0, 3)}
            print(f"OK   {name}: {info}")
            return True
        except Exception as exc:  # report the exact error
            res["checks"][name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                                   "traceback": traceback.format_exc()}
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            return False

    try:
        import torch
    except Exception as exc:
        print(f"FAIL import torch: {exc}")
        return 1
    res["torch"] = torch.__version__
    res["torch_cuda"] = torch.version.cuda
    res["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    print(f"torch {torch.__version__}  CUDA build {torch.version.cuda}  cuDNN {res['cudnn']}")

    def env():
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False")
        cap = torch.cuda.get_device_capability(0)
        arch = torch.cuda.get_arch_list()
        info = {"device": torch.cuda.get_device_name(0), "capability": f"sm_{cap[0]}{cap[1]}",
                "arch_list": arch, "has_sm_89": "sm_89" in arch, "has_sm_120": "sm_120" in arch,
                "total_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)}
        if f"sm_{cap[0]}{cap[1]}" not in arch and f"compute_{cap[0]}{cap[1]}" not in arch:
            raise RuntimeError(f"wheel has no kernels for sm_{cap[0]}{cap[1]}: {arch}")
        return info

    gpu_ok = check("cuda_device", env)
    if gpu_ok:
        torch.backends.cudnn.benchmark = True
        dev = torch.device("cuda:0")

        def fp32():
            _, losses, grad_ok = _train_steps(torch, dev, amp=False)
            if not grad_ok:
                raise RuntimeError("non-finite or missing gradients")
            return {"loss_first_last": [round(losses[0], 4), round(losses[-1], 4)]}

        def amp():
            _, losses, grad_ok = _train_steps(torch, dev, amp=True)
            if not grad_ok:
                raise RuntimeError("non-finite or missing gradients")
            return {"loss_first_last": [round(losses[0], 4), round(losses[-1], 4)]}

        def half_infer():
            net, _, _ = _train_steps(torch, dev, amp=False, steps=1)
            net.eval()
            x = torch.randn(1, 21, 64, 256, device=dev)
            with torch.no_grad():
                p32 = torch.softmax(net(x), 1)
                p16 = torch.softmax(net.half()(x.half()).float(), 1)
            torch.cuda.synchronize()
            return {"max_abs_dprob_fp16_vs_fp32": float((p32 - p16).abs().max())}

        gpu_ok &= check("gpu_train_fp32", fp32)
        gpu_ok &= check("gpu_train_amp_fp16", amp)
        gpu_ok &= check("gpu_infer_fp16", half_infer)

    def cpu():
        _, losses, grad_ok = _train_steps(torch, torch.device("cpu"), amp=False, steps=2)
        if not grad_ok:
            raise RuntimeError("non-finite or missing gradients")
        return {"loss_first_last": [round(losses[0], 4), round(losses[-1], 4)], "threads": torch.get_num_threads()}

    check("cpu_train_fp32", cpu)
    res["gpu_gate_passed"] = bool(gpu_ok)
    print("GPU GATE:", "PASSED" if gpu_ok else "FAILED")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
    return 0 if gpu_ok else 1


if __name__ == "__main__":
    sys.exit(main())
