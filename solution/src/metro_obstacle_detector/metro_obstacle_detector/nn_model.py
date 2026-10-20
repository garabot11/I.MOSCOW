"""Д3 network: a small U-Net for LiDAR range images (owner: ml-builder).

Input  (N, 21, 64, 256) float32: 7 channels per frame (interface D, ``nn_input.CHANNELS``) for the
current frame and the two previous ones.  Output (N, 3, 64, 256) logits for
{background, obstacle, dust} (``CLASSES``).

Only plain operators are used (3x3 / 1x1 convolutions, strided convolutions for down-sampling,
BatchNorm, ReLU, nearest up-sampling, addition / concatenation) so that the model runs with the
stock PyTorch kernels on any GPU the wheel supports and on the CPU, and exports to TorchScript.
Spatial size must be divisible by ``2 ** len(widths) - 1`` (8 for the default 4 levels).

``decoder="add"`` (default) projects the up-sampled features with a 1x1 convolution and adds the
skip connection (cheap at full resolution); ``decoder="concat"`` is the classic U-Net decoder
(about twice the compute).  See ``count_parameters`` / ``count_macs``.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import torch
from torch import nn
import torch.nn.functional as F

CLASSES = ("background", "obstacle", "dust")


def conv_bn_relu(cin: int, cout: int, k: int = 3, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class _Up(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int, mode: str):
        super().__init__()
        self.mode = mode
        if mode == "add":
            self.proj = conv_bn_relu(cin, cskip, k=1)
            self.fuse = conv_bn_relu(cskip, cout)
        elif mode == "concat":
            self.proj = nn.Identity()
            self.fuse = nn.Sequential(conv_bn_relu(cin + cskip, cout), conv_bn_relu(cout, cout))
        else:
            raise ValueError(f"unknown decoder mode {mode!r}")

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        if self.mode == "add":
            return self.fuse(self.proj(x) + skip)
        return self.fuse(torch.cat([x, skip], dim=1))


class RangeUNet(nn.Module):
    """U-Net over (rows, columns) of the cropped range image."""

    def __init__(self, in_channels: int = 21, n_classes: int = 3, widths: Sequence[int] = (24, 48, 96, 160),
                 decoder: str = "add"):
        super().__init__()
        widths = [int(w) for w in widths]
        if len(widths) < 2:
            raise ValueError("need at least two levels")
        self.config: Dict[str, object] = {"in_channels": int(in_channels), "n_classes": int(n_classes),
                                          "widths": list(widths), "decoder": decoder}
        self.stem = nn.Sequential(conv_bn_relu(in_channels, widths[0]), conv_bn_relu(widths[0], widths[0]))
        downs: List[nn.Module] = []
        for a, b in zip(widths[:-1], widths[1:]):
            downs.append(nn.Sequential(conv_bn_relu(a, b, stride=2), conv_bn_relu(b, b)))
        self.downs = nn.ModuleList(downs)
        ups: List[nn.Module] = []
        for i in range(len(widths) - 1, 0, -1):
            ups.append(_Up(widths[i], widths[i - 1], widths[i - 1], decoder))
        self.ups = nn.ModuleList(ups)
        self.head = nn.Conv2d(widths[0], n_classes, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []
        x = self.stem(x)
        for down in self.downs:
            skips.append(x)
            x = down(x)
        i = len(skips) - 1
        for up in self.ups:
            x = up(x, skips[i])
            i -= 1
        return self.head(x)


def build_model(config: Dict[str, object] | None = None) -> RangeUNet:
    cfg = dict(config or {})
    return RangeUNet(in_channels=int(cfg.get("in_channels", 21)), n_classes=int(cfg.get("n_classes", 3)),
                     widths=tuple(cfg.get("widths", (24, 48, 96, 160))), decoder=str(cfg.get("decoder", "add")))


def export_d3(model: RangeUNet, path: str, meta: Dict[str, object], example_shape=(1, 21, 64, 256)) -> Dict[str, str]:
    """Write ``path`` (TorchScript, fp32, CPU, ``meta.json`` in the extra files) and the state-dict
    fallback ``<stem>.state.pt`` (rebuilt with ``build_model`` when TorchScript cannot be loaded,
    e.g. by an older torch).  Returns both paths.  The traced module is checked against eager mode."""
    import json
    import os

    model = model.eval().to("cpu").float()
    meta = dict(meta)
    meta.setdefault("format", "d3_unet/v1")
    meta["model_config"] = dict(model.config)
    meta["n_parameters"] = count_parameters(model)
    meta["torch_version"] = torch.__version__
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    example = torch.randn(*example_shape)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
        ref = model(example)
        got = traced(example)
    if not torch.allclose(ref, got, atol=1e-4, rtol=1e-4):
        raise RuntimeError("traced model differs from the eager model")
    torch.jit.save(traced, path, _extra_files={"meta.json": json.dumps(meta, sort_keys=True)})
    stem, _ = os.path.splitext(path)
    state_path = stem + ".state.pt"
    torch.save({"format": "d3_unet_state/v1", "model_config": dict(model.config),
                "state_dict": {k: v.detach().clone() for k, v in model.state_dict().items()},
                "meta": json.loads(json.dumps(meta))}, state_path)
    return {"torchscript": path, "state_dict": state_path}


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def count_macs(model: nn.Module, shape=(1, 21, 64, 256)) -> int:
    """Multiply-accumulates of all convolutions for one forward pass (forward hooks, CPU)."""
    total = [0]

    def hook(mod, inp, out):
        k = mod.kernel_size[0] * mod.kernel_size[1]
        total[0] += int(out.numel() // out.shape[0]) * (mod.in_channels // mod.groups) * k

    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(torch.zeros(shape))
    finally:
        for h in hs:
            h.remove()
        model.train(was_training)
    return total[0]
