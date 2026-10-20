"""Emergency CPU-mode banner (Russian + English), big and readable in a terminal.

Used by ``device.DeviceManager.banner_text`` and by the ``metro`` console.  Pure
Python, no ROS / torch imports.

    python3 -m metro_obstacle_detector.banner [--state /results/logs/device.json] [--lang both|ru|en] [--color]

prints the banner of the device state written by ``python3 -m
metro_obstacle_detector.device --write-state`` (nothing when the GPU is in use).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from typing import Iterable, List, Optional

TEAM = "IRS-MIREA"
DEFAULT_CONTACT = "Telegram: IRS-MIREA"
WIDTH = 96

RED = "\033[1;97;41m"
YEL = "\033[1;33m"
RST = "\033[0m"

TEXT = {
    "ru": {
        "title": "ВНИМАНИЕ: GPU НЕДОСТУПЕН — РЕШЕНИЕ РАБОТАЕТ В АВАРИЙНОМ РЕЖИМЕ НА CPU",
        "title_forced": "РЕЖИМ CPU ВКЛЮЧЁН ВРУЧНУЮ (--cpu) — GPU НЕ ИСПОЛЬЗУЕТСЯ",
        "better": "С GPU работает нейросетевой детектор Д3 в каждом кадре (дополнительные свидетельства, на Falcon +22 кадра дальнего прохода) и выше частота обработки; на CPU Д3 считается каждый третий кадр.",
        "reason": "Причина:",
        "fix": "Как включить GPU (выполнить на хосте):",
        "contact": "Свяжитесь с командой {team} в {contact}",
    },
    "en": {
        "title": "WARNING: GPU UNAVAILABLE - THE SOLUTION RUNS IN AN EMERGENCY CPU MODE",
        "title_forced": "CPU MODE FORCED (--cpu) - THE GPU IS NOT USED",
        "better": "With the GPU the neural-network detector D3 runs on every frame (additional evidence, +22 frames of the far walk on the Falcon recording) and the frame rate is higher; on the CPU D3 runs on every third frame.",
        "reason": "Reason:",
        "fix": "How to enable the GPU (run on the host):",
        "contact": "Contact team {team} on {contact}",
    },
}


def _wrap(text: str, width: int, indent: str = "") -> List[str]:
    out: List[str] = []
    for para in str(text).splitlines() or [""]:
        if not para.strip():
            out.append("")
            continue
        lead = para[: len(para) - len(para.lstrip())]
        out.extend(textwrap.wrap(para.strip(), width=max(20, width - len(indent) - len(lead)),
                                 initial_indent=indent + lead, subsequent_indent=indent + lead + "  ",
                                 break_long_words=False, break_on_hyphens=False) or [indent + lead])
    return out


def _box(lines: Iterable[str], width: int, color: bool, title_rows: int) -> str:
    """Bars of '#' above and below, '# ' on the left; no right border, so long command lines stay intact."""
    bar = "#" * width
    rows = [bar, bar]
    for i, ln in enumerate(lines):
        if color and i < title_rows:
            rows.append("# " + RED + ln.ljust(width - 2) + RST)
        else:
            rows.append(("# " + ln).rstrip())
    rows += [bar, bar]
    if color:
        rows = [YEL + r + RST if r.startswith("##") else r for r in rows]
    return "\n".join(rows)


def render(reason: str, fix_hint: str, contact: str = DEFAULT_CONTACT, lang: str = "both",
           forced: bool = False, width: int = WIDTH, color: bool = False, reason_ru: Optional[str] = None) -> str:
    """The banner as text.  ``lang``: "both" (RU then EN), "ru" or "en".

    ``reason`` is shown in the English part, ``reason_ru`` (default: ``reason``) in the Russian one.
    Command lines of ``fix_hint`` are never wrapped (copy-paste must work); the box has no right border.
    """
    langs = ["ru", "en"] if lang == "both" else [lang if lang in TEXT else "en"]
    fix_lines = [ln.rstrip() for ln in (fix_hint or "-").splitlines()]
    inner = width - 2
    lines: List[str] = []
    title_rows = 0
    for k, code in enumerate(langs):
        t = TEXT[code]
        if k:
            lines.append("-" * inner)
        head = _wrap(t["title_forced" if forced else "title"], inner)
        if k == 0:
            title_rows = len(head)
        lines += head
        lines.append("")
        lines += _wrap(t["better"], inner)
        lines.append("")
        lines += _wrap(t["reason"], inner)
        lines += _wrap(((reason_ru or reason) if code == "ru" else reason) or "-", inner, "  ")
        lines.append("")
        lines += _wrap(t["fix"], inner)
        lines += ["  " + ln for ln in fix_lines]
        lines.append("")
        lines += _wrap(t["contact"].format(team=TEAM, contact=contact), inner)
    return _box(lines, width, color, title_rows)


def render_from_state(state: dict, lang: str = "both", color: bool = False, width: int = WIDTH) -> Optional[str]:
    """Banner for a ``DeviceManager.status_dict()``; None in GPU mode."""
    mode = state.get("mode")
    if mode not in ("cpu_emergency", "cpu_forced"):
        return None
    return render(state.get("reason", ""), state.get("fix_hint", ""), state.get("team_contact", DEFAULT_CONTACT), lang,
                  forced=mode == "cpu_forced", width=width, color=color, reason_ru=state.get("reason_ru") or None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="print the emergency CPU-mode banner of the current device state")
    ap.add_argument("--state", default=os.path.join(os.environ.get("METRO_RESULTS_DIR", "/results"), "logs", "device.json"))
    ap.add_argument("--lang", default="both", choices=["both", "ru", "en"])
    ap.add_argument("--color", action="store_true")
    ap.add_argument("--width", type=int, default=WIDTH)
    args = ap.parse_args(argv)
    try:
        with open(args.state, encoding="utf-8") as fh:
            state = json.load(fh)
    except Exception:
        return 0
    try:
        text = render_from_state(state, args.lang, args.color, args.width)
        if text:
            print(text, flush=True)
    except Exception as exc:  # never a traceback on the console
        print(f"(banner unavailable: {exc})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
