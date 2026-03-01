"""
nanoPOPIXA — Monitor d'entraînement terminal
Courbes Braille truecolor · live depuis train.log
Usage : python monitor.py   ou   popixa monitor
"""

import re
import os
import sys
import time
import argparse
from dataclasses import dataclass
from typing import Optional


# ─── Couleurs truecolor ───────────────────────────────────────────────────────
R   = "\033[0m"
B   = "\033[1m"

def fg(r, g, b): return f"\033[38;2;{r};{g};{b}m"

# Tuples (r,g,b) bruts — utilisés pour lerp_color()
TRAIN_A = (255,  80,  80)   # rouge  — début train loss
TRAIN_B = ( 80, 255, 120)   # vert   — fin train loss
VAL_A   = (180,  80, 255)   # violet — début val loss
VAL_B   = (  0, 220, 255)   # cyan   — fin val loss
LR_C    = (255, 180,   0)   # orange — learning rate
# Strings ANSI — utilisées directement pour l'affichage
INFO_C  = fg(160, 160, 200)
HDR_C   = fg(200, 120, 255)
BOX_C   = fg(120, 120, 180)

def lerp_color(a, b, t):
    """Interpolation linéaire entre deux couleurs RGB."""
    t = max(0.0, min(1.0, t))
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )


# ─── Structures de données ────────────────────────────────────────────────────
@dataclass
class Entry:
    iter:       int
    train_loss: float
    val_loss:   float
    lr:         float
    duration:   float   # secondes par itération (delta depuis la dernière éval)


def _fmt_time(secs: float) -> str:
    """Formate une durée en secondes → chaîne lisible."""
    if secs < 60:
        return f"{secs:.0f}s"
    if secs < 3600:
        m, s = divmod(int(secs), 60)
        return f"{m}m{s:02d}s"
    h, rem = divmod(int(secs), 3600)
    return f"{h}h{rem//60:02d}m"


# ─── Parser ───────────────────────────────────────────────────────────────────
_LOG_RE     = re.compile(
    r"iter\s+(\d+)\s*\|\s*train\s+([\d.]+)\s*\|\s*val\s+([\d.]+)"
    r"\s*\|\s*lr\s+([\d.eE+\-]+)\s*\|\s*([\d.]+)s"
)
_HEADER_RE  = re.compile(r"#\s*max_iters=(\d+)\s+eval_interval=(\d+)")


def parse_log(path: str) -> tuple[list[Entry], Optional[int], int]:
    """Lit le fichier log. Retourne (entries, max_iters, eval_interval)."""
    entries       = []
    max_iters     = None
    eval_interval = 1
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                mh = _HEADER_RE.match(line)
                if mh:
                    max_iters     = int(mh.group(1))
                    eval_interval = int(mh.group(2))
                    continue
                m = _LOG_RE.search(line)
                if m:
                    try:
                        entries.append(Entry(
                            iter       = int(m.group(1)),
                            train_loss = float(m.group(2)),
                            val_loss   = float(m.group(3)),
                            lr         = float(m.group(4)),
                            duration   = float(m.group(5)),
                        ))
                    except (ValueError, AttributeError):
                        continue
    except (FileNotFoundError, PermissionError):
        pass
    return entries, max_iters, eval_interval


# ─── Canvas Braille ────────────────────────────────────────────────────────────
_DOT = [
    [0x01, 0x08],   # row 0 : col 0, col 1
    [0x02, 0x10],   # row 1
    [0x04, 0x20],   # row 2
    [0x40, 0x80],   # row 3
]

class BrailleCanvas:
    """Canvas 2D → caractères Braille Unicode."""

    def __init__(self, w: int, h: int):
        """w, h en caractères (pixels = w*2, h*4)."""
        self.w = w
        self.h = h
        self.bits   = [[0] * w for _ in range(h)]
        self.colors = [[None] * w for _ in range(h)]

    def clear(self):
        for row in self.bits:
            row[:] = [0] * self.w
        for row in self.colors:
            row[:] = [None] * self.w

    def set_pixel(self, px: int, py: int, color: Optional[tuple] = None):
        """Allume le pixel (px, py) en coordonnées pixels."""
        cx, cy = px // 2, py // 4
        if 0 <= cx < self.w and 0 <= cy < self.h:
            self.bits[cy][cx]   |= _DOT[py % 4][px % 2]
            if color is not None:
                self.colors[cy][cx] = color

    def draw_curve(self, values: list[float],
                   y_min: float, y_max: float,
                   color_a: tuple, color_b: tuple):
        """Trace une courbe depuis `values`, points reliés par des lignes verticales."""
        if not values or y_max == y_min:
            return
        n   = len(values)
        pw  = self.w * 2
        ph  = self.h * 4

        prev_px: Optional[int] = None
        prev_py: Optional[int] = None

        for i, v in enumerate(values):
            px    = int(i / max(n - 1, 1) * (pw - 1))
            norm  = (v - y_min) / (y_max - y_min)
            py    = int((1.0 - norm) * (ph - 1))
            t     = i / max(n - 1, 1)
            color = lerp_color(color_a, color_b, t)

            # Relier verticalement au point précédent si même colonne pixel
            if prev_px is not None and px == prev_px and prev_py != py:
                for yy in range(min(prev_py, py), max(prev_py, py) + 1):
                    self.set_pixel(px, yy, color)
            else:
                self.set_pixel(px, py, color)

            prev_px, prev_py = px, py

    def render(self) -> list[str]:
        """Retourne une liste de lignes ANSI."""
        lines = []
        for cy in range(self.h):
            line = ""
            prev_color = None
            for cx in range(self.w):
                bits  = self.bits[cy][cx]
                color = self.colors[cy][cx]
                if color is not None and color != prev_color:
                    line += fg(*color)
                    prev_color = color
                elif color is None and prev_color is not None:
                    line += R
                    prev_color = None
                line += chr(0x2800 + bits)
            line += R
            lines.append(line)
        return lines


# ─── Layout ───────────────────────────────────────────────────────────────────
def _box_top(w: int, title: str = "") -> str:
    inner = w - 2
    if title:
        t   = f" {title} "
        pad = inner - 1 - len(t)   # -1 pour le ═ du préfixe ╔═
        pad = max(0, pad)
        return BOX_C + "╔═" + t + "═" * pad + "╗" + R
    return BOX_C + "╔" + "═" * inner + "╗" + R

def _box_mid(w: int, title: str = "") -> str:
    inner = w - 2
    if title:
        t   = f" {title} "
        pad = inner - 1 - len(t)   # -1 pour le ═ du préfixe ╠═
        pad = max(0, pad)
        return BOX_C + "╠═" + t + "═" * pad + "╣" + R
    return BOX_C + "╠" + "═" * inner + "╣" + R

def _box_bot(w: int) -> str:
    return BOX_C + "╚" + "═" * (w - 2) + "╝" + R

def _box_row(content: str, w: int) -> str:
    visible = re.sub(r"\033\[[^m]*m", "", content)
    pad     = max(0, w - 2 - len(visible))
    return BOX_C + "║" + R + content + " " * pad + BOX_C + "║" + R

def _fmt_y(v: float) -> str:
    """Format 5 chars pour les labels Y — s'adapte à la magnitude."""
    if v == 0:
        return " 0.00"
    if abs(v) >= 0.01:
        return f"{v:5.2f}"
    return f"{v:.0e}"   # ex: "4e-04" — toujours 5 chars

def _chart_section(canvas: BrailleCanvas,
                   y_min: float, y_max: float, w: int) -> list[str]:
    rows        = []
    lines_cv    = canvas.render()
    h           = canvas.h
    for i, line in enumerate(lines_cv):
        if i == 0:
            ytag = _fmt_y(y_max) + " " + BOX_C + "┤" + R
        elif i == h - 1:
            ytag = _fmt_y(y_min) + " " + BOX_C + "┤" + R
        else:
            ytag = "       " + BOX_C + "┤" + R
        rows.append(_box_row(ytag + line, w))
    return rows


def render_dashboard(entries: list[Entry], w: int = 72, log_path: str = "train.log",
                     max_iters: Optional[int] = None, eval_interval: int = 1) -> list[str]:
    """Construit le dashboard complet. Retourne une liste de lignes."""
    lines = []

    if not entries:
        lines.append(_box_top(w, "nanoPOPIXA Monitor"))
        lines.append(_box_row(INFO_C + f"  En attente de données…" + R, w))
        lines.append(_box_row(INFO_C + f"  Log : {log_path}" + R, w))
        lines.append(_box_row(INFO_C + f"  Lancez : popixa train --data_dir data/" + R, w))
        lines.append(_box_bot(w))
        return lines

    last = entries[-1]
    n    = len(entries)

    # Vitesse — moyenne des 10 dernières durées (delta par éval)
    recent   = entries[-min(n, 10):]
    avg_dur  = sum(e.duration for e in recent) / len(recent)
    if avg_dur > 0:
        speed = f"{eval_interval / avg_dur:.2f} it/s"
    else:
        speed = "–"

    # ETA — si max_iters connu
    if max_iters and avg_dur > 0:
        remaining_evals = max(0, (max_iters - last.iter) // eval_interval)
        eta_secs        = remaining_evals * avg_dur
        eta_str         = f"  ·  ETA {_fmt_time(eta_secs)}"
    else:
        eta_str = ""

    # Meilleur val loss
    best_val     = min(e.val_loss for e in entries)
    best_val_str = f"  ·  best {best_val:.4f}"

    train_arr = "↓" if n > 1 and last.train_loss < entries[-2].train_loss else "→"
    val_arr   = "↓" if n > 1 and last.val_loss   < entries[-2].val_loss   else "→"

    tc = fg(*lerp_color(TRAIN_A, TRAIN_B, min(1.0, n / 200)))
    vc = fg(*lerp_color(VAL_A,   VAL_B,   min(1.0, n / 200)))

    progress_str = (
        f"  iter {last.iter}"
        + (f"/{max_iters}" if max_iters else "")
        + f"  ·  lr {last.lr:.2e}"
        + f"  ·  {speed}"
        + eta_str
    )

    lines.append(_box_top(w, "nanoPOPIXA Monitor"))
    lines.append(_box_row(INFO_C + progress_str + R, w))
    lines.append(_box_row(
        f"  train " + tc + f"{last.train_loss:.4f} {train_arr}" + R
        + f"   val " + vc + f"{last.val_loss:.4f} {val_arr}" + R
        + INFO_C + best_val_str + R, w
    ))

    chart_w = w - 10
    chart_h = 4
    max_pts = chart_w * 2
    pts     = entries[-max_pts:] if n > max_pts else entries

    train_vals = [e.train_loss for e in pts]
    val_vals   = [e.val_loss   for e in pts]
    lr_vals    = [e.lr         for e in pts]

    # ── Train loss ────────────────────────────────────────────────────────────
    lines.append(_box_mid(w, "train loss"))
    c1   = BrailleCanvas(chart_w, chart_h)
    tmin = min(train_vals); tmax = max(train_vals)
    if tmax == tmin: tmax += 0.01
    c1.draw_curve(train_vals, tmin, tmax, TRAIN_A, TRAIN_B)
    lines.extend(_chart_section(c1, tmin, tmax, w))

    # ── Val loss ──────────────────────────────────────────────────────────────
    lines.append(_box_mid(w, "val loss"))
    c2   = BrailleCanvas(chart_w, chart_h)
    vmin = min(val_vals); vmax = max(val_vals)
    if vmax == vmin: vmax += 0.01
    c2.draw_curve(val_vals, vmin, vmax, VAL_A, VAL_B)
    lines.extend(_chart_section(c2, vmin, vmax, w))

    # ── Learning rate ─────────────────────────────────────────────────────────
    lines.append(_box_mid(w, "learning rate"))
    c3     = BrailleCanvas(chart_w, chart_h)
    lr_min = min(lr_vals); lr_max = max(lr_vals)
    if lr_max == lr_min: lr_max += 1e-7
    c3.draw_curve(lr_vals, lr_min, lr_max, LR_C, LR_C)
    lines.extend(_chart_section(c3, lr_min, lr_max, w))

    lines.append(_box_bot(w))
    return lines


# ─── Boucle principale ────────────────────────────────────────────────────────
def run_monitor(log_path: str, refresh: float):
    abs_path = os.path.abspath(log_path)
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()
    try:
        while True:
            entries, max_iters, eval_interval = parse_log(abs_path)
            try:
                cols = os.get_terminal_size().columns
            except OSError:
                cols = 80
            w     = min(cols, 90)
            lines = render_dashboard(entries, w=w, log_path=abs_path,
                                     max_iters=max_iters, eval_interval=eval_interval)
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.write("\n".join(lines) + "\n")
            sys.stdout.write(
                INFO_C
                + f"  ↺  {abs_path}  ·  rafraîchi toutes les {refresh}s"
                + "  ·  Ctrl+C pour quitter" + R + "\n"
            )
            sys.stdout.flush()
            time.sleep(refresh)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[?25h\033[2J\033[H")
        sys.stdout.flush()
        print(INFO_C + "  Monitor arrêté." + R)


# ─── Point d'entrée ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Monitor d'entraînement nanoPOPIXA")
    parser.add_argument("--log",     default="train.log")
    parser.add_argument("--refresh", type=float, default=1.0)
    args = parser.parse_args()
    run_monitor(args.log, args.refresh)


if __name__ == "__main__":
    main()
