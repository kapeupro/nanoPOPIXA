"""
nanoPOPIXA — Splash screen
Globe 3D (Lambert shading + continents rotatifs) + logo pixel art gradient
"""

import sys
import time
import math
import random

# ─── Truecolor ANSI ──────────────────────────────────────────────────────────
R = "\033[0m"
B = "\033[1m"
D = "\033[2m"

def fg(r, g, b): return f"\033[38;2;{r};{g};{b}m"

OCN_HI  = fg(40, 160, 230)   # ocean clair (face à la lumière)
OCN_MID = fg(15,  80, 160)   # ocean mi-ombre
OCN_DRK = fg(5,   35,  90)   # ocean sombre
LND_HI  = fg(60, 210,  80)   # continent clair
LND_MID = fg(25, 140,  50)   # continent mi-ombre
LND_DRK = fg(8,   60,  20)   # continent sombre
STAR_C  = fg(160, 160, 185)
SUB_C   = fg(90,   90, 115)

# Gradient du logo : violet → cyan (5 lignes)
GRAD = [
    fg(200,  20, 255),
    fg(140,  60, 255),
    fg( 70, 130, 255),
    fg( 10, 200, 255),
    fg(  0, 240, 220),
]

# ─── Globe 3D ─────────────────────────────────────────────────────────────────
# 23 colonnes × 11 lignes — RY = RX/2 pour compenser le ratio hauteur/largeur
# des caractères monospace (~2:1), ce qui donne un cercle à l'écran
W, H     = 23, 11
CX, CY   = 11.0, 5.0
RX, RY   = 10.0, 5.0

# Continents : (longitude_centre, latitude_centre, rayon_angulaire)
LANDS = [
    (0.4,  0.25, 0.60),
    (1.8, -0.15, 0.52),
    (3.0,  0.40, 0.58),
    (4.3, -0.20, 0.60),
    (5.5,  0.10, 0.48),
    (1.1, -0.55, 0.38),
]

def is_land(lon, lat, t):
    a = (lon + t) % (2 * math.pi)
    for ca, cla, r in LANDS:
        da = min(abs(a - ca), 2 * math.pi - abs(a - ca))
        if da * da + (lat - cla) ** 2 < r * r:
            return True
    return False

def globe_pixel(x, y, t):
    nx = (x - CX) / RX
    ny = (y - CY) / RY
    d2 = nx * nx + ny * ny
    if d2 > 1.0:
        return " "

    nz  = math.sqrt(max(0.0, 1.0 - d2))
    # Lumière depuis haut-gauche-devant (Lambert diffuse + ambiant)
    bri = max(0.0, -0.45*nx - 0.30*ny + 0.84*nz) * 0.85 + 0.15

    lon  = math.atan2(nx, nz)
    lat  = math.asin(max(-1.0, min(1.0, ny)))
    land = is_land(lon, lat, t)

    if land:
        if bri > 0.72: c = LND_HI  + "░"
        elif bri > 0.42: c = LND_MID + "▒"
        else:            c = LND_DRK + "▓"
    else:
        if bri > 0.68: c = OCN_HI  + "░"
        elif bri > 0.38: c = OCN_MID + "▒"
        else:            c = OCN_DRK + "▓"
    return c + R

def render_globe(t):
    return [
        "".join(globe_pixel(x, y, t) for x in range(W))
        for y in range(H)
    ]

# ─── Police pixel (7 × 5) ─────────────────────────────────────────────────────
FONT = {
    'P': ["██████ ", "██   ██", "██████ ", "██     ", "██     "],
    'O': [" █████ ", "██   ██", "██   ██", "██   ██", " █████ "],
    'I': ["███████", "  ██   ", "  ██   ", "  ██   ", "███████"],
    'X': ["██   ██", " ██ ██ ", "  ███  ", " ██ ██ ", "██   ██"],
    'A': ["  ███  ", " ██ ██ ", "███████", "██   ██", "██   ██"],
}

def logo():
    rows = [""] * 5
    for ch in "POPIXA":
        g = FONT.get(ch, ["       "] * 5)
        for r in range(5):
            rows[r] += g[r] + " "
    return rows  # chaque ligne = 48 chars (6 × 8)

# ─── Étoiles ──────────────────────────────────────────────────────────────────
random.seed(42)
_STARS = "".join(random.choice("        .....···*") for _ in range(70))

def star_row(f):
    n = len(_STARS)
    s = _STARS[f % n:] + _STARS[:f % n]
    out = STAR_C
    for c in s[:48]:
        out += (R + B + fg(255, 255, 255) + c + R + STAR_C) if c == "*" else c
    return out + R

# ─── Spinner ──────────────────────────────────────────────────────────────────
_SP = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
def spin(f): return GRAD[2] + _SP[f % len(_SP)] + R

# ─── Splash principale ────────────────────────────────────────────────────────
def splash(subtitle="Un LLM minimaliste · PyTorch · from scratch"):
    lo = logo()
    # Centrer le globe (23 chars) au-dessus du logo (1+48=49 chars visibles)
    PAD = (49 - W) // 2   # = 13

    sys.stdout.write("\033[?25l")   # cacher le curseur
    sys.stdout.flush()

    try:
        FPS = 12
        N   = int(FPS * 2.5)

        for frame in range(N + 1):
            gl = render_globe(frame * 0.14)

            lines = ["", " " + star_row(frame), ""]
            for l in gl:
                lines.append(" " * PAD + l)
            lines.append("")
            for r, row in enumerate(lo):
                lines.append(" " + B + GRAD[r] + row + R)
            lines.append("")
            lines.append("  " + spin(frame) + "  " + SUB_C + subtitle + R)
            lines.append("")

            sys.stdout.write("\n".join(lines) + "\n")
            sys.stdout.flush()

            if frame < N:
                time.sleep(1 / FPS)
                sys.stdout.write(f"\033[{len(lines)}A\r")
                sys.stdout.flush()

    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[?25h")   # réafficher le curseur
        sys.stdout.flush()


if __name__ == "__main__":
    splash()
