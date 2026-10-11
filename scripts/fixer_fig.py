"""Рисунок: керування в симуляторі fusion-sim, 100 нових розрядів (seed 2000), однакові в усіх режимах.
Цифри — зі звіту fusionsim teach у репозиторії симулятора (docs/MODELS.md, розд. 7; docs/FIXER.md, розд. 4).

    python work/fixer_fig.py presentation/figs/fixer.png
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID = "#0F172A", "#64748B", "#E2E8F0"
SLATE, AMB, IND, EME = "#94A3B8", "#D97706", "#4F46E5", "#059669"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})
modes = ["без\nкерування", "порадник\n(правила)", "вчитель\n(копія)", "учень\n(прилади)"]
cols = [SLATE, AMB, IND, EME]
prev = [31, 21, 5, 5]
ok = [61, 70, 86, 86]
w = [2.371, 2.376, 2.359, 2.374]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(wspace=0.3))
b = a1.bar(modes, prev, color=cols, width=0.62)
for bb, v in zip(b, prev):
    a1.text(bb.get_x() + bb.get_width() / 2, v + 0.6, str(v), ha="center", fontsize=14, fontweight="bold", color=INK)
a1.set_ylim(0, 36)
a1.set_title("Зриви, яким можна запобігти (не спонтанні)", fontsize=13, color=INK, loc="left")
a1.grid(axis="x", visible=False)
b = a2.bar(modes, ok, color=cols, width=0.62)
for bb, v, ww in zip(b, ok, w):
    a2.text(bb.get_x() + bb.get_width() / 2, v + 1.2, f"{v} %", ha="center", fontsize=14, fontweight="bold", color=INK)
    a2.text(bb.get_x() + bb.get_width() / 2, 6, f"W {ww:.3f}".replace(".", ","), ha="center", fontsize=10.5, color="white")
a2.set_ylim(0, 100)
a2.set_title("Розряди без зриву, % (W — енергія плазми, МДж)", fontsize=13, color=INK, loc="left")
a2.grid(axis="x", visible=False)
fig.savefig(sys.argv[1], dpi=200, bbox_inches="tight", facecolor="white")
print("→", sys.argv[1])
