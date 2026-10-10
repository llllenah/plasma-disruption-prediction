"""Графіки для білої презентації (з реальних прогонів) + вирізки скриншотів Test Run.

    python work/deck_figs.py <тека зі скриншотами> presentation/figs
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SHOTS_DIR, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
INK, MUTED, GRID = "#0F172A", "#64748B", "#E2E8F0"
IND, EME, ORA, VIO, AMB, RED, SLATE = "#4F46E5", "#059669", "#EA580C", "#7C3AED", "#D97706", "#DC2626", "#94A3B8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("→", name)


# ── 1. вирізки скриншотів Test Run (плитки результату) ──
def crop(src, box, name):
    p = SHOTS_DIR / src
    if p.exists():
        Image.open(p).convert("RGB").crop(box).save(OUT / name)
        print("→", name)


crop("13.png", (12, 470, 888, 1000), "ui_xgb_dynamics.png")       # XGBoost Dynamics + рядки розрядів
crop("14.png", (12, 470, 888, 660), "ui_xgb_physics.png")
crop("15.png", (12, 470, 888, 660), "ui_xgb_formula.png")
crop("6.png", (20, 600, 1112, 742), "ui_clean_big_01.png")         # наша чиста, 0,1 с
crop("7.png", (20, 600, 1112, 742), "ui_clean_big_003.png")        # наша чиста, 30 мс
crop("11.png", (8, 410, 573, 558), "ui_v8_new_engine.png")         # v8 на новому рушії
crop("13.png", (12, 620, 888, 1000), "ui_rows.png")                # як залік показує розряди

# ── 2. усі моделі на тих самих 200 розрядах інтерфейсу (0,1 с, 5 %) ──
ui = pd.DataFrame([
    ("XGB\nформули", 25.4, 16, 42, 5, 0.761, "teammate"), ("XGB\nфізика", 25.4, 16, 41, 6, 0.727, "teammate"),
    ("XGB\nдинаміка", 34.9, 22, 32, 9, 0.743, "teammate"), ("v6b", 57.1, 36, 26, 1, 0.267, "phys"),
    ("ens5", 60.3, 38, 25, 0, 0.232, "clean"), ("v6", 63.5, 40, 23, 0, 0.398, "phys"),
    ("clean_big\n(здача)", 63.5, 40, 23, 0, 0.227, "final"), ("nocs", 65.1, 41, 21, 1, 0.229, "nocs"),
    ("v8", 66.7, 42, 20, 1, 0.421, "phys")], columns=["model", "caught", "tp", "late", "missed", "thr", "kind"])
col = {"teammate": SLATE, "phys": IND, "clean": "#6EE7B7", "final": EME, "nocs": AMB}
fig, ax = plt.subplots(figsize=(12, 5.2))
bars = ax.bar(ui.model, ui.caught, color=[col[k] for k in ui.kind], width=0.62)
for b, (_, r) in zip(bars, ui.iterrows()):
    ax.text(b.get_x() + b.get_width() / 2, r.caught + 1.2, f"{r.caught:.1f} %".replace(".", ","), ha="center", fontsize=13, fontweight="bold", color=INK)
    ax.text(b.get_x() + b.get_width() / 2, 2.5, f"пропущ.\n{r.missed}", ha="center", fontsize=10,
            color=INK if r.kind in ("clean", "teammate") else "white")
ax.set_ylim(0, 78)
ax.set_ylabel("спіймано вчасно, %")
ax.tick_params(axis="x", labelsize=11)
ax.grid(axis="x", visible=False)
save(fig, "ui_compare.png")

# ── 3. еволюція моделей: об'єм офіційного заліку ──
ev = [("бейзлайн", 0.765, ORA, ""), ("v6\nфізика", 0.937, IND, ""), ("v8\nфізика", 0.944, IND, ""),
      ("nocs\nбез формул", 0.925, AMB, ""), ("clean\nбез підказок", 0.914, "#6EE7B7", ""),
      ("clean_big\n4500 розр.", 0.924, EME, "здача"), ("clean_9k\n9000 розр.", 0.928, "#6EE7B7", ""),
      ("ens5\nансамбль", 0.930, "#6EE7B7", "3 мс")]
fig, ax = plt.subplots(figsize=(12, 5.0))
xs = np.arange(len(ev))
bars = ax.bar(xs, [v for _, v, _, _ in ev], color=[c for _, _, c, _ in ev], width=0.62)
for x, (_, v, _, note) in zip(xs, ev):
    ax.text(x, v + 0.004, f"{v:.3f}".replace(".", ","), ha="center", fontsize=13, fontweight="bold", color=INK)
    if note:
        ax.text(x, 0.712, note, ha="center", fontsize=11, color="white", fontweight="bold")
ax.set_xticks(xs, [n for n, _, _, _ in ev], fontsize=10.5)
ax.set_ylim(0.7, 0.97)
ax.set_ylabel("об'єм заліку")
ax.grid(axis="x", visible=False)
save(fig, "evolution.png")

# ── 4. теплокарта «запас × бюджет» (clean_big, 4 набори) ──
grid = np.array([[99.1, 99.1, 99.6, 99.6], [97.8, 98.2, 99.6, 99.6], [39.9, 50.9, 71.9, 83.3],
                 [36.0, 47.8, 66.7, 81.6], [22.4, 29.8, 47.4, 61.0]])
fig, ax = plt.subplots(figsize=(7.2, 5.0))
im = ax.imshow(grid, cmap="Greens", vmin=10, vmax=100, aspect="auto")
ax.set_xticks(range(4), ["3 %", "5 %", "10 %", "15 %"])
ax.set_yticks(range(5), ["10 мс", "30 мс", "0,1 с", "0,3 с", "1 с"])
ax.set_xlabel("бюджет хибних тривог")
ax.set_ylabel("запас (lead)")
ax.grid(False)
for i in range(5):
    for j in range(4):
        v = grid[i, j]
        ax.text(j, i, f"{v:.1f}".replace(".", ","), ha="center", va="center", fontsize=14, fontweight="bold",
                color="white" if v > 60 else INK)
for s in ax.spines.values():
    s.set_visible(False)
save(fig, "heatmap.png")

# ── 5. оракул: λ як будильник ──
fig, ax = plt.subplots(figsize=(6.2, 4.6))
names, vals, cs = ["модель тривоги\n(v8)", "ImpactNet\nяк будильник", "справжня λ\n(оракул)"], [0.920, 0.767, 0.780], [EME, VIO, SLATE]
b = ax.bar(names, vals, color=cs, width=0.6)
for bb, v in zip(b, vals):
    ax.text(bb.get_x() + bb.get_width() / 2, v + 0.005, f"{v:.3f}".replace(".", ","), ha="center", fontsize=13, fontweight="bold", color=INK)
ax.set_ylim(0.6, 0.97)
ax.set_ylabel("об'єм як сигналізація")
ax.grid(axis="x", visible=False)
save(fig, "oracle.png")

# ── 6. реальний розряд: p(t) моделі тривоги + частки ImpactNet ──
data = Path("~/risk_data/r508").expanduser()
if (data / "measured.csv").exists():
    sys.path.insert(0, str(ROOT / "impactnet"))
    import drivers as D  # noqa: E402
    from impactnet import ImpactStep  # noqa: E402
    sys.path.insert(0, str(ROOT / "risk_model"))
    import targets as T  # noqa: E402
    m = pd.read_csv(data / "measured.csv")
    sh = pd.read_csv(data / "shots.csv").set_index("shot_id")
    sid = 5
    g = m[m.shot_id == sid].reset_index(drop=True)
    r = sh.loc[sid]
    t = g.t.to_numpy()
    # модель тривоги (фінальна здача) крок за кроком
    sys.path.insert(0, str(ROOT / "submission_clean_big"))
    import model as SUB  # noqa: E402
    mdl = SUB.Model()
    mdl.start_shot({"shot_id": sid, "dt": 0.005, "duration": float(t[-1]), "scenario": ""})
    p = np.array([mdl.predict(float(tt), dict(zip(D.CHANNELS, v))) for tt, v in zip(t, g[D.CHANNELS].to_numpy(float))])
    st = ImpactStep()
    outs = [st.step(dict(zip(D.CHANNELS, v))) for v in g[D.CHANNELS].to_numpy(float)]
    lam = np.array([o["risk_total"] for o in outs])
    shares = np.array([[o["share_pct"][k] for k in ("greenwald", "q95", "radiation", "troyon", "other")] for o in outs])
    _, _, lam_true = T.decompose(T.mechanism_risks(g))
    thr, tq = 0.23, float(r.t_tq_s)
    ia = int(np.argmax(p >= thr))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6.2), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.15], hspace=0.12))
    a1.plot(t, p, color=EME, lw=2.4, label="p моделі тривоги (clean_big)")
    a1.axhline(thr, color=MUTED, ls="--", lw=1.4, label="поріг під 5 % хибних ≈ 0,23")
    a1.axvline(tq, color=RED, lw=2.4, label="термічне гасіння")
    a1.axvline(t[ia], color=EME, lw=1.2, ls=":")
    a1.annotate(f"тривога за {tq - t[ia]:.2f} с до гасіння".replace(".", ","), xy=(t[ia], thr), xytext=(t[ia] - 2.4, 0.5), fontsize=12, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK))
    a1.set_ylim(0, 1.02)
    a1.set_ylabel("p (тривога)")
    a1.legend(loc="upper left", fontsize=10.5, frameon=False)
    keep = lam > 0.01
    sh_plot = np.where(keep[:, None], shares, np.nan)
    cols5 = [ORA, IND, AMB, "#06B6D4", SLATE]
    base = np.zeros(len(t))
    for k, (nm, c) in enumerate(zip(("Грінвальд", "q95", "радіація", "Тройон", "інші"), cols5)):
        a2.fill_between(t, base * keep, (base + np.nan_to_num(sh_plot[:, k])) * keep, color=c, alpha=0.75, lw=0, label=nm, step=None)
        base = base + np.nan_to_num(sh_plot[:, k])
    a2b = a2.twinx()
    a2b.plot(t, lam, color=INK, lw=2.0, label="λ ImpactNet")
    a2b.plot(t, lam_true, color=RED, lw=1.4, ls="--", label="справжня λ")
    a2b.set_ylim(0, 1.0)
    a2b.set_ylabel("λ, 1/с")
    a2b.grid(False)
    a2b.spines["right"].set_visible(True)
    a2.axvline(tq, color=RED, lw=2.4)
    a2.set_ylim(0, 100)
    a2.set_ylabel("частка механізму, %\n(де λ > 0,01)")
    a2.set_xlabel("час розряду, с")
    a2.set_xlim(max(0, tq - 6), tq + 0.4)
    h1, l1 = a2.get_legend_handles_labels()
    h2, l2 = a2b.get_legend_handles_labels()
    a2.legend(h1[:3] + h2, l1[:3] + l2, loc="upper left", fontsize=10.5, frameon=False, ncol=2)
    save(fig, "shot_trace.png")
    print(f"розряд {sid}: сценарій {r.scenario}, тривога за {tq - t[ia]:.2f} с, λ_max мережа {lam.max():.2f}, справжня {lam_true[t < tq].max():.2f}")
