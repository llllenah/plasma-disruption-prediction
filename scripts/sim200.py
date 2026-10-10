"""Прогін fusion-sim на 200 розрядів (v8 + ImpactNet): ROC AUC з кривих run.json і перевірка калібрування часток механізмів.

    python work/sim200.py "../run(1).json"
"""
import itertools
import json
import sys

import numpy as np
import pandas as pd

d = json.load(open(sys.argv[1], encoding="utf-8"))
print(d["run"], d["summary"]["mechanisms"]["confusion"])
for lead, cur in d["curve"]["curves"].items():
    pts = sorted({(c["fpr"], c["tpr"]) for c in cur} | {(0.0, 0.0), (1.0, 1.0)})
    f, t = np.array(pts).T
    # ступінчаста ROC: для кожного fpr беремо найбільший tpr
    fu = np.unique(f)
    tu = np.array([t[f == x].max() for x in fu])
    tu = np.maximum.accumulate(tu)
    print(f"lead {lead}: ROC AUC {np.trapezoid(tu, fu):.3f}")

M = ["greenwald", "q95", "radiation", "troyon", "other"]
rows = []
for s in d["shots"]:
    if s["disrupted"] and s.get("true_cause") and not s["spontaneous"] and not s["wall_contact"]:
        rows.append((s["shot_id"], s["true_cause"]["dominant"], [s["predicted_cause"]["share_pct"][m] for m in M],
                     [s["true_cause"]["share_pct"][m] for m in M]))
print("зривів з причиною:", len(rows))
y = np.array([M.index(r[1]) for r in rows])
P = np.array([r[2] for r in rows]) + 1e-9
print("точність як є:", (P.argmax(1) == y).mean().round(3), (P.argmax(1) == y).sum())


def acc(w, idx):
    return ((P[idx] * w).argmax(1) == y[idx]).mean()


grid = [0.5, 0.75, 1, 1.5, 2, 3, 4, 6]
ids = np.array([r[0] for r in rows])
# 5-кратна перевірка за розрядами: ваги механізмів підбираються на 4/5, точність — на решті
rng = np.random.default_rng(0)
fold = rng.permutation(len(rows)) % 5
hit = 0
ws = []
for k in range(5):
    tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
    best = max(((acc(np.array([1, wq, wr, 1, 1]), tr), -abs(np.log(wq)) - abs(np.log(wr)), wq, wr)
                for wq, wr in itertools.product(grid, grid)))
    w = np.array([1, best[2], best[3], 1, 1])
    ws.append((best[2], best[3]))
    hit += ((P[te] * w).argmax(1) == y[te]).sum()
print("ваги q95/радіація по фолдах:", ws)
print(f"точність з калібруванням (5-кратна перевірка): {hit}/{len(rows)} = {hit / len(rows):.3f}")
wall = max(((acc(np.array([1, wq, wr, 1, 1]), np.arange(len(rows))), -abs(np.log(wq)) - abs(np.log(wr)), wq, wr)
            for wq, wr in itertools.product(grid, grid)))
w = np.array([1, wall[2], wall[3], 1, 1])
pred = (P * w).argmax(1)
for j, m in enumerate(M[:3]):
    print(f"  {m}: було {((P.argmax(1) == y) & (y == j)).sum()}/{(y == j).sum()}, стало {((pred == y) & (y == j)).sum()}/{(y == j).sum()}")
print("ваги на всіх:", wall[2:], "точність", wall[0].round(3))

print("\nзриви з істинним кінком: частки модель (gw/q95/rad) | насправді (gw/q95/rad)")
for r in rows:
    if r[1] == "q95":
        print(f"  {r[0]:4d}  {r[2][0]:5.1f} {r[2][1]:5.1f} {r[2][2]:5.1f} | {r[3][0]:5.1f} {r[3][1]:5.1f} {r[3][2]:5.1f}")

# ── рисунок для презентації: перенесення моделей на інший рушій ──
if len(sys.argv) > 2:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    INK, MUTED, GRID = "#0F172A", "#64748B", "#E2E8F0"
    IND, EME, ORA, SLATE, PINK = "#4F46E5", "#059669", "#EA580C", "#94A3B8", "#DB2777"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})
    auc_tok = json.load(open(sys.argv[2], encoding="utf-8"))["v8"]
    auc_sim = {}
    for lead, cur in d["curve"]["curves"].items():
        pts = sorted({(c["fpr"], c["tpr"]) for c in cur} | {(0.0, 0.0), (1.0, 1.0)})
        f, t = np.array(pts).T
        fu = np.unique(f)
        auc_sim[float(lead)] = np.trapezoid(np.maximum.accumulate([t[f == x].max() for x in fu]), fu)
    leads = [(0.03, "30 мс"), (0.1, "0,1 с"), (0.3, "0,3 с"), (1.0, "1 с")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.35))
    x = np.arange(len(leads))
    for off, vals, c, lab in [(-0.19, [auc_tok[str(l)] for l, _ in leads], IND, "toksym, 747 розрядів (з шумом)"),
                              (0.19, [auc_sim[l] for l, _ in leads], EME, "fusion-sim, 200 розрядів (без шуму)")]:
        b = a1.bar(x + off, vals, 0.36, color=c, label=lab)
        for bb, v in zip(b, vals):
            a1.text(bb.get_x() + bb.get_width() / 2, v + 0.004, f"{v:.3f}".replace(".", ","), ha="center", fontsize=10.5, color=INK)
    a1.set_xticks(x, [n for _, n in leads])
    a1.set_ylim(0.8, 1.02)
    a1.set_ylabel("ROC AUC")
    a1.set_xlabel("запас (lead)")
    a1.legend(frameon=False, fontsize=10.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    a1.set_title("Модель тривоги v8 без донавчання", fontsize=13, color=INK, loc="left")
    names = ["Грінвальд", "кінк (q95)", "радіація"]
    # як у fusion-sim (максимум λ по всьому розряду, включно з прогрівом на старті) і виправлено (лише кроки з плазмою);
    # обидва — з онлайн-виходів моделі в measured.csv, work/fs_moment.py
    fsm = pd.read_csv("work/fs_moment.csv")
    shc = pd.read_csv(sys.argv[4]).set_index("shot_id")
    fsm = fsm[~fsm.sid.map(shc.spontaneous) & ~fsm.sid.map(shc.wall_contact)]
    yy = np.arange(3)[::-1]
    for off, col, c, lab in [(0.2, "fs", SLATE, "як рахував симулятор (з прогрівом на старті)"),
                             (-0.2, "ok", EME, "максимум λ лише там, де є плазма")]:
        v = [((fsm.true == m) & (fsm[col] == m)).sum() for m in ("greenwald", "q95", "radiation")]
        n = [(fsm.true == m).sum() for m in ("greenwald", "q95", "radiation")]
        a2.barh(yy + off, v, 0.36, color=c, label=lab)
        for yv, a, b in zip(yy + off, v, n):
            a2.text(a + 0.3, yv, f"{a} з {b}", va="center", fontsize=11.5, color=INK, fontweight="bold" if col == "ok" else "normal")
    a2.set_yticks(yy, names)
    a2.set_xlim(0, 20)
    a2.set_xticks([0, 4, 8, 12, 16, 20])
    a2.set_xlabel("зривів, визначених правильно")
    a2.grid(axis="y", visible=False)
    a2.legend(frameon=False, fontsize=10.5, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=1)
    tot_fs = (fsm.fs == fsm.true).sum()
    tot_ok = (fsm.ok == fsm.true).sum()
    a2.set_title(f"ImpactNet: механізм зриву, {tot_fs} → {tot_ok} з {len(fsm)}", fontsize=13, color=INK, loc="left")
    out = Path(sys.argv[3])
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print("→", out)
