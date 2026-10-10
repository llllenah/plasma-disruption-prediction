"""Онлайн-фільтр Калмана для 100 приладів toksym — шум прибирається крок за кроком.

Модель кожного каналу (з реверсу шуму, analysis/noise_reverse.py):
    вимір   y_t = x_t + v_t,   Var v = Ra + Rm·x²   (адитивний + множинний білий шум)
    сигнал  x_t = x_{t−1} + w_t, Var w = Qa + Qm·x²  (повільний рівень)
Якщо вимір відхиляється від прогнозу > JUMP·√S — це справжній стрибок (гасіння,
ELM, передвісник), фільтр переходить на вимір, а не згладжує його.
nan (відмова приладу) — лише прогноз.

Параметри оцінюються з самих даних, без істини:
    Cov(Δy_t, Δy_{t−1}) = −R,   Var(Δy) = Q + 2R
по бінах рівня сигналу → лінійна регресія на x².

    python model/kalman.py fit datasets/tr101 datasets/tr102 ...   → model/kalman_params.json
"""
import json
import sys
from pathlib import Path

import numpy as np

JUMP = 5.0
PARAMS_PATH = Path(__file__).resolve().parent / "kalman_params.json"


class OnlineKalman:
    """Фільтр для набору каналів одного розряду; step() — один крок 5 мс."""

    def __init__(self, params):
        self.names = list(params["names"])
        self.Ra = np.asarray(params["Ra"], float)
        self.Rm = np.asarray(params["Rm"], float)
        self.Qa = np.asarray(params["Qa"], float)
        self.Qm = np.asarray(params["Qm"], float)
        self.x = np.full(len(self.names), np.nan)
        self.P = np.zeros(len(self.names))

    def step_array(self, y):
        x, P = self.x, self.P
        new = np.isnan(x) & ~np.isnan(y)                        # перший нормальний вимір
        x = np.where(new, y, x)
        P = np.where(new, self.Ra + self.Rm * y * y, P)
        Pp = P + self.Qa + self.Qm * x * x
        R = self.Ra + self.Rm * x * x + 1e-18
        S = Pp + R
        innov = y - x
        miss = np.isnan(y)
        jump = ~miss & (np.abs(innov) > JUMP * np.sqrt(S))
        K = Pp / S
        upd = x + K * np.nan_to_num(innov)
        x_new = np.where(miss | new, x, np.where(jump, y, upd))
        P_new = np.where(miss, Pp, np.where(jump, R, (1 - K) * Pp))
        P_new = np.where(new, P, P_new)
        self.x, self.P = x_new, P_new
        return x_new

    def step(self, ch):
        """dict ім'я → значення (nan дозволено) → dict з відфільтрованими значеннями."""
        y = np.array([ch.get(n, np.nan) for n in self.names], dtype=float)
        x = self.step_array(y)
        out = dict(ch)
        out.update(zip(self.names, x.tolist()))
        return out


def load_params(path=PARAMS_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ═══════════════════ оцінка параметрів з даних (без істини) ═══════════════
def fit(srcs, n_bins=10):
    import pandas as pd

    frames = []
    for s in srcs:
        head = pd.read_csv(f"{s}/measured.csv", nrows=0).columns
        last = list(head).index("prog_p_ich_MW")
        chans = list(head[2:last + 1])
        frames.append(pd.read_csv(f"{s}/measured.csv", usecols=["shot_id", "t"] + chans)
                      .assign(shot_id=lambda d, s=s: s + ":" + d.shot_id.astype(str)))
    m = pd.concat(frames, ignore_index=True)
    out = dict(names=chans, Ra=[], Rm=[], Qa=[], Qm=[])
    rows = []
    for c in chans:
        y = m[c]
        d = y.groupby(m.shot_id).diff()
        dp = d.groupby(m.shot_id).shift()
        ok = d.notna() & dp.notna() & y.notna()
        # стрибки (гасіння, ELM) не мають потрапити в оцінку шуму
        mad = (d[ok] - d[ok].median()).abs().median() * 1.4826 + 1e-12
        ok &= (d.abs() < 6 * mad) & (dp.abs() < 6 * mad)
        if ok.sum() < 1000 or y[ok].std() == 0:
            for k in ("Ra", "Rm", "Qa", "Qm"):
                out[k].append(0.0)
            out["Ra"][-1] = 1e-12
            continue
        lvl = y[ok].abs()
        bins = pd.qcut(lvl, n_bins, duplicates="drop")
        R_b = -(d[ok] * dp[ok]).groupby(bins, observed=True).mean()
        V_b = (d[ok] ** 2).groupby(bins, observed=True).mean()
        L2 = (lvl ** 2).groupby(bins, observed=True).mean()
        A = np.c_[np.ones(len(L2)), L2.to_numpy()]

        def nnls2(target):
            w, *_ = np.linalg.lstsq(A, target, rcond=None)
            if w[1] < 0:                                   # лише адитивний
                return max(target.mean(), 0.0), 0.0
            if w[0] < 0:                                   # лише множинний
                return 0.0, max((target * L2).sum() / (L2 ** 2).sum(), 0.0)
            return w[0], w[1]

        Ra, Rm = nnls2(R_b.clip(lower=0).to_numpy())
        Qa, Qm = nnls2((V_b - 2 * (Ra + Rm * L2)).clip(lower=0).to_numpy())
        Ra = max(Ra, 1e-12)
        for k, v in zip(("Ra", "Rm", "Qa", "Qm"), (Ra, Rm, Qa, Qm)):
            out[k].append(float(v))
        lv = float(lvl.median())
        rows.append((c, np.sqrt(Ra + Rm * lv * lv) / max(lv, 1e-12), np.sqrt(Qa + Qm * lv * lv) / max(lv, 1e-12)))
    PARAMS_PATH.write_text(json.dumps(out), encoding="utf-8")
    print(f"збережено {PARAMS_PATH}, каналів {len(chans)}")
    print("приклади (відносний шум σ/рівень, відносна швидкість сигналу √Q/рівень за крок):")
    for c, sn, sq in rows[::7]:
        print(f"  {c:24s} шум {sn:.4f}   сигнал {sq:.4f}   → згладжування {'сильне' if sq < sn / 3 else 'слабке'}")


if __name__ == "__main__":
    if sys.argv[1] == "fit":
        fit(sys.argv[2:])
