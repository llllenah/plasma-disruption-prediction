"""Модель раннього попередження зриву: дві голови градієнтного бустингу.

Голова A навчена на горизонті ~0,1 с — ловить передвісник (locked mode,
Мірнов, просідання T_e), дає впевнену тривогу пізно. Голова B навчена на
горизонті ~1 с — оцінює близькість до операційних меж (f_GW, q, β,
P_rad/P_in, Dα) і дає ранню, але менш певну тривогу; її стиснуто в [0, w],
щоб ризиковий, але тихий розряд не перебивав справжній передвісник.

На спаді струму (уставка нижче 90 % максимуму) жереб не кидається — модель
мовчить. Відповідь — EMA і бігучий максимум: тривога — подія, не стан.
"""
import json
import math
from pathlib import Path

import numpy as np

from features import Features, feature_names

HERE = Path(__file__).resolve().parent


def _load():
    cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    z = np.load(HERE / "weights.npz")
    heads = {}
    for h in ("a", "b"):
        heads[h] = {k[2:]: z[k] for k in z.files if k.startswith(h + "_")}
    names = feature_names()
    # кожна голова — свій набір ознак (у старих здачах обидві беруть cfg["cols"])
    sel = {h: np.array([names.index(c) for c in cfg.get(f"cols_{h}", cfg["cols"])], dtype=np.int64)
           for h in ("a", "b")}
    return cfg, heads, sel


def _predict(fl, x):
    idx = fl["roots"].copy()
    for _ in range(int(fl["depth"])):
        lf = fl["leaf"][idx]
        if lf.all():
            break
        v = x[fl["feat"][idx]]
        go_left = np.where(np.isnan(v), fl["miss_left"][idx], v <= fl["thr"][idx])
        idx = np.where(lf, idx, np.where(go_left, fl["left"][idx], fl["right"][idx]))
    raw = float(fl["base"]) + float(fl["value"][idx].sum())
    return 1.0 / (1.0 + math.exp(-raw))


class Model:
    def __init__(self):
        self.cfg, self.heads, self.sel = _load()
        self.w = float(self.cfg["w"])
        self.alpha = float(self.cfg["ema"])
        self.i_rampdown = feature_names().index("rampdown")

    def start_shot(self, meta):
        self.fx = Features(meta.get("scenario", ""))
        self.ema = None
        self.peak = 0.0

    def predict(self, t, channels):
        f = np.array(self.fx.step(t, channels), dtype=float)
        if f[self.i_rampdown] > 0.5:
            p = 0.0
        else:
            f[~np.isfinite(f)] = np.nan
            a = _predict(self.heads["a"], f[self.sel["a"]])
            b = _predict(self.heads["b"], f[self.sel["b"]])
            p = 1.0 - (1.0 - a) * (1.0 - self.w * b)
        self.ema = p if self.ema is None else self.ema + self.alpha * (p - self.ema)
        self.peak = max(self.peak, self.ema)
        return self.peak


if __name__ == "__main__":
    from toksym_model import serve

    serve(Model())
