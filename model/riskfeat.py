"""Онлайн-нейромережа «вклад механізмів зриву» (risk_model/) як ознаки для v9 — лише numpy.

Ваги заморожені: мережу навчено на окремих сидах r501–508 / rs601–605, на tr*/trs* вона не вчилась.
На кожному кроці: 100 приладів → nan замінюємо останнім значенням → три EMA (30/100/300 мс) →
MLP → log Cₖ п'яти механізмів. Ознаки:
  nn_lam             λ = 1 − e^(−ΣC) — загальний ризик зараз;
  nn_logC_*          log-вклади густини, кінку, випромінювання, locked mode (β завжди ≈ 0 — не беремо);
  nn_lam_max         максимум λ від початку розряду;
  nn_lam_1s          λ, згладжена за 1 с;
  nn_Lam             накопичений ризик ∫λ dt (пуассон: P(зрив до t) = 1 − e^(−Λ)).
"""
from pathlib import Path

import numpy as np

DT = 0.005
NAMES = ["nn_lam", "nn_logC_density", "nn_logC_kink", "nn_logC_radiation", "nn_logC_lm",
         "nn_lam_max", "nn_lam_1s", "nn_Lam"]
_A1S = 1.0 - np.exp(-DT / 1.0)


def load(path=None):
    z = np.load(path or Path(__file__).resolve().parent / "risk_net.npz")
    return {k: z[k] for k in z.files}


def forward(W, Z):
    """Z (n, 300) нормовані входи → z = log C (n, 5)."""
    h = Z
    for i in range(3):
        h = h @ W[f"W{i}"].T + W[f"b{i}"]
        h = h / (1.0 + np.exp(-h))                      # SiLU
    return h @ W["W3"].T + W["b3"]


def prep(W, X):
    Z = np.clip((X - W["med"]) / W["sc"], -10, 10)
    return np.nan_to_num(Z, nan=0.0)


class RiskNetOnline:
    def __init__(self, W):
        self.W = W
        self.channels = [str(c) for c in W["channels"]]
        n = len(self.channels)
        self.alpha = np.repeat(1.0 - np.exp(-DT / W["taus"]), n)
        self.last = np.full(n, np.nan)
        self.ema = np.full(n * len(W["taus"]), np.nan)
        self.lam_max = 0.0
        self.lam_1s = np.nan
        self.Lam = 0.0

    def step(self, ch):
        x = np.array([ch.get(c, np.nan) for c in self.channels], dtype=float)
        self.last = np.where(np.isnan(x), self.last, x)
        xx = np.tile(self.last, len(self.W["taus"]))
        self.ema = np.where(np.isnan(self.ema), xx, self.ema + self.alpha * (xx - self.ema))
        z = forward(self.W, prep(self.W, self.ema)[None, :])[0]
        C = np.exp(z)
        lam = float(1.0 - np.exp(-C.sum()))
        self.lam_max = max(self.lam_max, lam)
        self.lam_1s = lam if self.lam_1s != self.lam_1s else self.lam_1s + _A1S * (lam - self.lam_1s)
        self.Lam += lam * DT
        return [lam, z[0], z[2], z[3], z[4], self.lam_max, self.lam_1s, self.Lam]


def offline(W, g):
    """Те саме для цілого розряду (DataFrame з колонками приладів, по кроках) — для навчальних ознак."""
    import pandas as pd
    x = g[[str(c) for c in W["channels"]]].ffill()
    X = np.concatenate([x.ewm(alpha=1 - np.exp(-DT / tau), adjust=False).mean().to_numpy() for tau in W["taus"]], 1)
    z = forward(W, prep(W, X))
    lam = 1.0 - np.exp(-np.exp(z).sum(1))
    lam_1s = pd.Series(lam).ewm(alpha=_A1S, adjust=False).mean().to_numpy()
    return np.c_[lam, z[:, 0], z[:, 2], z[:, 3], z[:, 4], np.maximum.accumulate(lam), lam_1s, np.cumsum(lam) * DT]
