"""ImpactNet — покроковий інференс на чистому numpy (без torch).

    from impactnet import ImpactStep
    st = ImpactStep()                 # новий об'єкт на КОЖЕН розряд (стан пам'яті — свій)
    out = st.step(channels)           # channels: dict «ім'я приладу → число» (nan = відмова приладу)

    python impactnet.py <measured.csv> <out.csv>    # прогнати цілий файл (pandas потрібен лише тут)

Крок: препроцесинг (drivers.Online: 100 приладів + 100 масок + 18 фізичних величин) → нормалізація →
MLP → 4 причинні згортки → GRU-комірка → LayerNorm → 5 нейронів + фізичне априорі → виходи.
"""
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import drivers as D  # noqa: E402

MECH = ("greenwald", "q95", "radiation", "troyon", "other")
N_CH = len(D.CHANNELS)
CONT = np.r_[np.arange(N_CH), np.arange(2 * N_CH, 2 * N_CH + len(D.PHYS))]     # не маски
_P = 0.3275911
_A = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)


def _erf(x):                                    # Абрамовіц–Стіган 7.1.26, похибка < 1,5·10⁻⁷
    s, a = np.sign(x), np.abs(x)
    t = 1.0 / (1.0 + _P * a)
    return s * (1.0 - (((((_A[4] * t + _A[3]) * t) + _A[2]) * t + _A[1]) * t + _A[0]) * t * np.exp(-a * a))


def _gelu(x):
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _ln(x, g, b):
    m = x.mean()
    return (x - m) / np.sqrt(((x - m) ** 2).mean() + 1e-5) * g + b


class ImpactStep:
    """Стан одного розряду. step(ch) → dict: share_pct (5, сума 100), risk (5), risk_total (λ),
    p_disrupt {0.1, 0.3, 1.0 с}, drivers (f_GW, q95, frad, βN з приладів)."""

    def __init__(self, path=HERE / "impact_net.npz"):
        z = np.load(path)
        self.W = W = {k: z[k] for k in z.files}
        self.dil = [int(d) for d in W["dilations"]]
        d = W["inp.3.weight"].shape[0]
        self.bufs = [np.zeros((d, 2 * dl)) for dl in self.dil]
        self.h = np.zeros(W["gru.weight_hh_l0"].shape[1])
        self.pre = D.Online(dict(zip(("ne", "w", "prad", "te"), W["calib"])))

    def _norm(self, x):
        W, x = self.W, x.copy()
        x[CONT] = np.clip((np.arcsinh(x[CONT] / W["prep_s"]) - W["prep_med"]) / W["prep_iqr"], -8, 8)
        return np.nan_to_num(x, nan=0.0)

    def step(self, ch):
        W = self.W
        x, drv = self.pre.step(ch)
        x = self._norm(x)
        drv = np.nan_to_num(drv, nan=0.0)
        z = W["inp.3.weight"] @ _gelu(W["inp.0.weight"] @ x + W["inp.0.bias"]) + W["inp.3.bias"]
        for i, dl in enumerate(self.dil):
            p = f"blocks.{i}."
            y = _ln(z, W[p + "norm.weight"], W[p + "norm.bias"])
            win = np.concatenate([self.bufs[i], y[:, None]], 1)                 # останні 2·dil + 1 кроків
            dw = (W[p + "dw.weight"][:, 0, :] * win[:, ::dl]).sum(1) + W[p + "dw.bias"]
            pw = W[p + "pw.weight"][:, :, 0] @ dw + W[p + "pw.bias"]
            h2 = pw.shape[0] // 2
            z = z + W[p + "out.weight"][:, :, 0] @ (pw[:h2] * _sig(pw[h2:])) + W[p + "out.bias"]
            self.bufs[i] = win[:, 1:]
        u = _ln(z, W["pre_gru.weight"], W["pre_gru.bias"])
        H = self.h.shape[0]
        gi = W["gru.weight_ih_l0"] @ u + W["gru.bias_ih_l0"]
        gh = W["gru.weight_hh_l0"] @ self.h + W["gru.bias_hh_l0"]
        r = _sig(gi[:H] + gh[:H])
        zg = _sig(gi[H:2 * H] + gh[H:2 * H])
        n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
        self.h = (1 - zg) * n + zg * self.h
        logits = W["head.weight"] @ _ln(self.h, W["post.weight"], W["post.bias"]) + W["head.bias"]
        fgw, q95, frad, bn = drv
        logits[:4] += (np.array([fgw, 2.2 / max(q95, 0.5), frad, bn / 2.8]) - W["c"]) / W["w_eff"]
        log_rp = np.array([3.0, 3, 3, 3, 1]) * -np.logaddexp(0.0, -logits)      # p · log σ
        rp = np.exp(log_rp)
        C = -np.log1p(-np.minimum(rp, 1 - 1e-6))
        log_C = np.where(rp < 1e-4, log_rp + rp / 2, np.log(np.maximum(C, 1e-30)))
        e = np.exp(log_C - log_C.max())
        lam = float(-np.expm1(-C.sum()))
        return dict(share_pct=dict(zip(MECH, 100 * e / e.sum())), risk=dict(zip(MECH, rp)), risk_total=lam,
                    p_disrupt={h: float(-np.expm1(-lam * h)) for h in (0.1, 0.3, 1.0)},
                    drivers=dict(fgw=fgw, q95=q95, frad=frad, beta_n=bn))


if __name__ == "__main__":
    import pandas as pd
    m = pd.read_csv(sys.argv[1])
    rows = []
    for sid, g in m.groupby("shot_id", sort=False):
        st = ImpactStep()
        vals, tt = g[D.CHANNELS].to_numpy(float), g.t.to_numpy()
        for t, v in zip(tt, vals):
            o = st.step(dict(zip(D.CHANNELS, v)))
            rows.append({"shot_id": sid, "t": t, **{f"share_pct_{k}": x for k, x in o["share_pct"].items()},
                         **{f"risk_{k}": x for k, x in o["risk"].items()}, "risk_total": o["risk_total"],
                         **{f"p_disrupt_{h}s": x for h, x in o["p_disrupt"].items()}})
    pd.DataFrame(rows).to_csv(sys.argv[2], index=False, float_format="%.6g")
    print(f"{len(rows)} кроків → {sys.argv[2]}")
