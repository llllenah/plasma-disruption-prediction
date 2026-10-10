"""ImpactNet без torch: експорт ваг у .npz і покроковий інференс на чистому numpy (для заліку).

    python risk_model/impact_numpy.py export risk_model/impact_net.pt risk_model/impact_net.npz
    python risk_model/impact_numpy.py bench                 # звірка з torch + час по етапах

Крок: препроцесинг (drivers.Online) → нормалізація → MLP → 4 причинні згортки (буфер 2·dil кроків)
→ GRU-комірка → LayerNorm → 5 нейронів + фізичне априорі → share_pct, risk, risk_total.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import drivers as D  # noqa: E402

_P = 0.3275911
_A = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)


def _erf(x):                                    # Абрамовіц–Стіган 7.1.26, похибка < 1,5·10⁻⁷
    s = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + _P * a)
    y = 1.0 - (((((_A[4] * t + _A[3]) * t) + _A[2]) * t + _A[1]) * t + _A[0]) * t * np.exp(-a * a)
    return s * y


def _gelu(x):
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _ln(x, g, b):
    m = x.mean()
    v = ((x - m) ** 2).mean()
    return (x - m) / np.sqrt(v + 1e-5) * g + b


def export(pt, out):
    import torch
    from impact_model import ImpactNet
    ck = torch.load(pt, weights_only=False)
    net = ImpactNet(ck["n_in"])
    net.load_state_dict(ck["state"])
    W = {k: v.detach().numpy().astype(np.float64) for k, v in net.state_dict().items()}
    W["w_eff"] = net.w.detach().numpy().astype(np.float64)
    W["dilations"] = np.array([b.dw.dilation[0] for b in net.blocks])
    prep = json.loads((HERE / "impact_prep.json").read_text())
    for k in ("s", "med", "iqr"):
        W[f"prep_{k}"] = np.array(prep[k])
    W["calib"] = np.array([prep["calib"][k] for k in ("ne", "w", "prad", "te")])
    np.savez(out, **W)
    print("збережено", out)


class ImpactStep:
    """Стан одного розряду; step(ch) → dict з 11 виходами."""

    def __init__(self, path=HERE / "impact_net.npz"):
        z = np.load(path)
        self.W = {k: z[k] for k in z.files}
        W = self.W
        self.dil = [int(d) for d in W["dilations"]]
        d = W["inp.3.weight"].shape[0]
        self.bufs = [np.zeros((d, 2 * dl)) for dl in self.dil]
        self.h = np.zeros(W["gru.weight_hh_l0"].shape[1])
        calib = dict(zip(("ne", "w", "prad", "te"), W["calib"]))
        self.pre = D.Online(calib)
        from train_impact import CONT
        self.cont = CONT

    def _norm(self, x):
        W = self.W
        x = x.copy()
        c = self.cont
        x[c] = np.clip((np.arcsinh(x[c] / W["prep_s"]) - W["prep_med"]) / W["prep_iqr"], -8, 8)
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
            win = np.concatenate([self.bufs[i], y[:, None]], 1)                 # [d, 2dl+1]
            k = W[p + "dw.weight"][:, 0, :]                                     # [d, 3]
            dw = (k * win[:, ::dl]).sum(1) + W[p + "dw.bias"]
            pw = W[p + "pw.weight"][:, :, 0] @ dw + W[p + "pw.bias"]
            h2 = pw.shape[0] // 2
            g = pw[:h2] * _sig(pw[h2:])
            z = z + W[p + "out.weight"][:, :, 0] @ g + W[p + "out.bias"]
            self.bufs[i] = win[:, 1:]
        u = _ln(z, W["pre_gru.weight"], W["pre_gru.bias"])
        H = self.h.shape[0]
        gi = W["gru.weight_ih_l0"] @ u + W["gru.bias_ih_l0"]
        gh = W["gru.weight_hh_l0"] @ self.h + W["gru.bias_hh_l0"]
        r = _sig(gi[:H] + gh[:H])
        zg = _sig(gi[H:2 * H] + gh[H:2 * H])
        n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
        self.h = (1 - zg) * n + zg * self.h
        hh = _ln(self.h, W["post.weight"], W["post.bias"])
        logits = W["head.weight"] @ hh + W["head.bias"]
        fgw, q95, frad, bn = drv
        px = np.array([fgw, 2.2 / max(q95, 0.5), frad, bn / 2.8])
        logits[:4] += (px - W["c"]) / W["w_eff"]
        power = np.array([3.0, 3, 3, 3, 1])
        log_rp = power * -np.logaddexp(0.0, -logits)                            # p · log σ
        rp = np.exp(log_rp)
        C = -np.log1p(-np.minimum(rp, 1 - 1e-6))
        log_C = np.where(rp < 1e-4, log_rp + rp / 2, np.log(np.maximum(C, 1e-30)))
        e = np.exp(log_C - log_C.max())
        lam = float(-np.expm1(-C.sum()))
        # імовірність зриву на горизонті h, якщо λ лишиться такою: 1 − e^(−λh) (нижня оцінка — перед зривом λ росте)
        return dict(share_pct=100 * e / e.sum(), risk=rp, risk_total=lam,
                    p_disrupt={h: float(-np.expm1(-lam * h)) for h in (0.1, 0.3, 1.0)})


def bench():
    import pandas as pd
    import torch
    from impact_model import ImpactNet
    from train_impact import DATA, transform
    npz = HERE / "impact_net.npz"
    if not npz.exists():
        export(HERE / "impact_net.pt", npz)
    prep = json.loads((HERE / "impact_prep.json").read_text())
    ck = torch.load(HERE / "impact_net.pt", weights_only=False)
    net = ImpactNet(ck["n_in"])
    net.load_state_dict(ck["state"])
    net.eval()
    m = pd.read_csv(DATA / "r508" / "measured.csv")
    g = m[m.shot_id == m.shot_id.iloc[0]].reset_index(drop=True)
    X, drv = D.offline(g, prep["calib"])
    with torch.no_grad():
        ref = net(torch.from_numpy(transform(X, prep)).float()[None], torch.from_numpy(np.nan_to_num(drv)).float()[None])
    ref_s = ref["share_pct"][0].numpy()
    ref_l = ref["risk_total"][0].numpy()
    rows = [{k: g[k].iat[i] for k in D.CHANNELS} for i in range(len(g))]
    st = ImpactStep(npz)
    tt, err_s, err_l = [], 0.0, 0.0
    import gc
    gc.disable()
    for i, ch in enumerate(rows):
        t0 = time.perf_counter()
        o = st.step(ch)
        tt.append(time.perf_counter() - t0)
        err_s = max(err_s, float(np.abs(o["share_pct"] - ref_s[i]).max()))
        err_l = max(err_l, abs(o["risk_total"] - float(ref_l[i])))
    gc.enable()
    t = np.array(tt[100:]) * 1e3
    print(f"numpy vs torch (повний розряд, {len(rows)} кроків): частки ≤ {err_s:.2e} п.п., λ ≤ {err_l:.2e}")
    print(f"крок «колонки → 11 виходів» на numpy: p50 {np.median(t):.3f} мс, p99 {np.percentile(t, 99):.3f} мс, "
          f"макс {t.max():.2f} мс; > 5 мс: {(t > 5).sum()} з {len(t)}")


if __name__ == "__main__":
    if sys.argv[1] == "export":
        export(sys.argv[2], sys.argv[3])
    else:
        bench()
