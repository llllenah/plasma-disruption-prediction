"""Передбачення для будь-якого measured.csv (102 колонки: shot_id, t і 100 приладів).

    python risk_model/predict.py <measured.csv> <out.csv> [risk_net.pt]

На кожен крок 5 мс — 11 чисел:
  share_pct_<механізм> (5, сума рівно 100), risk_<механізм> (5, від 0 до 1), risk_total (λ).
Лише минуле: EMA по кроках усередині розряду, як і на навчанні.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import CHANNELS, DT, TAUS  # noqa: E402
from targets import MECH  # noqa: E402
from train import Net, prep  # noqa: E402


def load_net(path):
    ck = torch.load(path, weights_only=False)
    net = Net(ck["n_in"])
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck["med"], ck["sc"]


def predict_shot(net, med, sc, g):
    x = g[CHANNELS].ffill()
    X = np.concatenate([x.ewm(alpha=1 - np.exp(-DT / tau), adjust=False).mean().to_numpy(np.float32) for tau in TAUS], 1)
    with torch.no_grad():
        z = net(torch.from_numpy(prep(X, med, sc))).numpy().astype(np.float64)
    C = np.exp(z)
    share = 100 * C / C.sum(1, keepdims=True)
    out = pd.DataFrame({"shot_id": g.shot_id.to_numpy(), "t": g.t.to_numpy()})
    for k, m in enumerate(MECH):
        out[f"share_pct_{m}"] = share[:, k]
    for k, m in enumerate(MECH):
        out[f"risk_{m}"] = 1 - np.exp(-C[:, k])
    out["risk_total"] = 1 - np.exp(-C.sum(1))
    return out


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    net, med, sc = load_net(sys.argv[3] if len(sys.argv) > 3 else Path(__file__).resolve().parent / "risk_net.pt")
    m = pd.read_csv(src)
    res = pd.concat([predict_shot(net, med, sc, g) for _, g in m.groupby("shot_id", sort=False)], ignore_index=True)
    res.to_csv(dst, index=False, float_format="%.6g")
    print(f"{len(res)} кроків → {dst}; сума часток: {res.filter(like='share_pct').sum(1).agg(['min', 'max']).round(9).tolist()}")
