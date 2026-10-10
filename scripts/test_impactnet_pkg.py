"""Перевірка теки impactnet/: numpy-крок == torch-forward на реальному розряді, час кроку, вивчені пороги."""
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PKG = Path(__file__).resolve().parents[1] / "impactnet"
sys.path.insert(0, str(PKG))
import drivers as D  # noqa: E402
from impact_model import ImpactNet  # noqa: E402
from impactnet import CONT, MECH, ImpactStep  # noqa: E402

m = pd.read_csv(Path("~/risk_data/r508/measured.csv").expanduser())
g = m[m.shot_id == 5].reset_index(drop=True)                 # розряд 5: зрив від Грінвальда
W = dict(np.load(PKG / "impact_net.npz"))
calib = dict(zip(("ne", "w", "prad", "te"), W["calib"]))
X, drv = D.offline(g, calib)
X[:, CONT] = np.clip((np.arcsinh(X[:, CONT] / W["prep_s"]) - W["prep_med"]) / W["prep_iqr"], -8, 8)
X = np.nan_to_num(X)
ck = torch.load(PKG / "impact_net.pt", weights_only=False)
net = ImpactNet(ck["n_in"], hidden=ck.get("hidden", 128))
net.load_state_dict(ck["state"])
net.eval()
with torch.no_grad():
    ref = net(torch.from_numpy(X).float()[None], torch.from_numpy(np.nan_to_num(drv)).float()[None])
st = ImpactStep()
rows = [dict(zip(D.CHANNELS, v)) for v in g[D.CHANNELS].to_numpy(float)]
gc.disable()
tt, err, outs = [], 0.0, []
for i, ch in enumerate(rows):
    t0 = time.perf_counter()
    o = st.step(ch)
    tt.append(time.perf_counter() - t0)
    outs.append(o)
    err = max(err, float(np.abs(np.array(list(o["share_pct"].values())) - ref["share_pct"][0, i].numpy()).max()))
gc.enable()
t = np.array(tt[100:]) * 1e3
print(f"numpy-крок == torch: макс. розбіжність часток {err:.1e} п.п. на {len(rows)} кроках")
print(f"час кроку: p50 {np.median(t):.2f} мс, p99 {np.percentile(t, 99):.2f} мс, макс {t.max():.2f} мс")
print("вивчені пороги c:", dict(zip(MECH[:4], net.c.detach().numpy().round(3))),
      " ширини w:", dict(zip(MECH[:4], net.w.detach().numpy().round(3))))
tr = 11.05
for dt in (3.0, 1.0, 0.3, 0.05):
    i = int(np.searchsorted(g.t, tr - dt))
    o = outs[i]
    print(f"за {dt} с до зриву: λ={o['risk_total']:.4f}, P(1с)={o['p_disrupt'][1.0]:.3f}, частки: " +
          ", ".join(f"{k} {v:.0f}%" for k, v in o["share_pct"].items()))
