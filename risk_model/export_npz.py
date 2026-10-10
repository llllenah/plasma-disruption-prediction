"""risk_net*.pt (torch) → model/risk_net.npz (numpy-ваги для онлайн-ознак v9) + звірка forward.

    python risk_model/export_npz.py risk_model/risk_net_bal0.0_s1.pt
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "risk_model"))
sys.path.insert(0, str(ROOT / "model"))
from build import CHANNELS, TAUS  # noqa: E402
from train import Net  # noqa: E402
import riskfeat  # noqa: E402

ck = torch.load(sys.argv[1], weights_only=False)
net = Net(ck["n_in"])
net.load_state_dict(ck["state"])
net.eval()
sd = net.state_dict()
W = {f"W{i}": sd[f"body.{j}.weight"].numpy().astype(np.float64) for i, j in enumerate((0, 3, 6, 8))}
W.update({f"b{i}": sd[f"body.{j}.bias"].numpy().astype(np.float64) for i, j in enumerate((0, 3, 6, 8))})
W.update(med=ck["med"].astype(np.float64), sc=ck["sc"].astype(np.float64),
         channels=np.array(CHANNELS), taus=np.array(TAUS, dtype=np.float64))
out = ROOT / "model" / "risk_net.npz"
np.savez(out, **W)
Z = np.random.default_rng(0).normal(size=(500, ck["n_in"])).astype(np.float32)
with torch.no_grad():
    ref = net(torch.from_numpy(Z)).numpy()
got = riskfeat.forward(riskfeat.load(out), Z.astype(np.float64))
print(f"{out}: розбіжність numpy vs torch {np.abs(ref - got).max():.1e}")
assert np.abs(ref - got).max() < 1e-4
