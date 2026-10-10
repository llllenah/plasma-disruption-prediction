"""ImpactNet як сигналізація проти v8 і submission_clean — на ТИХ САМИХ тестових розрядах (r508 + rs605).

    python risk_model/alarm_eval.py _e40            # тег навченої ImpactNet

Метрика — як у заліку (model/metric.py): тривога = перше p ≥ поріг; зрив спіймано, якщо тривога
не пізніше t_TQ − lead; поріг — під 5 % хибних тривог на тихих розрядах; спонтанні й контакт зі
стінкою виключено. Для ImpactNet p = P(зрив за 1 с) = 1 − e^(−λ·1 с), нуль на спаді струму,
EMA (0,1 на крок) і бігучий максимум — так само, як у моделей заліку.
Моделі заліку проганяються крок за кроком своїм model.py (паралельно по розрядах).
"""
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "risk_model"))
sys.path.insert(0, str(ROOT / "model"))
from metric import LEADS, report  # noqa: E402

DATA = Path(os.environ.get("RISK_DATA", "~/risk_data")).expanduser()
TEST = ["r508", "rs605"]
H = 1.0


def rampdown(g):
    pk = g.prog_ip_MA.cummax().to_numpy()
    return ((pk > 0.5) & (g.prog_ip_MA.to_numpy() < 0.9 * pk)).astype(float)


def post(p, rd, alpha=0.1):
    out = np.zeros_like(p)
    ema = peak = 0.0
    for i, (x, r) in enumerate(zip(p, rd)):
        x = 0.0 if r > 0.5 else x
        ema = x if i == 0 else ema + alpha * (x - ema)
        peak = max(peak, ema)
        out[i] = peak
    return out


_M = None


def _init(sub):
    global _M
    sys.path.insert(0, str(ROOT / sub))
    for m in ("model", "features", "features_v8_base", "riskfeat"):
        sys.modules.pop(m, None)
    import model as M
    _M = M


def _run_shot(g):
    m = _M.Model()
    m.start_shot({"shot_id": int(g.shot_id.iloc[0]), "dt": 0.005, "duration": float(g.t.iloc[-1]), "scenario": ""})
    from drivers import CHANNELS                       # лише 100 офіційних приладів, як на заліку
    chs = CHANNELS
    vals = g[chs].to_numpy(float)
    return np.array([m.predict(float(t), dict(zip(chs, v))) for t, v in zip(g.t.to_numpy(), vals)])


def shots_meta(name):
    return pd.read_csv(DATA / f"shots_{name}.csv").set_index("shot_id")


def impact_preds(tag):
    from impact_model import ImpactNet
    ck = torch.load(ROOT / "risk_model" / f"impact_net{tag}.pt", weights_only=False)
    net = ImpactNet(ck["n_in"], hidden=ck.get("hidden", 128))
    net.load_state_dict(ck["state"])
    net.eval()
    out = {}
    for n in TEST:
        z = np.load(DATA / f"seq_{n}.npz")
        off = np.r_[0, np.cumsum(z["lens"])]
        for i, sid in enumerate(z["sid"]):
            sl = slice(off[i], off[i + 1])
            with torch.no_grad():
                o = net(torch.from_numpy(z["X"][sl].astype(np.float32))[None], torch.from_numpy(z["drv"][sl])[None])
            out[(n, int(sid))] = o["risk_total"][0].numpy().astype(float)
    return out


def summarize(name, per_shot):
    rows = []
    for (src, sid), p in per_shot.items():
        r = shots_meta(src).loc[sid]
        if r.spontaneous or r.wall_contact:
            continue
        rows.append(dict(src=src, t=TS[(src, sid)], p=p, t_tq=float(r.t_tq_s) if r.disrupted else np.inf,
                         quiet=not bool(r.disrupted)))
    for mode, f in (("звичайний", lambda s: not s.startswith("rs")), ("shift", lambda s: s.startswith("rs")), ("разом", lambda s: True)):
        res = report([s for s in rows if f(s["src"])])
        print(f"{name:22s} {mode:9s} | об'єм {res['volume']:.3f} | TPR@5%: " +
              "  ".join(f"{l:g}с {res[l]['tpr_at_budget']:.3f}" for l in LEADS), flush=True)


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    meas = {n: pd.read_csv(DATA / n / "measured.csv") for n in TEST}
    groups = {(n, int(sid)): g.reset_index(drop=True) for n, m in meas.items() for sid, g in m.groupby("shot_id", sort=False)}
    TS = {k: g.t.to_numpy() for k, g in groups.items()}
    RD = {k: rampdown(g) for k, g in groups.items()}
    lam = impact_preds(tag)
    imp = {k: post(1 - np.exp(-lam[k] * H), RD[k]) for k in groups}
    print(f"тест: {len(groups)} розрядів (r508 + rs605), бюджет хибних тривог 5 %\n")
    summarize(f"ImpactNet{tag} P(1с)", imp)
    # оракул: СПРАВЖНЯ λ симулятора з істинного стану (стеля для будь-якої моделі, що оцінює λ)
    import targets as T
    orc = {}
    for k, g in groups.items():
        _, _, lt = T.decompose(T.mechanism_risks(g))
        orc[k] = post(1 - np.exp(-lt * H), RD[k])
    summarize("справжня λ (оракул)", orc)
    keys = list(groups)
    for sub in (() if os.environ.get("NO_SUBS") else ("submission_v8", "submission_clean")):
        if not (ROOT / sub).exists():
            continue
        with mp.get_context("spawn").Pool(min(64, os.cpu_count()), initializer=_init, initargs=(sub,)) as pool:
            ps = pool.map(_run_shot, [groups[k] for k in keys], chunksize=1)
        summarize(sub, dict(zip(keys, ps)))
