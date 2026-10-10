"""measured.csv + shots.csv → таблиця ознак (parquet) для навчання.

    python model/build_dataset.py datasets/tr101 features/tr101.parquet

Ознаки рахуються тим самим класом Features, що й на заліку, крок за кроком.
У файл іде кожен STRIDE-й крок (сусідні кроки майже тотожні), а останні 2 с
перед гасінням — кожен другий, щоб позитивів було більше.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib as _il, os as _os  # noqa: E402
_F = _il.import_module(_os.environ.get("FEATURES", "features"))  # FEATURES=features_nocs — без шпаргалки
Features, feature_names = _F.Features, _F.feature_names
import os  # noqa: E402
from kalman import load_params  # noqa: E402

KALMAN = load_params() if os.environ.get("KALMAN") == "1" else None  # KALMAN=1 — з фільтром Калмана

STRIDE = 4
DENSE_WINDOW = 2.0


def main(src, dst):
    t0 = time.time()
    sh = pd.read_csv(f"{src}/shots.csv")
    ts = pd.read_csv(f"{src}/measured.csv")
    ts = ts.loc[:, ~ts.columns.duplicated()]
    last = ts.columns.get_loc("prog_p_ich_MW")
    chans = list(ts.columns[2:last + 1])
    assert len(chans) == 100, len(chans)
    extra = [c for c in ("te0_keV", "w_th_MJ", "p_rad_MW", "p_input_MW", "ip_MA", "ne_bar_1e20") if c in ts.columns]

    names = feature_names()
    rows = []
    meta = sh.set_index("shot_id")
    for sid, g in ts.groupby("shot_id", sort=False):
        m = meta.loc[sid]
        fx = Features(m.scenario, kalman=KALMAN)
        tq = m.t_tq_s if m.disrupted and m.t_tq_s == m.t_tq_s else np.inf
        vals = g[chans].to_numpy(dtype=float)
        tt = g.t.to_numpy()
        ex = g[extra].to_numpy(dtype=float) if extra else None
        for i in range(len(g)):
            t = tt[i]
            if t >= tq:
                break
            f = fx.step(t, dict(zip(chans, vals[i])))
            dense = (tq - t) <= DENSE_WINDOW
            if i % STRIDE == 0 or (dense and i % 2 == 0):
                r = [sid, t, tq] + f
                if ex is not None:
                    r += list(ex[i])
                rows.append(r)
    cols = ["shot_id", "time", "t_tq"] + names + [f"true_{c}" for c in extra]
    df = pd.DataFrame(rows, columns=cols)
    df = df.merge(sh[["shot_id", "scenario", "disrupted", "spontaneous", "wall_contact", "t_trigger_s"]],
                  on="shot_id")
    df["src"] = Path(src).name
    df["shot_key"] = df.src + ":" + df.shot_id.astype(str)
    num = df.select_dtypes("float64").columns
    df[num] = df[num].astype("float32")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst)
    print(f"{src}: {len(df)} рядків, {df.shot_id.nunique()} розрядів, {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
