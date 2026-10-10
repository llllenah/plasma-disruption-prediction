"""Порахувати офлайн-метрику з даних прогону toksym score (--save-data).

    python model/eval_run.py runs/baseline_s1_data
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metric import fmt, report  # noqa: E402


def load(d):
    m = pd.read_csv(f"{d}/measured.csv", usecols=["shot_id", "t", "p"])
    s = pd.read_csv(f"{d}/shots.csv")
    return m, s


def shots_from(m, s):
    s = s.set_index("shot_id")
    out = []
    for sid, g in m.groupby("shot_id"):
        r = s.loc[sid]
        if bool(r.get("spontaneous", False)) or bool(r.get("wall_contact", False)):
            continue
        dis = bool(r.disrupted)
        tq = float(r.t_tq) if "t_tq" in r and r.t_tq == r.t_tq else float(r.get("t_tq_s", np.inf))
        out.append(dict(t=g.t.to_numpy(), p=g.p.to_numpy(), t_tq=tq if dis else np.inf, quiet=not dis))
    return out


if __name__ == "__main__":
    m, s = load(sys.argv[1])
    print(list(s.columns))
    print(fmt(report(shots_from(m, s))))
