"""Навчання і крос-валідація по сідах.

    python model/train.py cv  [H]      # leave-one-seed-out, метрика заліку
    python model/train.py fit [H] OUT  # навчити на всьому і зберегти

Мітка: y = 1, якщо до гасіння лишилося ≤ H с. Спонтанні зриви й контакт зі
стінкою з навчання прибрано: передбачати там нічого, а шум у мітках шкодить.
Рядки на спаді струму теж прибрано: там модель на заліку все одно мовчить.
"""
import glob
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib as _il, os as _os  # noqa: E402
_F = _il.import_module(_os.environ.get("FEATURES", "features"))  # FEATURES=features_nocs — без шпаргалки
feature_names = _F.feature_names
from metric import fmt, report  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FEAT_DIR = ROOT / __import__("os").environ.get("FEAT_DIR", "features")  # features_kf — з Калманом
FEATS = feature_names()
DROP = {"t"}  # абсолютний час не переноситься між програмами розрядів
# у nocs2 ці два _rel завжди порожні (в начале плато неон і ECH вимкнені) — sklearn на них падає
ALWAYS_DROP = {"x_gas_flow_ne_meas_rel", "x_ech_power_MW_rel"}
X_COLS = [c for c in FEATS if c not in DROP and c not in ALWAYS_DROP]
# абляція «наскільки модель спирається на програму симулятора»: DROP_GROUPS=scen,timers,program
_GROUPS = {
    "scen": [c for c in FEATS if c.startswith("scen_")],                       # мітка сценарію з meta
    "timers": ["t_flat", "t_neon_on", "t_heat_on", "t_ne_flat"],               # секунди від подій програми
    "program": ["neon_cmd", "neon_int", "prog_ne_slope", "raw_prog_ne_1e20", "ip_frac", "ramping_up",
                "raw_gas_flow_ne_meas", "raw_gas_flow_d2_meas"] + [c for c in FEATS if c.startswith("ne_over_prog")],
}
for _g in filter(None, __import__("os").environ.get("DROP_GROUPS", "").split(",")):
    X_COLS = [c for c in X_COLS if c not in _GROUPS[_g]]
# ознаки v3 (фаза програми + фізика меж) — лише для голови B, якщо HEAD_A_V2=1
V3_ONLY = {"t_neon_on", "neon_cmd", "neon_int", "t_heat_on", "prog_ne_slope", "t_ne_flat", "q95_hat", "r_gw",
           "r_beta", "r_q", "r_rad", "lam_hat", "lam_hat_max", "lam_int",
           "f_gw_prog",
           "nn_lam", "nn_logC_density", "nn_logC_kink", "nn_logC_radiation", "nn_logC_lm",   # v9: нейромережа
           "nn_lam_max", "nn_lam_1s", "nn_Lam"}
COLS_A = ([c for c in X_COLS if c not in V3_ONLY and not c.startswith("x_")]  # x_* — лише голова B
          if __import__("os").environ.get("HEAD_A_V2") == "1"
          else X_COLS)


EVAL_ONLY = __import__("os").environ.get("EVAL_ONLY")  # regex: тестуємо лише ці сиди, решта — завжди в навчанні
SUBSAMPLE = int(__import__("os").environ.get("SUBSAMPLE", "1"))  # >1: вдали от срыва брать каждую N-ю строку (память)


def load():
    fs = sorted(glob.glob(str(FEAT_DIR / "tr*.parquet")))
    parts = []
    for f in fs:
        d = pd.read_parquet(f)
        if SUBSAMPLE > 1:
            # розрідження лише для НАВЧАННЯ: усе, що близько до зриву, + кожен N-й рядок інакше.
            # Тестові сиди (EVAL_ONLY) зберігаються повністю — оцінка по всіх рядках.
            near = (d.t_tq - d.time) <= 6.0
            keep = (near | (np.arange(len(d)) % SUBSAMPLE == 0)).to_numpy()
            src = Path(f).stem
            if EVAL_ONLY and __import__("re").fullmatch(EVAL_ONLY, src):
                d = d.assign(_tr=keep)
            else:
                d = d[keep].assign(_tr=True)
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def make_model():
    # параметри можна перевизначити: HGB='{"max_leaf_nodes": 63, ...}'
    p = dict(max_iter=250, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
             l2_regularization=1.0, max_bins=128, early_stopping=False, random_state=0)
    p.update(__import__("json").loads(__import__("os").environ.get("HGB", "{}")))
    return HistGradientBoostingClassifier(**p)


def labels(df, H):
    return ((df.t_tq - df.time) <= H).astype(int)


def train_mask(df):
    m = ~(df.spontaneous | df.wall_contact) & (df.rampdown < 0.5)
    return m & df["_tr"] if "_tr" in df else m


def postprocess(p, rampdown):
    """p по кроках одного розряду → відповідь моделі: нуль на спаді, далі
    EMA і бігучий максимум (тривога — подія, не стан)."""
    out = np.zeros_like(p)
    ema, peak = 0.0, 0.0
    for i, (x, rd) in enumerate(zip(p, rampdown)):
        if rd > 0.5:
            x = 0.0
        ema = x if i == 0 else ema + 0.3 * (x - ema)
        peak = max(peak, ema)
        out[i] = peak
    return out


def eval_shots(df, p):
    df = df.assign(p=p)
    shots = []
    for _, g in df.groupby("shot_key", sort=False):
        if bool(g.spontaneous.iloc[0]) or bool(g.wall_contact.iloc[0]):
            continue
        dis = bool(g.disrupted.iloc[0])
        pp = postprocess(g.p.to_numpy(), g.rampdown.to_numpy())
        shots.append(dict(t=g.time.to_numpy(), p=pp, t_tq=float(g.t_tq.iloc[0]) if dis else np.inf,
                          quiet=not dis))
    return report(shots)




def eval_mask(df):
    return df.src.str.fullmatch(EVAL_ONLY).to_numpy() if EVAL_ONLY else np.ones(len(df), bool)


WEIGHTS = __import__("os").environ.get("WEIGHTS", "")   # "shot": кожен розряд має однакову сумарну вагу


def row_weights(d):
    """Ваги рядків: при WEIGHTS=shot розряд важить 1 незалежно від кількості рядків."""
    if WEIGHTS != "shot":
        return None
    n = d.groupby("shot_key").shot_key.transform("size").to_numpy(dtype=float)
    w = 1.0 / n
    return w * (len(w) / w.sum())


def oof_preds(df, H, cols=None):
    cols = cols or X_COLS
    oof = np.zeros(len(df))
    for s in sorted(df.src.unique()):
        if EVAL_ONLY and not __import__("re").fullmatch(EVAL_ONLY, s):
            continue
        tr = (df.src != s) & train_mask(df)
        te = df.src == s
        m = make_model().fit(df.loc[tr, cols], labels(df[tr], H), sample_weight=row_weights(df[tr]))
        oof[te.to_numpy()] = m.predict_proba(df.loc[te, cols])[:, 1]
    return oof


def combine(a, b, w):
    """Голова передвісника a — «впевнена» тривога; голова близькості b стиснута в [0, w]."""
    return 1.0 - (1.0 - a) * (1.0 - w * b)


def cv(H):
    df = load()
    print(f"рядків {len(df)}, сідів {df.src.nunique()}, H={H}")
    t0 = time.time()
    oof = oof_preds(df, H)
    print(f"cv {time.time() - t0:.0f} с")
    res = eval_shots(df, oof)
    print(fmt(res))
    return df, oof


def cv2(H_short, H_long):
    df = load()
    a = oof_preds(df, H_short, COLS_A)
    print(f"голова A: {len(COLS_A)} ознак, голова B: {len(X_COLS)} ознак")
    b = oof_preds(df, H_long)
    tag = f"{H_short:g}_{H_long:g}" + ("_a2" if COLS_A is not X_COLS else "") + __import__("os").environ.get("TAG", "")
    np.savez(FEAT_DIR / f"oof_{tag}.npz", a=a, b=b)
    ev = eval_mask(df)
    df, a, b = df[ev], a[ev], b[ev]
    print(f"оцінка на {df.shot_key.nunique()} розрядах, навчання — решта")
    shift = df.src.str.startswith("trs").to_numpy()
    for w in (0.7, 0.5, 0.3):
        p = combine(a, b, w)
        print(f"\n== A(H={H_short}) ⊕ {w}·B(H={H_long}) ==")
        print("звичайний:", fmt(eval_shots(df[~shift], p[~shift])))
        if shift.any():
            print("shift:    ", fmt(eval_shots(df[shift], p[shift])))
    print("\n== тільки A ==")
    print(fmt(eval_shots(df, a)))


def fit(H, out):
    df = load()
    tr = train_mask(df)
    m = make_model().fit(df.loc[tr, X_COLS], labels(df[tr], H))
    joblib.dump(dict(model=m, cols=X_COLS, H=H), out)
    print("збережено", out)


if __name__ == "__main__":
    mode = sys.argv[1]
    H = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    if mode == "cv":
        cv(H)
    elif mode == "cv2":
        cv2(float(sys.argv[2]), float(sys.argv[3]))
    else:
        fit(H, sys.argv[3])
