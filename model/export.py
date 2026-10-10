"""Навчити обидві голови на всіх даних і зібрати теку здачі.

    python model/export.py submission [H_short H_long W]

Дерева HistGradientBoosting перекладаються в плоскі масиви numpy (weights.npz),
щоб на заліку не тягнути sklearn і рахувати крок за десятки мікросекунд.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAME_NOCS = "two-head HGB (без формул шпаргалки)"
NOTES_NOCS = ("Голова A: предвестник срыва (H={h_short} с). Голова B: ранняя тревога (H={h_long} с). "
              "p = max(A, {w}·B). Признаки — сглаженные сырые каналы приборов, фаза программы и сценарий, "
              "Dα, Мирнов, SXR, ECE, детектор предвестника по седловой петле; на спаде тока p = 0.")


def flatten(model):
    """HGB → масиви вузлів усіх дерев з глобальними індексами."""
    feat, thr, left, right, miss_left, leaf, value, roots = [], [], [], [], [], [], [], []
    off = 0
    for (pred,) in model._predictors:
        n = pred.nodes
        roots.append(off)
        feat.append(n["feature_idx"].astype(np.int32))
        thr.append(n["num_threshold"].astype(np.float64))
        left.append((n["left"] + off).astype(np.int32))
        right.append((n["right"] + off).astype(np.int32))
        miss_left.append(n["missing_go_to_left"].astype(bool))
        leaf.append(n["is_leaf"].astype(bool))
        value.append(n["value"].astype(np.float64))
        off += len(n)
    depth = max(int(p.nodes["depth"].max()) for (p,) in model._predictors)
    return dict(feat=np.concatenate(feat), thr=np.concatenate(thr), left=np.concatenate(left),
                right=np.concatenate(right), miss_left=np.concatenate(miss_left),
                leaf=np.concatenate(leaf), value=np.concatenate(value),
                roots=np.array(roots, dtype=np.int32), depth=np.int32(depth),
                base=np.float64(np.ravel(model._baseline_prediction)[0]))


def predict_flat(fl, x):
    """Та сама функція, що в model.py: x — вектор ознак (nan дозволено)."""
    idx = fl["roots"].copy()
    for _ in range(int(fl["depth"])):
        lf = fl["leaf"][idx]
        if lf.all():
            break
        v = x[fl["feat"][idx]]
        go_left = np.where(np.isnan(v), fl["miss_left"][idx], v <= fl["thr"][idx])
        nxt = np.where(go_left, fl["left"][idx], fl["right"][idx])
        idx = np.where(lf, idx, nxt)
    raw = fl["base"] + fl["value"][idx].sum()
    return 1.0 / (1.0 + np.exp(-raw))


def main(out, h_short=0.1, h_long=1.0, w=0.5):
    df = train.load()
    tr = train.train_mask(df)
    cols = {"a": train.COLS_A, "b": train.X_COLS}      # голова A може мати менший набір (HEAD_A_V2=1)
    heads = {}
    for name, H in (("a", h_short), ("b", h_long)):
        X = df.loc[tr, cols[name]]
        m = train.make_model().fit(X, train.labels(df[tr], H), sample_weight=train.row_weights(df[tr]))
        fl = flatten(m)
        # звірка: плоский предиктор має збігатися з sklearn
        xs = X.sample(300, random_state=1).to_numpy(dtype=float)
        ref = m.predict_proba(xs)[:, 1]
        got = np.array([predict_flat(fl, r) for r in xs])
        err = np.abs(ref - got).max()
        print(f"голова {name} (H={H}): {len(fl['roots'])} дерев, глибина {fl['depth']}, розбіжність {err:.2e}")
        assert err < 1e-6
        heads[name] = fl

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    arrs = {f"{h}_{k}": v for h, fl in heads.items() for k, v in fl.items()}
    np.savez(out / "weights.npz", **arrs)
    (out / "config.json").write_text(json.dumps(dict(
        cols=train.X_COLS, cols_a=cols["a"], cols_b=cols["b"], combine=__import__("os").environ.get("COMBINE", "noisy_or"), w=w, h_short=h_short, h_long=h_long, ema=0.1), indent=1), encoding="utf-8")
    feat_mod = __import__("os").environ.get("FEATURES", "features")
    shutil.copyfile(ROOT / "model" / f"{feat_mod}.py", out / "features.py")
    shutil.copyfile(ROOT / "model" / "submission_model.py", out / "model.py")
    extra = []
    if feat_mod in ("features", "features_v9") and (ROOT / "model" / "q95_form.json").exists():  # форма плазми (v3)
        shutil.copyfile(ROOT / "model" / "q95_form.json", out / "q95_form.json")
        extra.append("q95_form.json")
    if feat_mod == "features_v9":                    # v8 як база + нейромережа вкладів механізмів
        shutil.copyfile(ROOT / "model" / "features.py", out / "features_v8_base.py")
        for f in ("riskfeat.py", "risk_net.npz"):
            shutil.copyfile(ROOT / "model" / f, out / f)
        extra += ["features_v8_base.py", "riskfeat.py", "risk_net.npz"]
    (out / "toksym.json").write_text(json.dumps(dict(
        schema=1, entry="model.py", files=["features.py", "weights.npz", "config.json"] + extra,
        requires=["numpy"], name=NAME_NOCS if feat_mod == "features_nocs" else "two-head HGB",
        notes=(NOTES_NOCS.format(h_short=h_short, h_long=h_long, w=w) if feat_mod == "features_nocs" else
               f"Голова A: передвісник (H={h_short} с). Голова B: близькість до меж (H={h_long} с), "
               f"стиснута ×{w}. Ознаки — онлайн EMA фізичних величин (f_GW, q, β, P_rad/P_in, Dα, сідлова петля), "
               "сценарій із meta, нуль на спаді струму.")), ensure_ascii=False, indent=1), encoding="utf-8")
    print("здача:", out)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], *(float(x) for x in a[1:]))
