"""Ансамбль (бегінг по розрядах) для здачі: N моделей кожної голови на бутстреп-вибірках РОЗРЯДІВ,
дерева склеюються в один плоский масив, листки × 1/N, base = середнє → вихід = середній логіт N моделей.
model.py здачі не змінюється (той самий формат weights.npz).

    FEATURES=features_clean COMBINE=max HEAD_A_V2=1 FEAT_DIR=features_nocs_9k ENS=5 \
        python model/export_ens.py submission_clean_ens 0.1 4.5 0.3
"""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export as E  # noqa: E402
import train  # noqa: E402

ENS = int(os.environ.get("ENS", "5"))


def fit_one(df, tr_idx, cols, H, seed):
    if True:                                       # послідовно: кожна модель бере всі ядра (OpenMP)
        d = df.loc[tr_idx]
        m = train.make_model().set_params(random_state=seed)
        m.fit(d[cols], train.labels(d, H), sample_weight=train.row_weights(d))
        return E.flatten(m), m


def merge(fls):
    n, off, out = len(fls), 0, {k: [] for k in ("feat", "thr", "left", "right", "miss_left", "leaf", "value", "roots")}
    for fl in fls:
        size = len(fl["feat"])
        for k in ("feat", "thr", "miss_left", "leaf"):
            out[k].append(fl[k])
        out["left"].append(fl["left"] + off)
        out["right"].append(fl["right"] + off)
        out["value"].append(fl["value"] / n)
        out["roots"].append(fl["roots"] + off)
        off += size
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["depth"] = np.int32(max(int(fl["depth"]) for fl in fls))
    res["base"] = np.float64(np.mean([float(fl["base"]) for fl in fls]))
    return res


def main(out, h_short=0.1, h_long=4.5, w=0.3):
    df = train.load()
    tr = train.train_mask(df)
    keys = df.loc[tr, "shot_key"].unique()
    rng = np.random.default_rng(0)
    boots = [rng.choice(keys, size=len(keys), replace=True) for _ in range(ENS)]
    by_key = df[tr].groupby("shot_key").indices
    base_idx = df.index[tr].to_numpy()
    idx = [np.concatenate([base_idx[by_key[k]] for k in b]) for b in boots]   # повтори розрядів = ваги бутстрепу
    cols = {"a": train.COLS_A, "b": train.X_COLS}
    heads = {}
    for name, H in (("a", h_short), ("b", h_long)):
        res = [fit_one(df, idx[e], cols[name], H, e) for e in range(ENS)]
        fl = merge([r[0] for r in res])
        xs = df.loc[tr, cols[name]].sample(300, random_state=1).to_numpy(dtype=float)
        ref = np.mean([r[1]._raw_predict(xs)[:, 0] for r in res], axis=0)
        got = np.array([np.log(p / (1 - p)) for p in (E.predict_flat(fl, x) for x in xs)])
        err = np.abs(ref - got).max()
        print(f"голова {name} (H={H}): ансамбль {ENS} × {len(res[0][0]['roots'])} дерев, розбіжність логіта {err:.1e}", flush=True)
        assert err < 1e-6
        heads[name] = fl
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "weights.npz", **{f"{h}_{k}": v for h, fl in heads.items() for k, v in fl.items()})
    (out / "config.json").write_text(json.dumps(dict(
        cols=train.X_COLS, cols_a=cols["a"], cols_b=cols["b"], combine=os.environ.get("COMBINE", "max"),
        w=w, h_short=h_short, h_long=h_long, ema=0.1, ensemble=ENS), indent=1), encoding="utf-8")
    shutil.copyfile(train.ROOT / "model" / f"{os.environ.get('FEATURES', 'features')}.py", out / "features.py")
    shutil.copyfile(train.ROOT / "model" / "submission_model.py", out / "model.py")
    man = json.loads((train.ROOT / "submission_clean" / "toksym.json").read_text(encoding="utf-8"))
    man["name"] = os.environ.get("NAME", f"two-head HGB (чиста, ансамбль {ENS})")
    (out / "toksym.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    print("здача:", out)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], *(float(x) for x in a[1:]))
