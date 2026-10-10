"""ImpactNet: дані → навчання → оцінка (ті самі рядки й метрики, що в MLP-еталона).

    python risk_model/train_impact.py build        # калібрування, нормалізація, послідовності по 5 мс
    python risk_model/train_impact.py train        # навчання + test + pred_test_impact.npz
    python risk_model/train_impact.py parity       # forward() == step() на реальному розряді

Нормалізація входів (218 = 100 приладів + 100 масок відмов + 18 фізичних величин):
  прилади й фізика: asinh(x / s) (s — медіана |x|: важкі хвости потужностей, нейтронів, тисків
  стискаються, знак зберігається) → (· − медіана) / IQR з train → обрізання ±8; маски — 0/1 як є.
Навчання: вікна 3,2 с (640 кроків по 5 мс), перші 0,64 с — прогрів стану без втрати; батч 24;
AdamW + OneCycle; відбір епохи за val. Мітки — справжні r_k^p (targets.py), кроки — як у build.py:
струм > IP_MIN, не спад, до запуску зриву.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drivers as D  # noqa: E402
import targets as T  # noqa: E402
from impact_model import IMPACT_NAMES, ImpactNet, impact_loss  # noqa: E402

DATA = Path(os.environ.get("RISK_DATA", "~/risk_data")).expanduser()
OUT = Path(__file__).resolve().parent
TRAIN = ["r501", "r502", "r503", "r504", "r505", "r506", "rs601", "rs602", "rs603"] + os.environ.get("TRAIN_EXTRA", "").split()
VAL, TEST = ["r507", "rs604"], ["r508", "rs605"]
ORDER = [T.MECH.index(m) for m in ("density", "kink", "radiation", "beta", "locked_mode")]   # → IMPACT_NAMES
INV = [ORDER.index(i) for i in range(5)]                                                    # назад до T.MECH
N_CH = len(D.CHANNELS)
CONT = np.r_[np.arange(N_CH), np.arange(2 * N_CH, 2 * N_CH + len(D.PHYS))]                 # не маски
WIN, BURN = 640, 128
BS = int(os.environ.get("BS", "24"))
SHIFT_P = float(os.environ.get("SHIFT_P", "0"))      # >0: частка вікон із shift-розрядів у батчі
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = int(os.environ.get("EPOCHS", "12"))
SEED = int(os.environ.get("SEED", "0"))
PREP = OUT / "impact_prep.json"


def transform(X, prep):
    X = X.copy()
    s, med, iqr = (np.array(prep[k]) for k in ("s", "med", "iqr"))
    X[:, CONT] = np.clip((np.arcsinh(X[:, CONT] / s) - med) / iqr, -8, 8)
    return np.nan_to_num(X, nan=0.0)


def shots_of(name):
    m = pd.read_csv(DATA / name / "measured.csv")
    sh = pd.read_csv(DATA / name / "shots.csv").set_index("shot_id")
    for sid, g in m.groupby("shot_id", sort=False):
        yield sid, g.reset_index(drop=True), sh.loc[sid]


def labels(g, r):
    risk = T.mechanism_risks(g, D.DT)[:, ORDER]
    t = g.t.to_numpy()
    pk = g.prog_ip_MA.cummax().to_numpy()
    ok = (g.ip_MA.to_numpy() > T.ph.IP_MIN) & ~(g.prog_ip_MA.to_numpy() < 0.9 * pk)
    if r.disrupted and r.t_trigger_s == r.t_trigger_s:
        ok &= t < r.t_trigger_s
    return risk.astype(np.float32), ok


def build():
    if not PREP.exists():
        cal_shots = [g for i, (_, g, _) in enumerate(shots_of("r501")) if i < 40] + \
                    [g for i, (_, g, _) in enumerate(shots_of("rs601")) if i < 40]
        calib = D.fit_calibration(cal_shots)
        Xs = np.concatenate([D.offline(g, calib)[0][::10] for g in cal_shots])[:, CONT]
        s = np.nanmedian(np.abs(Xs), 0) + 1e-6
        A = np.arcsinh(Xs / s)
        med = np.nanmedian(A, 0)
        q1, q3 = np.nanpercentile(A, [25, 75], 0)
        iqr = np.where(q3 - q1 > 1e-3, q3 - q1, np.maximum(np.nanstd(A, 0), 1e-3))
        PREP.write_text(json.dumps(dict(calib=calib, s=s.tolist(), med=med.tolist(), iqr=iqr.tolist()), indent=1))
        print("калібрування:", calib)
    prep = json.loads(PREP.read_text())
    for name in (os.environ["EXTRA_ONLY"].split() if os.environ.get("EXTRA_ONLY") else TRAIN + VAL + TEST):
        dst = DATA / f"seq_{name}.npz"
        if dst.exists():
            continue
        t0 = time.time()
        acc = {k: [] for k in ("X", "drv", "risk", "ok", "t")}
        sids, lens = [], []
        for sid, g, r in shots_of(name):
            X, drv = D.offline(g, prep["calib"])
            risk, ok = labels(g, r)
            acc["X"].append(transform(X, prep).astype(np.float16))
            acc["drv"].append(np.nan_to_num(drv, nan=0.0).astype(np.float32))
            acc["risk"].append(risk)
            acc["ok"].append(ok)
            acc["t"].append(g.t.to_numpy(np.float32))
            sids.append(sid)
            lens.append(len(g))
        np.savez(dst, **{k: np.concatenate(v) for k, v in acc.items()}, sid=np.array(sids), lens=np.array(lens))
        print(f"{name}: {len(sids)} розрядів, {sum(lens)} кроків [{time.time() - t0:.0f} с]", flush=True)


def load(names):
    shots = []
    for n in names:
        with np.load(DATA / f"seq_{n}.npz") as f:       # кожен масив читаємо з архіву ОДИН раз
            z = {k: f[k] for k in f.files}               # (z["X"] у циклі перечитував би весь масив)
        off = np.r_[0, np.cumsum(z["lens"])]
        sh = pd.read_csv(DATA / n / "shots.csv").set_index("shot_id")
        for i, sid in enumerate(z["sid"]):
            sl = slice(off[i], off[i + 1])
            s = dict(src=n, sid=int(sid), X=z["X"][sl], drv=z["drv"][sl], risk=z["risk"][sl], ok=z["ok"][sl], t=z["t"][sl])
            # подія для survival-навчання: останній крок перед запуском зриву (лише shots.csv — як на реальній машині);
            # «експертна причина» — механізм з найбільшим вкладом у цей момент (спонтанні → «інші»)
            r = sh.loc[int(sid)]
            okw = np.where(s["ok"])[0]
            s["ev"] = int(okw[-1]) if (r.disrupted and r.t_trigger_s == r.t_trigger_s and len(okw)) else -1
            s["cause"] = 4 if (s["ev"] < 0 or r.spontaneous) else int(np.argmax(s["risk"][s["ev"]]))
            shots.append(s)
    return shots


def batch(shots, rng):
    xs, ds, rs, ms = [], [], [], []
    sh_idx = [i for i, s in enumerate(shots) if s["src"].startswith("rs")]
    no_idx = [i for i, s in enumerate(shots) if not s["src"].startswith("rs")]
    for _ in range(BS):
        while True:
            if SHIFT_P > 0:                       # рівна (або задана) частка shift-вікон
                pool = sh_idx if rng.random() < SHIFT_P else no_idx
                s = shots[pool[rng.integers(len(pool))]]
            else:
                s = shots[rng.integers(len(shots))]
            n = len(s["t"])
            if n <= WIN:
                continue
            a = rng.integers(0, n - WIN)
            if s["ok"][a + BURN:a + WIN].any():
                break
        m = s["ok"][a:a + WIN].copy()
        m[:BURN] = False
        xs.append(s["X"][a:a + WIN]); ds.append(s["drv"][a:a + WIN]); rs.append(s["risk"][a:a + WIN]); ms.append(m)
    f = lambda v, dt: torch.from_numpy(np.stack(v).astype(dt))
    return tuple(f(v, np.float32).to(DEV) for v in (xs, ds, rs, ms))


@torch.no_grad()
def predict(net, shots):
    net.eval()
    out = []
    for s in shots:
        o = net(torch.from_numpy(s["X"].astype(np.float32))[None].to(DEV), torch.from_numpy(s["drv"])[None].to(DEV))
        out.append(o["log_C"][0].cpu().numpy().astype(np.float64))
    return out


def rows(shots, preds):
    """Ті самі рядки, що в MLP-еталона: мітка є і крок кратний 5 (25 мс)."""
    z, risk, meta = [], [], {"shot": [], "t": [], "src": []}
    for s, p in zip(shots, preds):
        k = s["ok"] & (np.arange(len(s["t"])) % 5 == 0)
        z.append(p[k][:, INV]); risk.append(s["risk"][k][:, INV].astype(np.float64))
        meta["shot"].append(np.full(k.sum(), s["sid"])); meta["t"].append(s["t"][k]); meta["src"].append(np.full(k.sum(), s["src"]))
    return np.concatenate(z), np.concatenate(risk), {k: np.concatenate(v) for k, v in meta.items()}


def train():
    from train import metrics
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    tr, va = load(TRAIN), load(VAL)
    net = ImpactNet(tr[0]["X"].shape[1], hidden=int(os.environ.get("HIDDEN", "128")),
                    p_drop=float(os.environ.get("DROP", "0.1")), arch=os.environ.get("ARCH", "gru")).to(DEV)
    print(f"параметрів: {sum(p.numel() for p in net.parameters())}, рецептивне поле згорток {net.receptive_field} кроків")
    z0, r0, _ = rows(va, predict(net, va))
    m0 = metrics(z0, r0)
    print(f"до навчання (чиста фізика з приладів): val λ R² {m0['lam_r2']:.3f}, частки {m0['share_mae_lamw']:.2f} п.п., "
          f"домін.@0,1 {m0['dom_acc@0.1']:.3f}", flush=True)
    steps = int(os.environ.get("STEPS", "150"))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=EPOCHS * steps, pct_start=0.15)
    best, best_state = np.inf, None
    for ep in range(EPOCHS):
        net.train()
        t0, tot = time.time(), 0.0
        for _ in range(steps):
            x, d, r, m = batch(tr, rng)
            loss = impact_loss(net(x, d), r, m)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        z, rk, _ = rows(va, predict(net, va))
        m = metrics(z, rk)
        score = m["share_mae_lamw"] + 100 * m["lam_mae"]
        print(f"епоха {ep + 1:2d} loss {tot / steps:.4f} | val: λ MAE {m['lam_mae']:.4f} R² {m['lam_r2']:.3f} "
              f"частки {m['share_mae_lamw']:.2f} п.п. домін.@0,1 {m['dom_acc@0.1']:.3f} | відбір {score:.3f} "
              f"[{time.time() - t0:.0f} с]", flush=True)
        if score < best:
            best, best_state = score, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    tag = os.environ.get("TAG", "")
    torch.save(dict(state={k: v.cpu() for k, v in net.state_dict().items()}, n_in=tr[0]["X"].shape[1],
                    hidden=net.hidden, arch=net.arch), OUT / f"impact_net{tag}.pt")
    del tr
    te = load(TEST)
    z, rk, meta = rows(te, predict(net, te))
    np.savez(DATA / f"pred_test_impact{tag}.npz", z=z, risk=rk, **meta)
    res = {"val": metrics(*rows(va, predict(net, va))[:2]), "test": metrics(z, rk)}
    print(json.dumps({k: {m: round(float(v), 4) for m, v in d.items()} for k, d in res.items()}, ensure_ascii=False, indent=1))
    print("пороги априорі після навчання: c =", net.c.detach().cpu().numpy().round(3), " w =", net.w.detach().cpu().numpy().round(3))
    (OUT / f"impact_metrics{tag}.json").write_text(json.dumps({k: {m: float(v) for m, v in d.items()} for k, d in res.items()}, indent=1))


def _pad(shots):
    L = max(len(s["t"]) for s in shots)
    f = lambda k, d: np.stack([np.pad(s[k].astype(d), [(0, L - len(s["t"]))] + [(0, 0)] * (s[k].ndim - 1)) for s in shots])
    return (torch.from_numpy(f("X", np.float32)).to(DEV), torch.from_numpy(f("drv", np.float32)).to(DEV),
            torch.from_numpy(f("ok", np.float32)).to(DEV))


def surv_terms(log_C, ok, shots, cause):
    """Пуассонівська log-правдоподібність розрядів: Σ_події log λ(t_події) − Σ ∫λ dt (λ — на секунду).
    cause=True: подія з відомою причиною k → log λ_k ≈ log C_k (конкуруючі ризики)."""
    S = torch.exp(torch.logsumexp(log_C, -1))                         # ΣC
    lam = -torch.expm1(-S)
    expo = (lam * ok).sum(1) * D.DT
    ev = torch.zeros(len(shots), device=log_C.device)
    for b, s in enumerate(shots):
        if s["ev"] >= 0:
            ev[b] = log_C[b, s["ev"], s["cause"]] if cause else torch.log(lam[b, s["ev"]].clamp_min(1e-30))
    return expo, ev


def train_surv():
    """Навчання БЕЗ міток λ: лише моменти зривів із shots.csv (+ причина зриву, якщо LOSS=surv_cause).
    Відбір епохи — за правдоподібністю на val (теж без λ). Справжня λ — лише для фінальної перевірки."""
    from train import metrics
    cause = os.environ.get("LOSS") == "surv_cause"
    prior = os.environ.get("PRIOR", "1") == "1"
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    tr, va = load(TRAIN), load(VAL)
    net = ImpactNet(tr[0]["X"].shape[1], physics_prior=prior).to(DEV)
    steps, B = int(os.environ.get("STEPS", "100")), 8
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=EPOCHS * steps, pct_start=0.15)

    def val_ll():
        net.eval()
        tot = 0.0
        with torch.no_grad():
            for i in range(0, len(va), 16):
                ch = va[i:i + 16]
                X, d, ok = _pad(ch)
                e, v = surv_terms(net(X, d)["log_C"], ok, ch, False)
                tot += float(v.sum() - e.sum())
        return tot / len(va)

    print(f"survival ({'з причиною' if cause else 'лише момент зриву'}, априорі {'так' if prior else 'ні'}); "
          f"val log-правдоподібність до навчання {val_ll():+.4f}", flush=True)
    best, best_state = -np.inf, None
    for ep in range(EPOCHS):
        net.train()
        t0, tot = time.time(), 0.0
        for _ in range(steps):
            ch = [tr[i] for i in rng.integers(0, len(tr), B)]
            X, d, ok = _pad(ch)
            e, v = surv_terms(net(X, d)["log_C"], ok, ch, cause)
            loss = (e.sum() - v.sum()) / B
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        ll = val_ll()
        z, rk, _ = rows(va, predict(net, va))
        m = metrics(z, rk)                                              # лише для звіту, НЕ для відбору
        print(f"епоха {ep + 1:2d} loss {tot / steps:.4f} | val logL {ll:+.4f} | (перевірка за істиною: λ R² {m['lam_r2']:.3f}, "
              f"частки {m['share_mae_lamw']:.2f} п.п., домін.@0,1 {m['dom_acc@0.1']:.3f}) [{time.time() - t0:.0f} с]", flush=True)
        if ll > best:
            best, best_state = ll, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    tag = os.environ.get("TAG", "")
    torch.save(dict(state={k: v.cpu() for k, v in net.state_dict().items()}, n_in=tr[0]["X"].shape[1], hidden=net.hidden),
               OUT / f"impact_net{tag}.pt")
    del tr
    te = load(TEST)
    z, rk, meta = rows(te, predict(net, te))
    np.savez(DATA / f"pred_test_impact{tag}.npz", z=z, risk=rk, **meta)
    print(json.dumps({"test": {k: round(float(v), 4) for k, v in metrics(z, rk).items()}}, ensure_ascii=False, indent=1))
    print("пороги априорі: c =", net.c.detach().cpu().numpy().round(3), " w =", net.w.detach().cpu().numpy().round(3))


def parity():
    """forward() по всьому розряду == step() по кроках (з повним онлайн-препроцесингом)."""
    prep = json.loads(PREP.read_text())
    ck = torch.load(OUT / "impact_net.pt", weights_only=False) if (OUT / "impact_net.pt").exists() else None
    sid, g, _ = next(shots_of("r508"))
    g = g.iloc[:800]
    X, drv = D.offline(g, prep["calib"])
    net = ImpactNet(X.shape[1])
    if ck:
        net.load_state_dict(ck["state"])
    else:                                           # без навченої — випадкові ваги голови й залишків,
        torch.manual_seed(1)                        # інакше звірка тривіальна (вихід = лише фізика)
        with torch.no_grad():
            net.head.weight.normal_(0, 0.3)
            for b in net.blocks:
                b.out.weight.normal_(0, 0.1)
    net.eval()
    torch.set_num_threads(1)
    rows_ = [{k: g[k].iat[i] for k in D.CHANNELS} for i in range(len(g))]
    with torch.no_grad():
        full = net(torch.from_numpy(transform(X, prep)).float()[None], torch.from_numpy(np.nan_to_num(drv)).float()[None])
        on, st = D.Online(prep["calib"]), net.init_state()
        err, tp, tn = 0.0, [], []
        for i in range(len(g)):
            t0 = time.perf_counter()
            x, d = on.step(rows_[i])
            xn, dn = torch.from_numpy(transform(x[None], prep)).float(), torch.from_numpy(np.nan_to_num(d))[None].float()
            t1 = time.perf_counter()
            o, st = net.step(xn, dn, st)
            t2 = time.perf_counter()
            tp.append(t1 - t0)
            tn.append(t2 - t1)
            err = max(err, float((o["share_pct"][0] - full["share_pct"][0, i]).abs().max()))
    q = lambda v: f"p50 {np.median(np.array(v[50:]) * 1e3):.2f} мс, p99 {np.percentile(np.array(v[50:]) * 1e3, 99):.2f} мс"
    print(f"forward vs step: макс. розбіжність часток {err:.2e} п.п. (частки в діапазоні "
          f"{float(full['share_pct'].min()):.1f}–{float(full['share_pct'].max()):.1f} %)")
    print(f"препроцесинг: {q(tp)} | мережа (torch step, 1 потік): {q(tn)}")


if __name__ == "__main__":
    {"build": build, "build_one": build, "train": train, "surv": train_surv, "parity": parity}[sys.argv[1]]()
