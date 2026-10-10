"""Донавчання ImpactNet на даних іншого рушія (fusion-sim, без шуму) і чесна перевірка на відкладених розрядах.

    python risk_model/finetune_fusionsim.py measured.csv shots.csv run.json

Розряди ділимо навпіл (стратифіковано за причиною зриву, фіксований сид). Донавчаємо з поточних ваг лише на
першій половині, атрибуцію рахуємо на другій — тим самим способом, що й fusion-sim: частки моделі в момент,
коли її власна λ найбільша (до гасіння). Два варіанти:
  surv        — лише моменти зривів із shots.csv (як на реальній машині, без жодної розмітки причин);
  surv_cause  — плюс причина зриву для навчальних розрядів (післяаварійний розбір; на перевірці не використовується).
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
from impact_model import ImpactNet  # noqa: E402
from train_impact import OUT, transform  # noqa: E402

MEAS, SHOTS, RUN = sys.argv[1:4]
NAMES = ["greenwald", "q95", "radiation", "troyon", "other"]
WIN, BURN, B = 800, 128, 8
EPOCHS, STEPS, LR = int(os.environ.get("EPOCHS", "6")), int(os.environ.get("STEPS", "60")), 3e-4
torch.manual_seed(0)
torch.set_num_threads(os.cpu_count())

prep = json.loads((OUT / "impact_prep.json").read_text())
sh = pd.read_csv(SHOTS).set_index("shot_id")
run = json.load(open(RUN, encoding="utf-8"))
true = {s["shot_id"]: s["true_cause"]["dominant"] for s in run["shots"] if s.get("true_cause")}
fs_pred = {s["shot_id"]: s["predicted_cause"]["dominant"] for s in run["shots"] if s.get("true_cause")}

# ── дані: читаємо 540 МБ частинами, кожен розряд перетворюємо одразу (у WSL лише 3 ГБ) ──
t0 = time.time()
shots, buf = {}, []


def flush(df):
    for sid, g in df.groupby("shot_id", sort=False):
        g = g.reset_index(drop=True)
        X, drv = D.offline(g, prep["calib"])
        t = g.t.to_numpy(np.float32)
        r = sh.loc[sid]
        pk = g.prog_ip_MA.cummax().to_numpy()
        ok = (g.mag_ip_MA.to_numpy() > T.ph.IP_MIN) & ~(g.prog_ip_MA.to_numpy() < 0.9 * pk)
        if r.disrupted:
            ok &= t < r.t_tq_s - 0.02            # момент запуску зриву невідомий — беремо 20 мс до гасіння
        okw = np.where(ok)[0]
        ev = int(okw[-1]) if (r.disrupted and len(okw)) else -1
        cause = NAMES.index(true[sid]) if sid in true else 4
        shots[int(sid)] = dict(sid=int(sid), X=transform(X, prep).astype(np.float16), drv=np.nan_to_num(drv).astype(np.float32),
                               ok=ok, t=t, ev=ev, cause=cause, scored=sid in true and not r.spontaneous and not r.wall_contact)


cols = None
for ch in pd.read_csv(MEAS, chunksize=150_000):
    ch = ch.drop(columns=[c for c in ch.columns if c in ("p", "model_lambda") or c.startswith("share_")])
    if buf:
        ch = pd.concat([buf.pop(), ch])
    last = ch.shot_id.iloc[-1]
    buf.append(ch[ch.shot_id == last])
    flush(ch[ch.shot_id != last])
    print(f"  розрядів {len(shots)} [{time.time() - t0:.0f} с]", flush=True)
flush(buf.pop())
print(f"дані: {len(shots)} розрядів, зарахованих зривів {sum(s['scored'] for s in shots.values())} [{time.time() - t0:.0f} с]")

# ── поділ навпіл, стратифіковано за причиною ──
rng = np.random.default_rng(0)
train_ids, test_ids = [], []
for c in range(5):
    ids = [s for s in sorted(shots) if shots[s]["scored"] and shots[s]["cause"] == c]
    rng.shuffle(ids)
    train_ids += ids[::2]
    test_ids += ids[1::2]
rest = [s for s in sorted(shots) if not shots[s]["scored"]]
rng.shuffle(rest)
train_ids += rest[::2]
test_ids += rest[1::2]
tr = [shots[i] for i in train_ids]
te = [shots[i] for i in sorted(test_ids) if shots[i]["scored"]]
print("перевірка: зривів", len(te), "за причиною", {n: sum(s["cause"] == k for s in te) for k, n in enumerate(NAMES)})


def load_net():
    ck = torch.load(OUT / "impact_net.pt", map_location="cpu")
    net = ImpactNet(ck["n_in"], hidden=ck["hidden"], arch=ck.get("arch", "gru"))
    net.load_state_dict(ck["state"])
    return net


@torch.no_grad()
def attribution(net, data):
    """Як у fusion-sim: частки в момент, коли власна λ моделі найбільша (кроки до гасіння)."""
    net.eval()
    pred = {}
    for s in data:
        o = net(torch.from_numpy(s["X"].astype(np.float32))[None], torch.from_numpy(s["drv"])[None])
        lc = o["log_C"][0].numpy().astype(np.float64)
        lam = -np.expm1(-np.exp(np.logaddexp.reduce(lc, 1)))
        k = np.where(s["ok"])[0]
        i = k[np.argmax(lam[k])]
        pred[s["sid"]] = int(np.argmax(lc[i]))
    return pred


def report(tag, pred):
    hit = {n: [0, 0] for n in NAMES}
    for s in te:
        hit[NAMES[s["cause"]]][1] += 1
        hit[NAMES[s["cause"]]][0] += pred[s["sid"]] == s["cause"]
    tot = sum(h[0] for h in hit.values())
    print(f"{tag}: {tot} з {len(te)} | " + ", ".join(f"{n} {a}/{b}" for n, (a, b) in hit.items() if b), flush=True)
    return dict(total=tot, n=len(te), by_cause={n: h for n, h in hit.items() if h[1]})


def window(s):
    n = len(s["t"])
    if n <= WIN:
        a = 0
    elif s["ev"] >= 0 and rng.random() < 0.5:
        a = int(np.clip(s["ev"] + 1 - WIN + rng.integers(0, 50), 0, n - WIN))
    else:
        a = int(rng.integers(0, n - WIN))
    sl = slice(a, a + WIN)
    m = s["ok"][sl].copy()
    m[:BURN] = False
    ev = s["ev"] - a if (s["ev"] >= a + BURN and s["ev"] < a + WIN) else -1
    return s["X"][sl], s["drv"][sl], m, ev


def finetune(cause):
    net = load_net()
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    for ep in range(EPOCHS):
        net.train()
        t1, tot = time.time(), 0.0
        for _ in range(STEPS):
            ch = [tr[i] for i in rng.integers(0, len(tr), B)]
            w = [window(s) for s in ch]
            L = max(len(x[0]) for x in w)
            pad = lambda a, d: np.stack([np.pad(v.astype(d), [(0, L - len(v))] + [(0, 0)] * (v.ndim - 1)) for v in a])
            X = torch.from_numpy(pad([x[0] for x in w], np.float32))
            d = torch.from_numpy(pad([x[1] for x in w], np.float32))
            ok = torch.from_numpy(pad([x[2] for x in w], np.float32))
            lc = net(X, d)["log_C"]
            S = torch.exp(torch.logsumexp(lc, -1))
            lam = -torch.expm1(-S)
            expo = (lam * ok).sum(1) * D.DT
            evl = torch.zeros(B)
            for b, (s, x) in enumerate(zip(ch, w)):
                if x[3] >= 0:
                    evl[b] = lc[b, x[3], s["cause"]] if cause else torch.log(lam[b, x[3]].clamp_min(1e-30))
            loss = (expo.sum() - evl.sum()) / B
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        print(f"  епоха {ep + 1} loss {tot / STEPS:.4f} [{time.time() - t1:.0f} с]", flush=True)
    return net


res = {}
base = load_net()
bp = attribution(base, te)
agree = np.mean([NAMES[bp[s["sid"]]] == fs_pred[s["sid"]] for s in te])
print(f"узгодженість нашого відтворення з fusion-sim (вихідна мережа): {agree:.2f}")
res["before"] = report("до донавчання", bp)
for mode in ("surv", "surv_cause"):
    print(f"донавчання: {mode}")
    net = finetune(mode == "surv_cause")
    res[mode] = report(f"після ({mode})", attribution(net, te))
    torch.save(dict(state=net.state_dict(), n_in=tr[0]["X"].shape[1], hidden=net.hidden, arch=net.arch),
               OUT / f"impact_net_fs_{mode}.pt")
res["split"] = dict(train=sorted(int(i) for i in train_ids), test=sorted(int(s["sid"]) for s in te), seed=0)
(OUT / "finetune_fusionsim.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
print(json.dumps({k: v for k, v in res.items() if k != "split"}, ensure_ascii=False))
