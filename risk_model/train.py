"""Нейромережа «вклад механізмів зриву»: 300 входів (100 приладів × 3 EMA) → 5 log-вкладів.

    python risk_model/train.py [BAL=0.7] [EPOCHS=20]

Вихідний шар — zₖ = log Cₖ. Звідси все інше виходить точно, без окремих голів:
    share_pct = 100 · softmax(z)            (сума рівно 100 %, бо softmax(log C) = C / ΣC)
    risk_k    = 1 − exp(−Cₖ) = rₖᵖ
    risk_total = 1 − exp(−ΣCₖ) = λ           (= 1 − Π(1 − rₖᵖ), формула симулятора)

Втрата:  Huber(z, log C) + Huber(logΣĈ, logΣC) + 2 · λ_true · KL(share_true ‖ share_pred)
  — перші дві вчать величину ризику (в лог-масштабі, щоб розрізняти 10⁻⁴ і 10⁻²),
  — KL вчить частки, але з вагою λ: частки при нульовому ризику — шум, їх не форсуємо.

Баланс: рядок належить групі «домінуючий механізм × рівень λ» (або «тихо», λ < 10⁻³);
ймовірність вибору рядка ∝ n_групи^(−BAL). BAL = 0 — природний розподіл (95 % «тихо»),
BAL = 1 — повний баланс (переучується на рідкісних групах); 0,7 — компроміс, перевірено на val.
Поділ — за сидами (цілими розрядами): train r501–506 + rs601–603, val r507 + rs604, test r508 + rs605.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import targets as T  # noqa: E402

DATA = Path(os.environ.get("RISK_DATA", "~/risk_data")).expanduser()
OUT = Path(__file__).resolve().parent
TRAIN = ["r501", "r502", "r503", "r504", "r505", "r506", "rs601", "rs602", "rs603"]
VAL = ["r507", "rs604"]
TEST = ["r508", "rs605"]
BAL = float(os.environ.get("BAL", "0.7"))
EPOCHS = int(os.environ.get("EPOCHS", "20"))
LAM_BINS = (1e-3, 1e-2, 0.1, 0.5)
LOG_EPS = 1e-6
SEED = int(os.environ.get("SEED", "0"))
torch.manual_seed(SEED)
np.random.seed(SEED)


def load(names):
    zs = [np.load(DATA / f"feat_{n}.npz") for n in names]
    X = np.concatenate([z["X"] for z in zs])
    risk = np.concatenate([z["risk"] for z in zs])
    meta = {k: np.concatenate([z[k] for z in zs]) for k in ("shot", "t")}
    meta["src"] = np.concatenate([np.full(len(z["X"]), n) for n, z in zip(names, zs)])
    return X, risk, meta


def groups(risk):
    _, share, lam = T.decompose(risk)
    dom = share.argmax(1)
    lev = np.digitize(lam, LAM_BINS)          # 0: <1e-3 (тихо), 1..4
    return np.where(lev == 0, 0, 1 + dom * 4 + (lev - 1)), lam


class Net(nn.Module):
    def __init__(self, n_in, h=256, p=0.1):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_in, h), nn.SiLU(), nn.Dropout(p),
                                  nn.Linear(h, h), nn.SiLU(), nn.Dropout(p),
                                  nn.Linear(h, h // 2), nn.SiLU(), nn.Linear(h // 2, 5))

    def forward(self, x):
        return self.body(x)                    # z = log C


def outputs(z):
    """z (log C) → share_pct, risk (5), risk_total — точно за формулою симулятора."""
    C = torch.exp(z)
    return 100 * torch.softmax(z, 1), 1 - torch.exp(-C), 1 - torch.exp(-C.sum(1))


def loss_fn(z, C_t):
    logC_t = torch.log(C_t + LOG_EPS)
    l_mag = F.smooth_l1_loss(z, logC_t)
    l_tot = F.smooth_l1_loss(torch.logsumexp(z, 1), torch.log(C_t.sum(1) + LOG_EPS))
    s_t = C_t / C_t.sum(1, keepdim=True).clamp_min(1e-30)
    kl = (s_t * (torch.log(s_t.clamp_min(1e-12)) - torch.log_softmax(z, 1))).sum(1)
    lam_t = 1 - torch.exp(-C_t.sum(1))
    return l_mag + l_tot + 2 * (lam_t * kl).mean()


def norm_stats(X):
    med = np.nanmedian(X, 0)
    q1, q3 = np.nanpercentile(X, [25, 75], 0)
    sc = q3 - q1
    sd = np.nanstd(X, 0)
    sc = np.where(sc > 1e-9 * (np.abs(med) + 1), sc, np.where(sd > 0, sd, 1.0))
    return med.astype(np.float32), sc.astype(np.float32)


def prep(X, med, sc):
    Z = (X - med) / sc
    return np.nan_to_num(np.clip(Z, -10, 10), nan=0.0).astype(np.float32)


def predict(net, Z, bs=8192):
    net.eval()
    with torch.no_grad():
        return np.concatenate([net(torch.from_numpy(Z[i:i + bs])).numpy() for i in range(0, len(Z), bs)])


def metrics(z, risk_t):
    """Якість на природному розподілі рядків (без балансу)."""
    C_t, share_t, lam_t = T.decompose(risk_t)
    C_p = np.exp(z)
    share_p = 100 * C_p / C_p.sum(1, keepdims=True)
    lam_p = 1 - np.exp(-C_p.sum(1))
    risk_p = 1 - np.exp(-C_p)
    r = {}
    r["lam_mae"] = np.abs(lam_p - lam_t).mean()
    r["lam_r2"] = 1 - ((lam_p - lam_t) ** 2).sum() / ((lam_t - lam_t.mean()) ** 2).sum()
    r["lam_logcorr"] = np.corrcoef(np.log(lam_p + 1e-6), np.log(lam_t + 1e-6))[0, 1]
    for k, m in enumerate(T.MECH):
        var = ((risk_t[:, k] - risk_t[:, k].mean()) ** 2).sum()
        # R² має сенс лише коли механізм справді змінюється (β у цих даних майже завжди ≈ 0)
        r[f"r2_{m}"] = 1 - ((risk_p[:, k] - risk_t[:, k]) ** 2).sum() / var if risk_t[:, k].max() > 1e-3 else np.nan
        r[f"mae_{m}"] = np.abs(risk_p[:, k] - risk_t[:, k]).mean()
    for thr in (0.01, 0.1):
        s = lam_t > thr
        r[f"share_mae_pp@{thr}"] = np.abs(share_p[s] - share_t[s]).sum(1).mean() / 2 if s.any() else np.nan
        r[f"dom_acc@{thr}"] = (share_p[s].argmax(1) == share_t[s].argmax(1)).mean() if s.any() else np.nan
    # зважена λ середня абсолютна помилка часток — головна метрика відбору (частки там, де ризик є)
    r["share_mae_lamw"] = (np.abs(share_p - share_t).sum(1) / 2 * lam_t).sum() / lam_t.sum()
    # тривога: чи відрізняє λ_pred рядки з λ_true > 0,1
    from sklearn.metrics import roc_auc_score
    r["auc_lam>0.1"] = roc_auc_score(lam_t > 0.1, lam_p)
    return r


def main():
    Xtr, Rtr, _ = load(TRAIN)
    Xva, Rva, _ = load(VAL)
    med, sc = norm_stats(Xtr)
    Ztr, Zva = prep(Xtr, med, sc), prep(Xva, med, sc)
    del Xtr, Xva
    g, lam = groups(Rtr)
    cnt = np.bincount(g, minlength=21)
    print(f"train: {len(Ztr)} рядків; групи (тихо / mech×рівень):", {int(k): int(v) for k, v in enumerate(cnt) if v})
    p = cnt[g].astype(float) ** (-BAL)
    p /= p.sum()
    eff = np.bincount(g, weights=p, minlength=21)
    print(f"BAL={BAL}: частка «тихо» у вибірці {eff[0]:.1%} (природно {cnt[0] / len(g):.1%})")

    C_tr = torch.from_numpy(T.decompose(Rtr)[0]).float()
    Ztr_t = torch.from_numpy(Ztr)
    net = Net(Ztr.shape[1])
    with torch.no_grad():
        net.body[-1].bias.copy_(torch.log(C_tr + LOG_EPS).mean(0))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    steps_per_epoch = len(Ztr) // 1024
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=EPOCHS * steps_per_epoch)
    best, best_state = np.inf, None
    for ep in range(EPOCHS):
        net.train()
        t0 = time.time()
        idx = np.random.choice(len(Ztr), size=steps_per_epoch * 1024, p=p)
        tot = 0.0
        for b in range(steps_per_epoch):
            j = torch.from_numpy(idx[b * 1024:(b + 1) * 1024])
            loss = loss_fn(net(Ztr_t[j]), C_tr[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        m = metrics(predict(net, Zva), Rva)
        score = m["share_mae_lamw"] + 100 * m["lam_mae"]
        print(f"епоха {ep + 1:2d}  loss {tot / steps_per_epoch:.4f} | val: λ MAE {m['lam_mae']:.4f}  R² {m['lam_r2']:.3f}"
              f"  частки (зваж. λ) {m['share_mae_lamw']:.2f} п.п.  домінуючий@0,1 {m['dom_acc@0.1']:.3f}"
              f"  [{time.time() - t0:.0f} с]", flush=True)
        if score < best:
            best, best_state = score, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    tag = os.environ.get("TAG", "")
    torch.save(dict(state=net.state_dict(), med=med, sc=sc, n_in=Ztr.shape[1], bal=BAL), OUT / f"risk_net{tag}.pt")
    res = {"val": metrics(predict(net, Zva), Rva)}
    del Ztr, Ztr_t
    Xte, Rte, mte = load(TEST)
    zte = predict(net, prep(Xte, med, sc))
    res["test"] = metrics(zte, Rte)
    np.savez(DATA / f"pred_test{tag}.npz", z=zte, risk=Rte, **mte)
    print(json.dumps({k: {m: round(float(v), 4) for m, v in d.items()} for k, d in res.items()}, ensure_ascii=False, indent=1))
    (OUT / f"metrics{tag}.json").write_text(json.dumps({k: {m: float(v) for m, v in d.items()} for k, d in res.items()}, indent=1))


if __name__ == "__main__":
    main()
