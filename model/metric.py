"""Офлайн-метрика за правилом заліку.

Тривога — перший момент, коли p ≥ поріг. Тихий розряд з тривогою — хибна.
Зрив спіймано вчасно, якщо тривога була не пізніше t_Q − lead. Спонтанні й
контакт зі стінкою виключено.

Об'єм: для кожного lead з LEADS — площа під кривою TPR(FPR) на FPR ∈ [0, 1],
далі середнє по lead. Це наш замінник «об'єму під поверхнею» з методички;
точної формули організатор не дав, тож дивимося також на TPR при FPR 5 %.
"""
import numpy as np

LEADS = (0.01, 0.03, 0.1, 0.3, 1.0)


def shot_summaries(shots):
    """shots: iterable of dict(t=array, p=array, t_tq=float|inf, quiet=bool).

    Повертає для кожного розряду max p (для тихих) або функцію «найбільший
    поріг, за якого тривога прийшла не пізніше t_Q − lead» (для зривів).
    """
    quiet_max, dis = [], []
    for s in shots:
        t, p = np.asarray(s["t"]), np.asarray(s["p"], dtype=float)
        p = np.nan_to_num(p, nan=0.0).clip(0, 1)
        if s["quiet"]:
            quiet_max.append(p.max() if len(p) else 0.0)
        else:
            run = np.maximum.accumulate(p) if len(p) else p
            dis.append((t, run, s["t_tq"]))
    return np.array(quiet_max), dis


def caught_level(dis, lead):
    """Для кожного зриву — максимум p до моменту t_Q − lead (вчасна тривога ⇔ поріг ≤ цього)."""
    out = []
    for t, run, tq in dis:
        # Як у заліку: запас (t_Q − t) порівнюється з lead без допуску, тож
        # тривога «рівно за lead» через похибку float часто стає пізньою.
        m = (tq - t) >= lead
        out.append(run[m][-1] if m.any() else 0.0)
    return np.array(out)


def tpr_at_fpr(quiet_max, lvl, fpr):
    # поріг — найменший, за якого частка тихих з max ≥ поріг не більша за fpr
    q = np.sort(quiet_max)[::-1]
    k = int(np.floor(fpr * len(q)))
    thr = (q[k] + 1e-9) if k < len(q) else 0.0
    return float((lvl >= thr).mean()), float(thr)


def roc_auc(quiet_max, lvl):
    # P(рівень зриву > max тихого) з половинкою за рівність
    q = np.sort(quiet_max)
    lt = np.searchsorted(q, lvl, side="left")
    le = np.searchsorted(q, lvl, side="right")
    return float(((lt + le) / 2).mean() / len(q))


def report(shots, budget=0.05):
    quiet_max, dis = shot_summaries(shots)
    res = {}
    for lead in LEADS:
        lvl = caught_level(dis, lead)
        tpr, thr = tpr_at_fpr(quiet_max, lvl, budget)
        res[lead] = dict(auc=roc_auc(quiet_max, lvl), tpr_at_budget=tpr, thr=thr)
    res["volume"] = float(np.mean([res[l]["auc"] for l in LEADS]))
    res["n_quiet"], res["n_dis"] = len(quiet_max), len(dis)
    return res


def fmt(res):
    lines = [f"volume (mean AUC over leads) = {res['volume']:.3f}   quiet={res['n_quiet']} dis={res['n_dis']}"]
    for lead in LEADS:
        r = res[lead]
        lines.append(f"  lead {lead:5.2f}s  AUC={r['auc']:.3f}  TPR@5%={r['tpr_at_budget']:.3f}")
    return "\n".join(lines)
