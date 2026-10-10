"""Розбір run.json з симулятора fusion-sim: чому «спіймано 25 %» і що з причинами зриву."""
import json
import sys

import numpy as np

d = json.load(open(sys.argv[1], encoding="utf-8"))
print("ключі:", list(d.keys()))
print("run:", d.get("run"), "\nsummary:", d.get("summary"), "\nrule:", d.get("rule"))
shots = d["shots"]
print("поля розряду:", list(shots[0].keys()))
lead = d.get("rule", {}).get("lead_time_s", 0.1)
thr = d["summary"]["threshold"]
dis, quiet = [], []
for s in shots:
    if s.get("spontaneous") or s.get("wall_contact"):
        continue
    st = np.array(s["staircase"]) if s.get("staircase") else np.zeros((1, 2))
    if s["disrupted"]:
        ok = st[:, 0] <= s["t_tq"] - lead + 1e-9
        lvl = st[ok, 1].max() if ok.any() else 0.0
        lvl30 = st[st[:, 0] <= s["t_tq"] - 0.03 + 1e-9, 1].max() if (st[:, 0] <= s["t_tq"] - 0.03 + 1e-9).any() else 0.0
        dis.append((s["shot_id"], s.get("scenario"), round(s["t_tq"], 3), s.get("verdict"), s.get("warning_s"), round(lvl, 3), round(lvl30, 3), round(st[:, 1].max(), 3)))
    else:
        quiet.append((s["shot_id"], s.get("scenario"), round(st[:, 1].max(), 3)))
q = np.array(sorted([x[2] for x in quiet], reverse=True))
print(f"\nтихих {len(quiet)}; найвищі max p тихих: {q[:6]}")
print(f"поріг {thr}: при бюджеті 5 % дозволено хибних = floor(0,05·{len(quiet)}) = {int(0.05 * len(quiet))}")
print("\nзриви: id, сценарій, t_tq, вердикт, попередження, рівень p до t_tq−0,1 с, до t_tq−30 мс, max p")
for x in dis:
    print("  ", x)


def tpr_at(levels, k_fp):
    th = q[k_fp] if k_fp < len(q) else 0.0           # поріг трохи вище (k_fp+1)-го тихого → k_fp хибних
    return np.mean([l > th for l in levels]), th


l01 = [x[5] for x in dis]
l30 = [x[6] for x in dis]
for k in (0, 1, 2, 3):
    t1, th = tpr_at(l01, k)
    t3, _ = tpr_at(l30, k)
    print(f"дозволено хибних {k} ({100 * k / len(quiet):.1f} %): поріг > {th:.3f} → вчасно при 0,1 с {t1:.0%}, при 30 мс {t3:.0%}")
auc = lambda L: float(np.mean([[1.0 if a > b else 0.5 if a == b else 0.0 for b in q] for a in L]))
print(f"\nROC AUC (рівень розрядів): 0,1 с {auc(l01):.3f}; 30 мс {auc(l30):.3f}")
# причини, якщо є
for key in ("causes", "cause", "impact", "shares", "factors"):
    if key in d:
        print(f"\n{key}:", json.dumps(d[key], ensure_ascii=False)[:1500])
ex = [s for s in shots if s["disrupted"]][:1]
for s in ex:
    print("\nприклад розряду (без staircase):", {k: v for k, v in s.items() if k != "staircase"})
