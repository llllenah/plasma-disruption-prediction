#!/bin/bash
# «Чистая» здача: без формул шпаргалки, без сценария из meta, без таймеров программы.
# Сборка → проверка совпадения признаков с features_nocs → официальный зачёт на 4 наборах по очереди.
cd "$(dirname "$0")/.."
export PATH="$HOME/venv/bin:$PATH"
S=submission_clean
while pgrep -f "model/train.py" >/dev/null; do sleep 30; done

FEATURES=features_clean COMBINE=max HEAD_A_V2=1 FEAT_DIR=features_nocs \
  python -W ignore model/export.py $S 0.1 4.5 0.3 2>&1 | grep -v -i warn
rm -rf $S/__pycache__
# модель не читает сценарий из meta
sed -i 's/Features(meta.get("scenario", ""))/Features()/' $S/model.py
sed -i 's/по фазе программы,$/по трендам,/; s/^сценарию и сигналам приборов/уставкам и сигналам приборов/' $S/model.py
python - <<'PY'
import json
p = "submission_clean/toksym.json"; m = json.load(open(p, encoding="utf-8"))
m["name"] = "two-head HGB (чистая: без шпаргалки, без сценария)"
m["notes"] = ("Голова A: предвестник срыва (H=0.1 с). Голова B: ранняя тревога (H=4.5 с). p = max(A, 0.3·B). "
              "Признаки — сглаженные сырые каналы приборов, уставки и команды актуаторов, Dα, Мирнов, SXR, ECE, "
              "детектор предвестника по седловой петле. Сценарий из meta не используется; на спаде тока p = 0.")
json.dump(m, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
c = json.load(open("submission_clean/config.json"))
bad = [x for x in c["cols_a"] + c["cols_b"] if x.startswith("scen_") or x in ("t_flat", "t_neon_on", "t_heat_on", "t_ne_flat")]
print("combine:", c["combine"], "| w:", c["w"], "| A:", len(c["cols_a"]), "| B:", len(c["cols_b"]), "| запрещённых колонок:", len(bad))
PY
echo "--- упоминания scenario / формул шпаргалки в коде здачи (должно быть 0):"
grep -c -E "scenario|scen_|f_gw|q95|beta|rad_frac|p_in|lam_|KAPPA|A_MINOR" $S/features.py $S/model.py
# признаки чистой версии == соответствующие признаки nocs на реальном разряде
python - <<'PY'
import sys, csv, math
sys.path.insert(0, "model")
import features_nocs as N, features_clean as C
rows = list(csv.DictReader(open("datasets/tr106/measured.csv")))
sid = rows[0]["shot_id"]; rows = [r for r in rows if r["shot_id"] == sid]
fn, fc = N.Features("density"), C.Features()
nn, cn = N.feature_names(), C.feature_names(); idx = [nn.index(c) for c in cn]
worst = 0.0
for r in rows:
    ch = {k: (float(v) if v not in ("", "nan") else float("nan")) for k, v in r.items() if k not in ("shot_id", "t")}
    t = float(r["t"]); a = fn.step(t, ch); b = fc.step(t, ch)
    for i, j in enumerate(idx):
        x, y = a[j], b[i]
        if not (x != x and y != y): worst = max(worst, abs(x - y))
print(f"признаков: {len(cn)}, шагов: {len(rows)}, макс. расхождение с nocs: {worst:.1e}")
PY
kit/participant/toksym-linux check $S | head -6

bash scripts/score_linuxfs.sh $S 1 clean_s1
bash scripts/score_linuxfs.sh $S 1 clean_shift_s1 --shift
bash scripts/score_linuxfs.sh $S 20261001 clean_s20261001
bash scripts/score_linuxfs.sh $S 42 clean_s42
python model/eval_pool.py nocs: runs/nocs_s1_data runs/nocs_shift_s1_data runs/nocs_s20261001_data runs/nocs_s42_data -- clean: runs/clean_s1_data runs/clean_shift_s1_data runs/clean_s20261001_data runs/clean_s42_data 2>&1 | tail -20
