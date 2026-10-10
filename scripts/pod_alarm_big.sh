#!/bin/bash
# Моделі тривоги на 4500 розрядах: генерація → ознаки (clean/nocs і v8) → експорт → офіційний залік (паралельно).
cd /workspace/h
export PYTHONWARNINGS=ignore
BIN=$PWD/kit/participant/toksym-linux
mkdir -p datasets features_nocs_big features_v8_big runs_pod
# 1. дані: старі сиди (ті самі розряди, що на ноутбуці) + нові
for s in $(seq 101 130); do [ -f datasets/tr$s/shots.csv ] || $BIN gen --shots 100 --seed $s --out datasets/tr$s --extras transport > /dev/null 2>&1 & done
for s in $(seq 201 215); do [ -f datasets/trs$s/shots.csv ] || $BIN gen --shots 100 --seed $s --shift --out datasets/trs$s --extras transport > /dev/null 2>&1 & done
wait; echo "дані: $(ls datasets | wc -l) наборів"
# 2. ознаки (крок за кроком тим самим класом, що на заліку), по процесу на набір
for d in datasets/tr*; do n=$(basename $d)
  [ -f features_nocs_big/$n.parquet ] || echo "features_nocs $d features_nocs_big/$n.parquet"
  [ -f features_v8_big/$n.parquet ] || echo "features $d features_v8_big/$n.parquet"
done | xargs -P 100 -L 1 sh -c 'FEATURES=$0 python model/build_dataset.py $1 $2 > /dev/null 2>&1'
echo "ознаки: $(ls features_nocs_big | wc -l) clean, $(ls features_v8_big | wc -l) v8"
# 3. експорт
FEATURES=features_clean COMBINE=max HEAD_A_V2=1 FEAT_DIR=features_nocs_big python model/export.py submission_clean_big 0.1 4.5 0.3 2>&1 | grep -v -i warn
sed -i 's/Features(meta.get("scenario", ""))/Features()/' submission_clean_big/model.py
python - <<'PY'
import json
p = "submission_clean_big/toksym.json"; m = json.load(open(p, encoding="utf-8"))
m["name"] = "two-head HGB (чистая, 4500 разрядов)"
json.dump(m, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
PY
FEATURES=features HEAD_A_V2=1 FEAT_DIR=features_v8_big python model/export.py submission_v8_big 0.1 4.5 0.5 2>&1 | grep -v -i warn
rm -rf submission_*/__pycache__
# 4. офіційний залік: 4 моделі × 4 набори одночасно
score() { $BIN score --submission $1 --shots 200 --seed $2 --lead 0.03 --budget 0.05 --report runs_pod/$3.json --save-data runs_pod/$3_data ${4:-} > runs_pod/$3.log 2>&1; }
for m in submission_clean submission_clean_big submission_v8 submission_v8_big; do
  k=${m#submission_}
  score $m 1 ${k}_s1 & score $m 1 ${k}_shift_s1 --shift & score $m 20261001 ${k}_s20261001 & score $m 42 ${k}_s42 &
done
wait
for m in clean clean_big v8 v8_big; do echo "== $m"; grep -h -E "Спіймано|Хибні" runs_pod/${m}_s1.log runs_pod/${m}_shift_s1.log runs_pod/${m}_s20261001.log runs_pod/${m}_s42.log; done
python model/eval_pool.py clean: runs_pod/clean_{s1,shift_s1,s20261001,s42}_data -- clean_big: runs_pod/clean_big_{s1,shift_s1,s20261001,s42}_data \
  -- v8: runs_pod/v8_{s1,shift_s1,s20261001,s42}_data -- v8_big: runs_pod/v8_big_{s1,shift_s1,s20261001,s42}_data 2>&1 | tail -8
echo ALARM_DONE
