#!/bin/bash
# Под №2 (CPU): 9000 розрядів → ознаки clean → clean_9k (одна модель) і clean_9k_ens5 (бегінг) →
# офіційний залік разом із поточною clean_big → парний бутстреп. Лог етапів — logs/pod2.txt.
cd /workspace/h
export PYTHONWARNINGS=ignore
# системний python3 тут 3.8 без бібліотек → python/python3 = 3.13 (для нашого коду і для здач у toksym score)
mkdir -p /opt/py/bin && ln -sf /usr/bin/python3.13 /opt/py/bin/python && ln -sf /usr/bin/python3.13 /opt/py/bin/python3
export PATH=/opt/py/bin:$PATH
BIN=$PWD/kit/participant/toksym-linux; chmod +x $BIN
mkdir -p datasets features_nocs_9k runs logs
NP=$(nproc)
# 1. дані (без додаткових колонок — моделі тривоги вони не потрібні)
{ for s in $(seq 101 160); do echo "tr$s $s"; done; for s in $(seq 201 230); do echo "trs$s $s --shift"; done; } |
  xargs -P $NP -L 1 sh -c '[ -f datasets/$0/shots.csv ] || '"$BIN"' gen --shots 100 --seed $1 --out datasets/$0 $2 > /dev/null 2>&1'
echo "дані: $(ls datasets | wc -l) наборів"
# 2. ознаки (той самий клас, що на заліку) → сирі measured.csv видаляємо, щоб не забити диск
ls datasets | xargs -P $NP -I{} sh -c '[ -f features_nocs_9k/{}.parquet ] || { FEATURES=features_nocs python model/build_dataset.py datasets/{} features_nocs_9k/{}.parquet > /dev/null 2>&1 && rm -f datasets/{}/measured.csv; }'
echo "ознаки: $(ls features_nocs_9k | wc -l)"
# 3. моделі
post() {   # як у чистій здачі: сценарій з meta не читається
  sed -i 's/Features(meta.get("scenario", ""))/Features()/' $1/model.py; rm -rf $1/__pycache__
  grep -c -E "scenario|scen_|f_gw|q95|beta|rad_frac|p_in|lam_" $1/features.py $1/model.py; }
FEATURES=features_clean COMBINE=max HEAD_A_V2=1 FEAT_DIR=features_nocs_9k python model/export.py submission_clean_9k 0.1 4.5 0.3 2>&1 | grep -v -i warn
python - <<'PY'
import json
m = json.load(open("submission_clean/toksym.json", encoding="utf-8")); m["name"] = "two-head HGB (чиста, 9000 розрядів)"
json.dump(m, open("submission_clean_9k/toksym.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
PY
post submission_clean_9k
FEATURES=features_clean COMBINE=max HEAD_A_V2=1 FEAT_DIR=features_nocs_9k ENS=5 NAME="two-head HGB (чиста, 9000 розрядів, ансамбль 5)" \
  python model/export_ens.py submission_clean_9k_ens5 0.1 4.5 0.3 2>&1 | grep -v -i warn
post submission_clean_9k_ens5
# 4. офіційний залік: 3 моделі × 4 набори одночасно
score() { $BIN score --submission $1 --shots 200 --seed $2 --lead 0.03 --budget 0.05 --report runs/$3.json --save-data runs/$3_data ${4:-} > runs/$3.log 2>&1; }
for m in submission_clean_big submission_clean_9k submission_clean_9k_ens5; do k=${m#submission_}
  score $m 1 ${k}_s1 & score $m 1 ${k}_shift_s1 --shift & score $m 20261001 ${k}_s20261001 & score $m 42 ${k}_s42 &
done
wait
for k in clean_big clean_9k clean_9k_ens5; do echo "== $k"; grep -h "Спіймано" runs/${k}_{s1,shift_s1,s20261001,s42}.log; done
python model/eval_pool.py clean_big: runs/clean_big_{s1,shift_s1,s20261001,s42}_data -- clean_9k: runs/clean_9k_{s1,shift_s1,s20261001,s42}_data \
  -- clean_9k_ens5: runs/clean_9k_ens5_{s1,shift_s1,s20261001,s42}_data 2>&1 | tail -5
for k in clean_9k clean_9k_ens5; do echo "== $k − clean_big"; python model/paired_boot.py $k clean_big s1 shift_s1 s20261001 s42 2>&1 | tail -7; done
echo POD2_DONE
