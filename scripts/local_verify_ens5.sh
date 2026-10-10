#!/bin/bash
# Офіційний залік фінальної моделі на НОУТБУЦІ (сид 42) — має збігтися з серверним прогоном.
cd "$(dirname "$0")/.."
export PATH="$HOME/venv/bin:$PATH"
mkdir -p ~/runs runs
kit/participant/toksym-linux score --submission submission_clean_9k_ens5 --shots 200 --seed 42 --lead 0.03 --budget 0.05 \
  --report ~/runs/ens5_local_s42.json > ~/runs/ens5_local_s42.log 2>&1
cp ~/runs/ens5_local_s42.json ~/runs/ens5_local_s42.log runs/
echo "--- ноутбук:"; grep -E "Спіймано|Пізні|Хибні|Крок|вклалися" ~/runs/ens5_local_s42.log
echo "--- сервер (той самий сид 42):"; grep -E "Спіймано|Пізні|Хибні|Крок|вклалися" pod_results/runs_pod2/clean_9k_ens5_s42.log
