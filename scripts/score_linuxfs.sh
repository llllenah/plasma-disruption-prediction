#!/bin/bash
# Залік з даними на диску Linux (~/runs) — без гальм /mnt/c. Звіт і лог копіюються в runs/.
# bash scripts/score_linuxfs.sh <тека здачі> <seed> <name> [ключі toksym...]
cd "$(dirname "$0")/.."
export PATH="$HOME/venv/bin:$PATH"
M=$1; S=$2; N=$3; shift 3
mkdir -p ~/runs runs
kit/participant/toksym-linux score --submission "$M" --shots 200 --seed "$S" --lead 0.03 --budget 0.05 \
  --report ~/runs/$N.json --save-data ~/runs/${N}_data "$@" > ~/runs/$N.log 2>&1
echo "== $N exit=$?"
cp ~/runs/$N.json ~/runs/$N.log runs/ 2>/dev/null
mkdir -p runs/${N}_data && cp ~/runs/${N}_data/*.csv runs/${N}_data/ 2>/dev/null
grep -E "Спіймано|Пізні|Хибні|Крок|вклалися|не відповіла" ~/runs/$N.log
