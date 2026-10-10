#!/bin/bash
# Данные для модели вкладов механизмов: измерения + истинное состояние плазмы (только для разметки).
# Хранятся на диске Linux (~/risk_data) — быстро и не засоряют /mnt/c.
cd "$(dirname "$0")/.."
BIN=$PWD/kit/participant/toksym-linux
mkdir -p ~/risk_data
gen() {  # имя сид [ключи]
  local n=$1 s=$2; shift 2
  [ -f ~/risk_data/$n/shots.csv ] && { echo "$n уже есть"; return; }
  "$BIN" gen --shots 100 --seed $s --out ~/risk_data/$n --extras transport,equilibrium "$@" 2>&1 | tail -1
}
for s in 501 502 503 504 505 506 507 508; do gen r$s $s; done           # обычный режим
for s in 601 602 603 604 605; do gen rs$s $s --shift; done              # shift
du -sh ~/risk_data
