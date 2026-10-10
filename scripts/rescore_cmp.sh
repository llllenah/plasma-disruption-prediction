#!/bin/bash
# clean vs clean_big на різних запасах: tp/n по кожному набору + разом (перерахунок звітів, без прогону).
cd "$(dirname "$0")/.."
export PATH="$HOME/venv/bin:$PATH"
for lead in 0.03 0.1 0.3 1.0; do
  echo "=== запас ${lead} с, бюджет 5 %"
  for m in clean clean_big; do
    tot=0; n=0; line=""
    for s in s1 shift_s1 s20261001 s42; do
      read tp nd <<< $(kit/participant/toksym-linux rescore pod_results/runs_pod/${m}_$s.json --lead $lead --budget 0.05 2>/dev/null \
        | python -c "import json,sys; d=json.load(sys.stdin)['summary']; print(d['tp'], d['n_disruptions_scored'])")
      line="$line  $s $tp/$nd"; tot=$((tot+tp)); n=$((n+nd))
    done
    printf "%-10s %s | разом %d/%d = %.1f %%\n" $m "$line" $tot $n $(python -c "print(100*$tot/$n)")
  done
done
