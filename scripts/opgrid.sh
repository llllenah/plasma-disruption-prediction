#!/bin/bash
# Робочі точки submission_clean_big: % спійманих вчасно (разом 4 набори) для запасу × бюджету.
cd "$(dirname "$0")/.."
export PATH="$HOME/venv/bin:$PATH"
printf "%-8s" "запас"; for b in 0.03 0.05 0.10 0.15; do printf "  бюджет %-5s" $b; done; echo
for lead in 0.01 0.03 0.1 0.3 1.0; do
  printf "%-8s" "${lead}с"
  for b in 0.03 0.05 0.10 0.15; do
    tot=0; n=0
    for s in s1 shift_s1 s20261001 s42; do
      read tp nd <<< $(kit/participant/toksym-linux rescore pod_results/runs_pod/clean_big_$s.json --lead $lead --budget $b 2>/dev/null \
        | python -c "import json,sys; d=json.load(sys.stdin)['summary']; print(d['tp'], d['n_disruptions_scored'])")
      tot=$((tot+tp)); n=$((n+nd))
    done
    printf "  %11s" "$(python -c "print(f'{100*$tot/$n:.1f} %')")"
  done; echo
done
