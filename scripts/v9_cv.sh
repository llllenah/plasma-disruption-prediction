#!/bin/bash
# CV v8 (бейзлайн) проти v9 (v8 + нейромережа вкладів механізмів) в однакових умовах.
# Чекає, поки закінчиться навчання ImpactNet (інакше не вистачить пам'яті).
cd "$(dirname "$0")/.."
export PATH="$HOME/venv/bin:$PATH"
while pgrep -f "impact_train.sh|train_impact.py|impact_numpy.py|risk_model/evaluate.py" >/dev/null; do sleep 30; done
for v in v8 v9; do
  F=features; [ $v = v9 ] && F=features_v9
  echo "=========== CV $v ==========="
  FEATURES=$F FEAT_DIR=features_$v HEAD_A_V2=1 SUBSAMPLE=2 EVAL_ONLY='tr10[1-6]|trs20[1-3]' TAG=_cmp \
    python -W ignore model/train.py cv2 0.1 4.5 2>&1 | grep -v -i warn | grep -A3 "голова\|0.5·B\|Killed\|Error"
done
