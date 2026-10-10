#!/bin/bash
# ImpactNet на 2500 розрядах (900 + 1600 нових), два сиди. Оцінка — на тих самих val/test, що й раніше.
cd /workspace/h
export RISK_DATA=/workspace/h/risk_data PYTHONWARNINGS=ignore
X="r509 r510 r511 r512 r513 r514 r515 r516 rs606 rs607 rs608 rs609 rs610 rs611 rs612 rs613"
for s in 0 1; do
  ( TRAIN_EXTRA="$X" EPOCHS=40 STEPS=300 SEED=$s TAG=_big_s$s python risk_model/train_impact.py train > logs/impact_big_s$s.log 2>&1
    python risk_model/evaluate.py _impact_big_s$s > logs/eval_big_s$s.log 2>&1 ) &
done
wait
echo BIG_DONE
