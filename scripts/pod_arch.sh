#!/bin/bash
# Порівняння архітектур ImpactNet на однакових даних (2500 розрядів), сид 0. GRU = вже навчена big_s0.
cd /workspace/h
export RISK_DATA=/workspace/h/risk_data PYTHONWARNINGS=ignore
X="r509 r510 r511 r512 r513 r514 r515 r516 rs606 rs607 rs608 rs609 rs610 rs611 rs612 rs613"
for a in lstm tcn mlp; do
  ( ARCH=$a TRAIN_EXTRA="$X" EPOCHS=40 STEPS=300 SEED=0 TAG=_arch_$a python risk_model/train_impact.py train > logs/impact_arch_$a.log 2>&1
    python risk_model/evaluate.py _impact_arch_$a > logs/eval_arch_$a.log 2>&1 ) &
done
wait
echo ARCH_DONE
