#!/bin/bash
# Survival-експеримент: ImpactNet без міток λ (лише моменти зривів), три варіанти паралельно на GPU.
cd /workspace/h
export RISK_DATA=/workspace/h/risk_data PYTHONWARNINGS=ignore
run() { local tag=$1; shift
  ( env "$@" TAG=$tag EPOCHS=30 python risk_model/train_impact.py surv > logs/impact$tag.log 2>&1
    python risk_model/evaluate.py _impact$tag > logs/eval$tag.log 2>&1 ) & }
run _surv        LOSS=surv
run _surv_cause  LOSS=surv_cause
run _surv_noprior_cause LOSS=surv_cause PRIOR=0
wait
echo SURV_DONE
