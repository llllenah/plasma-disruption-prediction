# Real-time Plasma Disruption Prediction & Cause Attribution

> **FMF KPI Hackathon, 9–11 Oct 2026 · tokamak simulator `toksym` (JET-like)**
>
> 🇺🇦 **Коротко:** модель тривоги без формул шпаргалки і без підказок симулятора піднімає об'єм заліку з **0,765 (бейзлайн) до 0,924** і ловить **~98 % зривів за 30 мс** при ≤ 5 % хибних тривог, крок 0,9 мс. Окрема нейромережа **ImpactNet** на кожному кроці пояснює, *чому* плазма зривається: частка Грінвальда, q95, радіації, Тройона та інших механізмів. Причину вгадано у **49 з 49** зривів.

---

## TL;DR

| | Organizers' baseline | **Our alarm model** (`submission/`) | Our physics model (reference) |
|---|---|---|---|
| Score volume (mean AUC over leads × FPR) | 0.765 | **0.924** [0.912–0.937] | 0.944 |
| Caught in time @ 30 ms lead, 5 % false alarms | — | **98.2 %** (224 / 228) | 95.2 % |
| Step latency (laptop, p50 / p99) | — | **0.9 / 3.3 ms** | 1.2 / 4.3 ms |
| Cheat-sheet physics formulas | yes | **no** | yes |
| Simulator hints (scenario label) | — | **no** | yes |

Official `toksym score`, 4 held-out sets (seeds 1, 1 `--shift`, 20261001, 42; 200 discharges each, 747 total).

**ImpactNet** (cause attribution, `impactnet/`): R² of total risk λ **0.996** (normal) / **0.966** (shift), dominant mechanism correct in **49 / 49** disruptions at the trigger moment, 0.6 ms per step in pure numpy.

---

## The problem

A **disruption** is a sudden loss of plasma confinement: the most damaging event in a tokamak. Every 5 ms the model receives **100 noisy diagnostic channels** (magnetics, interferometry, ECE, bolometry, SXR, gas, heating…) and must output a disruption probability using **only the past**. The simulator sweeps the alarm threshold to a false-alarm budget and counts disruptions caught at least *lead time* before the thermal quench. Lead time and budget are chosen by the team, so we optimise the **volume**: the average quality over all leads (10 ms … 1 s) and all budgets.

## Hypotheses we tested

| # | Hypothesis | Verdict |
|---|---|---|
| H1 | Two complementary detectors (late-but-sure *precursor* + *early warning*) beat one model | ✅ +0.15 volume over baseline |
| H2 | Textbook limit formulas (f_GW, q95, β_N, P_rad/P_in) are not needed: raw smoothed channels suffice | ✅ 0.924 without them (vs 0.944 with) |
| H3 | The scenario label in metadata is a *shortcut* (`none` ⇒ no scored disruption), not plasma physics | ✅ removing it costs ≈ 0.01, long-lead TPR even improves |
| H4 | Knowing the exact hazard λ is enough to raise alarms | ❌ even the **true** λ gives volume 0.78: a disruption is a random draw. Hence two models: *when* vs *why* |
| H5 | A physics prior makes mechanisms identifiable without risk labels | ✅ trained on disruption times only: 49/49 causes with the prior, 41/49 without |

## Solution 1: alarm model (`submission/`, `model/`)

```
100 channels ──► causal online features (EMA 50 ms / 250 ms / 1 s, trends, plateau maxima,
                 locked-mode & Mirnov precursor detector, actuator commands)       — no cheat-sheet formulas
            ├──► Head A  HistGradientBoosting: "thermal quench within 0.1 s"   (precursor)
            └──► Head B  HistGradientBoosting: "thermal quench within 4.5 s"   (early warning)
                 p = max(A, 0.3·B) → 0 during programmed ramp-down → EMA → running maximum
```
* Trees are flattened to numpy arrays; inference needs **only numpy**.
* Trained on **4 500 generated discharges**, split by seed; labels come only from `shots.csv`.
* Robust to any judge settings (fraction of disruptions caught in time, 4 sets):

| lead \ false alarms | 3 % | 5 % | 10 % | 15 % |
|---|---|---|---|---|
| 10 ms | 99.1 % | 99.1 % | 99.6 % | 99.6 % |
| 30 ms | 97.8 % | 98.2 % | 99.6 % | 99.6 % |
| 0.1 s | 39.9 % | 50.9 % | 71.9 % | 83.3 % |
| 0.3 s | 36.0 % | 47.8 % | 66.7 % | 81.6 % |
| 1 s | 22.4 % | 29.8 % | 47.4 % | 61.0 % |

## Solution 2: ImpactNet, *why* the plasma disrupts (`impactnet/`, `risk_model/`)

Exactly **5 output neurons**: Greenwald, q95 (kink), radiation, Troyon, other (locked mode). Each neuron is the simulator's own risk form `σ((x − c)/w)` + a learned correction, so

* `share_pct` = 100·Cₖ/ΣC with Cₖ = −log(1 − rₖᵖ): sums to exactly 100 %;
* `risk_total` λ = 1 − e^(−ΣC) = 1 − Π(1 − rₖᵖ): identical to the simulator's formula;
* `p_disrupt(h)` = 1 − e^(−λh).

Architecture: 218 inputs (100 channels + 100 dropout masks + 18 physics quantities) → MLP → causal dilated convolutions (155 ms) → GRU(128) → 5 neurons + physics prior. Labels are the true mechanism risks computed from the simulator's hidden state (`--extras`); the hidden state is **never** an input. Architecture study: GRU ≈ LSTM ≈ TCN > no memory; GRU chosen for O(1) streaming state. See [`impactnet/README.md`](impactnet/README.md).

## Closed-loop simulator ([lurbaby/plasma_simulator](https://github.com/lurbaby/plasma_simulator))

A teammate's fork of fusionsimulator.io (Rust/WASM engine by Daniel Burgess and the Columbia Fusion Research Center, GPL-3.0). The team added the organisers' `toksym` workflow (dataset, test run, CAUGHT / FALSE ALARMS report, CSV downloads) and live actuator control from a model, so the alarm model + ImpactNet + an advisor run in a closed loop.

* **Closed loop (v8 → ImpactNet → fixer student, 100 new discharges, seed 2000, same in every mode):** preventable (non-spontaneous) disruptions **31 → 5**, as many as the oracle teacher that sees the hidden state (rule-based advisor: 21). Stored energy unchanged (W 2.371 → 2.374 MJ). No discharge made worse: 26 saved, 0 broken (McNemar p = 3·10⁻⁸). The student picks one of 8 actions every 0.25 s from instruments only (ImpactNet outputs, 16 raw channels, current setpoint multipliers); the teacher runs 1 s rollouts on copies of the engine. Details: `docs/FIXER.md`, `docs/MODELS.md` in the simulator repo.
* Alarm model transfers to this noise-free engine **without retraining**: ROC AUC 0.902 at 0.1 s (0.934 on `toksym`), 0.970 at 30 ms (200 discharges, `results/fusionsim_v8_impact_200.json`).
* Cause attribution: the simulator reported 34 / 49 (kink 4 / 16). We found it took the cause at the model's maximum λ over the *whole* shot, which falls on the network warm-up at t = 5 ms, before there is plasma (default answer "Greenwald 97.6 %"). Restricting the search to steps with plasma (Ip above the simulator threshold, before the quench) gives **45 / 49** (Greenwald 16 / 16, kink 15 / 16, radiation 14 / 17) from the same online outputs in `measured.csv`, with no retraining (`scripts/fs_moment.py`, `results/fusionsim_attribution_moment.csv`).

## Validation & honesty

* Features are computed step by step from the past only: the same class runs in training and in scoring.
* Every model is compared on the **same 747 held-out discharges** with a **paired bootstrap**. The adoption rule (P > 0.9) was fixed in advance.
* Trained on one machine, re-scored on another: identical results (the simulator is deterministic).
* Removed on purpose: cheat-sheet formulas, scenario metadata, timers from program events.

## Repository layout

```
submission/           final alarm model (drop into toksym "Test run" or `toksym score --submission submission`)
submissions/v8_physics reference model with physics formulas (0.944)
model/                features, training, export, metric, paired bootstrap (alarm model)
impactnet/            ImpactNet inference package (numpy only) + README + simulator check script
risk_model/           ImpactNet training, labels, physics preprocessing, evaluation, alarm/oracle study
analysis/             simulator hazard formula, oracle & trigger-law studies
scripts/              data generation, official scoring, RunPod pipelines, benchmarks
results/              official score reports (summary CSV + gzipped full reports), ImpactNet evaluations
docs/                 defense presentation (presentation.pptx, 15 sections)
```

## Reproduce

```bash
# score the final model (simulator binary from the organizers' kit, not included)
toksym score --submission submission --shots 200 --seed 42 --lead 0.03 --budget 0.05
# ImpactNet on any measured.csv
python impactnet/impactnet.py measured.csv impact_out.csv
```
Data is not stored in the repo: every dataset is regenerated from its seed (`toksym gen --seed S`), see `scripts/`.

## Team
FMF KPI Hackathon 2026.
