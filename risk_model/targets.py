"""Розмітка: справжні ризики п'яти механізмів зриву на кожному кроці.

Рахуються з ІСТИННОГО стану плазми (колонки `gen --extras transport,equilibrium`) за формулою
симулятора з методички (розд. 4.6), константи відкалібровані на jet_dd_s1, де симулятор віддає
і стан, і свій disruption_risk (analysis/calibrate_physics.py):

    rᵢ = σ((xᵢ − cᵢ)/wᵢ),  i ∈ {густина f_GW, β_N/2,8, кінк 2,2/q95, випромінювання P_rad/P_in}
    r_lm = 5·10⁻⁴ (1 + 5 r_gw + 3 r_β + 2 r_q)                         (locked mode)
    1 − λ = (1 − r_gw³)(1 − r_β³)(1 − r_q³)(1 − r_rad³)(1 − r_lm)

Вклад механізму Cₖ = −log(1 − rₖᵖ) (p = 3 для чотирьох меж, 1 для locked mode), тоді
ΣC = −log(1 − λ) точно, а частка = Cₖ / ΣC.
Ці колонки — ЛИШЕ мітки: на вхід мережі вони не потрапляють.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import physics as ph  # noqa: E402

Q95_COEF = json.loads((ROOT / "analysis" / "physics_coef.json").read_text())["q95_coef"]
MECH = ("density", "beta", "kink", "radiation", "locked_mode")
TRUTH_COLS = ["ip_MA", "bt_T", "ne_bar_1e20", "w_th_MJ", "p_rad_MW", "p_input_MW",
              "axis_r_m", "axis_z_m", "xpoint_r_m", "xpoint_z_m"]
EPS = 1e-7


def mechanism_risks(g, dt=0.005):
    """g — DataFrame одного розряду з істинними колонками. → масив (n, 5) ризиків rₖᵖ."""
    ip = g.ip_MA.clip(lower=0.05)
    f_gw = ph.ema(g.ne_bar_1e20 / (ip / ph.GW_AREA), dt)
    bn = ph.ema(ph.BETA_C * g.w_th_MJ / (ip * g.bt_T) / 2.8, dt)
    p_in = g.p_input_MW.to_numpy()
    frad = ph.ema(np.where(p_in > 1.0, g.p_rad_MW.to_numpy() / np.clip(p_in, 1e-6, None), 0.0), dt)
    q95 = ph.q95_from_shape(g.bt_T, g.ip_MA, g.xpoint_r_m, g.xpoint_z_m, g.axis_r_m, g.axis_z_m, Q95_COEF)
    kq = 2.2 / np.clip(q95, 0.5, None)
    L = ph.LIMITS
    r1 = {"density": ph.sig(f_gw, *L["density"]), "beta": ph.sig(bn, *L["beta"]),
          "kink": ph.sig(kq, *L["kink"]), "radiation": ph.sig(frad, *L["radiation"])}
    r_lm = 5e-4 * (1 + 5 * r1["density"] + 3 * r1["beta"] + 2 * r1["kink"])
    risk = np.c_[r1["density"] ** 3, r1["beta"] ** 3, r1["kink"] ** 3, r1["radiation"] ** 3, r_lm]
    return np.clip(np.nan_to_num(risk, nan=0.0), 0.0, 1 - EPS)


def decompose(risk):
    """rₖᵖ (n, 5) → вклади C, частки у % (сума рівно 100), загальний ризик λ = 1 − e^(−ΣC)."""
    C = -np.log1p(-risk)
    tot = C.sum(1, keepdims=True)
    share = 100.0 * C / np.maximum(tot, 1e-30)
    return C, share, 1.0 - np.exp(-tot[:, 0])
