"""Физика срыва из методички (разд. 4.6) — общий модуль для build_shots_v3.

Всё считается по одному разряду (DataFrame одного shot_id, по времени).
Константы откалиброваны на jet_dd_s1, где есть и входы, и истинный риск
(analysis/calibrate_physics.py).
"""
import numpy as np
import pandas as pd

A_MINOR = 0.95                     # малый радиус JET, м
GW_AREA = np.pi * A_MINOR ** 2     # n_GW = Ip / (π a²)
BETA_C = 1.5917                    # β_N = BETA_C · W_th / (Ip · Bt), калибровка
TAU = 0.10                         # входы сигмоид сглажены 100 мс (q95 — нет)
IP_MIN = 0.24                      # ниже этого тока риск не считается

R_GEO = 2.96                       # большой радиус JET, м
# q95 = (Bt/Ip) · exp(c0 + c1·κ² + c2·δ): учебная зависимость q ∝ Bt/Ip × форма.
# κ = (z_оси − z_X)/a, δ = (R_geo − R_X)/a. Коэффициенты — calibrate_physics.py.

LIMITS = {                         # (центр, ширина) сигмоид
    "density": (1.00, 0.08),       # f_GW
    "beta": (0.85, 0.06),          # β_N / 2.8
    "kink": (0.90, 0.05),          # 2.2 / q95
    "radiation": (0.80, 0.08),     # P_rad / P_in
}


def sig(x, c, w):
    return 1.0 / (1.0 + np.exp(-(x - c) / w))


def ema(x, dt, tau=TAU):
    a = 1.0 - np.exp(-dt / tau)
    return pd.Series(np.asarray(x, dtype=float)).ewm(alpha=a, adjust=False).mean().to_numpy()


def shape(xr, xz, az):
    """Вытянутость κ и треугольность δ по X-точке и оси."""
    kappa = (np.asarray(az) - np.asarray(xz)) / A_MINOR
    delta = (R_GEO - np.asarray(xr)) / A_MINOR
    return kappa, delta


def q95_from_shape(bt, ip, xr, xz, ar, az, coef):
    kappa, delta = shape(xr, xz, az)
    c0, c1, c2 = coef
    return (np.asarray(bt) / np.clip(ip, 0.05, None)) * np.exp(c0 + c1 * kappa ** 2 + c2 * delta)


def hazard(f_gw, beta_n, q95, p_rad, p_in, ip, dt):
    """λ(t) и r_i³ по формуле симулятора. Все аргументы — массивы одного разряда."""
    fgw_s = ema(f_gw, dt)
    bn_s = ema(np.asarray(beta_n) / 2.8, dt)
    frad = np.where(np.asarray(p_in) > 1.0, np.asarray(p_rad) / np.clip(p_in, 1e-6, None), 0.0)
    frad_s = ema(frad, dt)
    kq = 2.2 / np.clip(q95, 0.5, None)
    r = {
        "density": sig(fgw_s, *LIMITS["density"]) ** 3,
        "beta": sig(bn_s, *LIMITS["beta"]) ** 3,
        "kink": sig(kq, *LIMITS["kink"]) ** 3,
        "radiation": sig(frad_s, *LIMITS["radiation"]) ** 3,
    }
    r_gw1, r_b1, r_q1 = (r["density"] ** (1 / 3), r["beta"] ** (1 / 3), r["kink"] ** (1 / 3))
    r_lm = 5e-4 * (1 + 5 * r_gw1 + 3 * r_b1 + 2 * r_q1)
    keep = (1 - r["density"]) * (1 - r["beta"]) * (1 - r["kink"]) * (1 - r["radiation"]) * (1 - r_lm)
    lam = 1 - keep
    lam = np.maximum(lam, 0.005 + np.minimum(r_lm, 0.01))
    lam = np.where(np.asarray(ip) > IP_MIN, lam, 0.0)
    return lam, r
