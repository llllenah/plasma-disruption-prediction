"""Фізичний препроцесинг для ImpactNet: 100 приладів → нормовані входи + маска відмов + фізика.

Лише минуле (причинно). Дві реалізації з однаковим результатом:
  offline(df)       — векторно по цілому розряду (для навчальних даних);
  Online().step(ch) — по кроку 5 мс (для інференсу); звірка — test_equal().

Фізика з методички (константи JET): f_GW, q95 (Bt/Ip × форма з котушок і петель потоку),
β_N (діамагнітна петля), P_rad/P_in (болометри / нагрів + омічна), W, dW/dt, τ_E з 0D-балансу,
τ_98 (IPB98(y,2)), H98, поріг L–H (Мартін) і запас P_in/P_LH, T_e з ECE, опір Спітцера,
nτT (Лоусон), дебаївська довжина. Масштаби проксі (хорда → n̄e тощо) калібруються на train
за істинним станом (fit_calibration), щоб априорі мережі стартувало з правильної фізики.
"""
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DT = 0.005
CHANNELS = json.loads((HERE / "channels.json").read_text())      # рівно 100 приладів заліку, у їхньому порядку
# JET (методичка)
R0, A_MIN, KAPPA, S_AREA, ZEFF, MASS = 2.89, 0.95, 1.70, 150.0, 1.6, 2.0
GW_AREA = math.pi * A_MIN ** 2
CHORD_M = 2 * A_MIN * KAPPA
BETA_C = 1.5917
BOLO = [f"bolo_{i:02d}" for i in range(1, 9)]
ECE_CORE = ("ece_01", "ece_02")
Q95_FORM = json.loads((HERE / "q95_form.json").read_text())
TAU_DRV = 0.1                   # симулятор згладжує входи сигмоїд 100 мс
TAU_DW = 0.05                   # згладжування W для похідної

PHYS = ["fgw", "q95", "kink", "frad", "beta_n", "p_in", "p_rad", "w_mj", "dwdt", "tau_e", "tau98", "h98",
        "p_lh", "lh_margin", "te_core", "eta_spitzer", "lawson", "debye"]
DRIVERS = ["fgw_s", "q95", "frad_s", "beta_n_s"]          # входи априорі (як у формулі симулятора)


def _a(tau):
    return 1.0 - math.exp(-DT / tau)


def _phys_inst(c, calib):
    """Миттєва фізика з (векторів або чисел) каналів c[name] після ffill."""
    ip = np.maximum(c["mag_ip_MA"], 0.1)
    bt = np.maximum(c["mag_bt_T"], 0.5)
    ne20 = calib["ne"] * c["ifr_chord_core"] / CHORD_M                 # n̄e, 10²⁰ м⁻³
    fgw = ne20 * GW_AREA / ip
    w = calib["w"] * c["mag_diamag_MJ"]
    beta_n = BETA_C * w / (ip * bt)
    p_oh = np.maximum(c["mag_vloop_V"] * c["mag_ip_MA"], 0.0)
    p_aux = sum(np.nan_to_num(c[k]) for k in ("nbi_power_MW", "ich_power_MW", "ech_power_MW"))
    p_in = p_aux + p_oh
    p_rad = calib["prad"] * sum(np.nan_to_num(c[b]) for b in BOLO)
    frad = np.where(p_in > 1.0, p_rad / np.maximum(p_in, 1e-6), 0.0)
    te = calib["te"] * 0.5 * (c[ECE_CORE[0]] + c[ECE_CORE[1]])        # кеВ
    return dict(ip=ip, bt=bt, ne20=ne20, fgw=fgw, w=w, beta_n=beta_n, p_in=p_in, p_rad=p_rad, frad=frad, te=te)


def _phys_rest(q, w_s, dwdt, form_z):
    """Решта фізики з миттєвих величин q, згладженої W і її похідної."""
    q95 = q["bt"] / q["ip"] * np.exp(Q95_FORM["w"][0] + form_z)
    p_net = np.maximum(q["p_in"] - dwdt, 0.3)
    tau_e = np.clip(w_s / p_net, 0.0, 10.0)
    n19 = np.maximum(q["ne20"] * 10, 0.1)
    tau98 = (0.0562 * q["ip"] ** 0.93 * q["bt"] ** 0.15 * n19 ** 0.41 * np.maximum(q["p_in"], 0.3) ** -0.69
             * R0 ** 1.97 * (A_MIN / R0) ** 0.58 * KAPPA ** 0.78 * MASS ** 0.19)
    p_lh = 0.0488 * np.maximum(q["ne20"], 0.01) ** 0.717 * q["bt"] ** 0.803 * S_AREA ** 0.941
    te = np.maximum(q["te"], 0.01)
    return dict(q95=q95, kink=2.2 / np.maximum(q95, 0.5), tau_e=tau_e, tau98=tau98, h98=tau_e / tau98,
                p_lh=p_lh, lh_margin=q["p_in"] / p_lh, eta_spitzer=ZEFF * te ** -1.5,
                lawson=q["ne20"] * tau_e * te, debye=7.43e3 * np.sqrt(te * 1e3 / np.maximum(q["ne20"] * 1e20, 1e17)))


def offline(df, calib):
    """df — один розряд (кроки 5 мс). → X (n, 200 + len(PHYS)) сирих входів, drivers (n, 4)."""
    raw = df[CHANNELS]
    mask = raw.isna().to_numpy(np.float32)
    c = raw.ffill()
    q = _phys_inst({k: c[k].to_numpy(float) for k in CHANNELS}, calib)
    import pandas as pd                     # лише для пакетного режиму; онлайн-кроку pandas не потрібен
    ew = lambda x, tau: pd.Series(x).ewm(alpha=_a(tau), adjust=False).mean().to_numpy()
    form_z = 0.0
    for k, name in enumerate(Q95_FORM["channels"]):
        v = ew(c[name].to_numpy(float), 0.05)
        form_z = form_z + (v - Q95_FORM["mu"][k]) / Q95_FORM["sd"][k] * Q95_FORM["w"][k + 1]
    w_s = ew(q["w"], TAU_DW)
    dwdt = ew(np.r_[0.0, np.diff(w_s)] / DT, TAU_DW)
    r = _phys_rest(q, w_s, dwdt, form_z)
    phys = {**q, **r, "w_mj": q["w"], "dwdt": dwdt, "te_core": q["te"]}
    P = np.c_[[phys[k] for k in PHYS]].T
    drv = np.c_[ew(q["fgw"], TAU_DRV), r["q95"], ew(q["frad"], TAU_DRV), ew(q["beta_n"], TAU_DRV)]
    return np.c_[c.to_numpy(float), mask, P], drv


class Online:
    """Той самий препроцесинг по одному кроку (стан — останні значення й EMA)."""

    def __init__(self, calib):
        self.calib = calib
        self.last = {k: np.nan for k in CHANNELS}
        self.ema = {}
        self.w_prev = None

    def _e(self, key, x, tau):
        o = self.ema.get(key)
        if o is None or o != o:
            self.ema[key] = x
        elif x == x:
            self.ema[key] = o + _a(tau) * (x - o)
        return self.ema[key]

    def step(self, ch):
        mask = []
        for k in CHANNELS:
            v = ch.get(k, np.nan)
            m = not (v == v)
            mask.append(float(m))
            if not m:
                self.last[k] = v
        c = self.last
        q = _phys_inst(c, self.calib)
        form_z = 0.0
        for k, name in enumerate(Q95_FORM["channels"]):
            v = self._e(("q", name), c[name], 0.05)
            form_z += (v - Q95_FORM["mu"][k]) / Q95_FORM["sd"][k] * Q95_FORM["w"][k + 1]
        w_s = self._e("w", float(q["w"]), TAU_DW)
        d = 0.0 if self.w_prev is None else (w_s - self.w_prev) / DT
        self.w_prev = w_s
        dwdt = self._e("dw", d, TAU_DW)
        r = _phys_rest(q, w_s, dwdt, form_z)
        phys = {**q, **r, "w_mj": q["w"], "dwdt": dwdt, "te_core": q["te"]}
        x = np.array([c[k] for k in CHANNELS] + mask + [float(phys[k]) for k in PHYS])
        drv = np.array([self._e("fgw", float(q["fgw"]), TAU_DRV), float(r["q95"]),
                        self._e("frad", float(q["frad"]), TAU_DRV), self._e("bn", float(q["beta_n"]), TAU_DRV)])
        return x, drv


def truth_drivers(g):
    """Істинні драйвери (колонки --extras) — лише для калібрування масштабів проксі."""
    import sys
    sys.path.insert(0, str(ROOT / "analysis"))
    import physics as ph
    coef = json.loads((ROOT / "analysis" / "physics_coef.json").read_text())["q95_coef"]
    ip = g.ip_MA.clip(lower=0.05)
    return dict(ne=g.ne_bar_1e20.to_numpy(), w=g.w_th_MJ.to_numpy(), prad=g.p_rad_MW.to_numpy(),
                te=g.te0_keV.to_numpy() if "te0_keV" in g else None,
                q95=ph.q95_from_shape(g.bt_T, g.ip_MA, g.xpoint_r_m, g.xpoint_z_m, g.axis_r_m, g.axis_z_m, coef))


def fit_calibration(shots):
    """Медіанні відношення істина / проксі на плато (масштаб хорда→n̄e, діамагнітна→W, болометри→P_rad, ECE→Te)."""
    num = {k: [] for k in ("ne", "w", "prad", "te")}
    for g in shots:
        ok = (g.ip_MA > 1.0).to_numpy()
        c = g[CHANNELS].ffill()
        tr = truth_drivers(g)
        prox = dict(ne=c.ifr_chord_core / CHORD_M, w=c.mag_diamag_MJ, prad=sum(c[b] for b in BOLO),
                    te=0.5 * (c[ECE_CORE[0]] + c[ECE_CORE[1]]))
        for k in num:
            if tr[k] is None:
                continue
            p = prox[k].to_numpy()
            s = ok & (np.abs(p) > 1e-6) & np.isfinite(p)
            num[k].append(tr[k][s] / p[s])
    return {k: float(np.median(np.concatenate(v))) if v else 1.0 for k, v in num.items()}


def test_equal(df, calib, n=400):
    X, D = offline(df.iloc[:n], calib)
    on = Online(calib)
    err = 0.0
    for i in range(n):
        x, d = on.step({k: df[k].iat[i] for k in CHANNELS})
        a, b = np.r_[x, d], np.r_[X[i], D[i]]
        both = np.isfinite(a) & np.isfinite(b)
        err = max(err, float(np.max(np.abs(a[both] - b[both]) / (np.abs(b[both]) + 1e-6), initial=0.0)))
        assert (np.isfinite(a) == np.isfinite(b)).all(), f"nan не збігся на кроці {i}"
    return err
