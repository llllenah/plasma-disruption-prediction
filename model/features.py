"""Онлайн-ознаки: крок за кроком, тільки минуле.

Той самий клас працює і в навчанні (прохід по measured.csv), і на заліку
(виклик на кожному кроці 5 мс). Так ознаки на train і на тесті збігаються за
побудовою, а не «за домовленістю».

Залежностей немає, крім стандартної бібліотеки: на заліку кожна мікросекунда
на кроці — наша.
"""
import json
import math
from pathlib import Path

try:  # фільтр Калмана для сирих каналів (model/kalman.py), необов'язковий
    from kalman import OnlineKalman
except ImportError:  # pragma: no cover
    OnlineKalman = None

DT = 0.005

# Геометрія JET — відома й опублікована (див. examples/model_baseline.py).
A_MINOR = 0.95
KAPPA = 1.70
CHORD_M = 2.0 * A_MINOR * KAPPA
GW_AREA = math.pi * A_MINOR ** 2

# Сталі часу згладжування, с.
TAU_FAST = 0.05
TAU_MID = 0.25
TAU_SLOW = 1.0

SCEN = ("density", "low_q", "neon", "heat_loss", "none")

# ── v3: фаза програми і фізика меж ──
BETA_C = 1.5917            # β_N = BETA_C · W / (Ip·Bt), калібровано на jet_dd_s1
NEON_ON = 0.05             # клапан неону «відкрито»
_QF = Path(__file__).resolve().parent / "q95_form.json"
Q95_FORM = json.loads(_QF.read_text()) if _QF.exists() else None   # форма плазми з котушок і петель потоку


# ── v7: корекція стрибків інтерферометра (fringe) ──
IFR_CH = ("ifr_chord_core", "ifr_chord_mid_in", "ifr_chord_mid_out", "ifr_chord_edge", "ifr_chord_far_edge")
# σ приросту кожної хорди за крок (робастна, з tr101–103: model/ifr_sigma.json)
IFR_SIGMA = {"ifr_chord_core": 0.01853, "ifr_chord_mid_in": 0.01483, "ifr_chord_mid_out": 0.01483,
             "ifr_chord_edge": 0.01112, "ifr_chord_far_edge": 0.00741}
FRINGE_Z, OTHERS_Z = 6.0, 3.0
# Перевірено на tr104 (analysis/fringe_study.py): корекція не зменшує розкид хорда/n̄_e
# (5,46 → 5,32 % на центральній, гірше на крайніх) — у моделі вимкнено.
FRINGE_ON = False


class FringeCorrector:
    """Стрибок приладу бачить ЛИШЕ одна хорда, справжню зміну густини — усі.

    Якщо приріст хорди > FRINGE_Z·σ, а всі інші хорди зрушили < OTHERS_Z·σ, це стрибок
    на ціле число смуг: він віднімається, і далі хорда йде з поправкою. Лише минуле.
    """

    def __init__(self):
        self.last = {}
        self.off = {c: 0.0 for c in IFR_CH}
        self.n = 0

    def step(self, ch):
        raw = {c: ch.get(c) for c in IFR_CH}
        z, d = {}, {}
        for c in IFR_CH:
            v, lv = raw[c], self.last.get(c)
            if _ok(v) and lv is not None:
                d[c] = v - lv
                z[c] = abs(d[c]) / IFR_SIGMA[c]
        if len(z) >= 3:
            for c in z:
                others = [z[o] for o in z if o != c]
                if z[c] > FRINGE_Z and max(others) < OTHERS_Z:
                    self.off[c] += d[c]
                    self.n += 1
        for c in IFR_CH:
            if _ok(raw[c]):
                self.last[c] = raw[c]
        out = dict(ch)
        for c in IFR_CH:
            if _ok(raw[c]):
                out[c] = raw[c] - self.off[c]
        return out


def _sig3(x, c, w):
    if x != x:
        return float("nan")
    z = (x - c) / w
    return (1.0 / (1.0 + math.exp(-max(min(z, 50.0), -50.0)))) ** 3

BOLO = tuple(f"bolo_{i:02d}" for i in range(1, 9))
SXR_CORE = ("sxr_04", "sxr_05")
ECE_CORE = ("ece_01", "ece_02")
MIRNOV = tuple(f"mag_mirnov_{i:02d}" for i in range(1, 9))
DALPHA = ("dalpha_div_inner", "dalpha_div_outer", "dalpha_midplane", "dalpha_xpoint")

# Сирі канали, які ще й згладжуємо самі по собі.
RAW = ("mag_ip_MA", "mag_bt_T", "mag_diamag_MJ", "mag_vloop_V", "coil_pf_shape_kA",
       "coil_pf_vert_kA", "ifr_chord_core", "ifr_chord_edge", "ifr_chord_far_edge",
       "gas_flow_d2_meas", "gas_flow_ne_meas", "vac_rga_mass20", "neutron_total",
       "div_probe_isat_out", "prog_ne_1e20")

# Похідні величини, для яких рахуємо три EMA і тренди.
DERIVED = ("f_gw", "f_gw_edge", "q_proxy", "beta_proxy", "rad_frac", "p_in", "ne_over_prog",
           "sxr_core", "ece_core", "dalpha", "mirnov_rms", "lm")


def _a(tau):
    return 1.0 - math.exp(-DT / tau)


A_F, A_M, A_S = _a(TAU_FAST), _a(TAU_MID), _a(TAU_SLOW)
# v8: детектор передвісника під відому модель шуму сідлової петлі
# (конфіг шуму: білий abs 0,005 + дрейф + AR(1) τ = 0,2 с; передвісник ≈ 0,04, росте як e^{20t})
A_20, A_100, A_500 = _a(0.02), _a(0.1), _a(0.5)


def _ok(v):
    return v is not None and v == v


def feature_names():
    names = ["t", "t_flat", "ip_frac", "rampdown", "ramping_up"]
    names += [f"scen_{s}" for s in SCEN] + ["scen_n"]
    for d in DERIVED:
        names += [f"{d}_f", f"{d}_m", f"{d}_s", f"{d}_trend", f"{d}_max"]
    names += [f"raw_{r}" for r in RAW]
    names += ["dalpha_peak_rate", "lm_max_f", "te_drop", "sxr_drop"]
    names += ["t_neon_on", "neon_cmd", "neon_int", "t_heat_on", "prog_ne_slope", "t_ne_flat",
              "q95_hat", "r_gw", "r_beta", "r_q", "r_rad", "lam_hat", "lam_hat_max", "lam_int"]
    names += ["f_gw_prog"]
    names += ["lm_hp20", "lm_hp50", "lm_growth", "mir_hp20", "mir_growth"]
    return names


class Features:
    """Стан ознак одного розряду."""

    def __init__(self, scenario="", kalman=None):
        # kalman — параметри model/kalman_params.json: тоді сирі канали спершу
        # очищаються онлайн-фільтром Калмана, і лише потім рахуються ознаки
        self.kf = OnlineKalman(kalman) if kalman else None
        self.fringe = FringeCorrector()
        parts = [p for p in (scenario or "").split("+") if p]
        self.scen = [1.0 if s in parts else 0.0 for s in SCEN]
        self.scen_n = float(len(parts))
        self.last = {}            # останнє справжнє значення кожного каналу
        self.ema = {}             # (назва, швидкість) -> значення
        self.vmax = {}
        self.prog_ip_peak = 0.0
        self.t_flat = None
        self.dalpha_peaks = 0.0   # EMA частоти піків D_alpha, 1/с
        self.dalpha_above = False
        self.n = 0
        self.t_neon_on = None
        self.neon_int = 0.0
        self.t_heat_on = None
        self.t_ne_flat = None
        self.lam_max = 0.0
        self.lam_int = 0.0

    def _get(self, ch, name):
        v = ch.get(name)
        if _ok(v):
            self.last[name] = v
            return v
        return self.last.get(name, float("nan"))

    def _ema(self, key, x, a):
        old = self.ema.get(key)
        if old is None or old != old:
            self.ema[key] = x
        elif x == x:
            self.ema[key] = old + a * (x - old)
        return self.ema[key]

    def step(self, t, ch):
        if self.kf is not None:
            ch = self.kf.step(ch)
        if FRINGE_ON:
            ch = self.fringe.step(ch)
        g = self._get
        ip = g(ch, "mag_ip_MA")
        bt = g(ch, "mag_bt_T")
        prog_ip = g(ch, "prog_ip_MA")
        self.n += 1

        if _ok(prog_ip):
            self.prog_ip_peak = max(self.prog_ip_peak, prog_ip)
        pk = max(self.prog_ip_peak, 1e-6)
        ip_frac = prog_ip / pk if _ok(prog_ip) else float("nan")
        # Спад: уставка впала нижче 90 % свого максимуму, а максимум уже солідний.
        rampdown = 1.0 if (self.prog_ip_peak > 0.5 and _ok(prog_ip) and prog_ip < 0.9 * pk) else 0.0
        ramping_up = 1.0 if (_ok(prog_ip) and prog_ip >= 0.9 * pk and self.n > 1
                             and self.ema.get(("prog_ip", "f"), prog_ip) < prog_ip - 1e-4) else 0.0
        self._ema(("prog_ip", "f"), prog_ip, A_F)
        if self.t_flat is None and self.prog_ip_peak > 0.5 and ip_frac >= 0.99 and not ramping_up:
            self.t_flat = t
        t_flat = t - self.t_flat if self.t_flat is not None else -1.0

        ipc = max(ip, 0.1) if _ok(ip) else float("nan")
        chord = g(ch, "ifr_chord_core")
        chord_e = g(ch, "ifr_chord_edge")
        f_gw = (chord / CHORD_M) * GW_AREA / ipc
        f_gw_edge = (chord_e / CHORD_M) * GW_AREA / ipc
        q_proxy = bt / ipc
        w = g(ch, "mag_diamag_MJ")
        beta_proxy = w / (ipc * max(bt, 0.5)) if _ok(bt) else float("nan")
        vloop = g(ch, "mag_vloop_V")
        p_aux = sum(x for x in (g(ch, "nbi_power_MW"), g(ch, "ich_power_MW"), g(ch, "ech_power_MW")) if _ok(x))
        p_oh = vloop * ip if (_ok(vloop) and _ok(ip)) else 0.0
        p_in = p_aux + max(p_oh, 0.0)
        bolo = [g(ch, b) for b in BOLO]
        bolo = [b for b in bolo if _ok(b)]
        p_rad = sum(bolo) if bolo else float("nan")
        rad_frac = p_rad / p_in if p_in > 1.0 else 0.0
        prog_ne = g(ch, "prog_ne_1e20")
        ne_over_prog = (chord / CHORD_M) / prog_ne if (_ok(prog_ne) and prog_ne > 0.05) else float("nan")
        sx = [g(ch, s) for s in SXR_CORE]
        sxr_core = sum(sx) / len(sx)
        ec = [g(ch, s) for s in ECE_CORE]
        ece_core = sum(ec) / len(ec)
        da = [g(ch, s) for s in DALPHA]
        da = [x for x in da if _ok(x)]
        dalpha = sum(da) / len(da) if da else float("nan")
        mv = [g(ch, s) for s in MIRNOV]
        mv = [x for x in mv if _ok(x)]
        mirnov_rms = math.sqrt(sum(x * x for x in mv) / len(mv)) if mv else float("nan")
        lm = abs(g(ch, "mag_saddle_lm"))

        vals = dict(f_gw=f_gw, f_gw_edge=f_gw_edge, q_proxy=q_proxy, beta_proxy=beta_proxy,
                    rad_frac=rad_frac, p_in=p_in, ne_over_prog=ne_over_prog, sxr_core=sxr_core,
                    ece_core=ece_core, dalpha=dalpha, mirnov_rms=mirnov_rms, lm=lm)

        out = [t, t_flat, ip_frac, rampdown, ramping_up] + self.scen + [self.scen_n]
        plateau = rampdown == 0.0 and self.prog_ip_peak > 0.5
        for d in DERIVED:
            x = vals[d]
            if x != x or abs(x) == float("inf"):
                x = float("nan")
            f = self._ema((d, "f"), x, A_F)
            m = self._ema((d, "m"), x, A_M)
            s = self._ema((d, "s"), x, A_S)
            if plateau and m == m:
                self.vmax[d] = max(self.vmax.get(d, m), m)
            out += [f, m, s, m - s, self.vmax.get(d, float("nan"))]
        for r in RAW:
            out.append(self._ema(("raw", r), g(ch, r), A_M))

        # Піки D_alpha: перетин угору рівня 1.5 × повільного середнього.
        base = self.ema.get(("dalpha", "s"), float("nan"))
        above = dalpha == dalpha and base == base and dalpha > 1.5 * base
        if above and not self.dalpha_above:
            self.dalpha_peaks += 1.0 / DT * A_S
        self.dalpha_peaks *= (1.0 - A_S)
        self.dalpha_above = above
        out.append(self.dalpha_peaks)

        lm_f = self.ema.get(("lm", "f"), float("nan"))
        self.vmax["lm_f"] = max(self.vmax.get("lm_f", 0.0), lm_f if lm_f == lm_f else 0.0)
        out.append(self.vmax["lm_f"])
        te_f, te_s = self.ema.get(("ece_core", "f")), self.ema.get(("ece_core", "s"))
        out.append(te_f / te_s - 1.0 if (te_s and te_s == te_s and abs(te_s) > 1e-6) else 0.0)
        sx_f, sx_s = self.ema.get(("sxr_core", "f")), self.ema.get(("sxr_core", "s"))
        out.append(sx_f / sx_s - 1.0 if (sx_s and sx_s == sx_s and abs(sx_s) > 1e-6) else 0.0)

        # ── v3: фаза програми ──
        neon_cmd = g(ch, "gas_valve_ne_cmd")
        neon_flow = g(ch, "gas_flow_ne_meas")
        if self.t_neon_on is None and _ok(neon_cmd) and neon_cmd > NEON_ON:
            self.t_neon_on = t
        if _ok(neon_flow):
            self.neon_int += max(neon_flow, 0.0) * DT
        if self.t_heat_on is None and p_aux > 1.0:
            self.t_heat_on = t
        pn_f = self._ema(("prog_ne", "f"), prog_ne, A_F)
        pn_s = self._ema(("prog_ne", "s"), prog_ne, A_S)
        slope = (pn_f - pn_s) if (pn_f == pn_f and pn_s == pn_s) else float("nan")
        if (self.t_ne_flat is None and t_flat >= 0 and slope == slope
                and abs(slope) < 0.01 * max(abs(pn_f), 1e-3)):
            self.t_ne_flat = t
        out += [t - self.t_neon_on if self.t_neon_on is not None else -1.0,
                self._ema(("neon_cmd", "m"), neon_cmd, A_M), self.neon_int,
                t - self.t_heat_on if self.t_heat_on is not None else -1.0,
                slope, t - self.t_ne_flat if self.t_ne_flat is not None else -1.0]

        # ── v3: фізика меж за формулою симулятора (методичка, розд. 4.6) ──
        q95 = float("nan")
        if Q95_FORM is not None and _ok(ipc) and _ok(bt):
            z = 0.0
            ok = True
            for k, name in enumerate(Q95_FORM["channels"]):
                v = self._ema(("q", name), g(ch, name), A_F)
                if v != v:
                    ok = False
                    break
                z += (v - Q95_FORM["mu"][k]) / Q95_FORM["sd"][k] * Q95_FORM["w"][k + 1]
            if ok:
                q95 = bt / ipc * math.exp(Q95_FORM["w"][0] + z)
        fgw_m = self.ema.get(("f_gw", "m"), float("nan"))
        bn_m = BETA_C * self.ema.get(("beta_proxy", "m"), float("nan"))
        rad_m = self.ema.get(("rad_frac", "m"), float("nan"))
        r_gw = _sig3(fgw_m, 1.00, 0.08)
        r_b = _sig3(bn_m / 2.8, 0.85, 0.06)
        r_q = _sig3(2.2 / q95 if (q95 == q95 and q95 > 0.5) else float("nan"), 0.90, 0.05)
        r_r = _sig3(rad_m, 0.80, 0.08)
        keep = 1.0
        for r in (r_gw, r_b, r_q, r_r):
            if r == r:
                keep *= 1.0 - r
        lam = 0.0 if (rampdown > 0.5 or not plateau) else 1.0 - keep
        self.lam_max = max(self.lam_max, lam)
        self.lam_int += lam * DT
        out += [q95, r_gw, r_b, r_q, r_r, lam, self.lam_max, self.lam_int]

        # ── v7: f_GW з уставки густини (не залежить від калібрування) і лічильник стрибків ──
        f_gw_prog = prog_ne * GW_AREA / ipc if (_ok(prog_ne) and _ok(ipc)) else float("nan")
        out.append(self._ema(("f_gw_prog", "m"), f_gw_prog, A_M))

        # ── v8: передвісник. Повільну складову (дрейф, AR τ 0,2 с) віднімаємо середнім за 0,5 с,
        # дивимось енергію залишку за 20 і 50 мс і швидкість її росту (мода росте експоненційно).
        lm_raw = g(ch, "mag_saddle_lm")
        base = self._ema(("lm_base", "s"), lm_raw, A_500)
        hp = lm_raw - base if (_ok(lm_raw) and base == base) else float("nan")
        e20 = self._ema(("lm_e", "20"), hp * hp if hp == hp else float("nan"), A_20)
        e50 = self._ema(("lm_e", "50"), hp * hp if hp == hp else float("nan"), A_F)
        e_slow = self._ema(("lm_e", "100"), e20, A_100)
        growth = (math.log(e20 + 1e-9) - math.log(e_slow + 1e-9)) if (e20 == e20 and e_slow == e_slow) else float("nan")
        mv_mean = sum(abs(x) for x in mv) / len(mv) if mv else float("nan")
        mbase = self._ema(("mir_base", "s"), mv_mean, A_500)
        mhp = mv_mean - mbase if (mv_mean == mv_mean and mbase == mbase) else float("nan")
        me20 = self._ema(("mir_e", "20"), mhp * mhp if mhp == mhp else float("nan"), A_20)
        me_slow = self._ema(("mir_e", "100"), me20, A_100)
        mgrowth = (math.log(me20 + 1e-12) - math.log(me_slow + 1e-12)) if (me20 == me20 and me_slow == me_slow) else float("nan")
        out += [math.sqrt(e20) if e20 == e20 else float("nan"), math.sqrt(e50) if e50 == e50 else float("nan"),
                growth, math.sqrt(me20) if me20 == me20 else float("nan"), mgrowth]
        return out
