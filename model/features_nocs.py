"""Онлайн-признаки модели (без формул из «Шпаргалки B» методички, даже косвенно).

Только сглаженные сырые каналы приборов, фаза программы (уставки) и сценарий, Dα, Мирнов,
SXR, ECE и детектор предвестника по седловой петле. Шаг за шагом, только прошлое —
одинаково при обучении и на зачёте.
"""
import math

DT = 0.005
TAU_FAST, TAU_MID, TAU_SLOW = 0.05, 0.25, 1.0          # постоянные EMA, с
NEON_ON = 0.05                                         # клапан неона «открыт»

SCEN = ("density", "low_q", "neon", "heat_loss", "none")
SXR_CORE = ("sxr_04", "sxr_05")
ECE_CORE = ("ece_01", "ece_02")
MIRNOV = tuple(f"mag_mirnov_{i:02d}" for i in range(1, 9))
DALPHA = ("dalpha_div_inner", "dalpha_div_outer", "dalpha_midplane", "dalpha_xpoint")

# Сырые каналы, которые просто сглаживаются (EMA 250 мс).
RAW = ("mag_ip_MA", "mag_bt_T", "mag_diamag_MJ", "mag_vloop_V", "coil_pf_shape_kA",
       "coil_pf_vert_kA", "ifr_chord_core", "ifr_chord_edge", "ifr_chord_far_edge",
       "gas_flow_d2_meas", "gas_flow_ne_meas", "vac_rga_mass20", "neutron_total",
       "div_probe_isat_out", "prog_ne_1e20")

# Величины, для которых считаем три EMA, тренд и максимум на плато.
DERIVED = ("ne_over_prog", "sxr_core", "ece_core", "dalpha", "mirnov_rms", "lm")


def _a(tau):
    return 1.0 - math.exp(-DT / tau)


A_F, A_M, A_S = _a(TAU_FAST), _a(TAU_MID), _a(TAU_SLOW)
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
    names += ["t_neon_on", "neon_cmd", "neon_int", "t_heat_on", "prog_ne_slope", "t_ne_flat"]
    names += ["lm_hp20", "lm_hp50", "lm_growth", "mir_hp20", "mir_growth"]
    return names


class Features:
    """Состояние признаков одного разряда."""

    def __init__(self, scenario="", kalman=None):
        parts = [p for p in (scenario or "").split("+") if p]
        self.scen = [1.0 if s in parts else 0.0 for s in SCEN]
        self.scen_n = float(len(parts))
        self.last = {}            # последнее нормальное значение каждого канала
        self.ema = {}
        self.vmax = {}
        self.prog_ip_peak = 0.0
        self.t_flat = None
        self.dalpha_peaks = 0.0
        self.dalpha_above = False
        self.n = 0
        self.t_neon_on = None
        self.neon_int = 0.0
        self.t_heat_on = None
        self.t_ne_flat = None

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
        g = self._get
        prog_ip = g(ch, "prog_ip_MA")
        self.n += 1

        # ── фаза программы по уставке тока ──
        if _ok(prog_ip):
            self.prog_ip_peak = max(self.prog_ip_peak, prog_ip)
        pk = max(self.prog_ip_peak, 1e-6)
        ip_frac = prog_ip / pk if _ok(prog_ip) else float("nan")
        rampdown = 1.0 if (self.prog_ip_peak > 0.5 and _ok(prog_ip) and prog_ip < 0.9 * pk) else 0.0
        ramping_up = 1.0 if (_ok(prog_ip) and prog_ip >= 0.9 * pk and self.n > 1
                             and self.ema.get(("prog_ip", "f"), prog_ip) < prog_ip - 1e-4) else 0.0
        self._ema(("prog_ip", "f"), prog_ip, A_F)
        if self.t_flat is None and self.prog_ip_peak > 0.5 and ip_frac >= 0.99 and not ramping_up:
            self.t_flat = t
        t_flat = t - self.t_flat if self.t_flat is not None else -1.0

        # ── величины из сырых каналов (без формул шпаргалки) ──
        chord = g(ch, "ifr_chord_core")
        prog_ne = g(ch, "prog_ne_1e20")
        ne_over_prog = chord / prog_ne if (_ok(chord) and _ok(prog_ne) and prog_ne > 0.05) else float("nan")
        sx = [g(ch, s) for s in SXR_CORE]
        sxr_core = sum(sx) / len(sx)
        ec = [g(ch, s) for s in ECE_CORE]
        ece_core = sum(ec) / len(ec)
        da = [x for x in (g(ch, s) for s in DALPHA) if _ok(x)]
        dalpha = sum(da) / len(da) if da else float("nan")
        mv = [x for x in (g(ch, s) for s in MIRNOV) if _ok(x)]
        mirnov_rms = math.sqrt(sum(x * x for x in mv) / len(mv)) if mv else float("nan")
        lm = abs(g(ch, "mag_saddle_lm"))
        vals = dict(ne_over_prog=ne_over_prog, sxr_core=sxr_core, ece_core=ece_core, dalpha=dalpha,
                    mirnov_rms=mirnov_rms, lm=lm)

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

        # ── спецпризнаки ──
        base = self.ema.get(("dalpha", "s"), float("nan"))          # пики Dα: пересечение 1,5 × фона
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

        # ── фаза программы: неон, нагрев, уставка плотности ──
        neon_cmd = g(ch, "gas_valve_ne_cmd")
        neon_flow = g(ch, "gas_flow_ne_meas")
        if self.t_neon_on is None and _ok(neon_cmd) and neon_cmd > NEON_ON:
            self.t_neon_on = t
        if _ok(neon_flow):
            self.neon_int += max(neon_flow, 0.0) * DT
        heat_on = any(_ok(x) and x > 1.0 for x in (g(ch, "nbi_power_MW"), g(ch, "ich_power_MW"), g(ch, "ech_power_MW")))
        if self.t_heat_on is None and heat_on:
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

        # ── детектор предвестника: медленную составляющую (дрейф, AR) убираем средним за 0,5 с,
        # смотрим энергию остатка за 20 и 50 мс и скорость её роста ──
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
