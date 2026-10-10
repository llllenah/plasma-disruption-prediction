"""ImpactNet: вклад механізмів у зрив. Вихідний шар — рівно 5 нейронів:

    0 greenwald  — межа Грінвальда,      драйвер f_GW
    1 q95        — низький запас q95,    драйвер 2,2/q95 (кінк)
    2 radiation  — радіаційний колапс,   драйвер P_rad/P_in
    3 troyon     — межа Тройона,         драйвер β_N/2,8
    4 other      — решта (locked mode тощо), лише з прихованого стану мережі

Нейрон k — логіт ризику у формі симулятора:  r_k = σ((x_k − c_k)/w_k + δ_k(h)),  ризик r_k^p
(p = 3 для чотирьох меж, 1 для «інших»). c, w — навчувані, ініціалізовані порогами симулятора
(перевірено на розкладі λ до 10⁻⁶): f_GW 1,00/0,08; 2,2/q95 0,90/0,05; frad 0,80/0,08; β_N/2,8 0,85/0,06.
З цих самих 5 значень, без додаткових нейронів:
    вклад       C_k = −log(1 − r_k^p)          (1 − λ = Π(1 − r_k^p) = e^(−ΣC))
    частка      share_pct = 100 · C_k / ΣC      (сума рівно 100 %)
    ризик       risk_k = r_k^p ∈ [0, 1]
    загальний   risk_total = λ = 1 − e^(−ΣC)

Будова (строго причинна; forward() по послідовності == step() по кроку):
    вхід → MLP → 4 причинні згорткові блоки (ядро 3, дилатації 1/2/4/8 → бачать 31 крок = 155 мс:
    швидкі передвісники) → GRU(128) — коротка пам'ять розряду → LayerNorm → Linear(128 → 5).
Голова стартує з нулів: необучена мережа = чиста фізика симулятора, «інші» ≈ 0,2 %.
Далі мережа вчить лише поправки δ(h) до фізики (шум приладів, затримки, калібрування).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

IMPACT_NAMES = ("greenwald", "q95", "radiation", "troyon", "other")
# (c0, w0) у змінних x = (f_GW, 2,2/q95, frad, β_N/2,8); p = 3
PRIOR = ((1.00, 0.08), (0.90, 0.05), (0.80, 0.08), (0.85, 0.06))
OTHER_P, OTHER_R0 = 1, 2e-3
LOG_EPS = 1e-6


class CausalConvBlock(nn.Module):
    """Pre-LN → depthwise причинна дилатована згортка → GLU → 1×1, залишкове з'єднання."""

    def __init__(self, d, kernel, dilation, p_drop):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.norm = nn.LayerNorm(d)
        self.dw = nn.Conv1d(d, d, kernel, dilation=dilation, groups=d)
        self.pw = nn.Conv1d(d, 2 * d, 1)
        self.out = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(p_drop)

    def _branch(self, y):
        return self.drop(self.out(F.glu(self.pw(self.dw(y)), dim=1)))

    def forward(self, x):                                   # [B, T, d]
        y = F.pad(self.norm(x).transpose(1, 2), (self.pad, 0))
        return x + self._branch(y).transpose(1, 2)

    def step(self, x, buf):                                 # x [B, d], buf [B, d, pad]
        win = torch.cat([buf, self.norm(x).unsqueeze(-1)], dim=-1)    # останні pad + 1 кроків
        return x + self._branch(win)[..., -1], win[..., 1:]


class ImpactNet(nn.Module):
    def __init__(self, n_in, d=96, hidden=128, kernel=3, dilations=(1, 2, 4, 8), mem_steps=200,
                 p_drop=0.1, physics_prior=True, arch="gru"):
        """arch — для порівняння архітектур (step() лише для gru):
        gru  — згортки 155 мс + GRU (основна);   lstm — те саме з LSTM;
        tcn  — лише причинні згортки, дилатації до 128 (бачать 511 кроків = 2,6 с), без рекурентності;
        mlp  — без пам'яті взагалі (кожен крок окремо)."""
        super().__init__()
        self.hidden, self.physics_prior, self.arch = hidden, physics_prior, arch
        if arch == "tcn":
            dilations = (1, 2, 4, 8, 16, 32, 64, 128)
        elif arch == "mlp":
            dilations = ()
        self.inp = nn.Sequential(nn.Linear(n_in, d), nn.GELU(), nn.Dropout(p_drop), nn.Linear(d, d))
        self.blocks = nn.ModuleList(CausalConvBlock(d, kernel, dl, p_drop) for dl in dilations)
        self.pre_gru = nn.LayerNorm(d)
        rnn = {"gru": nn.GRU, "lstm": nn.LSTM}.get(arch)
        self.gru = rnn(d, hidden, batch_first=True) if rnn else None
        h_out = hidden if rnn else d
        self.post = nn.LayerNorm(h_out)
        self.head = nn.Linear(h_out, len(IMPACT_NAMES))
        self.c = nn.Parameter(torch.tensor([p[0] for p in PRIOR]))
        self.w_raw = nn.Parameter(torch.log(torch.expm1(torch.tensor([p[1] for p in PRIOR]))))
        self.register_buffer("power", torch.tensor([3.0, 3.0, 3.0, 3.0, float(OTHER_P)]))
        self._init(mem_steps)

    def _init(self, mem_steps):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        for b in self.blocks:                           # залишкові гілки стартують як тотожність
            nn.init.zeros_(b.out.weight)
        H = self.hidden
        if self.gru is not None:
            ng = 3 if self.arch == "gru" else 4
            nn.init.xavier_uniform_(self.gru.weight_ih_l0)
            for g in range(ng):
                nn.init.orthogonal_(self.gru.weight_hh_l0[g * H:(g + 1) * H])
            nn.init.zeros_(self.gru.bias_ih_l0)
            nn.init.zeros_(self.gru.bias_hh_l0)
        with torch.no_grad():
            # chrono-init: гейт пам'яті (update у GRU, forget у LSTM) ≈ σ(b) → пам'ять ~ U(1, mem_steps) кроків
            if self.gru is not None:
                self.gru.bias_ih_l0[H:2 * H] = torch.log(torch.empty(H).uniform_(1, max(mem_steps - 1, 2)))
            self.head.weight.zero_()
            self.head.bias.zero_()
            self.head.bias[-1] = math.log(OTHER_R0 / (1 - OTHER_R0))
            if not self.physics_prior:
                self.head.weight.normal_(0.0, 0.01)

    @property
    def w(self):
        return F.softplus(self.w_raw) + 1e-3

    @property
    def receptive_field(self):
        return 1 + sum(b.pad for b in self.blocks)

    @staticmethod
    def physics_x(drivers):
        """drivers (f_GW, q95, frad, β_N) у своїх одиницях → змінні сигмоїд симулятора."""
        fgw, q95, frad, bn = drivers.unbind(-1)
        return torch.stack([fgw, 2.2 / q95.clamp_min(0.5), frad, bn / 2.8], -1)

    def _outputs(self, h, drivers):
        logits = self.head(h)                                               # 5 вихідних нейронів
        if self.physics_prior:
            phys = (torch.nan_to_num(self.physics_x(drivers)) - self.c) / self.w
            logits = logits + F.pad(phys, (0, 1))                           # «інші» — без фізики
        log_rp = self.power * F.logsigmoid(logits)                         # log r^p, стабільно для малих
        rp = log_rp.exp()
        C = -torch.log1p(-rp.clamp(max=1 - 1e-6))
        # log C: для малих r^p  −log(1−x) ≈ x(1 + x/2) → log C ≈ log x + x/2 (без втрати градієнта)
        log_C = torch.where(rp < 1e-4, log_rp + rp / 2, torch.log(C.clamp_min(1e-30)))
        return {"logits": logits, "r": torch.sigmoid(logits), "risk": rp, "log_C": log_C,
                "share_pct": 100 * torch.softmax(log_C, -1),
                "risk_total": -torch.expm1(-C.sum(-1))}

    def forward(self, x, drivers, h0=None):
        z = self.inp(x)
        for b in self.blocks:
            z = b(z)
        if self.gru is not None:
            h, hT = self.gru(self.pre_gru(z), h0)
        else:
            h, hT = z, None
        out = self._outputs(self.post(h), drivers)
        out["state"] = hT
        return out

    def init_state(self, batch=1):
        d = self.blocks[0].dw.in_channels
        return [torch.zeros(batch, d, b.pad) for b in self.blocks], torch.zeros(1, batch, self.hidden)

    def step(self, x, drivers, state):
        """x [B, n_in], drivers [B, 4] → (виходи, новий стан). O(1) на крок."""
        bufs, hg = state
        z = self.inp(x)
        new = []
        for b, buf in zip(self.blocks, bufs):
            z, nb = b.step(z, buf)
            new.append(nb)
        h, hg = self.gru(self.pre_gru(z).unsqueeze(1), hg)
        return self._outputs(self.post(h.squeeze(1)), drivers), (new, hg)


def impact_loss(out, risk_t, mask):
    """risk_t [B, T, 5] — справжні r_k^p (порядок IMPACT_NAMES), mask [B, T] — кроки, де мітка є.
    Величина ризику — у лог-шкалі (щоб розрізняти 10⁻⁴ і 10⁻²), частки — з вагою λ
    (частки при нульовому ризику — шум, їх не форсуємо)."""
    C_t = -torch.log1p(-risk_t.clamp(max=1 - 1e-7))
    lp = torch.logaddexp(out["log_C"], torch.full_like(out["log_C"], math.log(LOG_EPS)))
    lt = torch.log(C_t + LOG_EPS)
    l_mag = F.smooth_l1_loss(lp, lt, reduction="none").mean(-1)
    l_tot = F.smooth_l1_loss(torch.logaddexp(torch.logsumexp(out["log_C"], -1), torch.full_like(out["log_C"][..., 0], math.log(LOG_EPS))),
                             torch.log(C_t.sum(-1) + LOG_EPS), reduction="none")
    s_t = C_t / C_t.sum(-1, keepdim=True).clamp_min(1e-30)
    kl = (s_t * (torch.log(s_t.clamp_min(1e-12)) - torch.log_softmax(out["log_C"], -1))).sum(-1)
    lam_t = -torch.expm1(-C_t.sum(-1))
    l = l_mag + l_tot + 2 * lam_t * kl
    return (l * mask).sum() / mask.sum().clamp_min(1)
