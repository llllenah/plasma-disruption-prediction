"""Біла презентація для захисту: реальні скриншоти генерацій + графіки з прогонів (presentation/figs).

    python work/make_deck_white.py presentation/plasma_disruption.pptx
"""
import sys
from pathlib import Path

import qrcode
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "presentation/plasma_disruption.pptx")
FIGS = OUT.parent / "figs"
REPO = "github.com/llllenah/plasma-disruption-prediction"
FONT = "Arial"
BG, CARD, LINE = "FFFFFF", "F8FAFC", "E2E8F0"
TEXT, MUTED, DIM = "0F172A", "475569", "94A3B8"
IND, EME, ORA, VIO, AMB, RED, CYAN = "4F46E5", "059669", "EA580C", "7C3AED", "D97706", "DC2626", "0891B2"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
TOTAL = 14
_n = [0]


def rgb(h):
    return RGBColor.from_string(h)


def tb(slide, x, y, w, h, paras, anchor=MSO_ANCHOR.TOP):
    s = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = s.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", 0)
    for i, (content, o) in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = o.get("align", PP_ALIGN.LEFT)
        p.space_after = Pt(o.get("after", 0))
        for text, ro in (content if isinstance(content, list) else [(content, {})]):
            r = p.add_run()
            r.text = text
            f = r.font
            f.name = FONT
            f.size = Pt(ro.get("size", o.get("size", 15)))
            f.bold = ro.get("bold", o.get("bold", False))
            f.color.rgb = rgb(ro.get("color", o.get("color", TEXT)))
    return s


def box(slide, x, y, w, h, fill=CARD, line=LINE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06, lw=1.0):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = rgb(fill)
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(lw)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    return s


def card(slide, x, y, w, h, paras, pad=0.22, anchor=MSO_ANCHOR.TOP, line=LINE, fill=CARD):
    box(slide, x, y, w, h, fill, line)
    return tb(slide, x + pad, y + pad * 0.8, w - 2 * pad, h - 1.6 * pad, paras, anchor)


def stat(slide, x, y, w, h, big, small, color):
    card(slide, x, y, w, h, [(big, dict(size=30, bold=True, color=color, after=2)), (small, dict(size=12, color=MUTED))])


def pic(slide, name, x, y, w=None, h=None, border=False):
    path = FIGS / name
    W, H = Image.open(path).size
    if w and not h:
        h = w * H / W
    elif h and not w:
        w = h * W / H
    if border:
        box(slide, x - 0.04, y - 0.04, w + 0.08, h + 0.08, None, LINE, shape=MSO_SHAPE.RECTANGLE)
    slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))
    return w, h


def arrow(slide, x, y, w=0.4, h=0.3, shape=MSO_SHAPE.RIGHT_ARROW):
    box(slide, x, y, w, h, DIM, None, shape=shape)


def table(slide, x, y, w, rows, col_w, row_h=0.42, sizes=None, colors=None, fills=None):
    gs = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows)))
    tbl = gs.table
    tblPr = gs._element.graphic.graphicData.tbl.tblPr
    tblPr.set("firstRow", "0")
    tblPr.set("bandRow", "0")
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = rgb("EEF2FF" if i == 0 else (fills[i][j] if fills else ("FFFFFF" if i % 2 else CARD)))
            c.margin_left = c.margin_right = Inches(0.1)
            c.margin_top = c.margin_bottom = Inches(0.03)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = c.text_frame.paragraphs[0]
            c.text_frame.word_wrap = True
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = str(val)
            r.font.name = FONT
            r.font.size = Pt(sizes[i][j] if sizes else (12 if i == 0 else 13))
            r.font.bold = i == 0 or j == 0
            r.font.color.rgb = rgb(colors[i][j] if colors else (IND if i == 0 else TEXT))
    return gs


def new_slide(tag, title, color=IND, notes=""):
    _n[0] += 1
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb(BG)
    if tag:
        tb(s, 0.6, 0.38, 9, 0.3, [(tag, dict(size=12, bold=True, color=color))])
    if title:
        tb(s, 0.6, 0.64, 12.1, 0.8, [(title, dict(size=30, bold=True))])
    tb(s, 0.6, 7.05, 9, 0.3, [("Передбачення зриву плазми · ФМФ КПІ 2026", dict(size=9, color=DIM))])
    tb(s, 12.2, 7.05, 0.6, 0.3, [(f"{_n[0]}/{TOTAL}", dict(size=9, color=DIM, align=PP_ALIGN.RIGHT))])
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def caption(slide, x, y, w, text):
    tb(slide, x, y, w, 0.35, [(text, dict(size=10, color=DIM))])


# 1 ── титул
s = new_slide("ХАКАТОН ФМФ КПІ · 9–11 ЖОВТНЯ 2026", None, notes=(
    "≈30 с. Задача організаторів: за 100 приладами токамака в реальному часі передбачити зрив плазми. Ми зробили модель "
    "тривоги без формул і без підказок, нейромережу ImpactNet, яка пояснює причину, і перевірили все в симуляторі з "
    "автокеруванням."))
tb(s, 0.6, 1.45, 11.5, 2.3, [("Передбачаємо, пояснюємо і запобігаємо зриву плазми", dict(size=44, bold=True))])
tb(s, 0.6, 3.75, 8.4, 0.8, [("Модель тривоги в реальному часі · нейромережа причин ImpactNet · замкнений контур керування",
                             dict(size=17, color=MUTED))])
for i, (big, small, c) in enumerate((("0,924", "об'єм заліку (бейзлайн 0,765)", EME), ("98 %", "зривів спіймано за 30 мс", EME),
                                      ("49 / 49", "причин зривів названо вірно", VIO), ("11 → 5", "зривів з автокеруванням", ORA))):
    stat(s, 0.6 + i * 3.08, 4.85, 2.85, 1.3, big, small, c)
tb(s, 0.6, 6.45, 12, 0.4, [("Дмитрій Почко · Олександр Чорій · Ілля Симоненко · Олена Сергієнко · Софія Щербина · Арина Яковець",
                            dict(size=13, color=MUTED))])

# 2 ── задача
s = new_slide("ЗАДАЧА", "Побачити зрив раніше — і не лякати тихі розряди", notes=(
    "≈45 с. Кожні 5 мс модель отримує 100 приладів і видає ймовірність зриву, лише з минулого. Залік вважає зрив спійманим, "
    "якщо тривога прийшла щонайменше за lead до гасіння; поріг підбирається так, щоб хибних тривог на тихих розрядах було "
    "не більше бюджету. На скриншоті — як залік показує кожен розряд: заливка — бігучий максимум p, червона риска — гасіння, "
    "кольорова — тривога. Головна метрика — об'єм: середнє по всіх запасах і бюджетах."))
rows = (("100 приладів кожні 5 мс", "магнітні, інтерферометр, ECE, болометри, SXR, газ, нагрів"),
        ("Лише минуле", "модель працює онлайн, крок має вкластися в 5 мс"),
        ("Тривога ≥ lead до гасіння", "поріг підбирає залік під бюджет хибних тривог"),
        ("Метрика — об'єм", "середня якість по всіх запасах (10 мс…1 с) і бюджетах"))
for i, (h, t) in enumerate(rows):
    card(s, 0.6, 1.7 + i * 1.18, 4.9, 1.02, [(h, dict(size=15, bold=True, after=3)), (t, dict(size=12, color=MUTED))])
w, h = pic(s, "ui_rows.png", 5.9, 1.7, w=6.85, border=True)
caption(s, 5.9, 1.75 + h + 0.05, 6.9, "Test Run організаторів: рядок = розряд; заливка — p моделі, червона риска — гасіння, кольорова — тривога")

# 3 ── бейзлайн проти нас
s = new_slide("БЕЙЗЛАЙН ПРОТИ НАС", "0,765 → 0,924: без жодної формули і без підказок", EME, notes=(
    "≈45 с. Бейзлайн організаторів — поріг на частці межі Грінвальда плюс locked mode: об'єм 0,765, при 30 мс і 3 % хибних "
    "ловить 59 %. Наша фінальна модель без формул шпаргалки і без мітки сценарію — 0,924 і 98 % при 30 мс. Офіційний залік "
    "на 4 наборах, 747 розрядів, 228 зривів."))
card(s, 0.6, 1.7, 5.6, 4.3, [("Бейзлайн організаторів", dict(size=17, bold=True, color=ORA, after=8)),
                             ("0,765", dict(size=54, bold=True, color=ORA)), ("об'єм заліку", dict(size=13, color=MUTED, after=10)),
                             ("59 %", dict(size=34, bold=True, color=ORA)), ("зривів за 30 мс при 3 % хибних", dict(size=13, color=MUTED, after=12)),
                             ("Поріг на f_GW + сигнал locked mode: одна величина, без пам'яті; шум приладів смикає поріг; зриви від q95 і "
                              "радіації майже не бачить.", dict(size=13, color=TEXT))], pad=0.3)
card(s, 6.6, 1.7, 6.15, 4.3, [("Наша модель тривоги (здача)", dict(size=17, bold=True, color=EME, after=8)),
                              ("0,924", dict(size=54, bold=True, color=EME)), ("об'єм заліку  (+0,159)", dict(size=13, color=MUTED, after=10)),
                              ("98,2 %", dict(size=34, bold=True, color=EME)), ("зривів за 30 мс при 5 % хибних (224 / 228)", dict(size=13, color=MUTED, after=12)),
                              ("Без формул шпаргалки, без мітки сценарію; крок 0,9 мс на ноутбуці; по наборах 100 / 96,7 / 98,1 / 98,2 %.",
                               dict(size=13, color=TEXT))], pad=0.3, line=EME)
caption(s, 0.6, 6.15, 12, "Офіційний toksym score: сиди 1, 1 shift, 20261001, 42 · 747 розрядів · 228 зривів")

# 4 ── еволюція
s = new_slide("ЕВОЛЮЦІЯ МОДЕЛЕЙ", "Що ми пробували і чому здаємо clean_big", IND, notes=(
    "≈50 с. Шлях: моделі з фізикою і міткою сценарію давали до 0,944. Тімейт помітив, що мітка сценарію — підказка "
    "симулятора: scen_none гарантує, що зриву не буде. Ми прибрали формули шпаргалки і сценарій — 0,914, потім більше даних "
    "— 0,924. Ще більше даних і ансамбль дали 0,930, але ансамбль на ноутбуці робить крок 3 мс і кожен шостий крок не "
    "вкладається в 5 мс — тому здаємо clean_big."))
pic(s, "evolution.png", 0.6, 1.6, w=8.4)
for i, (h, t, c) in enumerate((("Фізика + сценарій", "v6 / v8: 0,937–0,944, але формули шпаргалки й мітка сценарію", IND),
                                ("Без підказок", "clean: 0,914 → +дані 0,924 (P = 0,97, парний бутстреп)", EME),
                                ("Відкинули ансамбль", "0,930, але 3 мс/крок і 16 % кроків > 5 мс на ноутбуці", RED))):
    card(s, 9.35, 1.6 + i * 1.6, 3.4, 1.42, [(h, dict(size=14, bold=True, color=c, after=4)), (t, dict(size=12, color=MUTED))])
caption(s, 0.6, 6.6, 12, "Об'єм офіційного заліку на тих самих 4 наборах · правило «беремо, якщо P > 0,9» зафіксоване наперед")

# 5 ── однакові 200 розрядів
s = new_slide("ГЕНЕРАЦІЇ: ОДНІ Й ТІ САМІ 200 РОЗРЯДІВ", "Пам'ять у часі важить більше, ніж фізичні ознаки", AMB, notes=(
    "≈50 с. Усі моделі прогнані в інтерфейсі організаторів на однакових 200 розрядах, запас 0,1 с, 5 % хибних. XGBoost без "
    "пам'яті — на фізичних ознаках і формулах — 25 %, з динамікою 35 %, і пропускає 5–9 зривів зовсім. Наші моделі з "
    "пам'яттю на трьох масштабах і детектором передвісника — 60–67 % і 0–1 пропущених. Медіана попередження в нас 1,2–1,4 с "
    "проти 0,7–0,9 с."))
pic(s, "ui_compare.png", 0.6, 1.55, w=8.6)
rows = [["", "XGBoost", "наші"], ["спіймано вчасно", "25–35 %", "57–67 %"], ["пропущено зовсім", "5–9", "0–1"],
        ["медіана попередження", "0,67–0,90 с", "0,94–1,42 с"], ["поріг p", "0,73–0,76", "0,23–0,42"]]
table(s, 9.5, 1.65, 3.25, rows, [1.45, 0.9, 0.9], row_h=0.55, sizes=[[11] * 3] + [[11, 12, 12]] * 4)
card(s, 9.5, 4.65, 3.25, 1.75, [("Висновок", dict(size=14, bold=True, color=AMB, after=4)),
                                ("головне — історія сигналу (EMA, тренди, передвісник), а не фізичні формули", dict(size=12, color=MUTED))])
caption(s, 0.6, 6.6, 12, "Test Run: 200 розрядів, 75 зривів (12 спонтанних виключено) · запас 0,1 с · бюджет 5 % · XGBoost — моделі Діми")

# 6 ── скриншоти
s = new_slide("ЩО МИ БАЧИЛИ В ГЕНЕРАЦІЯХ", "Ті самі розряди: XGBoost пізно, наша модель — за 1,4 с", AMB, notes=(
    "≈40 с. Ліворуч — скриншот XGBoost з динамікою: 22 спіймано, 32 пізно, 9 пропущено, поріг 0,743. Праворуч — наша здача "
    "на тих самих розрядах: 0,1 с — 63,5 %, медіана попередження 1,4 с, пропущених нуль; при 30 мс — 96,8 %."))
w, h = pic(s, "ui_xgb_dynamics.png", 0.6, 1.6, w=6.3, border=True)
caption(s, 0.6, 1.65 + h + 0.03, 6.3, "XGBoost + динаміка · 0,1 с: 22 вчасно, 32 пізно, 9 пропущено")
w2, h2 = pic(s, "ui_clean_big_01.png", 7.25, 1.6, w=5.5, border=True)
caption(s, 7.25, 1.65 + h2 + 0.03, 5.5, "наша здача clean_big · 0,1 с: 40 вчасно, 23 пізно, 0 пропущено")
y3 = 1.65 + h2 + 0.5
w3, h3 = pic(s, "ui_clean_big_003.png", 7.25, y3, w=5.5, border=True)
caption(s, 7.25, y3 + h3 + 0.05, 5.5, "наша здача clean_big · 30 мс: 61 вчасно, 2 пізно, 0 пропущено")
card(s, 7.25, y3 + h3 + 0.55, 5.5, 2.0, [("Різниця на тих самих розрядах", dict(size=15, bold=True, color=AMB, after=6)),
                                       ("XGBoost: 9 зривів не помічено зовсім, тривога в середньому за 0,9 с", dict(size=13, color=MUTED, after=4)),
                                       ("наша модель: пропущених 0, тривога в середньому за 1,4 с", dict(size=13, color=EME, bold=True))])

# 7 ── поріг
s = new_slide("ПОРІГ — НЕ НАЛАШТУВАННЯ МОДЕЛІ", "Поріг — не міра якості моделі", EME, notes=(
    "≈45 с. У XGBoost поріг 0,74, у нас 0,23 — це не означає, що хтось кращий: залік сам підбирає поріг під бюджет хибних "
    "тривог, а абсолютна шкала ймовірностей у моделей різна. Значення має лише, як модель розділяє зривні й тихі розряди. "
    "Тому журі може поставити будь-які запас і бюджет — модель перенавчати не треба. Теплокарта — частка спійманих вчасно "
    "для нашої здачі: до 30 мс — 98–100 %, для уникнення зриву при 10–15 % хибних — 67–82 %."))
pic(s, "heatmap.png", 0.6, 1.55, h=4.95)
for i, (h, t, c) in enumerate((("Поріг 0,74 vs 0,23", "різна шкала ймовірностей у моделей — поріг не міра якості", MUTED),
                                ("Пом'якшення зриву", "запас 10–30 мс при 5 % хибних: 98–99 % вчасно", EME),
                                ("Уникнення зриву", "запас 0,3 с при 10–15 % хибних: 67–82 % вчасно", AMB))):
    card(s, 8.25, 1.6 + i * 1.6, 4.5, 1.4, [(h, dict(size=15, bold=True, color=c, after=4)), (t, dict(size=12, color=MUTED))])
caption(s, 0.6, 6.6, 12, "Частка зривів, спійманих вчасно · фінальна здача · 4 офіційні набори (toksym rescore)")

# 8 ── реальний розряд
s = new_slide("РЕАЛЬНИЙ РОЗРЯД", "Тривога за 1,24 с — і названа причина: густина", EME, notes=(
    "≈45 с. Реальний розряд з тестового набору, сценарій density. Зверху — p нашої моделі тривоги: перетинає поріг за 1,24 "
    "с до гасіння. Знизу — ImpactNet: щойно ризик стає помітним, 98–100 % припадає на межу Грінвальда, і λ мережі "
    "збігається зі справжньою λ симулятора, яку мережа ніколи не бачила."))
pic(s, "shot_trace.png", 0.6, 1.55, w=8.9)
for i, (big, small, c) in enumerate((("1,24 с", "тривога до гасіння\n(поріг під 5 % хибних)", EME), ("Грінвальд", "98–100 % ризику —\nзрив від густини", ORA),
                                      ("0,90 = 0,90", "λ мережі = справжня λ\nу момент зриву", VIO))):
    stat(s, 9.85, 1.6 + i * 1.62, 2.9, 1.45, big, small, c)
caption(s, 0.6, 6.6, 12, "Розряд 5 тестового набору r508 (сценарій density) · модель тривоги clean_big · ImpactNet · справжня λ — з істинного стану")

# 9 ── модель тривоги
s = new_slide("РІШЕННЯ 1 · МОДЕЛЬ ТРИВОГИ", "Дві голови: передвісник і рання тривога", EME, notes=(
    "≈40 с. Ознаки онлайн: згладжування на трьох масштабах, тренди, детектор передвісника на сідловій петлі й Мірнові, "
    "команди актуаторів. Голова A — «зрив ≤ 0,1 с», голова B — «≤ 4,5 с». Максимум, нуль на спаді струму, згладжування, "
    "бігучий максимум. Дерева в numpy: 0,9 мс на крок."))
flow = ((0.6, "100 приладів", "кожні 5 мс", CYAN), (3.32, "Онлайн-ознаки", "EMA 50 мс / 250 мс / 1 с,\nтренди, передвісник,\nактуатори", VIO))
for x, h, t, c in flow:
    box(s, x, 2.4, 2.15, 2.2, "FFFFFF", c, lw=1.5)
    tb(s, x + 0.12, 2.5, 1.91, 2.0, [(h, dict(size=16, bold=True, color=c, after=6, align=PP_ALIGN.CENTER)),
                                     (t, dict(size=12, color=MUTED, align=PP_ALIGN.CENTER))], MSO_ANCHOR.MIDDLE)
arrow(s, 2.8, 3.35)
arrow(s, 5.52, 3.35)
for yy, h, t in ((1.85, "Голова A", "«зрив ≤ 0,1 с»\nпередвісник"), (3.85, "Голова B", "«зрив ≤ 4,5 с»\nрання тривога")):
    box(s, 6.0, yy, 2.5, 1.55, "FFFFFF", EME, lw=1.5)
    tb(s, 6.1, yy + 0.08, 2.3, 1.4, [(h, dict(size=16, bold=True, color=EME, after=4, align=PP_ALIGN.CENTER)),
                                     (t, dict(size=12, color=MUTED, align=PP_ALIGN.CENTER))], MSO_ANCHOR.MIDDLE)
arrow(s, 8.65, 3.35)
box(s, 9.2, 2.4, 3.55, 2.2, CARD, ORA, lw=1.5)
tb(s, 9.35, 2.5, 3.25, 2.0, [("p = max(A, 0,3·B)", dict(size=17, bold=True, color=ORA, after=6, align=PP_ALIGN.CENTER)),
                             ("0 на запланованому спаді струму\n→ згладжування → бігучий максимум", dict(size=12, color=MUTED, align=PP_ALIGN.CENTER))],
   MSO_ANCHOR.MIDDLE)
for i, (t, c) in enumerate((("лише numpy · 0,9 мс/крок", CYAN), ("4 500 розрядів, поділ за сидами", VIO),
                            ("без формул шпаргалки", ORA), ("без мітки сценарію", ORA))):
    box(s, 0.6 + i * 3.08, 5.6, 2.9, 0.48, "FFFFFF", c, radius=0.5)
    tb(s, 0.6 + i * 3.08, 5.6, 2.9, 0.48, [(t, dict(size=12, bold=True, color=c, align=PP_ALIGN.CENTER))], MSO_ANCHOR.MIDDLE)

# 10 ── фізика
s = new_slide("ФІЗИКА ЗРИВУ", "Зрив — кубик від чотирьох операційних меж", AMB, notes=(
    "≈50 с. Інтенсивність зриву λ = 1 − добуток (1 − rᵢ³): rᵢ — м'який поріг відстані до межі Грінвальда, кінку за q95, "
    "радіаційного колапсу і Тройона. Кожну межу можна відсунути актуатором, але дії конфліктують: струм тягне в обидва "
    "боки. Густина реагує за 0,3 с, неон — за 0,5 с, тож керувати треба завчасно."))
card(s, 0.6, 1.6, 12.15, 0.75, [([("λ = 1 − Π(1 − rᵢ³)·(1 − r_lm),   ", dict(size=19, bold=True, color=AMB)),
                                  ("rᵢ = σ((xᵢ − cᵢ)/wᵢ): далеко від межі ≈ 0, на межі 0,5, за межею → 1", dict(size=14, color=MUTED))], {})],
     anchor=MSO_ANCHOR.MIDDLE)
rows = [["механізм", "драйвер", "межа", "що зменшує ризик", "ціна"],
        ["Грінвальд", "f_GW = n̄ₑ / n_GW", "f_GW ≈ 1", "↓ уставка густини, ↓ D₂; ↑ Iₚ", "↑ Iₚ наближає кінк"],
        ["Кінк", "q95", "q95 ≈ 2,44", "↓ Iₚ; ↑ Bₜ, κ, δ", "↓ Iₚ наближає Грінвальда"],
        ["Радіація", "P_rad / P_in", "≈ 0,8", "неон → 0; ↑ нагрів; ↓ густина", "нагрів піднімає βN"],
        ["Тройон", "βN", "βN ≈ 2,4", "↓ нагрів; ↑ Iₚ, Bₜ", "менше утримання"]]
table(s, 0.6, 2.6, 12.15, rows, [1.8, 2.2, 1.6, 3.6, 2.95], row_h=0.6, sizes=[[12] * 5] + [[13] * 5] * 4)
tb(s, 0.6, 5.85, 12.1, 0.8, [([("Iₚ тягне в обидва боки", dict(bold=True, color=AMB)),
                               (" · густина реагує за ~0,3 с · неон накопичується за ~0,5 с — керувати треба завчасно, а не в момент тривоги",
                                dict(color=MUTED))], dict(size=14))])

# 11 ── ImpactNet
s = new_slide("РІШЕННЯ 2 · IMPACTNET", "ImpactNet називає причину кожного зриву", VIO, notes=(
    "≈50 с. Рівно 5 вихідних нейронів — Грінвальд, q95, радіація, Тройон, інші. Кожен — формула ризику симулятора плюс "
    "поправка мережі, тому частки складаються рівно в 100 %, а загальний ризик — та сама λ. Згортки бачать 155 мс, GRU дає "
    "пам'ять. Мітки — справжній ризик зі стану плазми, на вхід лише 100 приладів. 49 з 49 причин, R² 0,996, 0,6 мс."))
steps = (("218 входів", "100 приладів\n+ маски відмов\n+ 18 фіз. величин", CYAN), ("Згортки", "причинні,\nвікно 155 мс", IND),
         ("GRU (128)", "пам'ять\nрозряду", VIO), ("5 нейронів", "+ фізичне\nаприорі", ORA))
for i, (h, t, c) in enumerate(steps):
    x = 0.6 + i * 2.0
    box(s, x, 1.75, 1.7, 1.9, "FFFFFF", c, lw=1.5)
    tb(s, x + 0.08, 1.8, 1.54, 1.8, [(h, dict(size=14, bold=True, color=c, after=5, align=PP_ALIGN.CENTER)),
                                     (t, dict(size=11, color=MUTED, align=PP_ALIGN.CENTER))], MSO_ANCHOR.MIDDLE)
    if i < 3:
        arrow(s, x + 1.73, 2.55, w=0.24, h=0.28)
for i, (m, c) in enumerate((("Грінвальд", ORA), ("q95", IND), ("радіація", AMB), ("Тройон", CYAN), ("інші", DIM))):
    box(s, 0.6 + i * 1.57, 3.95, 1.45, 0.45, c, None, radius=0.5)
    tb(s, 0.6 + i * 1.57, 3.95, 1.45, 0.45, [(m, dict(size=12, bold=True, color="FFFFFF", align=PP_ALIGN.CENTER))], MSO_ANCHOR.MIDDLE)
card(s, 0.6, 4.65, 7.7, 1.75, [("частки = 100·Cₖ / ΣC,  Cₖ = −ln(1 − rₖᵖ)  →  сума рівно 100 %", dict(size=14, after=5)),
                               ("ризик λ = 1 − e^(−ΣC)  —  та сама формула, що в симуляторі", dict(size=14, after=5)),
                               ("P(зрив за h с) = 1 − e^(−λh)", dict(size=14))])
for i, (big, small, c) in enumerate((("49 / 49", "головна причина в момент зриву\n(20 Грінвальд, 15 q95, 14 радіація)", VIO),
                                      ("0,996", "R² загального ризику λ", EME), ("0,6 мс", "крок на ноутбуці, лише numpy", CYAN))):
    stat(s, 8.75, 1.75 + i * 1.57, 4.0, 1.4, big, small, c)
caption(s, 0.6, 6.6, 12, "200 тестових розрядів, яких мережа не бачила · порівняння архітектур: GRU ≈ LSTM ≈ TCN > без пам'яті")

# 12 ── інсайт
s = new_slide("ГОЛОВНИЙ ІНСАЙТ", "Точна λ не попереджає: потрібні дві моделі", VIO, notes=(
    "≈45 с. ImpactNet як будильник дає 0,77, але й справжня λ симулятора — лише 0,78: зрив — кидок кубика, тихі розряди теж "
    "підходять до межі. Тому «коли» — модель тривоги, «чому» — ImpactNet. І на справжньому реакторі λ ніхто не знає: ми "
    "навчили мережу лише з моментів зривів — 49 з 49 причин, але тільки з фізичним априорі; без нього 41."))
pic(s, "oracle.png", 0.6, 1.55, w=5.9)
card(s, 6.95, 1.6, 5.8, 1.55, [("Дві ролі", dict(size=15, bold=True, color=VIO, after=4)),
                               ("«коли» — модель тривоги: бачить передвісник після кидка кубика\n«чому» — ImpactNet: розкладає ризик на механізми",
                                dict(size=12, color=MUTED))])
rows = [["навчання ImpactNet", "причин\nвгадано", "R² λ"], ["з мітками λ (основна)", "49 / 49", "0,996"],
        ["лише моменти зривів + априорі", "49 / 49", "0,948"], ["без фізичного априорі", "41 / 49", "0,660"]]
table(s, 6.95, 3.4, 5.8, rows, [3.3, 1.3, 1.2], row_h=0.58, sizes=[[12] * 3] + [[13, 14, 14]] * 3)
caption(s, 6.95, 5.85, 5.8, "Як на справжньому реакторі: λ невідома, є лише моменти зривів")
caption(s, 0.6, 6.6, 6, "Об'єм як сигналізація, ті самі 200 тестових розрядів")

# 13 ── симулятор
s = new_slide("СИМУЛЯТОР З КЕРУВАННЯМ", "Модель сама керує розрядом: зривів удвічі менше", ORA, notes=(
    "≈50 с. Симулятор fusion-sim: модель під'єднана до розряду, панель показує ймовірність і частки механізмів, порадник "
    "пропонує дію, автокерування виконує її, коли p вище порогу. На скриншоті — v8 на новому рушії: поведінка та сама, "
    "пропущених 0. Замкнений контур на 40 розрядах: 11 зривів без керування, 5 з керуванням. Чесно: 40 розрядів і канали без "
    "шуму — наступний крок 200 розрядів і ціна керування."))
w, h = pic(s, "ui_v8_new_engine.png", 0.6, 1.6, w=6.2, border=True)
caption(s, 0.6, 1.65 + h + 0.03, 6.2, "v8 на новому рушії fusion-sim · 0,1 с: 25 вчасно, 24 пізно, 0 пропущено")
rows = [["загроза", "дія порадника"], ["Грінвальд", "густина × 0,85; D₂ × 0,5"], ["Кінк", "Iₚ до q95 ≈ 3 (≥ 1 МА)"],
        ["Радіація", "неон → 0; NBI + 5 МВт"], ["Тройон", "NBI × 0,8"]]
table(s, 0.6, 4.05, 6.2, rows, [1.8, 4.4], row_h=0.48, sizes=[[12, 12]] + [[13, 13]] * 4)
card(s, 7.2, 1.6, 5.55, 2.6, [("11 → 5", dict(size=60, bold=True, color=ORA)),
                              ("зривів на 40 розрядах (−55 %)", dict(size=16, bold=True, after=4)),
                              ("ті самі розряди без керування і з ним · v8 · поріг 0,25", dict(size=12, color=MUTED))], pad=0.3, line=ORA)
card(s, 7.2, 4.45, 5.55, 1.95, [("Цикл кожні 5 мс", dict(size=14, bold=True, after=4)),
                                ("прилади → тривога (модель) → причина (ImpactNet) → дія (порадник) → плазма реагує", dict(size=12, color=MUTED, after=6)),
                                ("попередньо: 40 розрядів, канали без шуму; далі — 200 розрядів і ціна керування", dict(size=12, color=AMB))])

# 14 ── підсумок
s = new_slide("ПІДСУМОК", "Коли, чому і що робити — у реальному часі", EME, notes=(
    "≈40 с. Модель тривоги без формул і підказок: 0,924 проти 0,765, 98 % за 30 мс. ImpactNet пояснює причину: 49 з 49. "
    "Симулятор з порадником удвічі зменшує зриви. Далі: контур на більшій вибірці і з шумом, ціна керування, довчання на "
    "реальних даних. Код і звіти — на GitHub. Дякуємо!"))
for i, (big, h, t, c) in enumerate((("0,924", "Коли", "модель тривоги без формул і підказок; 98 % зривів за 30 мс", EME),
                                    ("49 / 49", "Чому", "ImpactNet: частки 5 механізмів, та сама λ, що в симуляторі", VIO),
                                    ("11 → 5", "Що робити", "порадник і автокерування в замкненому контурі", ORA))):
    card(s, 0.6 + i * 3.0, 1.65, 2.8, 2.0, [(big, dict(size=30, bold=True, color=c, after=4)), (h, dict(size=16, bold=True, after=4)),
                                           (t, dict(size=12, color=MUTED))])
card(s, 0.6, 3.95, 8.8, 1.3, [("Далі", dict(size=15, bold=True, color=AMB, after=4)),
                             ("контур на 200+ розрядах і з шумом приладів · ціна керування (струм, енергія) · довчання ImpactNet лише з моментів зривів "
                              "на реальних даних", dict(size=12, color=MUTED))])
qr = qrcode.QRCode(border=1, box_size=10)
qr.add_data("https://" + REPO)
qr.make(fit=True)
qpath = FIGS / "repo_qr.png"
qr.make_image(fill_color="black", back_color="white").save(qpath)
box(s, 9.75, 1.65, 3.0, 3.0, "FFFFFF", LINE)
s.shapes.add_picture(str(qpath), Inches(9.9), Inches(1.8), Inches(2.7), Inches(2.7))
tb(s, 9.3, 4.8, 3.9, 0.9, [("Код, моделі, звіти заліку", dict(size=13, bold=True, align=PP_ALIGN.CENTER, after=2)),
                           (REPO, dict(size=10, color=IND, align=PP_ALIGN.CENTER))])
tb(s, 0.6, 5.7, 8, 0.5, [("Дякуємо! Питання?", dict(size=22, bold=True, color=EME))])

assert _n[0] == TOTAL, _n[0]
prs.save(OUT)
print("збережено", OUT)
