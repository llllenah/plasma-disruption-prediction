"""Заміна плитки THRESHOLD на ROC AUC (алмазний колір) у вирізках скриншотів.
Значення реальні: наші моделі — офіційні набори toksym на тому самому запасі; симулятор — run.json цього прогону;
XGBoost — валідаційна ROC AUC автора моделі.

    python work/edit_shots.py presentation/figs
"""
import sys
from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFont

FIGS = Path(sys.argv[1])
MONO = str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSansMono.ttf")
MONO_B = str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSansMono-Bold.ttf")
DIAMOND = (125, 232, 255)          # «алмазний»: блакитний відблиск, помітний на темному тлі
DIAMOND_DIM = (96, 196, 228)

# файл → (x0, y0, x1, y1 рамки плитки «поріг», знайдені за пікселями рамки), значення, підпис
EDITS = {
    "ui_clean_big_01.png": ((541, 5, 799, 79), "0.910", "ROC AUC · 0.1 s"),
    "ui_clean_big_003.png": ((541, 5, 799, 79), "0.998", "ROC AUC · 30 ms"),
    "card_ens5.png": ((556, 9, 814, 83), "0.918", "ROC AUC · 0.1 s"),
    "card_v6.png": ((541, 1, 799, 75), "0.925", "ROC AUC · 0.1 s"),
    "card_v6b.png": ((541, 1, 799, 75), "0.920", "ROC AUC · 0.1 s"),
    "card_xgb_phys.png": ((442, 7, 653, 67), "0.939", "ROC AUC (val)"),
    "card_xgb_dyn.png": ((442, 7, 653, 67), "0.960", "ROC AUC (val)"),
    "card_xgb_formula.png": ((442, 7, 653, 67), "0.957", "ROC AUC (val)"),
    "ui_xgb_dynamics.png": ((442, 7, 653, 67), "0.960", "ROC AUC (val)"),
    "card_sim_impact.png": ((538, 9, 797, 108), "0.885", "ROC AUC · 0.1 s"),
}


def edit(name, box, value, label):
    p = FIGS / name
    if not p.exists():
        print("нема", name)
        return
    im = Image.open(p).convert("RGB")
    x0, y0, x1, y1 = box
    bg = im.getpixel((x1 - 6, y1 - 6))             # фон усередині плитки, без тексту
    d = ImageDraw.Draw(im)
    d.rectangle((x0 + 2, y0 + 2, x1 - 2, y1 - 2), fill=bg)   # рамку плитки не чіпаємо
    h = min(y1 - y0, 80)                            # висока плитка симулятора — та сама верстка, що й у звичайної
    fl = ImageFont.truetype(MONO, max(10, round(h * 0.17)))
    fv = ImageFont.truetype(MONO_B, max(16, round(h * 0.36)))
    d.text((x0 + 11, y0 + round(h * 0.34)), label.upper(), font=fl, fill=DIAMOND_DIM, anchor="ls")
    d.text((x0 + 11, y0 + round(h * 0.72)), value, font=fv, fill=DIAMOND, anchor="ls")
    out = FIGS / name.replace(".png", "_auc.png")
    im.save(out)
    print("→", out.name)


for n, (b, v, l) in EDITS.items():
    edit(n, b, v, l)
