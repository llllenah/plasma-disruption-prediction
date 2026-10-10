"""Ознаки v9 = v8 + виходи нейромережі вкладів механізмів зриву (riskfeat.py, ваги заморожені).

У здачі v8 лежить поруч як features_v8_base.py; у model/ — це features.py.
"""
try:
    import features_v8_base as _B
except ImportError:
    import features as _B
import riskfeat as _R

_W = _R.load()


def feature_names():
    return _B.feature_names() + _R.NAMES


class Features(_B.Features):
    def __init__(self, scenario="", kalman=None):
        super().__init__(scenario, kalman)
        self.rn = _R.RiskNetOnline(_W)

    def step(self, t, ch):
        return super().step(t, ch) + self.rn.step(ch)
