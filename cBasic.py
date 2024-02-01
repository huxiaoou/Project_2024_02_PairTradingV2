class CInstruPair(object):
    def __init__(self, instru_a: str, instru_b: str):
        self.instru_a, self.instru_b = instru_a, instru_b

    @property
    def Id(self) -> str:
        return f"{self.instru_a}_{self.instru_b}"

    def get_instruments(self) -> list[str]:
        return [self.instru_a, self.instru_b]


class CCfgFactor(object):
    def __init__(self, factor_class: str, args: tuple):
        self.factor_class = factor_class.upper()
        self.args = args

    def get_factors(self) -> list[str]:
        return [f"{self.factor_class}{_:03d}" for _ in self.args]

    def get_factors_raw(self) -> list[str]:
        return [f"{self.factor_class}{_:03d}" for _ in self.args]


class CCfgFactorEWM(CCfgFactor):
    def __init__(self, args: tuple, fixed_bgn_date: str):
        self.fixed_bgn_date = fixed_bgn_date
        super().__init__(factor_class="EWM", args=args)

    def get_factors(self) -> list[str]:
        return [f"{self.factor_class}F{int(f * 100):02d}S{int(s * 100):02d}" for (f, s) in self.args]


class CCfgFactorMA(CCfgFactor):
    def __init__(self, win_mov_ave: int, **kwargs):
        self.win_mov_ave = win_mov_ave
        super().__init__(**kwargs)

    def get_factors(self) -> list[str]:
        fs = super().get_factors()
        return [s + f"MA{self.win_mov_ave:02d}" for s in fs]
