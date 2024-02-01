import datetime as dt
import numpy as np
import pandas as pd
from husfort.qutility import SFG
from husfort.qsqlite import CQuickSqliteLib, CLib1Tab1, CTable, CLibFactor
from husfort.qcalendar import CCalendar
from cBasic import CInstruPair
from cReturnsDiff import CLibDiffReturn


class CLibFactorExposure(CQuickSqliteLib):
    def __init__(self, factor: str, lib_save_dir: str):
        self.factor = factor
        lib_name = f"factor_exposure.{factor}.db"
        super().__init__(lib_name, lib_save_dir)

    def get_lib_struct(self) -> CLib1Tab1:
        return CLib1Tab1(
            lib_name=self.lib_name,
            table=CTable(
                {
                    "table_name": self.factor,
                    "primary_keys": {"trade_date": "TEXT", "pair": "TEXT"},
                    "value_columns": {"value": "REAL"},
                }
            )
        )


class CFactorExposure(object):
    def __init__(self, factor: str, factors_exposure_dir: str, instruments_pairs: list[CInstruPair]):
        self.factor = factor
        self.factors_exposure_dir = factors_exposure_dir
        self.instruments_pairs = instruments_pairs

    def cal(self, bgn_date: str, stp_date: str, calendar: CCalendar) -> pd.DataFrame:
        """

        :param bgn_date:
        :param stp_date:
        :param calendar:
        :return: a pd.DataFrame, with index = list[str["YYYYMMDD"]], columns = ["pair", factor]
        """
        pass

    @staticmethod
    def truncate_before_bgn(df: pd.DataFrame, bgn_date: str) -> pd.DataFrame:
        return df.truncate(before=bgn_date)

    def save(self, factor_exposure_df: pd.DataFrame, run_mode: str):
        lib_writer = CLibFactorExposure(self.factor, self.factors_exposure_dir).get_lib_writer(run_mode)
        lib_writer.update(update_df=factor_exposure_df, using_index=True)
        lib_writer.commit()
        lib_writer.close()
        return 0

    def main(self, run_mode: str, bgn_date: str, stp_date: str, calendar: CCalendar):
        factor_exposure_df = self.cal(bgn_date, stp_date, calendar)
        factor_exposure_df = self.truncate_before_bgn(factor_exposure_df, bgn_date)
        self.save(factor_exposure_df, run_mode)
        print(f"{dt.datetime.now()} [INF] Factor {SFG(self.factor)} is calculated")
        return 0


class _CFactorExposureEndogenous(CFactorExposure):
    def __init__(self, factor: str, diff_returns_dir: str, **kwargs):
        self.diff_returns_dir = diff_returns_dir
        super().__init__(factor=factor, **kwargs)

    def _get_base_date(self, bgn_date: str, calendar: CCalendar) -> str:
        pass

    def _get_diff_return(self, instru_pair: CInstruPair, base_date: str, stp_date: str) -> pd.DataFrame:
        lib_diff_return_reader = CLibDiffReturn(instru_pair, self.diff_returns_dir).get_lib_reader()
        pair_df = lib_diff_return_reader.read_by_conditions(
            conditions=[
                ("trade_date", ">=", base_date),
                ("trade_date", "<", stp_date),
            ], value_columns=["trade_date", "diff_return"]
        ).set_index("trade_date")
        return pair_df

    def _cal_factor(self, diff_ret_srs: pd.Series) -> pd.Series:
        pass

    def cal(self, bgn_date: str, stp_date: str, calendar: CCalendar) -> pd.DataFrame:
        base_date = self._get_base_date(bgn_date, calendar)
        dfs_list = []
        for instru_pair in self.instruments_pairs:
            pair_df = self._get_diff_return(instru_pair, base_date, stp_date)
            pair_df[self.factor] = self._cal_factor(pair_df["diff_return"])
            pair_df["pair"] = instru_pair.Id
            dfs_list.append(pair_df[["pair", self.factor]])
        df = pd.concat(dfs_list, axis=0, ignore_index=False)
        df.sort_values(by=["trade_date", "pair"], inplace=True)
        return df


class CFactorExposureLag(_CFactorExposureEndogenous):
    def __init__(self, lag: int, **kwargs):
        self.lag = lag
        super().__init__(**kwargs)

    def _get_base_date(self, bgn_date: str, calendar: CCalendar) -> str:
        return calendar.get_next_date(bgn_date, -self.lag)

    def _cal_factor(self, diff_ret_srs: pd.Series) -> pd.Series:
        return diff_ret_srs.shift(self.lag)


class CFactorExposureSUM(_CFactorExposureEndogenous):
    def __init__(self, win: int, **kwargs):
        self.win = win
        super().__init__(**kwargs)

    def _get_base_date(self, bgn_date: str, calendar: CCalendar) -> str:
        return calendar.get_next_date(bgn_date, -self.win + 1)

    def _cal_factor(self, diff_ret_srs: pd.Series) -> pd.Series:
        s: pd.Series = diff_ret_srs.rolling(window=self.win).sum()
        return s


class CFactorExposureEWM(_CFactorExposureEndogenous):
    def __init__(self, fast: float, slow: float, fix_base_date: str, **kwargs):
        self.fix_base_date = fix_base_date
        self.fast, self.slow = fast, slow
        super().__init__(**kwargs)

    def _get_base_date(self, bgn_date: str, calendar: CCalendar) -> str:
        return self.fix_base_date

    def _cal_factor(self, diff_ret_srs: pd.Series) -> pd.Series:
        fast_srs = diff_ret_srs.ewm(alpha=self.fast, adjust=False).mean()
        slow_srs = diff_ret_srs.ewm(alpha=self.slow, adjust=False).mean()
        return fast_srs - slow_srs


class CFactorExposureVOL(_CFactorExposureEndogenous):
    def __init__(self, win: int, **kwargs):
        self.win = win
        super().__init__(**kwargs)

    def _get_base_date(self, bgn_date: str, calendar: CCalendar) -> str:
        return calendar.get_next_date(bgn_date, -self.win + 1)

    def _cal_factor(self, diff_ret_srs: pd.Series) -> pd.Series:
        volatility: pd.Series = diff_ret_srs.rolling(window=self.win).std() * np.sqrt(250)
        return volatility


class CFactorExposureTNR(_CFactorExposureEndogenous):
    def __init__(self, win: int, **kwargs):
        self.win = win
        super().__init__(**kwargs)

    def _get_base_date(self, bgn_date: str, calendar: CCalendar) -> str:
        return calendar.get_next_date(bgn_date, -self.win + 1)

    def _cal_factor(self, diff_ret_srs: pd.Series) -> pd.Series:
        rng_sum_abs = diff_ret_srs.abs().rolling(window=self.win).sum()
        rng_sum = diff_ret_srs.rolling(window=self.win).sum()
        tnr: pd.Series = rng_sum / rng_sum_abs
        return tnr


class _CFactorExposureExogenous(CFactorExposure):
    pass


class CFactorExposureFromInstruExposureDiff(_CFactorExposureExogenous):
    def __init__(self, factor_exo: str, win_mov_ave: int, instru_factor_exposure_dir: str, **kwargs):
        self.factor_exo = factor_exo
        self.instru_factor_exposure_dir = instru_factor_exposure_dir
        self.win_mov_ave = win_mov_ave
        super().__init__(**kwargs)

    def cal(self, bgn_date: str, stp_date: str, calendar: CCalendar) -> pd.DataFrame:
        base_date = calendar.get_next_date(bgn_date, -self.win_mov_ave + 1)
        lib_instru_factor = CLibFactor(self.factor_exo, self.instru_factor_exposure_dir).get_lib_reader()
        instru_pair_dfs = []
        for instru_pair in self.instruments_pairs:
            instru_factor_exposure = {}
            for instru in instru_pair.get_instruments():
                instru_df = lib_instru_factor.read_by_conditions(
                    conditions=[
                        ("trade_date", ">=", base_date),
                        ("trade_date", "<", stp_date),
                        ("instrument", "=", instru)
                    ], value_columns=["trade_date", "value"]
                ).set_index("trade_date")
                instru_factor_exposure[instru] = instru_df["value"]
            pair_df = pd.DataFrame(instru_factor_exposure)
            pair_df["diff_exposure"] = (pair_df[instru_pair.instru_a] - pair_df[instru_pair.instru_b]).fillna(0)
            pair_df[self.factor] = pair_df["diff_exposure"].rolling(window=self.win_mov_ave).mean()
            pair_df["pair"] = instru_pair.Id
            pair_df = pair_df.truncate(before=bgn_date)
            instru_pair_dfs.append(pair_df[["pair", self.factor]])
        df = pd.concat(instru_pair_dfs, axis=0, ignore_index=False)
        df.sort_values(by=["trade_date", "pair"], inplace=True)
        return df
