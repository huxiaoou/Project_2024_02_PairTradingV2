from rich.progress import track
import datetime as dt
import multiprocessing as mp
import numpy as np
import pandas as pd
from itertools import product
from husfort.qutility import SFG, qtimer
from husfort.qcalendar import CCalendar
from husfort.qsqlite import CQuickSqliteLib, CLib1Tab1, CTable
from cBasic import CInstruPair
from cRegroups import CLibRegroups
from cReturnsDiff import CLibDiffReturn
from cMclrn import CMclrnModel, CLibPredictions


class CLibSimu(CQuickSqliteLib):
    def __init__(self, simu_id: str, lib_save_dir: str):
        self.simu_id = simu_id
        lib_name = f"simu.{self.simu_id}.db"
        super().__init__(lib_name, lib_save_dir)

    def get_lib_struct(self) -> CLib1Tab1:
        return CLib1Tab1(
            lib_name=self.lib_name,
            table=CTable(
                {
                    "table_name": "simu",
                    "primary_keys": {"trade_date": "TEXT"},
                    "value_columns": {
                        "rawRet": "REAL",
                        "dltWgt": "REAL",
                        "cost": "REAL",
                        "netRet": "REAL",
                        "nav": "REAL",
                    },
                }
            ),
        )


class CLibPortfolio(CQuickSqliteLib):
    def __init__(self, portfolio_id: str, lib_save_dir: str):
        self.portfolio_id = portfolio_id
        lib_name = f"portfolio.{self.portfolio_id}.db"
        super().__init__(lib_name, lib_save_dir)

    def get_lib_struct(self) -> CLib1Tab1:
        return CLib1Tab1(
            lib_name=self.lib_name,
            table=CTable(
                {
                    "table_name": "portfolio",
                    "primary_keys": {"trade_date": "TEXT"},
                    "value_columns": {
                        "netRet": "REAL",
                        "nav": "REAL",
                    },
                }
            ),
        )


class CSimuQuick(object):
    def __init__(self, simu_id: str, df: pd.DataFrame, sig: str, ret: str, cost_rate: float):
        """

        :param simu_id:
        :param df: has date-like index, with string format;
        :param sig: name of column which is used as signal
        :param ret: name of column which is used as raw return
        :param cost_rate:
        """
        self.simu_id = simu_id
        self.df = df
        self.sig = sig
        self.ret = ret
        self.cost_rate = cost_rate

    def __cal(self):
        self.df["rawRet"] = self.df[self.sig] * self.df[self.ret]
        self.df["dltWgt"] = self.df[self.sig] - self.df[self.sig].shift(1).fillna(0)
        self.df["cost"] = self.df["dltWgt"].abs() * self.cost_rate
        self.df["netRet"] = self.df["rawRet"] - self.df["cost"]
        self.df["nav"] = (self.df["netRet"] + 1).cumprod()
        return 0

    def __save(self, run_mode: str, simulations_dir: str):
        update_df = self.df[["rawRet", "dltWgt", "cost", "netRet", "nav"]]
        lib_simu_writer = CLibSimu(simu_id=self.simu_id, lib_save_dir=simulations_dir).get_lib_writer(run_mode)
        lib_simu_writer.update(update_df=update_df, using_index=True)
        lib_simu_writer.commit()
        lib_simu_writer.close()
        return 0

    def main(self, run_mode: str, simulations_dir: str):
        self.__cal()
        self.__save(run_mode=run_mode, simulations_dir=simulations_dir)
        return 0


def cal_simulations_quick(
    instru_pair: CInstruPair,
    delay: int,
    run_mode: str,
    bgn_date: str,
    stp_date: str,
    factors: list[str],
    cost_rate: float,
    regroups_dir: str,
    simulations_dir: str,
):
    lib_regroup_reader = CLibRegroups(instru_pair, delay, regroups_dir).get_lib_reader()
    df = lib_regroup_reader.read_by_conditions(
        conditions=[
            ("trade_date", ">=", bgn_date),
            ("trade_date", "<", stp_date),
        ],
        value_columns=["trade_date", "factor", "value"],
    )
    pivot_df = pd.pivot_table(data=df, index="trade_date", columns="factor", values="value")
    for factor in track(factors, description=f"{instru_pair.Id:>16s}"):
        simu_df = pivot_df[[factor, "diff_return"]].copy()
        simu_df["signal"] = np.sign(pivot_df[factor])
        simu_id = f"{instru_pair.Id}.{factor}.T{delay}"
        simu = CSimuQuick(simu_id=simu_id, df=simu_df, sig="signal", ret="diff_return", cost_rate=cost_rate)
        simu.main(run_mode=run_mode, simulations_dir=simulations_dir)
    return 0


@qtimer
def cal_simulations_instruments_pairs(
    instruments_pairs: list[CInstruPair], diff_ret_delays: list[int], proc_qty: int | None = None, **kwargs
):
    pool = mp.Pool(processes=proc_qty) if proc_qty else mp.Pool()
    for instru_pair, delay in product(instruments_pairs, diff_ret_delays):
        pool.apply_async(cal_simulations_quick, args=(instru_pair, delay), kwds=kwargs)
    pool.close()
    pool.join()
    return 0


class CSimuMclrn(object):
    def __init__(self, model: CMclrnModel):
        self.model_id = model.model_id
        self.instru_pair = model.instru_pair
        self.factors = model.factors
        self.delay = model.delay
        self.sig_method = model.sig_method

        self.signals = pd.DataFrame()
        self.signals_aligned = pd.DataFrame()
        self.weight_diff = pd.DataFrame()
        self.rets = pd.DataFrame()
        self.simu_rets = pd.DataFrame()

    def _get_dates(self, bgn_date: str, stp_date: str, calendar: CCalendar) -> tuple[list[str], list[str]]:
        ret_dates = calendar.get_iter_list(bgn_date, stp_date)
        sig_dates = calendar.shift_iter_dates(ret_dates, -self.delay)
        return sig_dates, ret_dates

    def _load_signal(self, sig_bgn_date: str, sig_end_date: str, predictions_dir: str):
        lib_pred = CLibPredictions(self.model_id, predictions_dir).get_lib_reader()
        pred_df = lib_pred.read_by_conditions(
            conditions=[
                ("trade_date", ">=", sig_bgn_date),
                ("trade_date", "<=", sig_end_date),
            ],
            value_columns=["trade_date", "value"],
        )
        signals: pd.DataFrame = pred_df.set_index("trade_date")
        if self.sig_method == "binary":
            self.signals = signals.applymap(lambda z: 2 * z - 1).fillna(0)
        else:  # self.sig_method == "continuous":
            self.signals = signals.applymap(lambda z: np.sign(z)).fillna(0)
        return 0

    def _align_signal(self, ret_dates: list[str], sig_dates: list[str]):
        bridge = pd.DataFrame({"ret_date": ret_dates, "sig_date": sig_dates})
        signals_aligned_df = pd.merge(left=bridge, right=self.signals, left_on="sig_date", right_index=True, how="left")
        self.signals_aligned = signals_aligned_df.set_index("ret_date").drop(axis=1, labels=["sig_date"]).fillna(0)
        return 0

    def _cal_weight_diff(self):
        self.weight_diff = self.signals_aligned - self.signals_aligned.shift(1).fillna(0)
        return 0

    def _load_ret(self, ret_bgn_date: str, ret_end_date: str, diff_returns_dir: str):
        lib_diff_return = CLibDiffReturn(self.instru_pair, diff_returns_dir).get_lib_reader()
        self.rets = (
            lib_diff_return.read_by_conditions(
                conditions=[
                    ("trade_date", ">=", ret_bgn_date),
                    ("trade_date", "<=", ret_end_date),
                ],
                value_columns=["trade_date", "diff_return"],
            )
            .set_index("trade_date")
            .fillna(0)
        )
        return 0

    def _check_shape(self):
        if self.signals_aligned.shape != self.rets.shape:
            print(
                f"{dt.datetime.now()} [ERR] signals and returns are not aligned"
                f" signals shape = {self.signals_aligned.shape},"
                f" returns shape = {self.rets.shape}"
            )
            raise ValueError
        return 0

    def _cal_ret(self, cost_rate: float):
        raw_ret: pd.Series = self.signals_aligned["value"] * self.rets["diff_return"]
        dlt_wgt_abs_sum: pd.Series = self.weight_diff.abs().sum(axis=1)
        cost: pd.Series = dlt_wgt_abs_sum * cost_rate
        net_ret: pd.Series = raw_ret - cost
        self.simu_rets = pd.DataFrame(
            {
                "rawRet": raw_ret,
                "dltWgt": dlt_wgt_abs_sum,
                "cost": cost,
                "netRet": net_ret,
                "cumNetRet": (net_ret + 1).cumprod(),
            }
        )
        return 0

    def _save(self, simulations_dir: str, run_mode: str):
        lib_simu_writer = CLibSimu(simu_id=self.model_id, lib_save_dir=simulations_dir).get_lib_writer(run_mode)
        lib_simu_writer.update(update_df=self.simu_rets, using_index=True)
        lib_simu_writer.commit()
        lib_simu_writer.close()
        return 0

    def main(
        self,
        run_mode: str,
        bgn_date: str,
        stp_date: str,
        calendar: CCalendar,
        cost_rate: float,
        predictions_dir: str,
        diff_returns_dir: str,
        simulations_dir: str,
        verbose: bool = False,
    ):
        sig_dates, ret_dates = self._get_dates(bgn_date, stp_date, calendar)
        self._load_signal(sig_bgn_date=sig_dates[0], sig_end_date=sig_dates[-1], predictions_dir=predictions_dir)
        self._align_signal(ret_dates, sig_dates)
        self._cal_weight_diff()
        self._load_ret(ret_bgn_date=ret_dates[0], ret_end_date=ret_dates[-1], diff_returns_dir=diff_returns_dir)
        self._check_shape()
        self._cal_ret(cost_rate)
        self._save(simulations_dir=simulations_dir, run_mode=run_mode)
        if verbose:
            print(
                f"\n{dt.datetime.now()} [INF] simulation for {SFG(self.model_id):.<48s}"
                f" from {SFG(bgn_date)} to {SFG(stp_date)} are calculated"
            )
        return 0


@qtimer
def cal_simulations_mclrn(
    call_multiprocess: bool, models_mclrn: list[CMclrnModel], proc_qty: int | None = None, **kwargs
):
    if call_multiprocess:
        pool = mp.Pool(processes=proc_qty) if proc_qty else mp.Pool()
        jobs = []
        for m in models_mclrn:
            s = CSimuMclrn(model=m)
            jobs.append(pool.apply_async(s.main, kwds=kwargs))
        pool.close()
        for job in track(jobs, description="Simulations for MclrnModels"):
            job.get()
        pool.join()
    else:
        for m in models_mclrn:
            s = CSimuMclrn(model=m)
            s.main(**kwargs)
    return 0


class CPortfolio(object):
    def __init__(
        self, portfolio_id: str, underlying_assets_ids: list[str], simulations_dir: str, portfolio_save_dir: str
    ):
        self.portfolio_id = portfolio_id
        self.underlying_assets_ids = underlying_assets_ids
        underlying_net_ret_data = {}
        for ml_model_id in self.underlying_assets_ids:
            lib_simu_reader = CLibSimu(simu_id=ml_model_id, lib_save_dir=simulations_dir).get_lib_reader()
            net_ret_df = lib_simu_reader.read(value_columns=["trade_date", "netRet"]).set_index("trade_date")
            underlying_net_ret_data[ml_model_id] = net_ret_df["netRet"]
        self.underlying_net_rets = pd.DataFrame(underlying_net_ret_data)
        self.net_ret: pd.Series = pd.Series(dtype=float)
        self.net_nav: pd.Series = pd.Series(dtype=float)
        self.portfolio_save_dir = portfolio_save_dir

    def cal_portfolio_ret_vanilla(self, bgn_date: str, stp_date: str):
        filter_dates = (self.underlying_net_rets.index >= bgn_date) & ((self.underlying_net_rets.index < stp_date))
        test_net_rets_df = self.underlying_net_rets.loc[filter_dates]
        self.net_ret = test_net_rets_df.mean(axis=1)
        self.net_nav = (self.net_ret + 1).cumprod()
        return 0

    @property
    def nav(self) -> pd.DataFrame:
        return pd.DataFrame({"net_ret": self.net_ret, "net_nav": self.net_nav})

    def save_portfolio(self, run_mode: str):
        lib_writer = CLibPortfolio(self.portfolio_id, self.portfolio_save_dir).get_lib_writer(run_mode=run_mode)
        lib_writer.update(update_df=self.nav, using_index=True)
        lib_writer.commit()
        lib_writer.close()
        return 0

    def main(self, bgn_date: str, stp_date: str, run_mode: str):
        self.cal_portfolio_ret_vanilla(bgn_date, stp_date)
        self.save_portfolio(run_mode=run_mode)
        print(f"[INF] {dt.datetime.now()} Portfolio of {SFG(self.portfolio_id)} are calculated")
        return 0


def create_portfolios(
    portfolios: dict[str, list[str]],
    bgn_date: str,
    stp_date: str,
    run_mode: str,
    simulations_dir: str,
    portfolio_save_dir: str,
    proc_qty: int | None = None,
):
    pool = mp.Pool(processes=proc_qty) if proc_qty else mp.Pool()
    for portfolio_id, underlying_asset_ids in portfolios.items():
        p = CPortfolio(
            portfolio_id=portfolio_id,
            underlying_assets_ids=underlying_asset_ids,
            simulations_dir=simulations_dir,
            portfolio_save_dir=portfolio_save_dir,
        )
        pool.apply_async(p.main, args=(bgn_date, stp_date, run_mode))
    pool.close()
    pool.join()
    return 0
