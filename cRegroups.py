from itertools import product
import datetime as dt
import pandas as pd
from husfort.qutility import SFG
from husfort.qcalendar import CCalendar
from husfort.qsqlite import CQuickSqliteLib, CLib1Tab1, CTable
from cBasic import CInstruPair
from cReturnsDiff import CLibDiffReturn
from cExposures import CLibFactorExposure


class CLibRegroups(CQuickSqliteLib):
    def __init__(self, instru_pair: CInstruPair, delay: int, lib_save_dir: str):
        self.instru_pair = instru_pair
        lib_name = f"regroups.{self.instru_pair.Id}.D{delay}.db"
        super().__init__(lib_name, lib_save_dir)

    def get_lib_struct(self) -> CLib1Tab1:
        return CLib1Tab1(
            lib_name=self.lib_name,
            table=CTable(
                {
                    "table_name": "regroups",
                    "primary_keys": {"trade_date": "TEXT", "factor": "TEXT"},
                    "value_columns": {"value": "REAL"},
                }
            )
        )


def cal_regroups(
        instru_pair: CInstruPair, delay: int,
        factors: list[str],
        run_mode: str, bgn_date: str, stp_date: str,
        diff_returns_dir: str,
        factors_exposure_dir: str,
        regroups_dir: str,
        calendar: CCalendar,

):
    diff_ret_iter_dates = calendar.get_iter_list(bgn_date, stp_date)
    fact_exp_iter_dates = calendar.shift_iter_dates(diff_ret_iter_dates, -delay)
    bridge_df = pd.DataFrame({"diff_ret_dates": diff_ret_iter_dates, "exposure_dates": fact_exp_iter_dates})

    # load factor exposure
    factors_exposure_dfs = []
    for factor in factors:
        lib_reader = CLibFactorExposure(factor, factors_exposure_dir).get_lib_reader()
        factor_exposure_df = lib_reader.read_by_conditions(conditions=[
            ("trade_date", ">=", fact_exp_iter_dates[0]),
            ("trade_date", "<=", fact_exp_iter_dates[-1]),
            ("pair", "=", instru_pair.Id),
        ], value_columns=["trade_date", "value"])
        factor_exposure_df["factor"] = factor
        factors_exposure_dfs.append(factor_exposure_df)
    factors_exposure_df = pd.concat(factors_exposure_dfs, axis=0, ignore_index=True)

    # align exposure
    merged_df = pd.merge(
        left=bridge_df, right=factors_exposure_df,
        left_on="exposure_dates", right_on="trade_date", how="right")
    aligned_exposure_df = merged_df[["diff_ret_dates", "factor", "value"]].rename(
        mapper={"diff_ret_dates": "trade_date"}, axis=1)

    # load diff return
    lib_reader = CLibDiffReturn(instru_pair, diff_returns_dir).get_lib_reader()
    diff_ret_df = lib_reader.read_by_conditions(conditions=[
        ("trade_date", ">=", diff_ret_iter_dates[0]),
        ("trade_date", "<=", diff_ret_iter_dates[-1]),
    ], value_columns=["trade_date", "diff_return"]).rename(mapper={"diff_return": "value"}, axis=1)
    diff_ret_df["factor"] = "diff_return"
    diff_ret_df = diff_ret_df[["trade_date", "factor", "value"]]

    # concat and sort
    update_df = pd.concat(objs=[aligned_exposure_df, diff_ret_df], axis=0, ignore_index=True)
    update_df.sort_values(by=["trade_date", "factor"], ascending=[True, True], inplace=True)

    # save to lib
    lib_regroups_writer = CLibRegroups(instru_pair, delay, regroups_dir).get_lib_writer(run_mode)
    lib_regroups_writer.update(update_df, using_index=False)
    lib_regroups_writer.commit()
    lib_regroups_writer.close()
    return 0


def cal_regroups_pairs(instruments_pairs: list[CInstruPair], diff_ret_delays: list[int], **kwargs):
    for (instru_pair, delay) in product(instruments_pairs, diff_ret_delays):
        cal_regroups(instru_pair, delay, **kwargs)
        print(f"{dt.datetime.now()} [INF] Pair {SFG(f'{instru_pair.Id}-T{delay}')} is calculated")
    return 0
