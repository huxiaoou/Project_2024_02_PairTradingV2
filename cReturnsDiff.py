import datetime as dt
import pandas as pd
from husfort.qutility import SFG
from husfort.qsqlite import CLibMajorReturn, CQuickSqliteLib, CLib1Tab1, CTable
from cBasic import CInstruPair


class CLibDiffReturn(CQuickSqliteLib):
    def __init__(self, instru_pair: CInstruPair, lib_save_dir: str):
        self.instru_pair = instru_pair
        lib_name = f"diff_return.{self.instru_pair.Id}.db"
        super().__init__(lib_name, lib_save_dir)

    def get_lib_struct(self) -> CLib1Tab1:
        return CLib1Tab1(
            lib_name=self.lib_name,
            table=CTable(
                {
                    "table_name": "diff_return",
                    "primary_keys": {"trade_date": "TEXT"},
                    "value_columns": {
                        "instru_a": "REAL",
                        "instru_b": "REAL",
                        "diff_return": "REAL",
                    },
                }
            ),
        )


def load_major_return(instru: str, bgn_date: str, stp_date: str, major_return_save_dir: str) -> pd.DataFrame:
    lib_reader = CLibMajorReturn(instrument=instru, lib_save_dir=major_return_save_dir).get_lib_reader()
    return_df = (
        lib_reader.read_by_conditions(
            conditions=[
                ("trade_date", ">=", bgn_date),
                ("trade_date", "<", stp_date),
            ],
            value_columns=["trade_date", "major_return"],
        )
        .set_index("trade_date")
        .rename(mapper={"major_return": instru}, axis=1)
    )
    lib_reader.close()
    return return_df


def cal_diff_returns(
    instru_pair: CInstruPair,
    major_return_save_dir: str,
    run_mode: str,
    bgn_date: str,
    stp_date: str,
    diff_returns_dir: str,
):
    instru_a, instru_b = instru_pair.instru_a, instru_pair.instru_b
    return_df_a = load_major_return(instru_a, bgn_date, stp_date, major_return_save_dir)
    return_df_b = load_major_return(instru_b, bgn_date, stp_date, major_return_save_dir)
    if len(return_df_a) != len(return_df_b):
        print(
            f"\n{dt.datetime.now()} [WRN] "
            f"length of {instru_a} = {len(return_df_a)} != length of {instru_b} = {len(return_df_b)}"
        )

    diff_return_df = pd.merge(left=return_df_a, right=return_df_b, left_index=True, right_index=True, how="outer")
    if len(return_df_a) != len(diff_return_df):
        print(f"\n{dt.datetime.now()} [ERR] length of {instru_a} != length of diff returns")
    diff_return_df.fillna(value=0, inplace=True)
    diff_return_df["diff_return"] = (diff_return_df[instru_a] - diff_return_df[instru_b]) * 0.5
    lib_writer = CLibDiffReturn(instru_pair, diff_returns_dir).get_lib_writer(run_mode=run_mode)
    lib_writer.update(update_df=diff_return_df, using_index=True)
    lib_writer.commit()
    lib_writer.close()
    print(f"{dt.datetime.now()} [INF] diff return of {SFG(f'{instru_pair.Id:>16s}')} are calculated", end="\r")
    return 0


def cal_diff_returns_pairs(instruments_pairs: list[CInstruPair], **kwargs):
    for instru_pair in instruments_pairs:
        cal_diff_returns(instru_pair=instru_pair, **kwargs)
    return 0
