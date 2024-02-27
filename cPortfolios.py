import multiprocessing as mp
from tkinter import NO
import numpy as np
import pandas as pd
from rich.progress import track
from husfort.qsqlite import CLibFactor
from cMclrn import CLibPredictions


class CPortfolioSignals:
    def __init__(self, portfolio_id: str, underlying_asset_ids: list[str], predictions_dir: str, signals_dir: str):
        self.portfolio_id = portfolio_id
        self.underlying_asset_ids = underlying_asset_ids
        self.predictions_dir = predictions_dir
        self.signals_dir = signals_dir
        self.signals: pd.DataFrame = pd.DataFrame()

    def load_mclrn_model_predictions(self, underlying_asset_id: str, bgn_date: str, stp_date: str) -> pd.DataFrame:
        lib_model_reader = CLibPredictions(
            model_id=underlying_asset_id, lib_save_dir=self.predictions_dir
        ).get_lib_reader()
        predictions = lib_model_reader.read_by_conditions(
            conditions=[
                ("trade_date", ">=", bgn_date),
                ("trade_date", "<", stp_date),
            ],
            value_columns=["trade_date", "value"],
        ).set_index("trade_date")
        return predictions

    def convert_predictions_to_instru_weight(self, underlying_asset_id: str, predictions: pd.DataFrame) -> pd.DataFrame:
        def __translate(sign: int) -> tuple[float, float]:
            if sign > 0:
                return (0.5, -0.5)
            elif sign < 0:
                return (-0.5, 0.5)
            else:
                return (0.0, 0.0)

        # "JM.DCE_I.DCE-T2-F50-W03-Ridge-A05"
        instru_pair, _, _, _, prototype, _ = underlying_asset_id.split("-")
        if prototype in ["Ridge"]:
            predictions["sign"] = predictions["value"].map(lambda z: np.sign(z))
        elif prototype in ["Logistic", "Svm"]:
            predictions["sign"] = predictions["value"].map(lambda z: 2 * z - 1)
        else:
            raise ValueError(f"prototype = {prototype} is not defined when translating to signals")
        instru_a, instru_b = instru_pair.split("_")
        predictions[instru_a], predictions[instru_b] = zip(*predictions["sign"].map(__translate))
        instru_weight = predictions[[instru_a, instru_b]].stack()
        instru_weight = instru_weight.reset_index().rename(mapper={"level_1": "instru", 0: "weight"}, axis=1)
        return instru_weight

    def merge_model(self, dfs: list[pd.DataFrame]):
        concat_df = pd.concat(dfs, axis=0, ignore_index=True)
        pivot_df = pd.pivot_table(data=concat_df, index="trade_date", columns="instru", values="weight", aggfunc="sum")
        pivot_df.fillna(0, inplace=True)
        instru_wgt_abs_sum = pivot_df.abs().sum(axis=1)
        weight_by_dates = pivot_df.div(instru_wgt_abs_sum, axis=0)
        self.signals = weight_by_dates.stack().reset_index().sort_values(by=["trade_date", "instru"])
        return 0

    def save(self, run_mode: str):
        lib_sig_writer = CLibFactor(factor=self.portfolio_id, lib_save_dir=self.signals_dir).get_lib_writer(
            run_mode=run_mode
        )
        lib_sig_writer.update(update_df=self.signals, using_index=False)
        lib_sig_writer.commit()
        lib_sig_writer.close()
        return 0

    def main(self, bgn_date: str, stp_date: str, run_mode: str):
        instru_weights: list[pd.DataFrame] = []
        for underlying_asset_id in self.underlying_asset_ids:
            predictions = self.load_mclrn_model_predictions(underlying_asset_id, bgn_date, stp_date)
            instru_weight = self.convert_predictions_to_instru_weight(underlying_asset_id, predictions=predictions)
            instru_weights.append(instru_weight)
        self.merge_model(instru_weights)
        self.save(run_mode=run_mode)
        return 0


def cal_portfolio_signals(
    portfolios: dict[str, list[str]],
    bgn_date: str,
    stp_date: str,
    run_mode: str,
    predictions_dir: str,
    signals_dir: str,
    call_multiprocess: bool,
    proc_qty: int | None,
):
    signals = [
        CPortfolioSignals(
            portfolio_id=portfolio_id,
            underlying_asset_ids=underlying_asset_ids,
            predictions_dir=predictions_dir,
            signals_dir=signals_dir,
        )
        for portfolio_id, underlying_asset_ids in portfolios.items()
    ]
    if call_multiprocess:
        pool = mp.Pool(processes=proc_qty) if proc_qty else mp.Pool()
        for signal in track(signals, description=f"Generating complex signals"):
            pool.apply_async(signal.main, args=(bgn_date, stp_date, run_mode))
        pool.close()
        pool.join()
    else:
        for signal in track(signals, description=f"Generating complex signals"):
            signal.main(bgn_date=bgn_date, stp_date=stp_date, run_mode=run_mode)
    return 0


def get_universe(portfolios: dict[str, list[str]]) -> list[str]:
    universe = []
    for _, underlying_asset_ids in portfolios.items():
        for underlying_asset_id in underlying_asset_ids:
            instru_pair = underlying_asset_id.split("-")[0]
            universe += instru_pair.split("_")
    return list(set(universe))


