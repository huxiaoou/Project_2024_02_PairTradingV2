import os
import datetime as dt
import pandas as pd
import tqdm
from tqdm.contrib.itertools import product
from husfort.qutility import SFG
from husfort.qevaluation import CNAV
from husfort.qplot import CPlotLines
from cBasic import CInstruPair
from cSimulations import CLibSimu


def cal_evaluations(simu_id: str, bgn_date: str, stp_date: str, simulations_dir: dir) -> dict:
    lib_simu_reader = CLibSimu(simu_id=simu_id, lib_save_dir=simulations_dir).get_lib_reader()
    net_ret_df = lib_simu_reader.read_by_conditions(conditions=[
        ("trade_date", ">=", bgn_date),
        ("trade_date", "<", stp_date),
    ], value_columns=["trade_date", "netRet"]).set_index("trade_date")
    nav = CNAV(input_srs=net_ret_df["netRet"], input_type="RET")
    nav.cal_all_indicators()
    d = nav.to_dict(save_type="eng")
    return d


def cal_evaluations_quick(instruments_pairs: list[CInstruPair], diff_ret_delays: list[int], factors: list[str],
                          evaluations_dir: str, **kwargs):
    eval_results = []
    for (instru_pair, delay, factor) in product(instruments_pairs, diff_ret_delays, factors,
                                                desc="Evaluation", colour="#006400", ascii=" o-"):
        simu_id = f"{instru_pair.Id}.{factor}.T{delay}"
        d = cal_evaluations(simu_id=simu_id, **kwargs)
        d.update({
            "instru_pair": instru_pair.Id,
            "factor": factor,
            "delay": f"T{delay}",
        })
        eval_results.append(d)
    eval_results_df = pd.DataFrame(eval_results)
    eval_results_file = "eval.quick.csv"
    eval_results_path = os.path.join(evaluations_dir, eval_results_file)
    eval_results_df.to_csv(eval_results_path, index=False, float_format="%.8f")
    return 0


def get_top_factors_for_instruments_pairs(top: int, evaluations_dir: str) -> dict[tuple[CInstruPair, int], list[str]]:
    eval_results_file = "eval.quick.csv"
    eval_results_path = os.path.join(evaluations_dir, eval_results_file)
    eval_results_df = pd.read_csv(eval_results_path)
    eval_results_df.sort_values(by=["instru_pair", "delay", "sharpeRatio"], ascending=[True, True, False], inplace=True)
    top_factors = {}
    for (pair_id, delay), sub_df in eval_results_df.groupby(by=["instru_pair", "delay"]):
        selected_factors = sub_df["factor"].head(n=top).tolist() + sub_df["factor"].tail(n=top).tolist()
        instru_pair = CInstruPair(*pair_id.split("_"))
        top_factors[(instru_pair, int(delay[-1]))] = selected_factors
    return top_factors


def __plot_instru_simu_quick(instru_pair: CInstruPair, delay: int, factors: list[str],
                             bgn_date: str, stp_date: str, simulations_dir: str, plot_save_dir: str):
    nav_data = {}
    for factor in factors:
        simu_id = f"{instru_pair.Id}.{factor}.T{delay}"
        lib_simu_reader = CLibSimu(simu_id=simu_id, lib_save_dir=simulations_dir).get_lib_reader()
        net_ret_df = lib_simu_reader.read_by_conditions(conditions=[
            ("trade_date", ">=", bgn_date),
            ("trade_date", "<", stp_date),
        ], value_columns=["trade_date", "nav"]).set_index("trade_date")
        nav_data[factor] = net_ret_df["nav"]
    nav_df = pd.DataFrame(nav_data)
    size_factors = len(factors)
    artist = CPlotLines(
        line_width=1, fig_save_dir=plot_save_dir, fig_save_type="pdf",
        line_style=["-", "-."] * (size_factors // 2) + ["-"] * (size_factors % 2),
        fig_name=f"simu_quick.{instru_pair.Id}.T{delay}.top", plot_df=nav_df,
        colormap="jet",
    )
    artist.plot()
    return 0


def plot_instru_simu_quick(instruments_pairs: list[CInstruPair], diff_ret_delays: list[int],
                           top_factors: dict[tuple[CInstruPair, int], list[str]],
                           **kwargs):
    for (instru_pair, delay) in product(instruments_pairs, diff_ret_delays,
                                        colour="green", ascii=" o-", desc="Plot top factors"):
        factors = top_factors[(instru_pair, delay)]
        __plot_instru_simu_quick(instru_pair, delay, factors=factors, **kwargs)
    return 0

# def cal_evaluations_mclrn(headers_mclrn: list[tuple[str, str]], evaluations_dir: str, **kwargs):
#     eval_results = []
#     for ml_model_id, desc in headers_mclrn:
#         d = cal_evaluations(simu_id=ml_model_id, **kwargs)
#         d.update({"model_id": ml_model_id, "desc": desc})
#         eval_results.append(d)
#     eval_results_df = pd.DataFrame(eval_results)
#     eval_results_file = "eval.mclrn.csv"
#     eval_results_path = os.path.join(evaluations_dir, eval_results_file)
#     eval_results_df.to_csv(eval_results_path, index=False, float_format="%.8f")
#     return 0
#
#
# def plot_simu_mclrn(headers_mclrn: list[tuple[str, str]],
#                     bgn_date: str, stp_date: str, simulations_dir: str, plot_save_dir: str):
#     nav_data = {}
#     for ml_model_id, desc in headers_mclrn:
#         lib_simu_reader = CLibSimu(simu_id=ml_model_id, lib_save_dir=simulations_dir).get_lib_reader()
#         net_ret_df = lib_simu_reader.read_by_conditions(conditions=[
#             ("trade_date", ">=", bgn_date),
#             ("trade_date", "<", stp_date),
#         ], value_columns=["trade_date", "netRet"]).set_index("trade_date")
#         nav_data[f"{ml_model_id}-{desc}"] = (net_ret_df["netRet"] + 1).cumprod()
#     nav_df = pd.DataFrame(nav_data)
#     artist = CPlotLines(
#         line_width=1, fig_save_dir=plot_save_dir, fig_save_type="PNG",
#         line_style=["-"] * 10 + ["-."] * 10,
#         fig_name=f"simu_mclrn", plot_df=nav_df,
#         colormap="jet",
#     )
#     artist.plot()
#     return 0
