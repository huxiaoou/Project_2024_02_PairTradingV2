import os
import multiprocessing as mp
import pandas as pd
from itertools import product as ittl_prod
from rich.progress import track
from husfort.qevaluation import CNAV
from husfort.qplot import CPlotLines
from cBasic import CInstruPair
from cSimulations import CLibSimu, CLibPortfolio


def __cal_evaluations(simu_id: str, bgn_date: str, stp_date: str, simulations_dir: str, lib_type: str) -> dict:
    if lib_type.lower() == "simu":
        lib_simu_reader = CLibSimu(simu_id=simu_id, lib_save_dir=simulations_dir).get_lib_reader()
    elif lib_type.lower() == "portfolio":
        lib_simu_reader = CLibPortfolio(portfolio_id=simu_id, lib_save_dir=simulations_dir).get_lib_reader()
    else:
        raise ValueError(f"argument lib_type={lib_type} is wrong, please check again")
    net_ret_df = lib_simu_reader.read_by_conditions(
        conditions=[
            ("trade_date", ">=", bgn_date),
            ("trade_date", "<", stp_date),
        ],
        value_columns=["trade_date", "netRet"],
    ).set_index("trade_date")
    nav = CNAV(input_srs=net_ret_df["netRet"], input_type="RET")
    nav.cal_all_indicators()
    d = nav.to_dict(save_type="eng")
    return d


def cal_evaluations_quick(
    instruments_pairs: list[CInstruPair], diff_ret_delays: list[int], factors: list[str], evaluations_dir: str, **kwargs
):
    eval_results = []
    for instru_pair, delay, factor in track(
        list(ittl_prod(instruments_pairs, diff_ret_delays, factors)), description="Evaluation"
    ):
        simu_id = f"{instru_pair.Id}.{factor}.T{delay}"
        d = __cal_evaluations(simu_id=simu_id, lib_type="simu", **kwargs)
        d.update(
            {
                "instru_pair": instru_pair.Id,
                "factor": factor,
                "delay": f"T{delay}",
            }
        )
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


def __plot_instru_simu_quick(
    instru_pair: CInstruPair,
    delay: int,
    factors: list[str],
    bgn_date: str,
    stp_date: str,
    simulations_dir: str,
    plot_save_dir: str,
):
    nav_data = {}
    for factor in factors:
        simu_id = f"{instru_pair.Id}.{factor}.T{delay}"
        lib_simu_reader = CLibSimu(simu_id=simu_id, lib_save_dir=simulations_dir).get_lib_reader()
        net_ret_df = lib_simu_reader.read_by_conditions(
            conditions=[
                ("trade_date", ">=", bgn_date),
                ("trade_date", "<", stp_date),
            ],
            value_columns=["trade_date", "nav"],
        ).set_index("trade_date")
        nav_data[factor] = net_ret_df["nav"]
    nav_df = pd.DataFrame(nav_data)
    size_factors = len(factors)
    artist = CPlotLines(
        line_width=1,
        fig_save_dir=plot_save_dir,
        fig_save_type="pdf",
        line_style=["-", "-."] * (size_factors // 2) + ["-"] * (size_factors % 2),
        fig_name=f"simu_quick.{instru_pair.Id}.T{delay}.top",
        plot_df=nav_df,
        colormap="jet",
    )
    artist.plot()
    return 0


def plot_instru_simu_quick(
    instruments_pairs: list[CInstruPair],
    diff_ret_delays: list[int],
    top_factors: dict[tuple[CInstruPair, int], list[str]],
    **kwargs,
):
    for instru_pair, delay in track(
        list(ittl_prod(instruments_pairs, diff_ret_delays)), description="Plot top factors"
    ):
        factors = top_factors[(instru_pair, delay)]
        __plot_instru_simu_quick(instru_pair, delay, factors=factors, **kwargs)
    return 0


def __process_for_eval_mclrn_model(ml_model_id: str, desc: str, **kwargs) -> dict:
    instru_pair, delay, facs, win, mclrn, subargs = desc.split("-")
    d = __cal_evaluations(simu_id=ml_model_id, lib_type="simu", **kwargs)
    d.update(
        {
            "modelId": ml_model_id,
            "instruPair": instru_pair,
            "delay": delay,
            "facs": facs,
            "win": win,
            "mclrn": mclrn,
            "subArgs": subargs,
        }
    )
    return d


def __process_for_eval_portfolio(portfolio_id: str, **kwargs) -> dict:
    d = __cal_evaluations(simu_id=portfolio_id, lib_type="portfolio", **kwargs)
    d.update({"portfolioId": portfolio_id})
    return d


def eval_mclrn_models(headers_mclrn: list[tuple[str, str]], evaluations_dir_mclrn: str, **kwargs):
    pool = mp.Pool()
    async_results = []
    for ml_model_id, desc in track(headers_mclrn, description="Evaluation for Mclrn"):
        job = pool.apply_async(__process_for_eval_mclrn_model, args=(ml_model_id, desc), kwds=kwargs)
        async_results.append(job)
    pool.close()
    pool.join()
    eval_results = [job.get() for job in async_results]
    eval_results_df = pd.DataFrame(eval_results)
    eval_results_file = "eval.mclrn.csv"
    eval_results_path = os.path.join(evaluations_dir_mclrn, eval_results_file)
    eval_results_df.to_csv(eval_results_path, index=False, float_format="%.8f")

    # --- find best subargs for each Mclrn
    top_data = {}
    for (instru_pair, delay, facs, win), sub_df in eval_results_df.groupby(by=["instruPair", "delay", "facs", "win"]):
        res = {}
        for mclrn, mclrn_df in sub_df.groupby(by="mclrn"):
            res[mclrn] = mclrn_df.set_index("subArgs")["sharpeRatio"].astype(float).idxmax()
        top_data[(instru_pair, delay, facs, win)] = res
    top_df = pd.DataFrame.from_dict(top_data, orient="index")
    top_df["instru_pair"], top_df["delay"], top_df["facs"], top_df["win"] = zip(*top_df.index)
    top_file = "eval.mclrn.top.csv"
    top_path = os.path.join(evaluations_dir_mclrn, top_file)
    top_df.to_csv(top_path, index=False)
    return 0


def eval_portfolios(
    portfolios: dict[str, list[str]],
    evaluations_dir_portfolios: str,
    verbose: bool,
    **kwargs,
):
    pool = mp.Pool()
    async_results = []
    for portfolio_id in track(portfolios, description="Evaluation for Portfolios"):
        job = pool.apply_async(__process_for_eval_portfolio, args=(portfolio_id,), kwds=kwargs)
        async_results.append(job)
    pool.close()
    pool.join()
    eval_results = [job.get() for job in async_results]
    eval_results_df = pd.DataFrame(eval_results)
    eval_results_file = "eval.portfolios.csv"
    eval_results_path = os.path.join(evaluations_dir_portfolios, eval_results_file)
    eval_results_df.to_csv(eval_results_path, index=False, float_format="%.8f")
    if verbose:
        print(eval_results_df)
    return 0


def __process_for_plot_simu_mclrn(
    mclrn_model_ids: list[str],
    bgn_date: str,
    stp_date: str,
    simulations_dir: str,
    plot_save_id: str,
    plot_save_dir: str,
    lib_type: str,
):
    nav_data = {}
    for ml_model_id in mclrn_model_ids:
        if lib_type.lower() == "simu":
            lib_simu_reader = CLibSimu(simu_id=ml_model_id, lib_save_dir=simulations_dir).get_lib_reader()
        elif lib_type.lower() == "portfolio":
            lib_simu_reader = CLibPortfolio(portfolio_id=ml_model_id, lib_save_dir=simulations_dir).get_lib_reader()
        else:
            raise ValueError(f"argument lib_type={lib_type} is wrong, please check again")
        net_ret_df = lib_simu_reader.read_by_conditions(
            conditions=[
                ("trade_date", ">=", bgn_date),
                ("trade_date", "<", stp_date),
            ],
            value_columns=["trade_date", "nav"],
        ).set_index("trade_date")
        nav_data[ml_model_id] = net_ret_df["nav"]
    nav_df = pd.DataFrame(nav_data)
    artist = CPlotLines(
        line_width=1,
        fig_save_dir=plot_save_dir,
        fig_save_type="pdf",
        line_style=["-", "-."] * (int(len(mclrn_model_ids) / 2) + 1),
        fig_name=f"simu_mclrn_{plot_save_id}",
        plot_df=nav_df,
        colormap="jet",
    )
    artist.plot()
    return 0


def plot_simu_mclrn_with_top_sharpe_by_instru_pair(
    bgn_date: str,
    stp_date: str,
    simulations_dir: str,
    evaluations_dir: str,
    model_prototypes: list[str],
    plot_save_dir: str,
):
    top_file = "eval.mclrn.top.csv"
    top_path = os.path.join(evaluations_dir, top_file)
    top_df = pd.read_csv(top_path)

    pool = mp.Pool()
    for r in top_df.itertuples(index=False):
        instru_pair, delay, facs, win = (
            getattr(r, "instru_pair"),
            getattr(r, "delay"),
            getattr(r, "facs"),
            getattr(r, "win"),
        )
        fix_id = "-".join([instru_pair, delay, facs, win])
        mclrn_model_ids = [fix_id + f"-{k}-{getattr(r, k)}" for k in model_prototypes]
        pool.apply_async(
            __process_for_plot_simu_mclrn,
            args=(mclrn_model_ids, bgn_date, stp_date, simulations_dir, fix_id, plot_save_dir, "simu"),
        )
    pool.close()
    pool.join()
    return 0


def plot_simu_mclrn_with_top_sharpe_by_mclrn_model(
    bgn_date: str,
    stp_date: str,
    simulations_dir: str,
    evaluations_dir: str,
    model_prototypes: list[str],
    plot_save_dir: str,
):
    top_file = "eval.mclrn.top.csv"
    top_path = os.path.join(evaluations_dir, top_file)
    top_df = pd.read_csv(top_path)

    pool = mp.Pool()
    for (delay, facs, win), sub_df in top_df.groupby(by=["delay", "facs", "win"]):
        for prototype in model_prototypes:
            mclrn_model_ids = []
            for instru_pair, subargs in zip(sub_df["instru_pair"], sub_df[prototype]):
                fix_id = "-".join([instru_pair, delay, facs, win])
                mclrn_model_ids.append(fix_id + f"-{prototype}-{subargs}")
            save_id = "-".join([prototype, delay, facs, win])
            pool.apply_async(
                __process_for_plot_simu_mclrn,
                args=(mclrn_model_ids, bgn_date, stp_date, simulations_dir, save_id, plot_save_dir, "simu"),
            )
    pool.close()
    pool.join()
    return 0


def plot_portfolios(
    portfolios: dict[str, list[str]],
    save_id: str,
    bgn_date: str,
    stp_date: str,
    simulations_dir: str,
    plot_save_dir: str,
):
    mclrn_model_ids = list(portfolios)
    __process_for_plot_simu_mclrn(
        mclrn_model_ids, bgn_date, stp_date, simulations_dir, save_id, plot_save_dir, "portfolio"
    )
    return 0
