import argparse


def parse_project_args():
    parser_main = argparse.ArgumentParser(description="Entry point of this project")
    parsers_sub = parser_main.add_subparsers(
        title="sub argument:switch",
        dest="switch",
        description="use this argument to go to call different functions",
    )

    # diff return
    parser_sub = parsers_sub.add_parser(name="diff", help="Calculate difference return of pair instruments")
    parser_sub.add_argument("--mode", type=str, help="overwrite or append", choices=("o", "a"), required=True)
    parser_sub.add_argument("--bgn", type=str, help="begin date, format = [YYYYMMDD]", required=True)
    parser_sub.add_argument("--stp", type=str, help="stop  date, format = [YYYYMMDD]")

    # factor exposure
    parser_sub = parsers_sub.add_parser(name="exposure", help="Calculate factor exposure")
    parser_sub.add_argument(
        "--factor",
        type=str,
        help="which factor to calculate",
        choices=(
            "lag",
            "sum",
            "ewm",
            "vol",
            "tnr",
            "basisa",
            "ctp",
            "cvp",
            "csp",
            "rsbr",
            "rslr",
            "skew",
            "mtms",
            "tsa",
            "tsld",
        ),
        required=True,
    )
    parser_sub.add_argument("--mode", type=str, help="overwrite or append", choices=("o", "a"), required=True)
    parser_sub.add_argument("--bgn", type=str, help="begin date, format = [YYYYMMDD]", required=True)
    parser_sub.add_argument("--stp", type=str, help="stop  date, format = [YYYYMMDD]")

    # regroup: diff ret and factor exposure
    parser_sub = parsers_sub.add_parser(name="regroups", help="Regroup factor exposure and diff return")
    parser_sub.add_argument("--mode", type=str, help="overwrite or append", choices=("o", "a"), required=True)
    parser_sub.add_argument("--bgn", type=str, help="begin date, format = [YYYYMMDD]", required=True)
    parser_sub.add_argument("--stp", type=str, help="stop  date, format = [YYYYMMDD]", required=True)

    # simulation quick
    parser_sub = parsers_sub.add_parser(name="simu-quick", help="Quick simulation for given factor and pairs")
    parser_sub.add_argument("--mode", type=str, help="overwrite or append", choices=("o", "a"), required=True)
    parser_sub.add_argument("--bgn", type=str, help="begin date, format = [YYYYMMDD]", required=True)
    parser_sub.add_argument("--stp", type=str, help="stop  date, format = [YYYYMMDD]", required=True)
    parser_sub.add_argument("--process", type=int, default=None, help="number of process")

    # evaluation quick
    parser_sub = parsers_sub.add_parser(name="eval-quick", help="Quick evaluation for given factor and pairs")
    parser_sub.add_argument("--bgn", type=str, help="begin date, format = [YYYYMMDD]", required=True)
    parser_sub.add_argument("--stp", type=str, help="stop  date, format = [YYYYMMDD]", required=True)
    parser_sub.add_argument("--top", type=int, help="factors with top n and bottom sharpe will be plot", default=5)

    # ------------------------
    # --- machine learning ---
    # ------------------------

    # train and predict
    parser_sub = parsers_sub.add_parser(name="mclrn", help="machine learning")
    parser_sub.add_argument("--mode", type=str, help="overwrite or append", choices=("o", "a"), required=True)
    parser_sub.add_argument("--bgn", type=str, help="begin date, format = [YYYYMMDD]", required=True)
    parser_sub.add_argument("--stp", type=str, help="stop  date, format = [YYYYMMDD]", required=True)
    parser_sub.add_argument("--process", type=int, default=None, help="number of process")

    # simulation
    parser_sub = parsers_sub.add_parser(name="simu-mclrn", help="simulation for machine learning")
    parser_sub.add_argument("--mode", type=str, help="overwrite or append", choices=("o", "a"), required=True)
    parser_sub.add_argument("--bgn", type=str, help="begin date, format = [YYYYMMDD]", required=True)
    parser_sub.add_argument("--stp", type=str, help="stop  date, format = [YYYYMMDD]", required=True)
    parser_sub.add_argument("--process", type=int, default=None, help="number of process")

    # evaluation
    parser_sub = parsers_sub.add_parser(name="eval-mclrn", help="evaluation for machine learning")
    parser_sub.add_argument("--bgn", type=str, help="begin date, format = [YYYYMMDD]", required=True)
    parser_sub.add_argument("--stp", type=str, help="stop  date, format = [YYYYMMDD]", required=True)

    # ------------------------
    # ------ portfolios ------
    # ------------------------
    # simulation
    parser_sub = parsers_sub.add_parser(name="simu-portfolios", help="simulations for portfolios")
    parser_sub.add_argument("--mode", type=str, help="overwrite or append", choices=("o", "a"), required=True)
    parser_sub.add_argument("--bgn", type=str, help="begin date, format = [YYYYMMDD]", required=True)
    parser_sub.add_argument("--stp", type=str, help="stop  date, format = [YYYYMMDD]", required=True)
    parser_sub.add_argument("--process", type=int, default=None, help="number of process")

    # evaluation
    parser_sub = parsers_sub.add_parser(name="eval-portfolios", help="evaluations for portfolios")
    parser_sub.add_argument("--bgn", type=str, help="begin date, format = [YYYYMMDD]", required=True)
    parser_sub.add_argument("--stp", type=str, help="stop  date, format = [YYYYMMDD]", required=True)

    # complex signals
    parser_sub = parsers_sub.add_parser(name="sig-complex-portfolios", help="complex signals for portfolios")
    parser_sub.add_argument("--mode", type=str, help="overwrite or append", choices=("o", "a"), required=True)
    parser_sub.add_argument("--bgn", type=str, help="begin date, format = [YYYYMMDD]", required=True)
    parser_sub.add_argument("--stp", type=str, help="stop  date, format = [YYYYMMDD]", required=True)
    parser_sub.add_argument("--process", type=int, default=None, help="number of process")

    # complex simulations
    parser_sub = parsers_sub.add_parser(name="simu-complex-portfolios", help="complex simulations for portfolios")
    parser_sub.add_argument("--bgn", type=str, help="begin date, format = [YYYYMMDD]", required=True)
    parser_sub.add_argument("--stp", type=str, help="stop  date, format = [YYYYMMDD]", required=True)
    parser_sub.add_argument("--process", type=int, default=None, help="number of process")

    # evaluation
    parser_sub = parsers_sub.add_parser(name="eval-complex-portfolios", help="complex evaluations for portfolios")
    parser_sub.add_argument("--bgn", type=str, help="begin date, format = [YYYYMMDD]", required=True)
    parser_sub.add_argument("--stp", type=str, help="stop  date, format = [YYYYMMDD]", required=True)

    return parser_main.parse_args()


if __name__ == "__main__":
    args = parse_project_args()
    if args.switch == "diff":
        from project_config import instruments_pairs
        from project_setup import diff_returns_dir, major_return_dir
        from cReturnsDiff import cal_diff_returns_pairs

        cal_diff_returns_pairs(
            instruments_pairs=instruments_pairs,
            major_return_save_dir=major_return_dir,
            run_mode=args.mode,
            bgn_date=args.bgn,
            stp_date=args.stp,
            diff_returns_dir=diff_returns_dir,
        )
    elif args.switch == "exposure":
        from husfort.qcalendar import CCalendar
        from project_setup import factors_exposure_dir, diff_returns_dir, instru_factor_exposure_dir, calendar_path
        from project_config import instruments_pairs, config_factor, CCfgFactorMA, CCfgFactorEWM, CCfgFactor

        calendar = CCalendar(calendar_path)
        factor_cfgs = config_factor[args.factor]
        if args.factor == "lag":
            from cExposures import CFactorExposureLag

            if isinstance(factor_cfgs, CCfgFactor):
                for lag, factor in zip(factor_cfgs.args, factor_cfgs.get_factors()):
                    factor_exposure = CFactorExposureLag(
                        lag=lag,
                        factor=factor,
                        diff_returns_dir=diff_returns_dir,
                        factors_exposure_dir=factors_exposure_dir,
                        instruments_pairs=instruments_pairs,
                    )
                    factor_exposure.main(run_mode=args.mode, bgn_date=args.bgn, stp_date=args.stp, calendar=calendar)
            else:
                raise TypeError
        elif args.factor == "sum":
            from cExposures import CFactorExposureSUM

            if isinstance(factor_cfgs, CCfgFactor):
                for win, factor in zip(factor_cfgs.args, factor_cfgs.get_factors()):
                    factor_exposure = CFactorExposureSUM(
                        win=win,
                        factor=factor,
                        diff_returns_dir=diff_returns_dir,
                        factors_exposure_dir=factors_exposure_dir,
                        instruments_pairs=instruments_pairs,
                    )
                    factor_exposure.main(run_mode=args.mode, bgn_date=args.bgn, stp_date=args.stp, calendar=calendar)
            else:
                raise TypeError
        elif args.factor == "ewm":
            from cExposures import CFactorExposureEWM

            if isinstance(factor_cfgs, CCfgFactorEWM):
                for (fast, slow), factor in zip(factor_cfgs.args, factor_cfgs.get_factors()):
                    factor_exposure = CFactorExposureEWM(
                        fast=fast,
                        slow=slow,
                        fix_base_date=factor_cfgs.fixed_bgn_date,
                        factor=factor,
                        diff_returns_dir=diff_returns_dir,
                        factors_exposure_dir=factors_exposure_dir,
                        instruments_pairs=instruments_pairs,
                    )
                    factor_exposure.main(run_mode=args.mode, bgn_date=args.bgn, stp_date=args.stp, calendar=calendar)
            else:
                raise TypeError
        elif args.factor == "vol":
            from cExposures import CFactorExposureVOL

            if isinstance(factor_cfgs, CCfgFactor):
                for win, factor in zip(factor_cfgs.args, factor_cfgs.get_factors()):
                    factor_exposure = CFactorExposureVOL(
                        win=win,
                        factor=factor,
                        diff_returns_dir=diff_returns_dir,
                        factors_exposure_dir=factors_exposure_dir,
                        instruments_pairs=instruments_pairs,
                    )
                    factor_exposure.main(run_mode=args.mode, bgn_date=args.bgn, stp_date=args.stp, calendar=calendar)
            else:
                raise TypeError
        elif args.factor == "tnr":
            from cExposures import CFactorExposureTNR

            if isinstance(factor_cfgs, CCfgFactor):
                for win, factor in zip(factor_cfgs.args, factor_cfgs.get_factors()):
                    factor_exposure = CFactorExposureTNR(
                        win=win,
                        factor=factor,
                        diff_returns_dir=diff_returns_dir,
                        factors_exposure_dir=factors_exposure_dir,
                        instruments_pairs=instruments_pairs,
                    )
                    factor_exposure.main(run_mode=args.mode, bgn_date=args.bgn, stp_date=args.stp, calendar=calendar)
            else:
                raise TypeError
        elif args.factor in ["basisa", "ctp", "cvp", "csp", "rsbr", "rslr", "skew", "mtms", "tsa", "tsld"]:
            from cExposures import CFactorExposureFromInstruExposureDiff

            if isinstance(factor_cfgs, CCfgFactorMA):
                for factor_exo, factor in zip(factor_cfgs.get_factors_raw(), factor_cfgs.get_factors()):
                    factor_exposure = CFactorExposureFromInstruExposureDiff(
                        factor_exo=factor_exo,
                        win_mov_ave=factor_cfgs.win_mov_ave,
                        instru_factor_exposure_dir=instru_factor_exposure_dir,
                        factor=factor,
                        factors_exposure_dir=factors_exposure_dir,
                        instruments_pairs=instruments_pairs,
                    )
                    factor_exposure.main(run_mode=args.mode, bgn_date=args.bgn, stp_date=args.stp, calendar=calendar)
            else:
                raise TypeError

        else:
            print(f"... [ERR] factor = {args.factor}")
            raise ValueError
    elif args.switch == "regroups":
        from husfort.qcalendar import CCalendar
        from project_setup import factors_exposure_dir, diff_returns_dir, regroups_dir, calendar_path
        from project_config import instruments_pairs, factors, diff_ret_delays
        from cRegroups import cal_regroups_pairs

        calendar = CCalendar(calendar_path)
        cal_regroups_pairs(
            instruments_pairs=instruments_pairs,
            diff_ret_delays=diff_ret_delays,
            factors=factors,
            run_mode=args.mode,
            bgn_date=args.bgn,
            stp_date=args.stp,
            diff_returns_dir=diff_returns_dir,
            factors_exposure_dir=factors_exposure_dir,
            regroups_dir=regroups_dir,
            calendar=calendar,
        )
    elif args.switch == "simu-quick":
        from project_setup import regroups_dir, simulations_dir_quick
        from project_config import cost_rate_quick, instruments_pairs, factors, diff_ret_delays
        from cSimulations import cal_simulations_instruments_pairs

        cal_simulations_instruments_pairs(
            proc_qty=args.process,
            instruments_pairs=instruments_pairs,
            diff_ret_delays=diff_ret_delays,
            run_mode=args.mode,
            bgn_date=args.bgn,
            stp_date=args.stp,
            factors=factors,
            cost_rate=cost_rate_quick,
            regroups_dir=regroups_dir,
            simulations_dir=simulations_dir_quick,
        )
    elif args.switch == "eval-quick":
        from project_setup import simulations_dir_quick, evaluations_dir_quick
        from project_config import instruments_pairs, factors, diff_ret_delays
        from cEvaluations import cal_evaluations_quick, get_top_factors_for_instruments_pairs, plot_instru_simu_quick

        cal_evaluations_quick(
            instruments_pairs=instruments_pairs,
            diff_ret_delays=diff_ret_delays,
            factors=factors,
            bgn_date=args.bgn,
            stp_date=args.stp,
            evaluations_dir=evaluations_dir_quick,
            simulations_dir=simulations_dir_quick,
        )
        top_factors = get_top_factors_for_instruments_pairs(top=args.top, evaluations_dir=evaluations_dir_quick)
        plot_instru_simu_quick(
            instruments_pairs=instruments_pairs,
            diff_ret_delays=diff_ret_delays,
            top_factors=top_factors,
            bgn_date=args.bgn,
            stp_date=args.stp,
            plot_save_dir=evaluations_dir_quick,
            simulations_dir=simulations_dir_quick,
        )
    elif args.switch == "mclrn":
        from project_setup import models_dir, predictions_dir, regroups_dir, calendar_path
        from project_config_mclrn import models_mclrn
        from husfort.qcalendar import CCalendar
        from cMclrn import cal_mclrn_train_and_predict

        calendar = CCalendar(calendar_path)
        cal_mclrn_train_and_predict(
            call_multiprocess=True,
            models_mclrn=models_mclrn,
            proc_qty=args.process,
            run_mode=args.mode,
            bgn_date=args.bgn,
            stp_date=args.stp,
            calendar=calendar,
            regroups_dir=regroups_dir,
            models_dir=models_dir,
            predictions_dir=predictions_dir,
        )
    elif args.switch == "simu-mclrn":
        from project_setup import predictions_dir, diff_returns_dir, simulations_dir_mclrn, calendar_path
        from project_config import cost_rate_mclrn
        from project_config_mclrn import models_mclrn
        from husfort.qcalendar import CCalendar
        from cSimulations import cal_simulations_mclrn

        calendar = CCalendar(calendar_path)
        cal_simulations_mclrn(
            call_multiprocess=True,
            models_mclrn=models_mclrn,
            proc_qty=args.process,
            run_mode=args.mode,
            bgn_date=args.bgn,
            stp_date=args.stp,
            calendar=calendar,
            cost_rate=cost_rate_mclrn,
            predictions_dir=predictions_dir,
            diff_returns_dir=diff_returns_dir,
            simulations_dir=simulations_dir_mclrn,
        )
    elif args.switch == "eval-mclrn":
        from project_setup import (
            simulations_dir_mclrn,
            evaluations_dir_mclrn,
            evaluations_dir_mclrn_by_instru_pair,
            evaluations_dir_mclrn_by_mclrn_model,
        )
        from project_config_mclrn import models_mclrn, headers_mclrn, model_prototypes
        from cEvaluations import (
            eval_mclrn_models,
            plot_simu_mclrn_with_top_sharpe_by_instru_pair,
            plot_simu_mclrn_with_top_sharpe_by_mclrn_model,
        )

        eval_mclrn_models(
            headers_mclrn=headers_mclrn,
            bgn_date=args.bgn,
            stp_date=args.stp,
            evaluations_dir_mclrn=evaluations_dir_mclrn,
            simulations_dir=simulations_dir_mclrn,
        )
        plot_simu_mclrn_with_top_sharpe_by_instru_pair(
            bgn_date=args.bgn,
            stp_date=args.stp,
            simulations_dir=simulations_dir_mclrn,
            evaluations_dir=evaluations_dir_mclrn,
            model_prototypes=model_prototypes,
            plot_save_dir=evaluations_dir_mclrn_by_instru_pair,
        )
        plot_simu_mclrn_with_top_sharpe_by_mclrn_model(
            bgn_date=args.bgn,
            stp_date=args.stp,
            simulations_dir=simulations_dir_mclrn,
            evaluations_dir=evaluations_dir_mclrn,
            model_prototypes=model_prototypes,
            plot_save_dir=evaluations_dir_mclrn_by_mclrn_model,
        )
    elif args.switch == "simu-portfolios":
        from project_setup import simulations_dir_mclrn, simulations_dir_portfolios
        from project_config_portfolio import portfolios
        from cSimulations import create_portfolios

        create_portfolios(
            portfolios=portfolios,
            bgn_date=args.bgn,
            stp_date=args.stp,
            run_mode=args.mode,
            simulations_dir=simulations_dir_mclrn,
            portfolio_save_dir=simulations_dir_portfolios,
        )
    elif args.switch == "eval-portfolios":
        from project_setup import simulations_dir_portfolios, evaluations_dir_portfolios
        from project_config_portfolio import portfolios
        from cEvaluations import eval_portfolios, plot_portfolios_db

        eval_portfolios(
            portfolios=portfolios,
            evaluations_dir_portfolios=evaluations_dir_portfolios,
            verbose=True,
            simu_save_type="db",
            eval_save_id="simple",
            bgn_date=args.bgn,
            stp_date=args.stp,
            simulations_dir=simulations_dir_portfolios,
        )
        plot_portfolios_db(
            portfolios=portfolios,
            save_id="portfolios.simple",
            bgn_date=args.bgn,
            stp_date=args.stp,
            simulations_dir=simulations_dir_portfolios,
            plot_save_dir=evaluations_dir_portfolios,
        )
    elif args.switch == "sig-complex-portfolios":
        from project_setup import predictions_dir, signals_dir_portfolios
        from cPortfolios import cal_portfolio_signals
        from project_config_portfolio import portfolios

        cal_portfolio_signals(
            call_multiprocess=True,
            proc_qty=args.process,
            portfolios=portfolios,
            bgn_date=args.bgn,
            stp_date=args.stp,
            run_mode=args.mode,
            predictions_dir=predictions_dir,
            signals_dir=signals_dir_portfolios,
        )
    elif args.switch == "simu-complex-portfolios":
        from project_setup import (
            calendar_path,
            instru_info_path,
            major_minor_dir,
            market_data_dir,
            available_universe_dir,
        )
        from project_setup import signals_dir_portfolios, simulations_complex_dir_portfolios
        from project_config_portfolio import portfolios, cost_rate, init_cash
        from husfort.qsimulation import cal_multiple_complex_simulations
        from cPortfolios import get_universe

        universe = get_universe(portfolios=portfolios)
        cal_multiple_complex_simulations(
            signal_ids=list(portfolios),
            universe=universe,
            init_cash=init_cash,
            cost_rate=cost_rate,
            simu_bgn_date=args.bgn,
            simu_stp_date=args.stp,
            signals_dir=signals_dir_portfolios,
            simulations_save_dir=simulations_complex_dir_portfolios,
            calendar_path=calendar_path,
            instru_info_path=instru_info_path,
            market_data_dir=market_data_dir,
            major_minor_dir=major_minor_dir,
            available_universe_dir=available_universe_dir,
            call_multiprocess=True,
            proc_qty=args.process,
            save_trades_and_positions=True,
        )
    elif args.switch == "eval-complex-portfolios":
        from project_setup import simulations_complex_dir_portfolios, evaluations_dir_portfolios
        from project_config_portfolio import portfolios
        from cEvaluations import eval_portfolios, plot_portfolios_csv

        eval_portfolios(
            portfolios=portfolios,
            evaluations_dir_portfolios=evaluations_dir_portfolios,
            verbose=True,
            simu_save_type="csv",
            eval_save_id="complex",
            bgn_date=args.bgn,
            stp_date=args.stp,
            simulations_dir=simulations_complex_dir_portfolios,
        )
        plot_portfolios_csv(
            portfolios=portfolios,
            save_id="portfolios.complex",
            bgn_date=args.bgn,
            stp_date=args.stp,
            simulations_dir=simulations_complex_dir_portfolios,
            plot_save_dir=evaluations_dir_portfolios,
        )
    else:
        raise ValueError("Not a right input for subparser")
