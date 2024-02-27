from husfort.qutility import get_twin_dir
import os
import platform

if platform.system() == "Windows":
    major_return_dir = r"E:\Deploy\Data\Futures\by_instrument"
    major_minor_dir = r"E:\Deploy\Data\Futures\by_instrument"
    market_data_dir = r"E:\Deploy\Data\Futures\by_instrument\by_instru_md"
    instru_info_path = r"E:\Deploy\Data\Futures\InstrumentInfo3.csv"
    calendar_path = r"E:\Deploy\Data\Calendar\cne_calendar.csv"
    instru_factor_exposure_dir = r"E:\Deploy\Data\ForProjects\cta3\factors_exposure\raw"
    available_universe_dir = r"E:\Deploy\Data\ForProjects\cta3\available_universe"
    save_root_dir = r"E:\ProjectsData"
else:
    major_return_dir = "/home/huxo/Deploy/Data/Futures/by_instrument"
    major_minor_dir = "/home/huxo/Deploy/Data/Futures/by_instrument"
    market_data_dir = "/home/huxo/Deploy/Data/Futures/by_instrument/by_instru_md"
    instru_info_path = "/home/huxo/Deploy/Data/Futures/InstrumentInfo3.csv"
    calendar_path = "/home/huxo/Deploy/Data/Calendar/cne_calendar.csv"
    instru_factor_exposure_dir = "/home/huxo/Deploy/Data/ForProjects/cta3/factors_exposure/raw"
    available_universe_dir = "/home/huxo/Deploy/Data/ForProjects/cta3/available_universe"
    save_root_dir = "/home/huxo/Research/Data"

project_save_dir = get_twin_dir(save_root_dir, src=".")
diff_returns_dir = os.path.join(project_save_dir, "diff_returns")
factors_exposure_dir = os.path.join(project_save_dir, "factors_exposure")
regroups_dir = os.path.join(project_save_dir, "regroups")
simulations_dir = os.path.join(project_save_dir, "simulations")
evaluations_dir = os.path.join(project_save_dir, "evaluations")

simulations_dir_quick = os.path.join(simulations_dir, "quick")
simulations_dir_mclrn = os.path.join(simulations_dir, "mclrn")
simulations_dir_portfolios = os.path.join(simulations_dir, "portfolios")

evaluations_dir_quick = os.path.join(evaluations_dir, "quick")
evaluations_dir_mclrn = os.path.join(evaluations_dir, "mclrn")
evaluations_dir_portfolios = os.path.join(evaluations_dir, "portfolios")

evaluations_dir_mclrn_by_instru_pair = os.path.join(evaluations_dir_mclrn, "by_instru_pair")
evaluations_dir_mclrn_by_mclrn_model = os.path.join(evaluations_dir_mclrn, "by_mclrn_model")

models_dir = os.path.join(project_save_dir, "models")
predictions_dir = os.path.join(project_save_dir, "predictions")
signals_dir = os.path.join(project_save_dir, "signals")
simulations_complex_dir = os.path.join(project_save_dir, "simulations_complex")

signals_dir_portfolios = os.path.join(signals_dir, "portfolios")
simulations_complex_dir_portfolios = os.path.join(simulations_complex_dir, "portfolios")

if __name__ == "__main__":
    from husfort.qutility import check_and_mkdir

    check_and_mkdir(project_save_dir)
    check_and_mkdir(diff_returns_dir)
    check_and_mkdir(factors_exposure_dir)
    check_and_mkdir(regroups_dir)
    check_and_mkdir(simulations_dir)
    check_and_mkdir(evaluations_dir)

    check_and_mkdir(simulations_dir_quick)
    check_and_mkdir(simulations_dir_mclrn)
    check_and_mkdir(simulations_dir_portfolios)

    check_and_mkdir(evaluations_dir_quick)
    check_and_mkdir(evaluations_dir_mclrn)
    check_and_mkdir(evaluations_dir_portfolios)

    check_and_mkdir(evaluations_dir_mclrn_by_instru_pair)
    check_and_mkdir(evaluations_dir_mclrn_by_mclrn_model)

    check_and_mkdir(models_dir)
    check_and_mkdir(predictions_dir)
    check_and_mkdir(signals_dir)
    check_and_mkdir(simulations_complex_dir)

    check_and_mkdir(signals_dir_portfolios)
    check_and_mkdir(simulations_complex_dir_portfolios)
