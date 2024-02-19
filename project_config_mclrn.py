from project_config import instruments_pairs, factors
from cMclrn import (CMclrnModel, CMclrnBatchRidge, CMclrnBatchLogistic)
from cEvaluations import get_top_factors_for_instruments_pairs
from project_setup import evaluations_dir_quick

# --- init
models_mclrn: list[CMclrnModel] = []
delays = [2]
trn_wins = [3, 6, 12]
top = 5
ridge_alphas = [(0.01, 0.1, 1.0, 10.0, 100)]
logistic_cvs = [3, 5, 10]

# --- load top factors
top_factors = get_top_factors_for_instruments_pairs(top=top, evaluations_dir=evaluations_dir_quick)

# --- add Ridge
batch_generator = CMclrnBatchRidge(ridge_alphas=ridge_alphas,
                                   instruments_pairs=instruments_pairs, delays=delays, trn_wins=trn_wins,
                                   all_factors=factors, top_factors=top_factors)
batch_generator.append_batch(models_mclrn=models_mclrn)

# --- add Logistic
batch_generator = CMclrnBatchLogistic(cvs=logistic_cvs,
                                      instruments_pairs=instruments_pairs, delays=delays, trn_wins=trn_wins,
                                      all_factors=factors, top_factors=top_factors)
batch_generator.append_batch(models_mclrn=models_mclrn)

# models_mclrn: list[CMclrnModel] = [
#     CMclrnMlp(
#         model_id="M02", desc="MultiLayerPerception",
#         pairs=instruments_pairs, delay=2, factors=factors, y_lbl="diff_return",
#         sig_method="binary", trn_win=3
#     ),
#     CMclrnSvc(
#         model_id="M03", desc="SupportVectorMachine",
#         pairs=instruments_pairs, delay=2, factors=factors, y_lbl="diff_return",
#         sig_method="binary", trn_win=3
#     ),
#     CMclrnDt(
#         model_id="M04", desc="DecisionTree",
#         pairs=instruments_pairs, delay=2, factors=factors, y_lbl="diff_return",
#         sig_method="binary", trn_win=3
#     ),
#     CMclrnKn(
#         model_id="M05", desc="KNeighbor",
#         pairs=instruments_pairs, delay=2, factors=factors, y_lbl="diff_return",
#         sig_method="binary", trn_win=3
#     ),
#     CMclrnAdaboost(
#         model_id="M06", desc="AdaBoost",
#         pairs=instruments_pairs, delay=2, factors=factors, y_lbl="diff_return",
#         sig_method="binary", trn_win=3
#     ),
#     CMclrnGb(
#         model_id="M07", desc="GradientBoosting",
#         pairs=instruments_pairs, delay=2, factors=factors, y_lbl="diff_return",
#         sig_method="binary", trn_win=3
#     ),
# ]


headers_mclrn = [(m.model_id, m.desc) for m in models_mclrn]

if __name__ == "__main__":
    print("-" * 32)
    print(headers_mclrn)
    print("-" * 32)
    print(f"size of models:{len(headers_mclrn)}")
