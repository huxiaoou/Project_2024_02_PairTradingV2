from project_config import instruments_pairs, factors
from cMclrn import (
    CMclrnModel,
    CMclrnBatchRidge,
    CMclrnBatchLogistic,
    CMclrnBatchMlp,
    CMclrnBatchSvm,
    CMclrnBatchDt,
    CMclrnBatchKn,
    CMclrnBatchAb,
    CMclrnBatchGb,
)
from cEvaluations import get_top_factors_for_instruments_pairs
from project_setup import evaluations_dir_quick

# --- init
delays = [2]
trn_wins = [3, 6, 12]
top = 5
ridge_alphas = [(0.01, 0.1, 1.0, 10.0, 100)]
logistic_cs = [7]
mlp_args = {
    "M0": (10, 10, 10),
    "M1": (50, 50, 50),
    "M2": (100, 100, 100),
    "M3": (10, 10, 10, 10, 10),
    "M4": (50, 50, 50, 50, 50),
    "M5": (100, 100, 100, 100, 100),
}
svm_args = {
    "M0": (0.1, 2),
    "M1": (0.1, 3),
    "M2": (0.1, 4),
    "M3": (1.0, 2),
    "M4": (1.0, 3),
    "M5": (1.0, 4),
    "M6": (10.0, 2),
    "M7": (10.0, 3),
    "M8": (10.0, 4),
}
dt_args = {
    "M0": 2,
    "M1": 3,
    "M2": 5,
    "M3": None,
}
kn_args = {
    "M0": (5, "uniform", 1),
    "M1": (10, "uniform", 1),
    "M2": (20, "uniform", 1),
    "M3": (5, "distance", 1),
    "M4": (10, "distance", 1),
    "M5": (20, "distance", 1),
    "M6": (5, "uniform", 2),
    "M7": (10, "uniform", 2),
    "M8": (20, "uniform", 2),
    "M9": (5, "distance", 2),
    "MX": (10, "distance", 2),
    "MA": (20, "distance", 2),
}
ab_args = {
    "M0": (10, 0.2),
    "M1": (50, 0.2),
    "M2": (100, 0.2),
    "M3": (10, 1.0),
    "M4": (50, 1.0),
    "M5": (100, 1.0),
    "M6": (10, 2),
    "M7": (50, 2),
    "M8": (100, 2),
}
gb_args = {
    "M0": (50, 0.05),
    "M1": (100, 0.05),
    "M2": (200, 0.05),
    "M3": (50, 0.1),
    "M4": (100, 0.1),
    "M5": (200, 0.1),
    "M6": (50, 0.5),
    "M7": (100, 0.5),
    "M8": (200, 0.5),
}
model_prototypes = ["Ab", "Dt", "Gb", "Kn", "Logistic", "Mlp", "Ridge", "Svm"]
models_mclrn: list[CMclrnModel] = []

# --- load top factors
top_factors = get_top_factors_for_instruments_pairs(top=top, evaluations_dir=evaluations_dir_quick)

# --- add Ridge
batch_generator = CMclrnBatchRidge(
    ridge_alphas=ridge_alphas,
    instruments_pairs=instruments_pairs,
    delays=delays,
    trn_wins=trn_wins,
    all_factors=factors,
    top_factors=top_factors,
)
models_mclrn += batch_generator.gen_batch()

# --- add Logistic
batch_generator = CMclrnBatchLogistic(
    logistic_cs=logistic_cs,
    instruments_pairs=instruments_pairs,
    delays=delays,
    trn_wins=trn_wins,
    all_factors=factors,
    top_factors=top_factors,
)
models_mclrn += batch_generator.gen_batch()

# --- add MultiLayerPerception
batch_generator = CMclrnBatchMlp(
    mlp_args=mlp_args,
    instruments_pairs=instruments_pairs,
    delays=delays,
    trn_wins=trn_wins,
    all_factors=factors,
    top_factors=top_factors,
)
models_mclrn += batch_generator.gen_batch()

# --- add SupportVectorMachine
batch_generator = CMclrnBatchSvm(
    svm_args=svm_args,
    instruments_pairs=instruments_pairs,
    delays=delays,
    trn_wins=trn_wins,
    all_factors=factors,
    top_factors=top_factors,
)
models_mclrn += batch_generator.gen_batch()

# --- add DecisionTree
batch_generator = CMclrnBatchDt(
    dt_args=dt_args,
    instruments_pairs=instruments_pairs,
    delays=delays,
    trn_wins=trn_wins,
    all_factors=factors,
    top_factors=top_factors,
)
models_mclrn += batch_generator.gen_batch()

# --- add KNeighbor
batch_generator = CMclrnBatchKn(
    kn_args=kn_args,
    instruments_pairs=instruments_pairs,
    delays=delays,
    trn_wins=trn_wins,
    all_factors=factors,
    top_factors=top_factors,
)
models_mclrn += batch_generator.gen_batch()

# --- add Adaboost
batch_generator = CMclrnBatchAb(
    ab_args=ab_args,
    instruments_pairs=instruments_pairs,
    delays=delays,
    trn_wins=trn_wins,
    all_factors=factors,
    top_factors=top_factors,
)
models_mclrn += batch_generator.gen_batch()

# --- add GradientBoosting
batch_generator = CMclrnBatchGb(
    gb_args=gb_args,
    instruments_pairs=instruments_pairs,
    delays=delays,
    trn_wins=trn_wins,
    all_factors=factors,
    top_factors=top_factors,
)
models_mclrn += batch_generator.gen_batch()

headers_mclrn = [(m.model_id, m.desc) for m in models_mclrn]

if __name__ == "__main__":
    print("-" * 32)
    print(headers_mclrn)
    print("-" * 32)
    print(f"size of models:{len(headers_mclrn)}")
