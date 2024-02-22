from cBasic import CInstruPair, CCfgFactor, CCfgFactorEWM, CCfgFactorMA

instruments_pairs: list[CInstruPair] = [
    CInstruPair(instru_a="A.DCE", instru_b="M.DCE"),
    CInstruPair(instru_a="A.DCE", instru_b="Y.DCE"),
    CInstruPair(instru_a="M.DCE", instru_b="Y.DCE"),
    CInstruPair(instru_a="M.DCE", instru_b="RM.CZC"),
    CInstruPair(instru_a="OI.CZC", instru_b="P.DCE"),
    CInstruPair(instru_a="OI.CZC", instru_b="Y.DCE"),
    CInstruPair(instru_a="P.DCE", instru_b="Y.DCE"),
    CInstruPair(instru_a="AG.SHF", instru_b="AU.SHF"),
    CInstruPair(instru_a="AL.SHF", instru_b="ZN.SHF"),
    CInstruPair(instru_a="CU.SHF", instru_b="ZN.SHF"),
    CInstruPair(instru_a="CU.SHF", instru_b="AL.SHF"),
    CInstruPair(instru_a="HC.SHF", instru_b="RB.SHF"),
    CInstruPair(instru_a="I.DCE", instru_b="RB.SHF"),
    CInstruPair(instru_a="JM.DCE", instru_b="J.DCE"),
    CInstruPair(instru_a="JM.DCE", instru_b="I.DCE"),
    CInstruPair(instru_a="SF.CZC", instru_b="SM.CZC"),
    CInstruPair(instru_a="C.DCE", instru_b="CS.DCE"),
    CInstruPair(instru_a="BU.SHF", instru_b="TA.CZC"),
    CInstruPair(instru_a="L.DCE", instru_b="PP.DCE"),
    CInstruPair(instru_a="L.DCE", instru_b="V.DCE"),
    CInstruPair(instru_a="MA.CZC", instru_b="V.DCE"),
    CInstruPair(instru_a="BU.SHF", instru_b="FU.SHF"),  # "FU.SHF" since 20180716
]

config_factor: dict[str, CCfgFactor] = {
    "lag": CCfgFactor(factor_class="lag", args=(0, 1, 2, 3, 5)),
    "sum": CCfgFactor(factor_class="sum", args=(2, 3, 5, 10, 15, 20)),
    "ewm": CCfgFactorEWM(
        args=(
            (0.90, 0.60),
            (0.90, 0.30),
            (0.90, 0.20),
            (0.90, 0.10),
            (0.90, 0.05),
            (0.60, 0.30),
            (0.60, 0.20),
            (0.60, 0.10),
            (0.60, 0.05),
            (0.60, 0.02),
            (0.30, 0.20),
            (0.30, 0.10),
            (0.30, 0.05),
            (0.30, 0.02),
            (0.30, 0.01),
        ),
        fixed_bgn_date="20160104",
    ),
    "vol": CCfgFactor(factor_class="vol", args=(2, 3, 5, 10, 15, 20)),
    "tnr": CCfgFactor(factor_class="tnr", args=(2, 3, 5, 10, 15, 20)),  # trend to noise ratio
    "basisa": CCfgFactorMA(factor_class="basisa", args=(60, 120), win_mov_ave=5),
    "ctp": CCfgFactorMA(factor_class="ctp", args=(120,), win_mov_ave=5),
    "cvp": CCfgFactorMA(factor_class="cvp", args=(120,), win_mov_ave=5),
    "csp": CCfgFactorMA(factor_class="csp", args=(120,), win_mov_ave=5),
    "rsbr": CCfgFactorMA(factor_class="rsbr", args=(10,), win_mov_ave=5),
    "rslr": CCfgFactorMA(factor_class="rslr", args=(20,), win_mov_ave=5),
    "skew": CCfgFactorMA(factor_class="skew", args=(10,), win_mov_ave=5),
    "mtms": CCfgFactorMA(factor_class="mtms", args=(240,), win_mov_ave=5),
    "tsa": CCfgFactorMA(factor_class="tsa", args=(120, 180), win_mov_ave=5),
    "tsld": CCfgFactorMA(factor_class="tsld", args=(240,), win_mov_ave=5),
}

factors = []
for _, v in config_factor.items():
    factors += v.get_factors()

pairs_qty = len(instruments_pairs)
factors_qty = len(factors)
diff_ret_delays = [1, 2]
cost_rate_quick = 0e-4
cost_rate_mclrn = 5e-4

if __name__ == "__main__":
    print(f"quantity of pairs   = {pairs_qty}")
    print(f"quantity of factors = {factors_qty}")
    print(f"| {'SN':>3s} | {'Factor':<18s} |")
    for i, f in enumerate(factors):
        print(f"| {i:>3d} | {f:<18s} |")
