from cBasic import CInstruPair

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

# config_factor = {
#     "lag": {
#         "args": (1, 2, 3, 4, 5)
#     },
#     "ewm": {
#         "args": (
#             (0.90, 0.60), (0.90, 0.30), (0.90, 0.20), (0.90, 0.10), (0.90, 0.05),
#             (0.60, 0.30), (0.60, 0.20), (0.60, 0.10), (0.60, 0.05), (0.60, 0.02),
#             (0.30, 0.20), (0.30, 0.10), (0.30, 0.05), (0.30, 0.02), (0.30, 0.01),
#         ),
#         "fix_base_date": "20160104",
#     },
#     "volatility": {
#         "args": (
#             (5, 3), (10, 3), (20, 3),
#             (5, 10), (10, 10), (20, 10),
#         )
#     },
#     "tnr": {  # trend to noise ratio
#         "args": (
#             (5, 3), (10, 3), (20, 3),
#             (5, 10), (10, 10), (20, 10),
#         )
#     },
#     "basisa": {
#         "args": (60, 120,)
#     },
#     "ctp": {
#         "args": (120,)
#     },
#     "cvp": {
#         "args": (120,)
#     },
#     "csp": {
#         "args": (120,)
#     },
#     "rsbr": {
#         "args": (10,)
#     },
#     "rslr": {
#         "args": (20,)
#     },
#     "skew": {
#         "args": (10,)
#     },
#     "mtm": {
#         "args": (1,)
#     },
#     "mtms": {
#         "args": (240,)
#     },
#     "tsa": {
#         "args": (120, 180)
#     },
#     "tsld": {
#         "args": (240,)
#     },
# }
#
# factors_lag = [f"LAG{_:02d}" for _ in config_factor["lag"]["args"]]
# factors_ewm = [f"F{int(fast * 100):02d}S{int(slow * 100):02d}" for (fast, slow) in config_factor["ewm"]["args"]]
# factors_vty = [f"VTY{_:02d}K{k:02d}" for (_, k) in config_factor["volatility"]["args"]]
# factors_tnr = [f"TNR{_:02d}K{k:02d}" for (_, k) in config_factor["tnr"]["args"]]
#
# factors_basisa = [f"BASISA{_:03d}" for _ in config_factor["basisa"]["args"]]
# factors_ctp = [f"CTP{_:03d}" for _ in config_factor["ctp"]["args"]]
# factors_cvp = [f"CVP{_:03d}" for _ in config_factor["csp"]["args"]]
# factors_csp = [f"CSP{_:03d}" for _ in config_factor["cvp"]["args"]]
# factors_rsbr = [f"RSBR{_:03d}" for _ in config_factor["rsbr"]["args"]]
# factors_rslr = [f"RSLR{_:03d}" for _ in config_factor["rslr"]["args"]]
# factors_skew = [f"SKEW{_:03d}" for _ in config_factor["skew"]["args"]]
# factors_mtm = [f"MTM" for _ in config_factor["mtm"]["args"]]
# factors_mtms = [f"MTMS{_:03d}" for _ in config_factor["mtms"]["args"]]
# factors_tsa = [f"TSA{_:03d}" for _ in config_factor["tsa"]["args"]]
# factors_tsld = [f"TSLD{_:03d}" for _ in config_factor["tsld"]["args"]]
#
# factors = factors_lag + factors_ewm + factors_vty + factors_tnr + \
#           factors_basisa + \
#           factors_ctp + factors_cvp + factors_csp + \
#           factors_rsbr + factors_rslr + \
#           factors_skew + \
#           factors_mtm + factors_mtms + \
#           factors_tsa + factors_tsld
#
pairs_qty = len(instruments_pairs)
# factors_qty = len(factors)
# diff_ret_delays = [1, 2]
# cost_rate = 0e-4

if __name__ == "__main__":
    print(f"quantity of pairs   = {pairs_qty}")
    # print(f"quantity of factors = {factors_qty}")
