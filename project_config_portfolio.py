g0 = [
    "JM.DCE_I.DCE-T2-F50-W03-Ridge-A05",
    "I.DCE_RB.SHF-T2-F50-W03-Ridge-A05",
    "JM.DCE_J.DCE-T2-F50-W03-Ridge-A05",
    "SF.CZC_SM.CZC-T2-F50-W03-Ridge-A05",
    "MA.CZC_V.DCE-T2-F50-W03-Ridge-A05",
]
g1 = [
    "I.DCE_RB.SHF-T2-F50-W03-Svm-M0",
    "A.DCE_M.DCE-T2-F50-W03-Svm-M0",
    "JM.DCE_I.DCE-T2-F50-W03-Svm-M3",
]
g2 = [
    "A.DCE_M.DCE-T2-F50-W03-Logistic-CS07",
    "I.DCE_RB.SHF-T2-F50-W03-Logistic-CS07",
    "MA.CZC_V.DCE-T2-F50-W03-Logistic-CS07",
    "P.DCE_Y.DCE-T2-F50-W03-Logistic-CS07",
]
portfolios: dict[str, list[str]] = {
    "P00": g0,
    "P01": g1,
    "P02": g2,
    "P03": g0 + g1,
    "P04": g0 + g1 + g2,
}
