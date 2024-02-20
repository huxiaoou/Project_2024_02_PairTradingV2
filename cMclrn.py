import os
import datetime as dt
import numpy as np
import scipy.stats as sps
import skops.io as sio
import pandas as pd
import multiprocessing as mp
import itertools as ittl
from rich.progress import track
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from husfort.qcalendar import CCalendar
from husfort.qutility import check_and_mkdir, SFG, SFY
from husfort.qsqlite import CQuickSqliteLib, CLib1Tab1, CTable
from cBasic import CInstruPair
from cRegroups import CLibRegroups


class CLibPredictions(CQuickSqliteLib):
    def __init__(self, model_id: str, lib_save_dir: str):
        self.model_id = model_id
        lib_name = f"{self.model_id}.db"
        super().__init__(lib_name, lib_save_dir)

    def get_lib_struct(self) -> CLib1Tab1:
        return CLib1Tab1(
            lib_name=self.lib_name,
            table=CTable(
                {
                    "table_name": "predictions",
                    "primary_keys": {"trade_date": "TEXT"},
                    "value_columns": {"value": "REAL"},
                }
            )
        )


class CMclrnModel(object):
    def __init__(self, model_id: str, desc: str,
                 instru_pair: CInstruPair, delay: int, factors: list[str], y_lbl: str,
                 sig_method: str,
                 trn_win: int, days_per_month: int = 20, normalize_alpha: float = 0.05, random_state: int = 0):
        self.model_id, self.desc = model_id, desc
        self.instru_pair = instru_pair
        self.delay = delay
        self.factors, self.y_lbl = factors, y_lbl
        self.train_win, self.days_per_month = trn_win, days_per_month
        self.random_state = random_state

        # ---
        if sig_method not in ["binary", "continuous"]:
            print(f"{dt.datetime.now()} [ERR] sig method = {SFY(sig_method)} is illegal")
            raise ValueError
        self.sig_method = sig_method

        # ---
        if (normalize_alpha > 0.5) or (normalize_alpha <= 0):
            print(f"{dt.datetime.now()} [ERR] alpha = {normalize_alpha}")
            raise ValueError
        self.normalize_alpha = normalize_alpha

        # ---
        self.core_data: pd.DataFrame = pd.DataFrame()
        self.prototype_model = None
        self.fitted_estimator = None

    @staticmethod
    def get_iter_dates(bgn_date: str, stp_date: str,
                       calendar: CCalendar, shifts: list[int] = None) -> tuple[list[str], tuple]:
        iter_dates = calendar.get_iter_list(bgn_date, stp_date)
        shift_dates = tuple(calendar.shift_iter_dates(iter_dates, s) for s in shifts)
        return iter_dates, shift_dates

    @staticmethod
    def is_model_update_date(this_date: str, next_date: str) -> bool:
        return this_date[0:6] != next_date[0:6]

    @staticmethod
    def is_2_days_to_next_month(this_date: str, next_date_1: str, next_date_2: str) -> bool:
        return (this_date[0:6] == next_date_1[0:6]) and (this_date[0:6] != next_date_2[0:6])

    @staticmethod
    def is_last_iter_date(trade_date: str, iter_dates: list[str]) -> bool:
        return trade_date == iter_dates[-1]

    @property
    def train_win_size(self) -> int:
        return self.train_win * self.days_per_month

    def _get_train_df(self, end_date: str) -> pd.DataFrame:
        return self.core_data.truncate(after=end_date).tail(n=self.train_win_size)

    def _get_predict_df(self, bgn_date: str, end_date: str) -> pd.DataFrame:
        return self.core_data.truncate(before=bgn_date, after=end_date)[self.factors]

    @staticmethod
    def _fillna(df: pd.DataFrame, aver: pd.Series) -> pd.DataFrame:
        return df.fillna(aver).fillna(0)

    @staticmethod
    def _get_rng(df_fill_nan: pd.DataFrame, alpha: float) -> np.ndarray:
        df_none_inf = df_fill_nan.replace(to_replace=[np.inf, -np.inf], value=np.nan).dropna(axis=0, how="any")
        qu = np.quantile(df_none_inf, 1 - alpha / 2, axis=0)
        ql = np.quantile(df_none_inf, alpha / 2, axis=0)
        rng = (qu - ql) / 2
        return rng

    @staticmethod
    def _replace_inf(df_fill_nan: pd.DataFrame, ub: np.ndarray, lb: np.ndarray) -> pd.DataFrame:
        df_fill_nan = df_fill_nan[df_fill_nan < ub].fillna(ub)
        df_fill_nan = df_fill_nan[df_fill_nan > lb].fillna(lb)
        return df_fill_nan

    @staticmethod
    def _cal_sd(rng: np.ndarray, alpha: float):
        return rng / sps.norm.ppf(1 - alpha / 2)

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        aver = df.median()
        df_fill_nan = self._fillna(df, aver=aver)

        rng = self._get_rng(df_fill_nan, alpha=self.normalize_alpha)
        ub, lb = aver + rng, aver - rng
        df_fill_inf = self._replace_inf(df_fill_nan, ub=ub, lb=lb)

        sd = self._cal_sd(rng, alpha=self.normalize_alpha)
        df_norm: pd.DataFrame = (df_fill_inf - aver) / sd
        df_norm.fillna(value=0, inplace=True)
        return df_norm

    def _transform_y(self, y_srs: pd.Series) -> pd.Series:
        if self.sig_method == "binary":
            return y_srs.map(lambda z: 1 if z >= 0 else 0)
        else:  # self.sig_method == "continuous":
            return y_srs

    def _norm_and_trans(self, train_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = self._normalize(train_df[self.factors])
        y = self._transform_y(train_df[self.y_lbl])
        return x.values, y.values

    def _fit(self, x: np.ndarray, y: np.ndarray):
        self.fitted_estimator = self.prototype_model.fit(x, y)
        return 0

    def _apply_model(self, predict_df: pd.DataFrame) -> pd.DataFrame:
        x = self._normalize(predict_df[self.factors])
        p = self.fitted_estimator.predict(X=x.values)
        pred = pd.DataFrame(data={"pred": p}, index=predict_df.index)
        return pred

    def _save_model(self, month_id: str, models_dir: str):
        model_file = f"{month_id}.{self.model_id}.skops"
        month_dir = os.path.join(models_dir, month_id)
        model_path = os.path.join(month_dir, model_file)
        check_and_mkdir(month_dir)
        sio.dump(self.fitted_estimator, model_path)
        return 0

    def _load_model(self, month_id: str, models_dir: str) -> bool:
        model_file = f"{month_id}.{self.model_id}.skops"
        month_dir = os.path.join(models_dir, month_id)
        model_path = os.path.join(month_dir, model_file)
        if os.path.exists(model_path):
            self.fitted_estimator = sio.load(model_path, trusted=True)
            return True
        else:
            # print(f"{dt.datetime.now()} [WRN] failed to load model for {SFY(self.model_id)} at {SFY(month_id)}")
            return False

    def load_data(self, bgn_date: str, stp_date: str, regroups_dir: str):
        lib_reader = CLibRegroups(self.instru_pair, self.delay, regroups_dir).get_lib_reader()
        df = lib_reader.read_by_conditions(conditions=[
            ("trade_date", ">=", bgn_date),
            ("trade_date", "<", stp_date),
        ], value_columns=["trade_date", "factor", "value"])
        self.core_data = pd.pivot_table(data=df, index="trade_date", columns="factor", values="value")
        self.core_data = self.core_data[self.factors + [self.y_lbl]]
        return 0

    def train(self, bgn_date: str, stp_date: str, calendar: CCalendar, models_dir: str):
        iter_dates, (next_dates,) = self.get_iter_dates(bgn_date, stp_date, calendar, shifts=[1])
        seq, description = list(zip(iter_dates, next_dates)), f"{SFY('Training  ')} for {self.desc:.<60s}"
        for (this_date, next_date) in track(seq, description=description):
            if self.is_model_update_date(this_date, next_date):
                train_df = self._get_train_df(end_date=this_date)
                x, y = self._norm_and_trans(train_df)
                self._fit(x, y)
                self._save_model(month_id=this_date[0:6], models_dir=models_dir)
        return 0

    def predict(self, bgn_date, stp_date, calendar: CCalendar, models_dir: str) -> pd.DataFrame:
        iter_dates, (next_dates_1, next_dates_2) = self.get_iter_dates(bgn_date, stp_date, calendar, shifts=[1, 2])
        month_dates: list[str] = []
        sub_predictions: list[pd.DataFrame] = []
        seq, description = (list(zip(iter_dates, next_dates_1, next_dates_2)),
                            f"{SFG('Prediction')} for {self.desc:.<60s}")
        for (this_date, next_date_1, next_date_2) in track(seq, description):
            month_dates.append(this_date)
            if (self.is_2_days_to_next_month(this_date, next_date_1, next_date_2)
                    or self.is_last_iter_date(this_date, iter_dates)):
                this_month = next_date_1[0:6]
                prev_month = calendar.get_next_month(this_month, s=-1)
                if self._load_model(month_id=prev_month, models_dir=models_dir):
                    pred_bgn_date, pred_end_date = month_dates[0], this_date
                    predict_input_df = self._get_predict_df(bgn_date=pred_bgn_date, end_date=pred_end_date)
                    sub_predictions.append(self._apply_model(predict_input_df))
                month_dates.clear()  # prepare for next month
        predictions = pd.concat(sub_predictions, axis=0, ignore_index=False).reset_index()
        return predictions

    def save_prediction(self, df: pd.DataFrame, run_mode: str, predictions_dir: str):
        lib_pred_writer = CLibPredictions(self.model_id, predictions_dir).get_lib_writer(run_mode)
        lib_pred_writer.update(update_df=df, using_index=False)
        lib_pred_writer.commit()
        lib_pred_writer.close()
        return 0

    def main(self, run_mode: str, bgn_date: str, stp_date: str, calendar: CCalendar,
             regroups_dir: str, models_dir: str, predictions_dir: str):
        self.load_data(bgn_date, stp_date, regroups_dir)
        self.train(bgn_date, stp_date, calendar, models_dir)
        predictions = self.predict(bgn_date, stp_date, calendar, models_dir)
        self.save_prediction(predictions, run_mode, predictions_dir)
        return 0


class CMclrnRidge(CMclrnModel):
    def __init__(self, alphas: tuple, fit_intercept: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.prototype_model = RidgeCV(alphas=self.alphas, fit_intercept=self.fit_intercept)


class CMclrnLogistic(CMclrnModel):
    def __init__(self, cv: int,
                 cs: int = 10, fit_intercept: bool = False, penalty: str = "l2",
                 max_iter: int = 5000, **kwargs):
        super().__init__(**kwargs)
        self.cv = cv
        self.cs = cs
        self.fit_intercept = fit_intercept
        self.penalty = penalty
        self.max_iter = max_iter
        self.prototype_model = LogisticRegressionCV(
            Cs=self.cs, cv=self.cv,
            fit_intercept=self.fit_intercept, penalty=self.penalty,
            max_iter=self.max_iter, random_state=self.random_state)


class CMclrnMlp(CMclrnModel):
    def __init__(self, hidden_layer_size: tuple[int], max_iter: int = 5000, **kwargs):
        self.hidden_layer_size = hidden_layer_size
        self.max_iter = max_iter
        self.prototype_model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_size,
            max_iter=self.max_iter, random_state=self.random_state)
        super().__init__(**kwargs)


class CMclrnSvc(CMclrnModel):
    def __init__(self, c: float, degree: int, **kwargs):
        self.c = c
        self.degree = degree
        self.prototype_model = SVC(C=self.c, degree=self.degree, random_state=self.random_state)
        super().__init__(**kwargs)


class CMclrnDt(CMclrnModel):
    def __init__(self, max_depth: int, **kwargs):
        self.max_depth = max_depth
        self.prototype_model = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
        super().__init__(**kwargs)


class CMclrnKn(CMclrnModel):
    def __init__(self, n_neighbors: int, weights: str = "distance", p: int = 1, **kwargs):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self.prototype_model = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights, p=self.p)
        super().__init__(**kwargs)


class CMclrnAdaboost(CMclrnModel):
    def __init__(self, n_estimators: int, learning_rate: float, **kwargs):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.prototype_model = AdaBoostClassifier(
            n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            random_state=self.random_state)
        super().__init__(**kwargs)


class CMclrnGb(CMclrnModel):
    def __init__(self, n_estimators: int, learning_rate: float, **kwargs):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.prototype_model = GradientBoostingClassifier(
            n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            random_state=self.random_state)
        super().__init__(**kwargs)


def cal_mclrn_train_and_predict(call_multiprocess: bool, models_mclrn: list[CMclrnModel], proc_qty: int = None,
                                **kwargs):
    if call_multiprocess:
        pool = mp.Pool(processes=proc_qty) if proc_qty else mp.Pool()
        for m in models_mclrn:
            pool.apply_async(m.main, kwds=kwargs)
        pool.close()
        pool.join()
    else:
        for m in models_mclrn:
            m.main(**kwargs)
    return 0


class CMclrnBatch(object):
    def __init__(self, instruments_pairs: list[CInstruPair], delays: list[int], trn_wins: list[int],
                 all_factors: list[str],
                 top_factors: dict[tuple[CInstruPair, int], list[str]]):
        self.instruments_pairs = instruments_pairs
        self.delays = delays
        self.trn_wins = trn_wins
        self.all_factors = all_factors
        self.top_factors = top_factors

    def append_batch(self, models_mclrn: list[CMclrnModel]):
        for instru_pair, delay, trn_win in ittl.product(self.instruments_pairs, self.delays, self.trn_wins):
            sel_factors = self.top_factors[(instru_pair, delay)]
            for factors in [self.all_factors, sel_factors]:
                self.core(models_mclrn=models_mclrn,
                          instru_pair=instru_pair, delay=delay, factors=factors, trn_win=trn_win)
        return 0

    def core(self, models_mclrn: list[CMclrnModel],
             instru_pair: CInstruPair, delay: int, trn_win: int, factors: list[str]):
        pass

    @staticmethod
    def get_fix_id(instru_pair: CInstruPair, delay: int, trn_win: int, factors: list[str]) -> str:
        return f"P-{instru_pair}-T{delay}-F{len(factors):02d}-W{trn_win:02d}"

    @staticmethod
    def get_model_id(sn: int) -> str:
        return f"M{sn:04d}"


class CMclrnBatchRidge(CMclrnBatch):
    def __init__(self, ridge_alphas: list[tuple], **kwargs):
        self.ridge_alphas = ridge_alphas
        super().__init__(**kwargs)

    def core(self, models_mclrn: list[CMclrnModel],
             instru_pair: CInstruPair, delay: int, trn_win: int, factors: list[str]):
        model_sn: int = len(models_mclrn)
        for alphas in self.ridge_alphas:
            model_id = self.get_model_id(sn=model_sn)
            desc = self.get_fix_id(instru_pair, delay, trn_win, factors) + f"-Ridge-A{len(alphas)}"
            m = CMclrnRidge(
                alphas=alphas, model_id=model_id, desc=desc,
                instru_pair=instru_pair, delay=delay, factors=factors, y_lbl="diff_return",
                sig_method="continuous", trn_win=trn_win,
            )
            models_mclrn.append(m)
            model_sn += 1
        return 0


class CMclrnBatchLogistic(CMclrnBatch):
    def __init__(self, cvs: list[int], **kwargs):
        self.cvs = cvs
        super().__init__(**kwargs)

    def core(self, models_mclrn: list[CMclrnModel],
             instru_pair: CInstruPair, delay: int, trn_win: int, factors: list[str]):
        model_sn: int = len(models_mclrn)
        for cv in self.cvs:
            model_id = self.get_model_id(sn=model_sn)
            desc = self.get_fix_id(instru_pair, delay, trn_win, factors) + f"-Logistic-CV{cv:02d}"
            m = CMclrnLogistic(
                cv=cv,
                model_id=model_id, desc=desc,
                instru_pair=instru_pair, delay=delay, factors=factors, y_lbl="diff_return",
                sig_method="binary", trn_win=trn_win,
            )
            models_mclrn.append(m)
            model_sn += 1
        return 0
