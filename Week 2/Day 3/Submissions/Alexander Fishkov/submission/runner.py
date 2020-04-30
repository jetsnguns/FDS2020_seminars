from abc import ABC, abstractmethod
import json
import time
from importlib import import_module

from dask_ml.model_selection import GridSearchCV
import dask.dataframe as dd
import joblib
import pandas as pd

from dask_ml.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV as SkGridSearchCV


# Hack for DaskML bug: impossible to get best params without refit=True
def _get_best_params_score(grigsearch):
    df_cv_res = pd.DataFrame(grigsearch.cv_results_)
    best_params = df_cv_res.loc[df_cv_res["rank_test_score"] == 1, "params"].iloc[0]
    mean_test_score = df_cv_res.loc[df_cv_res["rank_test_score"] == 1, "mean_test_score"].iloc[0]

    return df_cv_res, best_params, mean_test_score


class ExperimentRunner:
    """
    This class encapsulates data preparation and running the CV experiment.
    Assumed to be run in presence of a dask client (and possibly a cluster)
    """
    def __init__(self, input_path, target_col, model_desc):
        self.input_path = input_path
        self.target_col = target_col
        self.model_desc = model_desc

    def load_data(self):
        df = dd.read_hdf(self.input_path, '/data', mode='r')

        self.X = df.drop(self.target_col, axis=1)
        self.y = df[self.target_col]

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size=float(
                                                                                    self.model_desc.get("test_size",
                                                                                                        0.5)),
                                                                                random_state=int(
                                                                                    self.model_desc.get("seed", 31337)))

    def _create_model(self, **kwargs):
        module_name, class_name = self.model_desc["algorithm_name"].rsplit('.', 1)
        return getattr(import_module(module_name), class_name)(**kwargs)

    def run(self):
        self.load_data()
        self.split_data()

        # nulls = X_train.isnull().sum()
        # total_nulls = nulls.sum()
        # if total_nulls > 0:
        #    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #        print(nulls[nulls > 0], "total: ", total_nulls)

        reg = self._create_model()
        gs = GridSearchCV(reg, self.model_desc["params_grid"], cv=self.model_desc["num_folds"], n_jobs=-1, refit=False)

        start_time = time.monotonic()
        with joblib.parallel_backend("dask", scatter=[self.X_train, self.y_train]):
            gs.fit(self.X_train, self.y_train)
        finish_time = time.monotonic()
        gs_time = finish_time - start_time
        print("Searching for marameters for [{}]".format(self.model_desc["algorithm_name"]))
        print("GridSearchCV time: {}".format(gs_time))
        cv_res, best_params, gs_score = _get_best_params_score(gs)
        print("GridSearchCV score: {}".format(gs_score))
        print("Best params: {}".format(best_params))

        start_time = time.monotonic()
        regr_best = self._create_model(**best_params).fit(self.X_train, self.y_train)
        finish_time = time.monotonic()
        test_time = finish_time - start_time
        print("Final training time: {}".format(test_time))

        test_score = float(regr_best.score(self.X_test, self.y_test))

        print("Test score: {}".format(test_score))

        ans = {"algorithm_name": self.model_desc["algorithm_name"],
               "gs_time": gs_time,
               "gs_score": gs_score,
               "test_time": test_time,
               "test_score": test_score}

        if "output_path" in self.model_desc:
            with open(self.model_desc["output_path"], "w") as fp:
                json.dump(ans, fp)

        return ans


class PersistExperimentRunner(ExperimentRunner):
    """
    Persist all data in memory for compatibility with some algorithms.
    """
    def load_data(self):
        super().load_data()
        self.X = self.X.compute()
        self.y = self.y.compute()


class ArrayExperimentRunner(ExperimentRunner):
    """
    Convert data to dask arrays. Some dask ans sklearn algorithms do not support dask DataFrame.
    """
    def load_data(self):
        super().load_data()
        self.X = self.X.to_dask_array(lengths=True)
        self.y = self.y.to_dask_array(lengths=True)


runners = {
    "dask": ExperimentRunner,
    "dask_array": ArrayExperimentRunner,
    "sklearn": PersistExperimentRunner
}


def get_experiment_runner(path, target, desc):
    m = desc["mode"]
    return runners[m](input_path=path,
                      target_col=target,
                      model_desc=desc)
