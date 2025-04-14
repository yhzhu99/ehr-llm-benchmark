from catboost import CatBoostClassifier, CatBoostRegressor


class CatBoost():
    def __init__(self, **params):
        """params is a dict
        seed: int, random seed
        n_estimators: int, number of trees
        learning_rate: float, learning rate
        max_depth: int, depth of trees
        """
        task = params['task']
        seed = params['seed']
        learning_rate = params['learning_rate']
        n_estimators = params['n_estimators']
        max_depth = params['max_depth']
        self.task = task

        if task in ["mortality", "readmission"]:
            self.model = CatBoostClassifier(random_state=seed, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, verbose=None, silent=True, allow_writing_files=False, loss_function="CrossEntropy")
        elif task == "los":
            self.model = CatBoostRegressor(random_state=seed, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, verbose=None, silent=True, allow_writing_files=False, loss_function="RMSE")
        else:
            raise ValueError("Task must be either 'outcome', 'readmission' or 'los'.")

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        if self.task in ["mortality", "readmission"]:
            return self.model.predict_proba(x)[:, 1]
        elif self.task == "los":
            return self.model.predict(x)
        else:
            raise ValueError("Task must be either 'outcome', 'readmission' or 'los'.")