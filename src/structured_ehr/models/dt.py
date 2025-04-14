from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class DT():
    def __init__(self, **params):
        """params is a dict
        seed: int, random seed
        n_estimators: int, number of trees
        max_depth: int, depth of trees
        """
        task = params['task']
        seed = params['seed']
        max_depth = params['max_depth']
        self.task = task

        if task in ["mortality", "readmission"]:
            self.model = DecisionTreeClassifier(random_state=seed, max_depth=max_depth)
        elif task == "los":
            self.model = DecisionTreeRegressor(random_state=seed,  max_depth=max_depth)
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