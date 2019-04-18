import xgboost as xgb

# https://xgboost.readthedocs.io/en/latest/parameter.html
param_dist = {'objective':'binary:logistic', 'n_estimators':2, "metrics":"auc", "eval_metric":["auc"], "n_jobs": 3}

clf = xgb.XGBModel(**param_dist)

clf.get_params(deep=True)


from sklearn.datasets import load_iris
import xgboost as xgb
iris = load_iris()
DTrain = xgb.DMatrix(iris.data, iris.target)
x_parameters = {"max_depth":2}
result = xgb.cv(x_parameters, DTrain)

result.shape[0]

result.iloc[-1, 0]

result.iloc[-1].to_dict()


idx_name = [i for i in result.columns if i.startswith("test") and i.endswith("mean")][0]

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
X, y = load_iris(return_X_y=True)
X.columns
clf = LogisticRegression(random_state=0, solver='lbfgs',
                         multi_class='multinomial').fit(X, y)
clf.predict(X[:2, :])

import pandas as pd
df = pd.DataFrame(clf.coef_.T)
df["variable"] = list("abcd")
clf.intercept_
clf.predict_proba(X[:2, :])