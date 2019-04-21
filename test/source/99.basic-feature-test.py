# 1 基本配置项
## 1.1 导基本包
import sys
import os
import configparser
import logging
import json
import re

import numpy as np
import pandas as pd
import datetime

## 1.2 配置外部包和数据路径
tool_path = '/Users/long.li/Documents/work/project/data-analytics-utopia'
sys.path.append(tool_path)

## 1.3 加载分析通用工具包(环境路径上一步添加)
import utopia as mt
# import source.MdVarSelect as vs
from utopia import sf
import utopia.tool as tf

## 1.4 加载项目配置相关项
conf = configparser.ConfigParser()
conf.read('conf/config.ini', encoding="utf-8")
conf_ssh = tf.pub_geti_from_conf(conf, 'ssh')
conf_db = tf.pub_geti_from_conf(conf, 'postgre')

## 1.5 配置日志项
log_theme = "tool-test"
mt.set_logger(postfix=log_theme, logerr="error", loginfo="run", path='log', logname=None)
logger = logging.getLogger('root')

## 1.6 其他分析相关配置
### https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
target = 'y'
miss_set = [999]
split = "split"
vars_remove = []
vars_rm_sp = []

# xgb test
dataset_train_y = np.load("dataset_train_y.npy")
dataset_train_proc = pd.read_csv("data/dataset_preproc_train.csv")

import xgboost as xgb
DTrain = xgb.DMatrix(dataset_train_proc, dataset_train_y)
x_parameters = {"max_depth":2,"objective":"binary:logistic", "eval_metric":"auc"}
result = xgb.cv(x_parameters, DTrain, nfold = 5, num_boost_round = 100, early_stopping_rounds = 5)

result.iloc[-1, 0],  result.iloc[-1].to_dict()


# hyperopt
from hyperopt import fmin, tpe, Trials, STATUS_OK, hp, space_eval
import numpy as np

dataset_train_y = np.load("dataset_train_y.npy")
dataset_train_proc = pd.read_csv("data/dataset_preproc_train.csv")

fspace = {"learning_rate":hp.loguniform("learning_rate", np.log(10**-5), 0),
          "max_depth":hp.choice("max_depth", [2,3,4])}

kw = {"objective":"binary:logistic", "metric":"auc", "cv":5, "ntree":200, "early_stopping_rounds":5}

obj_optimizer = mt.MdHyperopt(model = "xgb", space = fspace, minimize = False, target_type = "b", **kw)

objective = obj_optimizer.hyperopt_objfunc_factory(X = dataset_train_proc, y = dataset_train_y, **obj_optimizer.kw)
trials = Trials()
best = fmin(objective, obj_optimizer.space, algo=tpe.suggest, max_evals=5, trials=trials)

for trial in trials.trials:
    print(trial['misc']['vals'], trial['misc']['idxs'])


dft1 = pd.DataFrame([{k:v[0] for k, v in t['misc']['vals'].items()} for t in trials.trials])

dft2 = pd.DataFrame([t['result']['eval'] for t in trials.trials])

pd.concat([dft1, dft2], axis=1,ignore_index=True)

dict(dict(best, a= 3), **space_eval(fspace, best))

ns = 0
vars_p_round = 8
vars_all = list(range(100))
while ns < len(vars_all):
    print(vars_all[ns:(ns + vars_p_round)])
    ns = ns + vars_p_round


a = [1]
a.pop()
a == []

import pandas as pd
df = pd.DataFrame({"a":[1,2,3], "b":[4,5,6]})
df.__len__()

a = set()
type(a) is list or type(a) is  set

# test append operate
import codecs
import json
with codecs.open('{}/{}'.format("log", 'stat_hyperopt.log'), 'w', 'utf-8') as f:
    pass

# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/scorer.py

# feature test
import numpy as np
dataset_train_y = np.load("data/dataset_train_y.npy")
dataset_train_proc = pd.read_csv("data/dataset_preproc_train.csv")
dataset_train_proc.shape, dataset_train_y.shape

import sklearn.linear_model as glm
params = {"fit_intercept" :True, "Cs":[0.1], "n_jobs":3, "cv":5, "solver":"saga", "penalty":"l1", "max_iter":100}
predictor_t = glm.LogisticRegressionCV(**params)
predictor_t.fit(dataset_train_proc, dataset_train_y)

result2 = predictor_t.predict_proba(dataset_train_proc)

np.mean(result2[:,1])

# 222
from scipy.special import expit
import codecs
import json
len(predictor_t.coef_[0])
predictor_t.intercept_
coef = predictor_t.coef_
np.append(coef, predictor_t.intercept_)

result = np.matmul(np.array(dataset_train_proc), predictor_t.coef_.T) + predictor_t.intercept_
np.mean(expit(result))

with codecs.open("model/model_lr_t.json", 'w', 'utf-8') as f:
    json.dump(list(coef[0]), f, ensure_ascii=False)

with codecs.open("model/model_lr_t.json") as f:
    coef = json.load(f)

coef = np.array(coef).reshape(-1,1)
result = np.matmul(np.array(dataset_train_proc), coef) + predictor_t.intercept_
np.mean(expit(result))

result = np.array(0)
coef = dict(zip(dataset_train_proc.columns, predictor_t.coef_[0]))
for k in dataset_train_proc.columns:
    result = result + dataset_train_proc[k] * coef[k]

np.mean(expit(result + predictor_t.intercept_))



X = dataset_train_proc
X = {k: np.array(X[k]) for k in set(obj_predictor.model_coef.keys()) & set(X.columns)}
X["intercept"] = 1  # 改变了原数据集
result = np.array(0)
for k, v in obj_predictor.model_coef.items():
    result = X[k] * v

X[k] * v

if self.target_type == "b":
    result = expit(result)
