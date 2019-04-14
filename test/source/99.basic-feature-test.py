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

"FX01_"[:5][-1]