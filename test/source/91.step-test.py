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
import utopia.MdVarSelect as vs
import utopia.MdModelSelection as ms
import utopia.statfun as sf
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

########################################################
# 2. load data
dataset_model = pd.read_csv("data/dataset_model.csv")
dataset_model["split"] = np.where(np.random.uniform(size=len(dataset_model)) < 0.7, "train", "test")

variables = list(set(dataset_model.columns) - set(vars_remove + [target, split]) - set(vars_rm_sp))
flag_train = np.array(dataset_model["split"] == "train")
flag_test = ~flag_train


# 3. split train and test
dataset_train_X = dataset_model[variables][flag_train]
dataset_train_y = np.array(dataset_model[target][flag_train])
np.save("dataset_train_y.npy", dataset_train_y)
dataset_test_X = dataset_model[variables][flag_test]
dataset_test_y = np.array(dataset_model[target][flag_test])
########################################################
# 4. eda & preprocess
###

# {v for k,v in dataset_train_X.dtypes.astype(str).to_dict().items()}
logger.info(dataset_train_X.shape)
obj_processor = mt.MdProcess(miss_set = miss_set, cat_proc = "onehot")
obj_processor.fit(dataset_train_X, dataset_train_y, save = "stat")

obj_processor.export("json")
_, var_desc = obj_processor.view()

dataset_train_proc = obj_processor.predict(dataset_train_X)
pd.DataFrame(dataset_train_proc).to_csv("data/dataset_preproc_train.csv", index=False)
pd.unique(pd.DataFrame(dataset_train_proc).dtypes)
#

########################################################

# 5. 变量选择
import numpy as np
dataset_train_y = np.load("data/dataset_train_y.npy")
dataset_train_proc = pd.read_csv("data/dataset_preproc_train.csv")
dataset_train_proc.shape

selector = {"IV":{"threshold": 0.01}, "Xgb":{"top":5}}
params = {"groups":2, "ntree":50, "learning_rate":0.1, "splitter":"optimal","max_depth":4}

obj_selector = mt.mv.MdVarSelectApi(target_type = "b", selector = selector, **params)
obj_selector.fit(X = dataset_train_proc, y = dataset_train_y, save = "stat")
obj_selector.varsele_get_perf()


########################################################
# 6. 变量变换
import numpy as np
dataset_train_y = np.load("dataset_train_y.npy")
dataset_train_proc = pd.read_csv("data/dataset_preproc_train.csv")

obj_transformer_woe = mt.vt.MdVarTransWoe(var_desc = var_desc)
obj_transformer_woe.fit(X = dataset_train_proc, y = dataset_train_y)

stat_vartrans_woe, _ = obj_transformer_woe.view()
obj_transformer_woe.export(lang = "json")

import codecs
with codecs.open('{}/{}'.format("model", 'model_vartrans_woe.json')) as f:
    model_vartrans_woe = json.load(f)

import json_tricks
with codecs.open('{}/{}'.format("model", 'model_vartrans_woe.json')) as f:
    model_vartrans_woe = json_tricks.load(f)

########################################################
# 7. 超参数调优
from hyperopt import hp
import numpy as np

dataset_train_y = np.load("data/dataset_train_y.npy")
dataset_train_proc = pd.read_csv("data/dataset_preproc_train.csv")

fspace = {"learning_rate":hp.loguniform("learning_rate", np.log(10**-3), 0),
          "min_child_weight":hp.choice('min_child_weight',[30, 50, 100, 150])}

kw = {"objective":"binary:logistic", "metric":"auc", "cv":5, "ntree":200, "early_stopping_rounds":5,"max_depth":3}

obj_optimizer = mt.MdHyperopt(model = "xgb", space = fspace,  minimize = False, target_type = "b", **kw)
obj_optimizer.fit(X = dataset_train_proc, y = dataset_train_y, max_evals=5)

obj_optimizer.save(save = "stat")
obj_optimizer.stat_hyperopt
obj_optimizer.best

########################################################
# 7. 模型训练与评估
# 列转行，新的列名为decile
import numpy as np
miss_set = [999]
dataset_train_y = np.load("dataset_train_y.npy")
dataset_train_proc = pd.read_csv("data/dataset_preproc_train.csv")
obj_predictor = mt.ms.MdModelSelectionXgb()
obj_predictor.fit(dataset_train_proc, dataset_train_y)

obj_predictor.save(save = "model")

